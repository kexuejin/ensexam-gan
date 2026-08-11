#!/usr/bin/env python3
"""Audit frozen primary mb/ms support separation on train-only data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.audit_dual_input_support_separation import (  # noqa: E402
    AuditError,
    auc_rank,
    balanced_indices,
    fit_closed_form_ridge,
    fold_for_name,
    read_json,
    read_rgb,
    repo_path,
    sha256_file,
    validate_file_artifact,
    validate_prediction_set,
)
from scripts.analysis.build_sign_separated_residual_patch_index import (  # noqa: E402
    effective_train_filenames,
    sha256_rows,
)


PLAN_PATH = Path("docs/spatial-primary-mask-support-prerequisite-v1.json")
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
OUTPUT_PATH = Path(
    "outputs/spatial-primary-mask-support-prerequisite-20260811/audit.json"
)
ACTIVE_ITERATION_ID = "monotonic-residual-erase-support"
PREREGISTRATION_ID = "spatial_primary_mask_support_preregistration"
DIAGNOSTIC_ID = "spatial_primary_mask_support_train_only_diagnostic"
MASK_CHANNELS = ("mb", "ms", "mb_minus_ms", "mb_times_ms")


def validate_authority(ledger: dict[str, Any]) -> None:
    program = ledger.get("program", {})
    if (
        program.get("product_default") != "artifacts/current-primary"
        or program.get("promotion_state") != "disabled"
        or program.get("reserved_blind_state") != "disabled"
    ):
        raise AuditError("quality-loop program authority changed")
    active = ledger.get("active_iteration", {})
    if (
        active.get("id") != ACTIVE_ITERATION_ID
        or active.get("terminal") != "PREREQUISITE_NEEDED"
        or active.get("first_gate") != "scut_inner_val15"
    ):
        raise AuditError("active iteration authority changed")
    prerequisites = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if prerequisites.get(PREREGISTRATION_ID) != "passed":
        raise AuditError("spatial mask preregistration is not passed")
    if prerequisites.get(DIAGNOSTIC_ID) != "pending":
        raise AuditError("spatial mask diagnostic is not pending")
    if (
        prerequisites.get("dual_input_support_separation_train_only_diagnostic")
        != "passed"
    ):
        raise AuditError("prior dual-input KILL is not recorded")


def validate_plan(plan: dict[str, Any]) -> None:
    exact = {
        "schema_version": 1,
        "iteration_id": ACTIVE_ITERATION_ID,
        "state": "preregistered_pending_mask_materialization",
        "next_boundary_on_pass": "mask_aware_data_and_training_preflight_only",
    }
    for key, expected in exact.items():
        if plan.get(key) != expected:
            raise AuditError(f"plan field changed: {key}")

    representation = plan.get("representation", {})
    if representation != {
        "channels": list(MASK_CHANNELS),
        "feature_count": 4,
        "mask_maps_are_pixel_aligned": True,
        "single_causal_change": (
            "page_broadcast_pipeline_context_to_frozen_pixel_aligned_primary_masks"
        ),
    }:
        raise AuditError("registered mask representation changed")

    diagnostic = plan.get("diagnostic", {})
    required_diagnostic = {
        "ablation_features": [
            "second_stage_r",
            "second_stage_g",
            "second_stage_b",
        ],
        "auc_tie_policy": "average_rank",
        "class_balance": "equal_per_page_target_lighter_and_preserve",
        "fold_assignment": "int(sha256(utf8_basename), 16) % 5",
        "fold_count": 5,
        "full_feature_count": 4,
        "lambda": 1.0,
        "luma_definition": "arithmetic_mean_rgb_gray_units",
        "max_samples_per_class_per_page": 1024,
        "numeric_type": "float64",
        "optimizer": "none_closed_form_ridge",
        "primary_mask_channels": list(MASK_CHANNELS),
        "ridge_intercept": "unpenalized",
        "standardization": "fitting_fold_mean_and_standard_deviation_only",
        "threshold_derivation": "prohibited",
    }
    for key, expected in required_diagnostic.items():
        if diagnostic.get(key) != expected:
            raise AuditError(f"diagnostic field changed: {key}")

    required_acceptance = {
        "full_auc_ablation_margin_min": 0.03,
        "full_fold_auc_min": 0.55,
        "full_mean_fold_auc_min": 0.65,
        "macro_median_page_auc_min": 0.6,
        "positive_mean_above_preserve_min_folds": 4,
        "required_fold_count": 5,
        "required_terminal_on_pass": "PASS",
    }
    if plan.get("acceptance") != required_acceptance:
        raise AuditError("acceptance contract changed")

    authorization = plan.get("authorization", {})
    forbidden = {
        "candidate_checkpoint",
        "candidate_inference",
        "development_gate",
        "holdout40",
        "inner_val15",
        "model_training",
        "promotion",
        "reserved_blind",
        "scut115",
        "visual_review",
    }
    if any(authorization.get(name) is not False for name in forbidden):
        raise AuditError("a candidate or quality surface was opened")
    if (
        authorization.get("mask_materialization") is not True
        or authorization.get("mask_materialization_reads_targets") is not False
        or authorization.get("separability_diagnostic_target_decode_roles")
        != ["train"]
    ):
        raise AuditError("mask or train-label authorization changed")


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise AuditError(f"mask decode failed: {path}")
    if image.dtype != np.uint8 or not np.isfinite(image).all():
        raise AuditError(f"invalid mask values: {path}")
    return image


def validate_label_set(
    label_dir: Path, file_names: list[str], spec: dict[str, Any]
) -> dict[str, Any]:
    if (
        spec.get("count") != len(file_names)
        or Path(str(spec.get("directory")))
        != Path("data-links/samples/SCUT-HW5K-mixed-20260729/train/all_labels")
    ):
        raise AuditError("train label set specification changed")
    rows = []
    for file_name in file_names:
        path = label_dir / file_name
        if not path.is_file():
            raise AuditError(f"missing train label: {file_name}")
        rows.append(f"{file_name} {sha256_file(path)}")
    content_sha256 = sha256_rows(sorted(rows))
    if content_sha256 != spec.get("content_sha256"):
        raise AuditError("train label content hash changed")
    return {"count": len(rows), "content_sha256": content_sha256}


def validate_materialization(
    *,
    repo_root: Path,
    plan: dict[str, Any],
    expected_names: list[str],
) -> tuple[Path, Path, dict[str, dict[str, Any]], dict[str, Any]]:
    output_root = repo_path(
        repo_root, plan["mask_materialization"]["output_root"]
    )
    if not output_root.is_dir():
        raise AuditError("missing mask materialization output")
    if {path.name for path in output_root.iterdir()} != {"mb", "ms", "manifest.json"}:
        raise AuditError("mask materialization output surface changed")
    mb_dir = output_root / "mb"
    ms_dir = output_root / "ms"
    if not mb_dir.is_dir() or not ms_dir.is_dir():
        raise AuditError("mask materialization directories are missing")

    manifest_path = output_root / "manifest.json"
    manifest = read_json(manifest_path)
    exact = {
        "schema_version": 1,
        "terminal": "PASS",
        "provenance": "utils.page_inference.infer_full_page",
        "target_access": False,
        "train_count": 275,
        "page_overlap": 32,
        "batch_size": 8,
        "maps": ["mb", "ms"],
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise AuditError(f"mask manifest field changed: {key}")
    if manifest.get("source_manifest") != plan["data"]["manifest"]:
        raise AuditError("mask manifest source role changed")
    if manifest.get("primary_config") != plan["evidence"]["current_primary_config"]:
        raise AuditError("mask manifest primary config changed")
    if (
        manifest.get("primary_checkpoint")
        != plan["evidence"]["current_primary_checkpoint"]
    ):
        raise AuditError("mask manifest primary checkpoint changed")

    pages = manifest.get("pages")
    if not isinstance(pages, list) or len(pages) != len(expected_names):
        raise AuditError("mask manifest page count changed")
    by_file: dict[str, dict[str, Any]] = {}
    required_page_keys = {
        "file",
        "source_sha256",
        "mb_sha256",
        "ms_sha256",
        "height",
        "width",
    }
    for row in pages:
        if not isinstance(row, dict) or set(row) != required_page_keys:
            raise AuditError("mask manifest page schema changed")
        file_name = str(row["file"])
        if file_name in by_file:
            raise AuditError(f"duplicate mask manifest page: {file_name}")
        if (
            not isinstance(row["height"], int)
            or not isinstance(row["width"], int)
            or row["height"] <= 0
            or row["width"] <= 0
        ):
            raise AuditError(f"invalid mask dimensions: {file_name}")
        by_file[file_name] = row
    if sorted(by_file) != sorted(expected_names):
        raise AuditError("mask manifest identities changed")

    expected_png_names = sorted(f"{Path(name).stem}.png" for name in expected_names)
    for map_name, directory in (("mb", mb_dir), ("ms", ms_dir)):
        entries = sorted(directory.iterdir(), key=lambda path: path.name)
        if [path.name for path in entries] != expected_png_names or not all(
            path.is_file() for path in entries
        ):
            raise AuditError(f"{map_name} filenames or count changed")
        for file_name, row in by_file.items():
            path = directory / f"{Path(file_name).stem}.png"
            if sha256_file(path) != row[f"{map_name}_sha256"]:
                raise AuditError(f"{map_name} content hash changed: {file_name}")

    summary = {
        "manifest_path": str(manifest_path.relative_to(repo_root)),
        "manifest_sha256": sha256_file(manifest_path),
        "train_count": len(by_file),
        "provenance": manifest["provenance"],
        "target_access": manifest["target_access"],
        "mb_content_sha256": sha256_rows(
            sorted(
                f"{file_name} {row['mb_sha256']}"
                for file_name, row in by_file.items()
            )
        ),
        "ms_content_sha256": sha256_rows(
            sorted(
                f"{file_name} {row['ms_sha256']}"
                for file_name, row in by_file.items()
            )
        ),
    }
    return mb_dir, ms_dir, by_file, summary


def build_page(
    *,
    file_name: str,
    source_path: Path,
    label_dir: Path,
    second_stage_dir: Path,
    mb_dir: Path,
    ms_dir: Path,
    mask_row: dict[str, Any],
    margin_gray: float,
    sample_cap: int,
) -> dict[str, Any]:
    if sha256_file(source_path) != mask_row["source_sha256"]:
        raise AuditError(f"source image hash changed: {file_name}")
    source = read_rgb(source_path)
    second_stage = read_rgb(second_stage_dir / f"{Path(file_name).stem}.png")
    target = read_rgb(label_dir / file_name)
    mb = read_gray(mb_dir / f"{Path(file_name).stem}.png")
    ms = read_gray(ms_dir / f"{Path(file_name).stem}.png")
    height, width = source.shape[:2]
    if (
        second_stage.shape != source.shape
        or target.shape != source.shape
        or mb.shape != (height, width)
        or ms.shape != (height, width)
        or mask_row["height"] != height
        or mask_row["width"] != width
    ):
        raise AuditError(f"image or mask shape mismatch: {file_name}")

    target_luma = target.astype(np.float32).mean(axis=2)
    second_float = second_stage.astype(np.float32)
    second_luma = second_float.mean(axis=2)
    positive_mask = (target_luma - second_luma) > margin_gray
    positive_indices, preserve_indices = balanced_indices(
        positive_mask, file_name, sample_cap
    )
    indices = np.concatenate([positive_indices, preserve_indices])
    labels = np.concatenate(
        [
            np.ones(len(positive_indices), dtype=np.int8),
            -np.ones(len(preserve_indices), dtype=np.int8),
        ]
    )

    mb_values = mb.reshape(-1)[indices].astype(np.float32) / 255.0
    ms_values = ms.reshape(-1)[indices].astype(np.float32) / 255.0
    features = np.column_stack(
        [mb_values, ms_values, mb_values - ms_values, mb_values * ms_values]
    ).astype(np.float32, copy=False)
    ablation_features = (
        second_float.reshape(-1, 3)[indices] / 255.0
    ).astype(np.float32, copy=False)
    if (
        features.shape[1] != 4
        or ablation_features.shape[1] != 3
        or not np.isfinite(features).all()
        or not np.isfinite(ablation_features).all()
    ):
        raise AuditError(f"invalid feature matrix: {file_name}")

    sample_digest = hashlib.sha256()
    sample_digest.update(file_name.encode("utf-8"))
    sample_digest.update(positive_indices.astype("<i8").tobytes())
    sample_digest.update(preserve_indices.astype("<i8").tobytes())
    return {
        "file": file_name,
        "fold": fold_for_name(file_name),
        "features": features,
        "ablation_features": ablation_features,
        "labels": labels,
        "samples_per_class": len(positive_indices),
        "positive_pixel_count": int(positive_mask.sum()),
        "preserve_pixel_count": int((~positive_mask).sum()),
        "sample_sha256": sample_digest.hexdigest(),
        "mask_sha256": {
            "mb": mask_row["mb_sha256"],
            "ms": mask_row["ms_sha256"],
        },
        "mask_stats": {
            "mb_min": int(mb.min()),
            "mb_max": int(mb.max()),
            "mb_mean": float(mb.mean()),
            "ms_min": int(ms.min()),
            "ms_max": int(ms.max()),
            "ms_mean": float(ms.mean()),
        },
    }


def concatenate_pages(
    pages: list[dict[str, Any]], feature_key: str
) -> tuple[np.ndarray, np.ndarray]:
    features = np.concatenate([page[feature_key] for page in pages], axis=0)
    labels = np.concatenate([page["labels"] for page in pages], axis=0)
    return features, labels


def mask_channel_strata(
    train_pages: list[dict[str, Any]],
    test_pages: list[dict[str, Any]],
    scores_by_name: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fit_features = np.concatenate(
        [page["features"] for page in train_pages], axis=0
    ).astype(np.float64, copy=False)
    for channel_index, name in enumerate(MASK_CHANNELS):
        cuts = np.quantile(fit_features[:, channel_index], [0.25, 0.5, 0.75])
        bins: list[dict[str, Any]] = []
        for bin_index in range(4):
            labels_parts: list[np.ndarray] = []
            scores_parts: list[np.ndarray] = []
            for page in test_pages:
                membership = (
                    np.searchsorted(
                        cuts, page["features"][:, channel_index], side="right"
                    )
                    == bin_index
                )
                if membership.any():
                    labels_parts.append(page["labels"][membership])
                    scores_parts.append(scores_by_name[page["file"]][membership])
            if labels_parts:
                labels = np.concatenate(labels_parts)
                scores = np.concatenate(scores_parts)
                positive_count = int((labels > 0).sum())
                preserve_count = int((labels < 0).sum())
                auc = (
                    auc_rank(labels, scores)
                    if positive_count and preserve_count
                    else None
                )
            else:
                positive_count = 0
                preserve_count = 0
                auc = None
            bins.append(
                {
                    "bin": bin_index,
                    "sample_count": positive_count + preserve_count,
                    "positive_count": positive_count,
                    "preserve_count": preserve_count,
                    "auc": auc,
                }
            )
        result[name] = {"fit_quartile_cuts": cuts.tolist(), "bins": bins}
    return result


def evaluate_fold(
    pages: list[dict[str, Any]], fold: int, ridge_lambda: float
) -> dict[str, Any]:
    train_pages = [page for page in pages if page["fold"] != fold]
    test_pages = [page for page in pages if page["fold"] == fold]
    if not train_pages or not test_pages:
        raise AuditError(f"empty fitting or held-out fold: {fold}")
    train_full, train_labels = concatenate_pages(train_pages, "features")
    test_full, test_labels = concatenate_pages(test_pages, "features")
    full_scores, full_fit = fit_closed_form_ridge(
        train_full, train_labels, test_full, ridge_lambda
    )
    train_ablation, _ = concatenate_pages(train_pages, "ablation_features")
    test_ablation, _ = concatenate_pages(test_pages, "ablation_features")
    ablation_scores, ablation_fit = fit_closed_form_ridge(
        train_ablation, train_labels, test_ablation, ridge_lambda
    )

    positive = test_labels > 0
    scores_by_name: dict[str, np.ndarray] = {}
    page_results: list[dict[str, Any]] = []
    offset = 0
    for page in test_pages:
        count = len(page["labels"])
        page_scores = full_scores[offset : offset + count]
        scores_by_name[page["file"]] = page_scores
        page_results.append(
            {
                "file": page["file"],
                "auc": auc_rank(page["labels"], page_scores),
                "positive_score_mean": float(
                    page_scores[page["labels"] > 0].mean()
                ),
                "preserve_score_mean": float(
                    page_scores[page["labels"] < 0].mean()
                ),
                "samples_per_class": page["samples_per_class"],
            }
        )
        offset += count
    full_auc = auc_rank(test_labels, full_scores)
    ablation_auc = auc_rank(test_labels, ablation_scores)
    return {
        "fold": fold,
        "fit_page_count": len(train_pages),
        "test_page_count": len(test_pages),
        "full_auc": full_auc,
        "ablation_auc": ablation_auc,
        "auc_margin": full_auc - ablation_auc,
        "positive_score_mean": float(full_scores[positive].mean()),
        "preserve_score_mean": float(full_scores[~positive].mean()),
        "page_auc_median": float(
            np.median([row["auc"] for row in page_results])
        ),
        "full_fit": full_fit,
        "ablation_fit": ablation_fit,
        "mask_channel_strata": mask_channel_strata(
            train_pages, test_pages, scores_by_name
        ),
        "pages": page_results,
    }


def run_audit(
    *,
    repo_root: Path,
    plan_path: Path = PLAN_PATH,
    ledger_path: Path = LEDGER_PATH,
) -> dict[str, Any]:
    plan_file = repo_path(repo_root, str(plan_path))
    ledger_file = repo_path(repo_root, str(ledger_path))
    plan = read_json(plan_file)
    ledger = read_json(ledger_file)
    validate_plan(plan)
    validate_authority(ledger)

    evidence = plan["evidence"]
    file_evidence_names = (
        "current_primary_checkpoint",
        "current_primary_config",
        "dual_input_kill",
        "role_plan",
        "second_stage_checkpoint",
        "second_stage_metrics",
    )
    paths = {
        name: validate_file_artifact(repo_root, evidence[name], name)
        for name in file_evidence_names
    }
    manifest_path = validate_file_artifact(
        repo_root, plan["data"]["manifest"], "train manifest"
    )
    manifest_rows = [
        row.strip()
        for row in manifest_path.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    file_names = [Path(row).name for row in manifest_rows]
    if (
        len(file_names) != plan["data"]["effective_train_count"]
        or len(file_names) != len(set(file_names))
    ):
        raise AuditError("train manifest count or uniqueness changed")

    role_wrapper = read_json(paths["role_plan"])
    base_role_artifact = role_wrapper.get("evidence", {}).get("base_role_contract")
    if not isinstance(base_role_artifact, dict):
        raise AuditError("monotonic role plan lacks its base role contract")
    base_role_path = validate_file_artifact(
        repo_root, base_role_artifact, "base role contract"
    )
    if sorted(file_names) != effective_train_filenames(repo_root, base_role_path):
        raise AuditError("train manifest no longer matches frozen roles")

    second_stage_dir, second_prediction_summary = validate_prediction_set(
        repo_root,
        evidence["second_stage_prediction_set"],
        file_names,
        "second-stage prediction set",
    )
    label_dir = repo_path(repo_root, evidence["train_label_set"]["directory"])
    label_summary = validate_label_set(
        label_dir, file_names, evidence["train_label_set"]
    )
    mb_dir, ms_dir, mask_rows, materialization_summary = validate_materialization(
        repo_root=repo_root, plan=plan, expected_names=file_names
    )

    pages: list[dict[str, Any]] = []
    margin = float(plan["data"]["target_lighter_margin_gray"])
    sample_cap = int(plan["diagnostic"]["max_samples_per_class_per_page"])
    for index, manifest_row in enumerate(manifest_rows, start=1):
        file_name = Path(manifest_row).name
        pages.append(
            build_page(
                file_name=file_name,
                source_path=repo_path(repo_root, manifest_row),
                label_dir=label_dir,
                second_stage_dir=second_stage_dir,
                mb_dir=mb_dir,
                ms_dir=ms_dir,
                mask_row=mask_rows[file_name],
                margin_gray=margin,
                sample_cap=sample_cap,
            )
        )
        if index % 25 == 0 or index == len(manifest_rows):
            print(f"decoded train-only mask pages {index}/{len(manifest_rows)}", flush=True)

    fold_counts = {
        fold: sum(page["fold"] == fold for page in pages) for fold in range(5)
    }
    if any(count == 0 for count in fold_counts.values()):
        raise AuditError("page hash assignment produced an empty fold")
    ridge_lambda = float(plan["diagnostic"]["lambda"])
    folds = [evaluate_fold(pages, fold, ridge_lambda) for fold in range(5)]
    full_aucs = [float(fold["full_auc"]) for fold in folds]
    ablation_aucs = [float(fold["ablation_auc"]) for fold in folds]
    page_aucs = [
        float(page["auc"]) for fold in folds for page in fold["pages"]
    ]
    positive_above_preserve = sum(
        fold["positive_score_mean"] > fold["preserve_score_mean"]
        for fold in folds
    )
    aggregates = {
        "full_mean_fold_auc": float(np.mean(full_aucs)),
        "full_min_fold_auc": min(full_aucs),
        "ablation_mean_fold_auc": float(np.mean(ablation_aucs)),
        "full_auc_ablation_margin": float(
            np.mean(full_aucs) - np.mean(ablation_aucs)
        ),
        "macro_median_page_auc": float(np.median(page_aucs)),
        "positive_mean_above_preserve_folds": positive_above_preserve,
    }
    contract = plan["acceptance"]
    conditions = {
        "full_mean_fold_auc": (
            aggregates["full_mean_fold_auc"]
            >= contract["full_mean_fold_auc_min"]
        ),
        "every_fold_auc": (
            aggregates["full_min_fold_auc"] >= contract["full_fold_auc_min"]
        ),
        "macro_median_page_auc": (
            aggregates["macro_median_page_auc"]
            >= contract["macro_median_page_auc_min"]
        ),
        "positive_mean_above_preserve": (
            positive_above_preserve
            >= contract["positive_mean_above_preserve_min_folds"]
        ),
        "full_auc_ablation_margin": (
            aggregates["full_auc_ablation_margin"]
            >= contract["full_auc_ablation_margin_min"]
        ),
    }
    terminal = "PASS" if all(conditions.values()) else "KILL"
    page_records = [
        {
            key: page[key]
            for key in (
                "file",
                "fold",
                "samples_per_class",
                "positive_pixel_count",
                "preserve_pixel_count",
                "sample_sha256",
                "mask_sha256",
                "mask_stats",
            )
        }
        for page in pages
    ]
    return {
        "schema_version": 1,
        "terminal": terminal,
        "iteration_id": ACTIVE_ITERATION_ID,
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_file),
        "ledger_path": str(ledger_path),
        "real_data_access": True,
        "target_decode_roles": ["train"],
        "training_started": False,
        "checkpoint_generated": False,
        "candidate_inference_started": False,
        "quality_gate_started": False,
        "promotion_enabled": False,
        "reserved_blind_authorized": False,
        "feature_count": 4,
        "feature_names": list(MASK_CHANNELS),
        "ablation_feature_names": plan["diagnostic"]["ablation_features"],
        "train_page_count": len(pages),
        "fold_page_counts": fold_counts,
        "materialization": materialization_summary,
        "second_stage_predictions": second_prediction_summary,
        "train_labels": label_summary,
        "sample_count": sum(len(page["labels"]) for page in pages),
        "page_samples": page_records,
        "folds": folds,
        "aggregates": aggregates,
        "acceptance": {
            "contract": contract,
            "conditions": conditions,
            "passed": all(conditions.values()),
        },
        "next_boundary": (
            plan["next_boundary_on_pass"]
            if terminal == "PASS"
            else "close_exact_spatial_primary_mask_support_diagnostic"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path = repo_path(args.repo_root, str(args.output))
    if output_path.parent.exists():
        print(
            f"terminal=PREREQUISITE_NEEDED reason=output directory already exists: {output_path.parent}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    try:
        result = run_audit(
            repo_root=args.repo_root,
            plan_path=args.plan,
            ledger_path=args.ledger,
        )
    except (AuditError, OSError, ValueError) as error:
        result = {
            "schema_version": 1,
            "terminal": "PREREQUISITE_NEEDED",
            "reason": str(error),
            "training_started": False,
            "checkpoint_generated": False,
            "candidate_inference_started": False,
            "quality_gate_started": False,
            "promotion_enabled": False,
        }
        return_code = 2
    else:
        return_code = 0 if result["terminal"] == "PASS" else 1
    output_path.parent.mkdir(parents=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"terminal={result['terminal']} output={output_path}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
