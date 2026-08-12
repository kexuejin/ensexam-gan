#!/usr/bin/env python3
"""Audit train-only reconstruction-stage disagreement before model training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

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
from scripts.analysis.materialize_reconstruction_stage_disagreement_train_only import (  # noqa: E402
    CHANNELS,
)


PLAN_PATH = Path("docs/reconstruction-stage-disagreement-prerequisite-v1.json")
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
OUTPUT_PATH = Path(
    "outputs/reconstruction-stage-disagreement-prerequisite-20260812/audit.json"
)
ACTIVE_ITERATION_ID = "monotonic-residual-erase-support"
PREREGISTRATION_ID = "materially_new_target_free_support_preregistration"
DIAGNOSTIC_ID = "reconstruction_stage_disagreement_train_only_diagnostic"


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
        raise AuditError("stage-disagreement preregistration is not passed")
    if prerequisites.get(DIAGNOSTIC_ID) != "pending":
        raise AuditError("stage-disagreement diagnostic is not pending")
    if prerequisites.get("spatial_primary_mask_support_train_only_diagnostic") != "passed":
        raise AuditError("prior spatial-mask KILL is not recorded")


def validate_plan(plan: dict[str, Any]) -> None:
    exact = {
        "schema_version": 1,
        "iteration_id": ACTIVE_ITERATION_ID,
        "state": "preregistered_pending_stage_disagreement_materialization",
        "next_boundary_on_pass": (
            "stage_disagreement_data_and_training_preflight_only"
        ),
    }
    for key, expected in exact.items():
        if plan.get(key) != expected:
            raise AuditError(f"plan field changed: {key}")
    representation = plan.get("representation", {})
    if representation != {
        "channels": list(CHANNELS),
        "feature_count": 4,
        "no_final_prediction_rgb": True,
        "no_masks_or_page_scalars": True,
        "no_source_rgb": True,
        "single_causal_change": (
            "frozen_primary_multiscale_reconstruction_stage_disagreement"
        ),
    }:
        raise AuditError("registered stage-disagreement representation changed")
    diagnostic = plan.get("diagnostic", {})
    required_diagnostic = {
        "ablation_features": [
            "second_stage_r",
            "second_stage_g",
            "second_stage_b",
        ],
        "auc_tie_policy": "average_rank",
        "class_balance": "equal_per_page_target_lighter_and_preserve",
        "feature_normalization": (
            "divide_gray_unit_stage_disagreement_channels_by_255"
        ),
        "fold_assignment": "int(sha256(utf8_basename), 16) % 5",
        "fold_count": 5,
        "full_feature_count": 4,
        "lambda": 1.0,
        "luma_definition": "arithmetic_mean_rgb_gray_units",
        "max_samples_per_class_per_page": 1024,
        "numeric_type": "float64",
        "optimizer": "none_closed_form_ridge",
        "ridge_intercept": "unpenalized",
        "stage_disagreement_channels": list(CHANNELS),
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
        authorization.get("stage_disagreement_materialization") is not True
        or authorization.get("stage_disagreement_materialization_reads_targets")
        is not False
        or authorization.get("separability_diagnostic_target_decode_roles")
        != ["train"]
    ):
        raise AuditError("stage-disagreement authorization changed")


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


def close_float(actual: float, expected: Any) -> bool:
    try:
        expected_float = float(expected)
    except (TypeError, ValueError):
        return False
    return math.isfinite(expected_float) and math.isclose(
        actual, expected_float, rel_tol=0.0, abs_tol=1e-6
    )


def load_stage_npz(
    path: Path, *, expected_shape: tuple[int, int] | None = None
) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != set(CHANNELS):
                raise AuditError(f"stage NPZ channels changed: {path}")
            channels = {
                name: np.asarray(payload[name]) for name in CHANNELS
            }
    except (OSError, ValueError) as error:
        raise AuditError(f"stage NPZ decode failed: {path}") from error
    for name, values in channels.items():
        if (
            values.dtype != np.float32
            or values.ndim != 2
            or (expected_shape is not None and values.shape != expected_shape)
            or not np.isfinite(values).all()
        ):
            raise AuditError(f"invalid {name} stage map: {path}")
    return channels


def validate_materialization(
    *, repo_root: Path, plan: dict[str, Any], expected_names: list[str]
) -> tuple[Path, dict[str, dict[str, Any]], dict[str, Any]]:
    spec = plan["stage_disagreement_materialization"]
    output_root = repo_path(repo_root, spec["output_root"])
    if not output_root.is_dir():
        raise AuditError("missing stage-disagreement materialization output")
    if {path.name for path in output_root.iterdir()} != {"pages", "manifest.json"}:
        raise AuditError("stage-disagreement output surface changed")
    page_dir = output_root / "pages"
    if not page_dir.is_dir():
        raise AuditError("stage-disagreement page directory is missing")
    manifest_path = output_root / "manifest.json"
    manifest = read_json(manifest_path)
    exact = {
        "schema_version": 1,
        "terminal": "PASS",
        "provenance": spec["provenance"],
        "target_access": False,
        "train_count": 275,
        "page_overlap": 32,
        "batch_size": 8,
        "dtype": "float32",
        "encoding": "one_compressed_npz_per_page_with_exact_channel_keys",
        "channels": list(CHANNELS),
        "channel_definitions": spec["channel_definitions"],
        "derive_per_patch_before_overlap_fusion": True,
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise AuditError(f"stage manifest field changed: {key}")
    plan_file = repo_path(
        repo_root, "docs/reconstruction-stage-disagreement-prerequisite-v1.json"
    )
    if manifest.get("plan") != {
        "path": str(plan_file.relative_to(repo_root)),
        "sha256": sha256_file(plan_file),
    }:
        raise AuditError("stage manifest plan changed")
    comparisons = {
        "source_manifest": plan["data"]["manifest"],
        "primary_config": plan["evidence"]["current_primary_config"],
        "primary_checkpoint": plan["evidence"]["current_primary_checkpoint"],
        "generator_source": plan["evidence"]["generator_source"],
        "page_inference_source": plan["evidence"]["page_inference_source"],
    }
    for key, expected in comparisons.items():
        if manifest.get(key) != expected:
            raise AuditError(f"stage manifest {key} changed")

    pages = manifest.get("pages")
    if not isinstance(pages, list) or len(pages) != len(expected_names):
        raise AuditError("stage manifest page count changed")
    by_file: dict[str, dict[str, Any]] = {}
    required_keys = {
        "file",
        "source_sha256",
        "npz_sha256",
        "height",
        "width",
        "channels",
    }
    for row in pages:
        if not isinstance(row, dict) or set(row) != required_keys:
            raise AuditError("stage manifest page schema changed")
        file_name = str(row["file"])
        if file_name in by_file:
            raise AuditError(f"duplicate stage manifest page: {file_name}")
        if (
            not isinstance(row["height"], int)
            or not isinstance(row["width"], int)
            or row["height"] <= 0
            or row["width"] <= 0
        ):
            raise AuditError(f"invalid stage dimensions: {file_name}")
        if not isinstance(row["channels"], dict) or set(row["channels"]) != set(
            CHANNELS
        ):
            raise AuditError(f"invalid stage channel summary: {file_name}")
        by_file[file_name] = row
    if sorted(by_file) != sorted(expected_names):
        raise AuditError("stage manifest identities changed")

    expected_npz_names = sorted(f"{Path(name).stem}.npz" for name in expected_names)
    entries = sorted(page_dir.iterdir(), key=lambda path: path.name)
    if [path.name for path in entries] != expected_npz_names or not all(
        path.is_file() for path in entries
    ):
        raise AuditError("stage NPZ filenames or count changed")
    content_rows: list[str] = []
    for file_name, row in by_file.items():
        path = page_dir / f"{Path(file_name).stem}.npz"
        actual_hash = sha256_file(path)
        if actual_hash != row["npz_sha256"]:
            raise AuditError(f"stage NPZ content hash changed: {file_name}")
        channels = load_stage_npz(
            path, expected_shape=(int(row["height"]), int(row["width"]))
        )
        for name, values in channels.items():
            summary = row["channels"].get(name)
            if not isinstance(summary, dict) or set(summary) != {"min", "max", "mean"}:
                raise AuditError(f"invalid {name} stage summary: {file_name}")
            if not (
                close_float(float(values.min()), summary["min"])
                and close_float(float(values.max()), summary["max"])
                and close_float(float(values.mean()), summary["mean"])
            ):
                raise AuditError(f"{name} stage statistics changed: {file_name}")
        content_rows.append(f"{file_name} {actual_hash}")
    return page_dir, by_file, {
        "manifest_path": str(manifest_path.relative_to(repo_root)),
        "manifest_sha256": sha256_file(manifest_path),
        "train_count": len(by_file),
        "provenance": manifest["provenance"],
        "target_access": manifest["target_access"],
        "content_sha256": sha256_rows(sorted(content_rows)),
    }


def build_page(
    *,
    file_name: str,
    source_path: Path,
    label_dir: Path,
    second_stage_dir: Path,
    page_dir: Path,
    stage_row: dict[str, Any],
    margin_gray: float,
    sample_cap: int,
) -> dict[str, Any]:
    if sha256_file(source_path) != stage_row["source_sha256"]:
        raise AuditError(f"source image hash changed: {file_name}")
    source = read_rgb(source_path)
    second_stage = read_rgb(second_stage_dir / f"{Path(file_name).stem}.png")
    target = read_rgb(label_dir / file_name)
    if source.shape != second_stage.shape or source.shape != target.shape:
        raise AuditError(f"source, second-stage, or target shape mismatch: {file_name}")
    height, width = source.shape[:2]
    if stage_row["height"] != height or stage_row["width"] != width:
        raise AuditError(f"stage manifest shape changed: {file_name}")
    channels = load_stage_npz(
        page_dir / f"{Path(file_name).stem}.npz",
        expected_shape=(height, width),
    )
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
    features = np.column_stack(
        [channels[name].reshape(-1)[indices] / 255.0 for name in CHANNELS]
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
        raise AuditError(f"invalid stage feature matrix: {file_name}")
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
        "npz_sha256": stage_row["npz_sha256"],
    }


def concatenate_pages(
    pages: list[dict[str, Any]], feature_key: str
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([page[feature_key] for page in pages], axis=0),
        np.concatenate([page["labels"] for page in pages], axis=0),
    )


def channel_strata(
    train_pages: list[dict[str, Any]],
    test_pages: list[dict[str, Any]],
    scores_by_name: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fit_features = np.concatenate(
        [page["features"] for page in train_pages], axis=0
    ).astype(np.float64, copy=False)
    for channel_index, name in enumerate(CHANNELS):
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
        "stage_channel_strata": channel_strata(
            train_pages, test_pages, scores_by_name
        ),
        "pages": page_results,
    }


def run_audit(
    *, repo_root: Path, plan_path: Path = PLAN_PATH, ledger_path: Path = LEDGER_PATH
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
        "generator_source",
        "page_inference_source",
        "role_plan",
        "second_stage_checkpoint",
        "second_stage_metrics",
        "spatial_mask_kill",
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
    page_dir, stage_rows, materialization_summary = validate_materialization(
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
                page_dir=page_dir,
                stage_row=stage_rows[file_name],
                margin_gray=margin,
                sample_cap=sample_cap,
            )
        )
        if index % 25 == 0 or index == len(manifest_rows):
            print(
                f"decoded train-only stage pages {index}/{len(manifest_rows)}",
                flush=True,
            )
    fold_counts = {
        fold: sum(page["fold"] == fold for page in pages) for fold in range(5)
    }
    if any(count == 0 for count in fold_counts.values()):
        raise AuditError("page hash assignment produced an empty fold")
    folds = [
        evaluate_fold(pages, fold, float(plan["diagnostic"]["lambda"]))
        for fold in range(5)
    ]
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
                "npz_sha256",
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
        "feature_names": list(CHANNELS),
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
            else "close_exact_reconstruction_stage_disagreement_diagnostic"
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
            "terminal=PREREQUISITE_NEEDED "
            f"reason=output directory already exists: {output_path.parent}",
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
