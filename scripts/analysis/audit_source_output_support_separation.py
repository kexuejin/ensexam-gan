#!/usr/bin/env python3
"""Audit raw source plus frozen-output support before model training."""

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
    metric_float,
    read_json,
    read_metric_rows,
    read_rgb,
    repo_path,
    sha256_file,
    validate_file_artifact,
    validate_label_set,
    validate_prediction_set,
)
from scripts.analysis.build_sign_separated_residual_patch_index import (  # noqa: E402
    effective_train_filenames,
)


PLAN_PATH = Path("docs/source-output-support-prerequisite-v1.json")
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
OUTPUT_PATH = Path(
    "outputs/source-output-support-prerequisite-20260812/audit.json"
)
ACTIVE_ITERATION_ID = "monotonic-residual-erase-support"
PREREGISTRATION_ID = "materially_new_support_successor_preregistration"
DIAGNOSTIC_ID = "source_output_support_train_only_diagnostic"
CHANNELS = (
    "source_r",
    "source_g",
    "source_b",
    "second_stage_r",
    "second_stage_g",
    "second_stage_b",
)
ABLATION_CHANNELS = CHANNELS[3:]


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
    statuses = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if statuses.get(PREREGISTRATION_ID) != "passed":
        raise AuditError("source-output preregistration is not passed")
    if statuses.get(DIAGNOSTIC_ID) != "pending":
        raise AuditError("source-output diagnostic is not pending")
    if statuses.get("reconstruction_stage_disagreement_train_only_diagnostic") != "passed":
        raise AuditError("prior reconstruction-stage KILL is not recorded")


def validate_plan(plan: dict[str, Any]) -> None:
    exact = {
        "schema_version": 1,
        "iteration_id": ACTIVE_ITERATION_ID,
        "state": "preregistered_pending_source_output_train_only_diagnostic",
        "next_boundary_on_pass": (
            "source_output_support_data_and_training_preflight_only"
        ),
    }
    for key, expected in exact.items():
        if plan.get(key) != expected:
            raise AuditError(f"plan field changed: {key}")
    data = plan.get("data", {})
    required_data = {
        "effective_train_count": 275,
        "effective_train_domain_counts": {"hw5k": 253, "scut": 22},
        "effective_train_filename_sha256": (
            "e9ac4d6f700f41ef3a9b7c3f04ce0593f593324a881a0f9fc387901a497f9039"
        ),
        "second_stage_prediction_dir": (
            "outputs/archive/sign-separated-residual-repair-20260810/"
            "train275-frozen-pipeline/pred"
        ),
        "split": "train",
        "target_access": "diagnostic_labels_only",
        "target_lighter_margin_gray": 2.0,
    }
    for key, expected in required_data.items():
        if data.get(key) != expected:
            raise AuditError(f"data field changed: {key}")
    if plan.get("representation") != {
        "channels": list(CHANNELS),
        "feature_count": 6,
        "no_masks_page_scalars_stages_or_primary_rgb": True,
        "no_threshold_neighborhood_or_model_transform": True,
        "single_causal_change": (
            "add_raw_source_rgb_to_frozen_second_stage_rgb_support_evidence"
        ),
        "source_and_output_are_target_free": True,
    }:
        raise AuditError("registered source-output representation changed")
    diagnostic = plan.get("diagnostic", {})
    required_diagnostic = {
        "ablation_features": list(ABLATION_CHANNELS),
        "auc_tie_policy": "average_rank",
        "class_balance": "equal_per_page_target_lighter_and_preserve",
        "coordinate_rank": (
            "splitmix64(row_major_index XOR "
            "first_64_bits(sha256(utf8_basename))) then unsigned_hash_y_x"
        ),
        "feature_normalization": "divide_raw_uint8_rgb_channels_by_255",
        "fold_assignment": "int(sha256(utf8_basename), 16) % 5",
        "fold_count": 5,
        "full_feature_count": 6,
        "lambda": 1.0,
        "luma_definition": "arithmetic_mean_rgb_gray_units",
        "max_samples_per_class_per_page": 1024,
        "numeric_type": "float64",
        "optimizer": "none_closed_form_ridge",
        "ridge_intercept": "unpenalized",
        "sampling_count": (
            "min(1024, target_lighter_count, preserve_count) per class per page"
        ),
        "standardization": "fitting_fold_mean_and_standard_deviation_only",
        "target_encoding": {"preserve": -1, "target_lighter": 1},
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
        authorization.get("diagnostic_implementation") is not True
        or authorization.get("train_target_decode_for_diagnostic_labels_only")
        is not True
        or authorization.get("diagnostic_output_only")
        != str(OUTPUT_PATH)
    ):
        raise AuditError("diagnostic authorization changed")
    if plan.get("planned_implementation") != {
        "audit_output": str(OUTPUT_PATH),
        "audit_script": (
            "scripts/analysis/audit_source_output_support_separation.py"
        ),
        "test": "tests/test_source_output_support_prerequisite.py",
    }:
        raise AuditError("planned implementation changed")


def build_page(
    *,
    file_name: str,
    source_path: Path,
    label_dir: Path,
    second_stage_dir: Path,
    primary_row: dict[str, str],
    second_stage_row: dict[str, str],
    margin_gray: float,
    sample_cap: int,
) -> dict[str, Any]:
    source = read_rgb(source_path)
    second_stage = read_rgb(
        second_stage_dir / f"{Path(file_name).stem}.png"
    )
    target = read_rgb(label_dir / file_name)
    if source.shape != second_stage.shape or source.shape != target.shape:
        raise AuditError(f"image shape mismatch for {file_name}")
    if sha256_file(source_path) != primary_row.get("image_sha256"):
        raise AuditError(f"source image hash changed for {file_name}")
    if (
        metric_float(second_stage_row, "base_edit_threshold", file_name) != 12.0
        or metric_float(second_stage_row, "second_delta_threshold", file_name)
        != 32.0
        or metric_float(second_stage_row, "dark_threshold", file_name) != 0.0
    ):
        raise AuditError(f"second-stage protocol changed for {file_name}")

    source_float = source.astype(np.float32)
    second_float = second_stage.astype(np.float32)
    target_float = target.astype(np.float32)
    positive_mask = (
        target_float.mean(axis=2) - second_float.mean(axis=2)
    ) > margin_gray
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
    source_flat = source_float.reshape(-1, 3)[indices] / 255.0
    second_flat = second_float.reshape(-1, 3)[indices] / 255.0
    features = np.column_stack([source_flat, second_flat]).astype(
        np.float32, copy=False
    )
    if features.shape != (len(indices), len(CHANNELS)):
        raise AuditError(f"source-output feature shape changed for {file_name}")
    if not np.isfinite(features).all():
        raise AuditError(f"non-finite source-output feature for {file_name}")
    sample_digest = hashlib.sha256()
    sample_digest.update(file_name.encode("utf-8"))
    sample_digest.update(positive_indices.astype("<i8").tobytes())
    sample_digest.update(preserve_indices.astype("<i8").tobytes())
    return {
        "file": file_name,
        "fold": fold_for_name(file_name),
        "features": features,
        "ablation_features": features[:, 3:],
        "labels": labels,
        "samples_per_class": len(positive_indices),
        "positive_pixel_count": int(positive_mask.sum()),
        "preserve_pixel_count": int((~positive_mask).sum()),
        "sample_sha256": sample_digest.hexdigest(),
    }


def concatenate_pages(
    pages: list[dict[str, Any]], feature_key: str
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([page[feature_key] for page in pages], axis=0),
        np.concatenate([page["labels"] for page in pages], axis=0),
    )


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
    page_results: list[dict[str, Any]] = []
    offset = 0
    for page in test_pages:
        count = len(page["labels"])
        page_scores = full_scores[offset : offset + count]
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
        "current_second_stage_checkpoint",
        "dual_input_kill",
        "primary_metrics",
        "reconstruction_stage_kill",
        "role_plan",
        "second_stage_metrics",
        "spatial_mask_kill",
    )
    paths = {
        name: validate_file_artifact(repo_root, evidence[name], name)
        for name in file_evidence_names
    }
    data = plan["data"]
    manifest_path = validate_file_artifact(
        repo_root, data["manifest"], "train manifest"
    )
    manifest_rows = [
        row.strip()
        for row in manifest_path.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    file_names = [Path(row).name for row in manifest_rows]
    if (
        len(file_names) != data["effective_train_count"]
        or len(file_names) != len(set(file_names))
    ):
        raise AuditError("train manifest count or uniqueness changed")
    role_wrapper = read_json(paths["role_plan"])
    base_role_artifact = role_wrapper.get("evidence", {}).get(
        "base_role_contract"
    )
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
    primary_rows = read_metric_rows(paths["primary_metrics"])
    second_rows = read_metric_rows(paths["second_stage_metrics"])
    if set(primary_rows) != set(file_names) or set(second_rows) != set(file_names):
        raise AuditError("pipeline metric identities changed")
    label_dir = repo_path(repo_root, evidence["train_label_set"]["directory"])
    label_summary = validate_label_set(
        label_dir, file_names, evidence["train_label_set"]
    )

    pages: list[dict[str, Any]] = []
    margin = float(data["target_lighter_margin_gray"])
    sample_cap = int(plan["diagnostic"]["max_samples_per_class_per_page"])
    for index, manifest_row in enumerate(manifest_rows, start=1):
        file_name = Path(manifest_row).name
        pages.append(
            build_page(
                file_name=file_name,
                source_path=repo_path(repo_root, manifest_row),
                label_dir=label_dir,
                second_stage_dir=second_stage_dir,
                primary_row=primary_rows[file_name],
                second_stage_row=second_rows[file_name],
                margin_gray=margin,
                sample_cap=sample_cap,
            )
        )
        if index % 25 == 0 or index == len(manifest_rows):
            print(
                f"decoded train-only source-output pages {index}/{len(manifest_rows)}",
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
    page_aucs = [float(page["auc"]) for fold in folds for page in fold["pages"]]
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
        "feature_count": len(CHANNELS),
        "feature_names": list(CHANNELS),
        "ablation_feature_names": list(ABLATION_CHANNELS),
        "train_page_count": len(pages),
        "fold_page_counts": fold_counts,
        "second_stage_predictions": second_prediction_summary,
        "train_labels": label_summary,
        "sample_count": sum(len(page["labels"]) for page in pages),
        "page_samples": [
            {
                key: page[key]
                for key in (
                    "file",
                    "fold",
                    "samples_per_class",
                    "positive_pixel_count",
                    "preserve_pixel_count",
                    "sample_sha256",
                )
            }
            for page in pages
        ],
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
            else "close_exact_source_output_support_diagnostic"
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
