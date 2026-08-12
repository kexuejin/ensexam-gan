#!/usr/bin/env python3
"""Audit raw frozen second-stage alpha support before model training."""

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
    metric_float,
    read_json,
    read_metric_rows,
    repo_path,
    sha256_file,
    validate_file_artifact,
    validate_label_set,
    validate_prediction_set,
)
from scripts.analysis.build_sign_separated_residual_patch_index import (  # noqa: E402
    effective_train_filenames,
)
from scripts.analysis.materialize_second_stage_alpha_train_only import (  # noqa: E402
    OUTPUT_ROOT,
    RAW_ALPHA_KEY,
)

PLAN_PATH = Path("docs/second-stage-alpha-support-prerequisite-v1.json")
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
OUTPUT_PATH = Path("outputs/second-stage-alpha-support-prerequisite-20260812/audit.json")
ACTIVE_ITERATION_ID = "monotonic-residual-erase-support"
CHANNELS = (
    "second_stage_r",
    "second_stage_g",
    "second_stage_b",
    "raw_second_stage_alpha",
)
ABLATION_CHANNELS = CHANNELS[:3]


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
    if statuses.get("materially_new_support_successor_preregistration_v2") != "passed":
        raise AuditError("second-stage alpha preregistration is not passed")
    if statuses.get("second_stage_alpha_support_train_only_diagnostic") != "pending":
        raise AuditError("diagnostic is not pending")
    if statuses.get("source_output_support_train_only_diagnostic") != "passed":
        raise AuditError("prior source-output KILL is not recorded")


def validate_plan(plan: dict[str, Any]) -> None:
    exact = {
        "schema_version": 1,
        "iteration_id": ACTIVE_ITERATION_ID,
        "state": "preregistered_pending_second_stage_alpha_materialization",
        "next_boundary_on_pass": "second_stage_alpha_data_and_training_preflight_only",
    }
    for key, expected in exact.items():
        if plan.get(key) != expected:
            raise AuditError(f"plan field changed: {key}")
    if plan.get("representation") != {
        "channels": list(CHANNELS),
        "feature_count": 4,
        "no_clean_candidate_or_internal_feature": True,
        "no_masks_page_scalars_stages_source_or_primary_rgb": True,
        "no_threshold_neighborhood_or_component_transform": True,
        "single_causal_change": (
            "add_frozen_second_stage_prethreshold_raw_alpha_to_output_rgb_support_evidence"
        ),
    }:
        raise AuditError("registered alpha representation changed")
    data = plan.get("data", {})
    required_data = {
        "effective_train_count": 275,
        "effective_train_domain_counts": {"hw5k": 253, "scut": 22},
        "effective_train_filename_sha256": (
            "e9ac4d6f700f41ef3a9b7c3f04ce0593f593324a881a0f9fc387901a497f9039"
        ),
        "primary_prediction_dir": (
            "outputs/archive/sign-separated-residual-repair-20260810/train275-primary/pred"
        ),
        "second_stage_prediction_dir": (
            "outputs/archive/sign-separated-residual-repair-20260810/train275-frozen-pipeline/pred"
        ),
        "split": "train",
        "target_lighter_margin_gray": 2.0,
    }
    for key, expected in required_data.items():
        if data.get(key) != expected:
            raise AuditError(f"data field changed: {key}")
    diagnostic = plan.get("diagnostic", {})
    required_diagnostic = {
        "ablation_features": list(ABLATION_CHANNELS),
        "auc_tie_policy": "average_rank",
        "class_balance": "equal_per_page_target_lighter_and_preserve",
        "coordinate_rank": (
            "splitmix64(row_major_index XOR first_64_bits(sha256(utf8_basename))) "
            "then unsigned_hash_y_x"
        ),
        "feature_normalization": "raw_alpha_in_unit_interval_and_rgb_divided_by_255",
        "fold_assignment": "int(sha256(utf8_basename), 16) % 5",
        "fold_count": 5,
        "full_feature_count": 4,
        "lambda": 1.0,
        "luma_definition": "arithmetic_mean_rgb_gray_units",
        "max_samples_per_class_per_page": 1024,
        "numeric_type": "float64",
        "optimizer": "none_closed_form_ridge",
        "ridge_intercept": "unpenalized",
        "sampling_count": "min(1024, target_lighter_count, preserve_count) per class per page",
        "standardization": "fitting_fold_mean_and_standard_deviation_only",
        "target_encoding": {"preserve": -1, "target_lighter": 1},
        "threshold_derivation": "prohibited",
    }
    for key, expected in required_diagnostic.items():
        if diagnostic.get(key) != expected:
            raise AuditError(f"diagnostic field changed: {key}")
    if plan.get("acceptance") != {
        "full_auc_ablation_margin_min": 0.03,
        "full_fold_auc_min": 0.55,
        "full_mean_fold_auc_min": 0.65,
        "macro_median_page_auc_min": 0.6,
        "positive_mean_above_preserve_min_folds": 4,
        "required_fold_count": 5,
        "required_terminal_on_pass": "PASS",
    }:
        raise AuditError("acceptance contract changed")
    auth = plan.get("authorization", {})
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
    if any(auth.get(name) is not False for name in forbidden):
        raise AuditError("a candidate or quality surface was opened")
    if (
        auth.get("second_stage_alpha_materialization") is not True
        or auth.get("second_stage_alpha_materialization_reads_targets") is not False
        or auth.get("separability_diagnostic_target_decode_roles") != ["train"]
    ):
        raise AuditError("alpha authorization changed")
    if plan.get("planned_implementation") != {
        "audit_output": str(OUTPUT_PATH),
        "audit_script": "scripts/analysis/audit_second_stage_alpha_support.py",
        "materialization_output": str(OUTPUT_ROOT),
        "materializer": "scripts/analysis/materialize_second_stage_alpha_train_only.py",
        "test": "tests/test_second_stage_alpha_support_prerequisite.py",
    }:
        raise AuditError("planned implementation changed")


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise AuditError(f"image decode failed: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_raw_alpha_npz(
    path: Path, *, expected_shape: tuple[int, int] | None = None
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {RAW_ALPHA_KEY}:
            raise AuditError("raw alpha NPZ keys changed")
        raw_alpha = payload[RAW_ALPHA_KEY]
    if raw_alpha.dtype != np.float32:
        raise AuditError("raw alpha dtype changed")
    if raw_alpha.ndim != 2:
        raise AuditError("raw alpha must be 2D")
    if expected_shape is not None and raw_alpha.shape != expected_shape:
        raise AuditError("raw alpha shape changed")
    if not np.isfinite(raw_alpha).all():
        raise AuditError("raw alpha contains non-finite values")
    if float(raw_alpha.min()) < 0.0 or float(raw_alpha.max()) > 1.0:
        raise AuditError("raw alpha escaped unit interval")
    return raw_alpha


def build_page(
    *,
    file_name: str,
    second_stage_dir: Path,
    label_dir: Path,
    alpha_dir: Path,
    alpha_manifest_row: dict[str, Any],
    second_stage_row: dict[str, str],
    margin_gray: float,
    sample_cap: int,
) -> dict[str, Any]:
    second_stage = read_rgb(second_stage_dir / f"{Path(file_name).stem}.png")
    target = read_rgb(label_dir / file_name)
    if second_stage.shape != target.shape:
        raise AuditError(f"image shape mismatch for {file_name}")
    raw_alpha = load_raw_alpha_npz(
        alpha_dir / f"{Path(file_name).stem}.npz",
        expected_shape=second_stage.shape[:2],
    )
    if (
        int(alpha_manifest_row["height"]) != int(raw_alpha.shape[0])
        or int(alpha_manifest_row["width"]) != int(raw_alpha.shape[1])
    ):
        raise AuditError(f"raw alpha manifest shape changed for {file_name}")
    if float(alpha_manifest_row["raw_alpha_min"]) != float(raw_alpha.min()):
        raise AuditError(f"raw alpha manifest min changed for {file_name}")
    if float(alpha_manifest_row["raw_alpha_max"]) != float(raw_alpha.max()):
        raise AuditError(f"raw alpha manifest max changed for {file_name}")
    if abs(float(alpha_manifest_row["raw_alpha_mean"]) - float(raw_alpha.mean())) > 1e-7:
        raise AuditError(f"raw alpha manifest mean changed for {file_name}")
    if (
        metric_float(second_stage_row, "base_edit_threshold", file_name) != 12.0
        or metric_float(second_stage_row, "second_delta_threshold", file_name) != 32.0
        or metric_float(second_stage_row, "dark_threshold", file_name) != 0.0
    ):
        raise AuditError(f"second-stage protocol changed for {file_name}")

    second_float = second_stage.astype(np.float32)
    target_float = target.astype(np.float32)
    positive_mask = (target_float.mean(axis=2) - second_float.mean(axis=2)) > margin_gray
    positive_indices, preserve_indices = balanced_indices(positive_mask, file_name, sample_cap)
    indices = np.concatenate([positive_indices, preserve_indices])
    labels = np.concatenate(
        [
            np.ones(len(positive_indices), dtype=np.int8),
            -np.ones(len(preserve_indices), dtype=np.int8),
        ]
    )
    second_features = second_float.reshape(-1, 3)[indices] / 255.0
    alpha_feature = raw_alpha.reshape(-1, 1)[indices]
    features = np.column_stack([second_features, alpha_feature]).astype(
        np.float32, copy=False
    )
    if features.shape != (len(indices), len(CHANNELS)):
        raise AuditError(f"second-stage alpha feature shape changed for {file_name}")
    if not np.isfinite(features).all():
        raise AuditError(f"non-finite second-stage alpha feature for {file_name}")

    sample_digest = hashlib.sha256()
    sample_digest.update(file_name.encode("utf-8"))
    sample_digest.update(positive_indices.astype("<i8").tobytes())
    sample_digest.update(preserve_indices.astype("<i8").tobytes())
    return {
        "file": file_name,
        "fold": fold_for_name(file_name),
        "features": features,
        "ablation_features": features[:, :3],
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
                "positive_score_mean": float(page_scores[page["labels"] > 0].mean()),
                "preserve_score_mean": float(page_scores[page["labels"] < 0].mean()),
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
        "page_auc_median": float(np.median([row["auc"] for row in page_results])),
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
        "primary_metrics",
        "role_plan",
        "second_stage_inference_source",
        "second_stage_metrics",
        "source_output_kill",
    )
    paths = {
        name: validate_file_artifact(repo_root, evidence[name], name)
        for name in file_evidence_names
    }
    data = plan["data"]
    manifest_path = validate_file_artifact(repo_root, data["manifest"], "train manifest")
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
    base_role_artifact = role_wrapper.get("evidence", {}).get("base_role_contract")
    if not isinstance(base_role_artifact, dict):
        raise AuditError("monotonic role plan lacks its base role contract")
    base_role_path = validate_file_artifact(
        repo_root, base_role_artifact, "base role contract"
    )
    if sorted(file_names) != effective_train_filenames(repo_root, base_role_path):
        raise AuditError("train manifest no longer matches frozen roles")

    primary_dir, primary_prediction_summary = validate_prediction_set(
        repo_root,
        evidence["primary_prediction_set"],
        file_names,
        "primary prediction set",
    )
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
    label_summary = validate_label_set(label_dir, file_names, evidence["train_label_set"])

    alpha_root = repo_path(repo_root, plan["second_stage_alpha_materialization"]["output_root"])
    alpha_entries = {entry.name for entry in alpha_root.iterdir()} if alpha_root.is_dir() else set()
    if alpha_entries != {"manifest.json", "pages"}:
        raise AuditError("alpha output surface changed")
    alpha_manifest_path = alpha_root / "manifest.json"
    if not alpha_manifest_path.is_file():
        raise AuditError("alpha materialization manifest is missing")
    alpha_manifest = read_json(alpha_manifest_path)
    required_manifest = {
        "schema_version": 1,
        "terminal": "PASS",
        "provenance": "current_second_stage_erasemap_alpha_head_sigmoid",
        "target_access": False,
        "train_count": len(file_names),
        "tile_size": 160,
        "stride": 160,
        "batch_size": 32,
        "dtype": "float32",
        "encoding": "one_compressed_npz_per_page_with_exact_raw_alpha_key",
        "raw_alpha_key": RAW_ALPHA_KEY,
        "overlap_fusion": "arithmetic_mean_of_raw_patch_alpha_before_any_threshold",
        "output_root": str(alpha_root.relative_to(repo_root)),
        "pages_directory": "pages",
    }
    for key, expected in required_manifest.items():
        if alpha_manifest.get(key) != expected:
            raise AuditError(f"alpha materialization manifest changed: {key}")
    for key, expected_path in {
        "plan": plan_file,
        "source_manifest": manifest_path,
        "primary_config": paths["current_primary_config"],
        "primary_checkpoint": paths["current_primary_checkpoint"],
        "second_stage_checkpoint": paths["current_second_stage_checkpoint"],
        "second_stage_inference_source": paths["second_stage_inference_source"],
    }.items():
        row = alpha_manifest.get(key)
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise AuditError(f"alpha materialization {key} contract changed")
        if row["path"] != str(expected_path.relative_to(repo_root)):
            raise AuditError(f"alpha materialization {key} path changed")
        if row["sha256"] != sha256_file(expected_path):
            raise AuditError(f"alpha materialization {key} sha256 changed")

    alpha_rows = alpha_manifest.get("pages")
    if not isinstance(alpha_rows, list) or len(alpha_rows) != len(file_names):
        raise AuditError("alpha materialization page count changed")
    alpha_dir = alpha_root / "pages"
    if not alpha_dir.is_dir():
        raise AuditError("alpha materialization pages are missing")
    page_entries = list(alpha_dir.iterdir())
    if any(entry.is_dir() for entry in page_entries):
        raise AuditError("alpha page directory contains subdirectories")
    expected_npz_names = sorted(f"{Path(file_name).stem}.npz" for file_name in file_names)
    actual_npz_names = sorted(entry.name for entry in page_entries)
    if actual_npz_names != expected_npz_names:
        raise AuditError("alpha page filenames changed")
    alpha_by_name: dict[str, dict[str, Any]] = {}
    for row in alpha_rows:
        if not isinstance(row, dict) or set(row) != {
            "file",
            "height",
            "npz_sha256",
            "raw_alpha_max",
            "raw_alpha_mean",
            "raw_alpha_min",
            "source_prediction_sha256",
            "width",
        }:
            raise AuditError("alpha materialization page schema changed")
        file_name = str(row["file"])
        if file_name in alpha_by_name:
            raise AuditError(f"duplicate alpha materialization page: {file_name}")
        if file_name not in file_names:
            raise AuditError("alpha materialization identities changed")
        height = row["height"]
        width = row["width"]
        if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
            raise AuditError(f"invalid alpha summary dimensions: {file_name}")
        for key in ("raw_alpha_min", "raw_alpha_max", "raw_alpha_mean"):
            value = row[key]
            if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
                raise AuditError(f"invalid alpha summary value: {file_name}")
            if float(value) < 0.0 or float(value) > 1.0:
                raise AuditError(f"alpha summary value escaped unit interval: {file_name}")
        npz_path = alpha_dir / f"{Path(file_name).stem}.npz"
        if not npz_path.is_file():
            raise AuditError(f"missing raw alpha NPZ: {file_name}")
        if sha256_file(npz_path) != row["npz_sha256"]:
            raise AuditError(f"raw alpha content hash changed: {file_name}")
        primary_prediction_path = primary_dir / f"{Path(file_name).stem}.png"
        if sha256_file(primary_prediction_path) != row["source_prediction_sha256"]:
            raise AuditError(f"alpha source prediction hash changed: {file_name}")
        alpha_by_name[file_name] = row
    if sorted(alpha_by_name) != sorted(file_names):
        raise AuditError("alpha materialization identities changed")

    pages: list[dict[str, Any]] = []
    margin = float(data["target_lighter_margin_gray"])
    sample_cap = int(plan["diagnostic"]["max_samples_per_class_per_page"])
    for index, file_name in enumerate(file_names, start=1):
        pages.append(
            build_page(
                file_name=file_name,
                second_stage_dir=second_stage_dir,
                label_dir=label_dir,
                alpha_dir=alpha_dir,
                alpha_manifest_row=alpha_by_name[file_name],
                second_stage_row=second_rows[file_name],
                margin_gray=margin,
                sample_cap=sample_cap,
            )
        )
        if index % 25 == 0 or index == len(file_names):
            print(
                f"decoded train-only second-stage alpha pages {index}/{len(file_names)}",
                flush=True,
            )

    fold_counts = {fold: sum(page["fold"] == fold for page in pages) for fold in range(5)}
    if any(count == 0 for count in fold_counts.values()):
        raise AuditError("page hash assignment produced an empty fold")
    folds = [evaluate_fold(pages, fold, float(plan["diagnostic"]["lambda"])) for fold in range(5)]
    full_aucs = [float(fold["full_auc"]) for fold in folds]
    ablation_aucs = [float(fold["ablation_auc"]) for fold in folds]
    page_aucs = [float(page["auc"]) for fold in folds for page in fold["pages"]]
    positive_above_preserve = sum(
        fold["positive_score_mean"] > fold["preserve_score_mean"] for fold in folds
    )
    aggregates = {
        "full_mean_fold_auc": float(np.mean(full_aucs)),
        "full_min_fold_auc": min(full_aucs),
        "ablation_mean_fold_auc": float(np.mean(ablation_aucs)),
        "full_auc_ablation_margin": float(np.mean(full_aucs) - np.mean(ablation_aucs)),
        "macro_median_page_auc": float(np.median(page_aucs)),
        "positive_mean_above_preserve_folds": positive_above_preserve,
    }
    contract = plan["acceptance"]
    conditions = {
        "full_mean_fold_auc": aggregates["full_mean_fold_auc"] >= contract["full_mean_fold_auc_min"],
        "every_fold_auc": aggregates["full_min_fold_auc"] >= contract["full_fold_auc_min"],
        "macro_median_page_auc": (
            aggregates["macro_median_page_auc"] >= contract["macro_median_page_auc_min"]
        ),
        "positive_mean_above_preserve": (
            positive_above_preserve >= contract["positive_mean_above_preserve_min_folds"]
        ),
        "full_auc_ablation_margin": (
            aggregates["full_auc_ablation_margin"] >= contract["full_auc_ablation_margin_min"]
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
        "primary_predictions": primary_prediction_summary,
        "second_stage_predictions": second_prediction_summary,
        "materialization": {
            "manifest_path": str(alpha_manifest_path.relative_to(repo_root)),
            "manifest_sha256": sha256_file(alpha_manifest_path),
            "target_access": alpha_manifest["target_access"],
            "count": len(alpha_rows),
            "content_sha256": hashlib.sha256(
                "\n".join(
                    sorted(f"{row['file']} {row['npz_sha256']}" for row in alpha_rows)
                ).encode("utf-8") + b"\n"
            ).hexdigest(),
        },
        "train_labels": label_summary,
        "sample_count": sum(len(page["labels"]) for page in pages),
        "page_samples": [
            {
                **{
                    key: page[key]
                    for key in (
                        "file",
                        "fold",
                        "samples_per_class",
                        "positive_pixel_count",
                        "preserve_pixel_count",
                        "sample_sha256",
                    )
                },
                "npz_sha256": alpha_by_name[page["file"]]["npz_sha256"],
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
            else "close_exact_second_stage_alpha_support_diagnostic"
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
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"terminal={result['terminal']} output={output_path}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
