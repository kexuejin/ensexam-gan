#!/usr/bin/env python3
"""Audit train-only dual-input support separation before model training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.build_sign_separated_residual_patch_index import (  # noqa: E402
    effective_train_filenames,
    sha256_rows,
)


PLAN_PATH = Path("docs/dual-input-support-separation-prerequisite-v1.json")
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
OUTPUT_PATH = Path(
    "outputs/dual-input-support-separation-prerequisite-20260811/audit.json"
)
ACTIVE_ITERATION_ID = "monotonic-residual-erase-support"
PREREGISTRATION_ID = (
    "monotonic_residual_erase_support_separation_preregistration"
)
DIAGNOSTIC_ID = "dual_input_support_separation_train_only_diagnostic"
NEXT_PREREGISTRATION_ID = "spatial_primary_mask_support_preregistration"


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object: {path}")
    return value


def repo_path(repo_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise AuditError(f"path must stay repository-relative: {value}")
    return repo_root / relative


def validate_file_artifact(
    repo_root: Path, artifact: dict[str, Any], label: str
) -> Path:
    if set(artifact) != {"path", "sha256"}:
        raise AuditError(f"{label} must contain path and sha256")
    path = repo_path(repo_root, str(artifact["path"]))
    if not path.is_file():
        raise AuditError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != artifact["sha256"]:
        raise AuditError(f"{label} sha256 changed")
    return path


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
        raise AuditError("dual-input preregistration is not passed")
    if prerequisites.get(DIAGNOSTIC_ID) != "passed":
        raise AuditError("train-only diagnostic KILL is not recorded")
    if prerequisites.get(NEXT_PREREGISTRATION_ID) != "pending":
        raise AuditError("next support preregistration is not pending")


def validate_plan(plan: dict[str, Any]) -> None:
    exact = {
        "schema_version": 1,
        "iteration_id": ACTIVE_ITERATION_ID,
        "state": "preregistered_pending_train_only_diagnostic",
        "next_boundary_on_pass": (
            "dual_input_support_data_and_training_preflight_only"
        ),
    }
    for key, expected in exact.items():
        if plan.get(key) != expected:
            raise AuditError(f"plan field changed: {key}")
    representation = plan.get("representation", {})
    if (
        representation.get("feature_count") != 13
        or not representation.get("page_features_are_target_free")
        or not representation.get("pixel_features_are_target_free")
        or len(representation.get("channels", [])) != 13
    ):
        raise AuditError("registered representation changed")
    diagnostic = plan.get("diagnostic", {})
    required_diagnostic = {
        "fold_count": 5,
        "lambda": 1.0,
        "max_samples_per_class_per_page": 1024,
        "optimizer": "none_closed_form_ridge",
        "numeric_type": "float64",
        "luma_definition": "arithmetic_mean_rgb_gray_units",
        "threshold_derivation": "prohibited",
    }
    for key, expected in required_diagnostic.items():
        if diagnostic.get(key) != expected:
            raise AuditError(f"diagnostic field changed: {key}")
    if diagnostic.get("ablation_features") != [
        "second_stage_r",
        "second_stage_g",
        "second_stage_b",
    ]:
        raise AuditError("ablation feature set changed")
    acceptance = plan.get("acceptance", {})
    required_acceptance = {
        "full_auc_ablation_margin_min": 0.03,
        "full_fold_auc_min": 0.55,
        "full_mean_fold_auc_min": 0.65,
        "macro_median_page_auc_min": 0.6,
        "positive_mean_above_preserve_min_folds": 4,
        "required_fold_count": 5,
        "required_terminal_on_pass": "PASS",
    }
    if acceptance != required_acceptance:
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


def read_metric_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_file: dict[str, dict[str, str]] = {}
    for row in rows:
        file_name = Path(row.get("file", "")).name
        if not file_name or file_name in by_file:
            raise AuditError(f"invalid or duplicate metric row: {file_name}")
        by_file[file_name] = row
    return by_file


def validate_prediction_set(
    repo_root: Path, spec: dict[str, Any], expected_names: list[str], label: str
) -> tuple[Path, dict[str, Any]]:
    expected_keys = {
        "content_sha256",
        "count",
        "directory",
        "filename_sha256",
    }
    if set(spec) != expected_keys:
        raise AuditError(f"{label} specification changed")
    directory = repo_path(repo_root, str(spec["directory"]))
    paths = sorted(directory.glob("*.png"), key=lambda path: path.name)
    names = [path.name for path in paths]
    expected_png_names = sorted(f"{Path(name).stem}.png" for name in expected_names)
    if names != expected_png_names or len(paths) != int(spec["count"]):
        raise AuditError(f"{label} filenames or count changed")
    filename_sha256 = sha256_rows(names)
    content_rows = [f"{path.name} {sha256_file(path)}" for path in paths]
    content_sha256 = sha256_rows(sorted(content_rows))
    if filename_sha256 != spec["filename_sha256"]:
        raise AuditError(f"{label} filename hash changed")
    if content_sha256 != spec["content_sha256"]:
        raise AuditError(f"{label} content hash changed")
    return directory, {
        "count": len(paths),
        "filename_sha256": filename_sha256,
        "content_sha256": content_sha256,
    }


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise AuditError(f"image decode failed: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def fold_for_name(file_name: str) -> int:
    digest = hashlib.sha256(file_name.encode("utf-8")).hexdigest()
    return int(digest, 16) % 5


def splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    z = values.astype(np.uint64, copy=True)
    z ^= np.uint64(seed)
    z += np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def lowest_ranked_indices(
    indices: np.ndarray, file_name: str, count: int
) -> np.ndarray:
    if count <= 0 or count > len(indices):
        raise AuditError("invalid deterministic sample count")
    seed = int(
        hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:16], 16
    )
    keys = splitmix64(indices, seed)
    if len(indices) > count:
        selected = np.argpartition(keys, count - 1)[:count]
        indices = indices[selected]
        keys = keys[selected]
    order = np.lexsort((indices, keys))
    return indices[order].astype(np.int64, copy=False)


def balanced_indices(
    positive_mask: np.ndarray, file_name: str, cap: int
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(positive_mask.reshape(-1))
    preserve = np.flatnonzero(~positive_mask.reshape(-1))
    count = min(cap, len(positive), len(preserve))
    if count <= 0:
        raise AuditError(f"page lacks both support classes: {file_name}")
    return (
        lowest_ranked_indices(positive, file_name, count),
        lowest_ranked_indices(preserve, file_name, count),
    )


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels > 0
    positive_count = int(positive.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise AuditError("AUC requires both classes")
    if not np.isfinite(scores).all():
        raise AuditError("non-finite diagnostic score")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        sorted_ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = sorted_ranks
    rank_sum = float(ranks[positive].sum())
    baseline = positive_count * (positive_count + 1) / 2.0
    return (rank_sum - baseline) / (positive_count * negative_count)


def fit_closed_form_ridge(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    train = np.asarray(train_features, dtype=np.float64)
    test = np.asarray(test_features, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.float64)
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise AuditError("non-finite feature matrix")
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    train_standard = (train - mean) / scale
    test_standard = (test - mean) / scale
    train_design = np.column_stack(
        [train_standard, np.ones(len(train_standard), dtype=np.float64)]
    )
    test_design = np.column_stack(
        [test_standard, np.ones(len(test_standard), dtype=np.float64)]
    )
    penalty = np.eye(train_design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    system = train_design.T @ train_design + ridge_lambda * penalty
    target = train_design.T @ labels
    try:
        weights = np.linalg.solve(system, target)
    except np.linalg.LinAlgError as error:
        raise AuditError("closed-form ridge system is singular") from error
    scores = test_design @ weights
    if not np.isfinite(weights).all() or not np.isfinite(scores).all():
        raise AuditError("non-finite ridge result")
    return scores, {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "weights": weights.tolist(),
    }


def concatenate_pages(
    pages: list[dict[str, Any]], feature_indices: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    features = np.concatenate(
        [page["features"][:, feature_indices] for page in pages], axis=0
    )
    labels = np.concatenate([page["labels"] for page in pages], axis=0)
    return features, labels


def gate_feature_strata(
    train_pages: list[dict[str, Any]],
    test_pages: list[dict[str, Any]],
    scores_by_name: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    names = (
        "copy_mask_cov8",
        "primary_edit_px",
        "primary_p95_edit_delta",
        "second_stage_gate_ratio",
    )
    for name in names:
        fit_values = np.asarray(
            [page["gate_features"][name] for page in train_pages],
            dtype=np.float64,
        )
        cuts = np.quantile(fit_values, [0.25, 0.5, 0.75])
        bins: list[dict[str, Any]] = []
        for bin_index in range(4):
            members = [
                page
                for page in test_pages
                if int(
                    np.searchsorted(
                        cuts,
                        page["gate_features"][name],
                        side="right",
                    )
                )
                == bin_index
            ]
            if members:
                labels = np.concatenate(
                    [page["labels"] for page in members], axis=0
                )
                scores = np.concatenate(
                    [scores_by_name[page["file"]] for page in members],
                    axis=0,
                )
                auc = auc_rank(labels, scores)
            else:
                auc = None
            bins.append(
                {
                    "bin": bin_index,
                    "page_count": len(members),
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
    full_indices = list(range(13))
    ablation_indices = [3, 4, 5]
    train_full, train_labels = concatenate_pages(train_pages, full_indices)
    test_full, test_labels = concatenate_pages(test_pages, full_indices)
    full_scores, full_fit = fit_closed_form_ridge(
        train_full, train_labels, test_full, ridge_lambda
    )
    train_ablation, _ = concatenate_pages(train_pages, ablation_indices)
    test_ablation, _ = concatenate_pages(test_pages, ablation_indices)
    ablation_scores, ablation_fit = fit_closed_form_ridge(
        train_ablation, train_labels, test_ablation, ridge_lambda
    )
    full_auc = auc_rank(test_labels, full_scores)
    ablation_auc = auc_rank(test_labels, ablation_scores)
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
        "gate_feature_strata": gate_feature_strata(
            train_pages, test_pages, scores_by_name
        ),
        "pages": page_results,
    }


def metric_float(row: dict[str, str], key: str, file_name: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise AuditError(f"invalid {key} for {file_name}") from error
    if not math.isfinite(value):
        raise AuditError(f"non-finite {key} for {file_name}")
    return value


def build_page(
    *,
    repo_root: Path,
    file_name: str,
    source_path: Path,
    label_dir: Path,
    primary_dir: Path,
    second_stage_dir: Path,
    primary_row: dict[str, str],
    second_stage_row: dict[str, str],
    margin_gray: float,
    sample_cap: int,
) -> dict[str, Any]:
    source = read_rgb(source_path)
    primary = read_rgb(primary_dir / f"{Path(file_name).stem}.png")
    second_stage = read_rgb(
        second_stage_dir / f"{Path(file_name).stem}.png"
    )
    target = read_rgb(label_dir / file_name)
    if not (
        source.shape == primary.shape == second_stage.shape == target.shape
    ):
        raise AuditError(f"image shape mismatch for {file_name}")
    if sha256_file(source_path) != primary_row.get("image_sha256"):
        raise AuditError(f"source image hash changed for {file_name}")
    if (
        metric_float(second_stage_row, "base_edit_threshold", file_name)
        != 12.0
        or metric_float(
            second_stage_row, "second_delta_threshold", file_name
        )
        != 32.0
        or metric_float(second_stage_row, "dark_threshold", file_name) != 0.0
    ):
        raise AuditError(f"second-stage protocol changed for {file_name}")

    source_float = source.astype(np.float32)
    primary_float = primary.astype(np.float32)
    second_float = second_stage.astype(np.float32)
    target_float = target.astype(np.float32)
    primary_delta = np.abs(primary_float - source_float).mean(axis=2)
    gate_features = {
        "copy_mask_cov8": metric_float(
            primary_row, "copy_mask_cov8", file_name
        ),
        "primary_edit_px": float((primary_delta >= 12.0).sum()),
        "primary_p95_edit_delta": float(np.percentile(primary_delta, 95)),
        "second_stage_gate_ratio": metric_float(
            second_stage_row, "gate_ratio", file_name
        ),
    }
    if not all(math.isfinite(value) for value in gate_features.values()):
        raise AuditError(f"non-finite gate feature for {file_name}")
    target_luma = target_float.mean(axis=2)
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
    primary_flat = primary_float.reshape(-1, 3)[indices] / 255.0
    second_flat = second_float.reshape(-1, 3)[indices] / 255.0
    signed_delta = second_flat - primary_flat
    page_values = np.asarray(list(gate_features.values()), dtype=np.float32)
    page_matrix = np.broadcast_to(page_values, (len(indices), 4))
    features = np.column_stack(
        [primary_flat, second_flat, signed_delta, page_matrix]
    ).astype(np.float32, copy=False)
    if features.shape[1] != 13 or not np.isfinite(features).all():
        raise AuditError(f"invalid feature matrix for {file_name}")
    sample_digest = hashlib.sha256()
    sample_digest.update(file_name.encode("utf-8"))
    sample_digest.update(positive_indices.astype("<i8").tobytes())
    sample_digest.update(preserve_indices.astype("<i8").tobytes())
    return {
        "file": file_name,
        "fold": fold_for_name(file_name),
        "features": features,
        "labels": labels,
        "gate_features": gate_features,
        "samples_per_class": len(positive_indices),
        "positive_pixel_count": int(positive_mask.sum()),
        "preserve_pixel_count": int((~positive_mask).sum()),
        "sample_sha256": sample_digest.hexdigest(),
    }


def validate_label_set(
    label_dir: Path, file_names: list[str], spec: dict[str, Any]
) -> dict[str, Any]:
    if (
        spec.get("count") != len(file_names)
        or Path(str(spec.get("directory"))) != Path(
            "data-links/samples/SCUT-HW5K-mixed-20260729/train/all_labels"
        )
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
        "current_second_stage_checkpoint",
        "killed_checkpoint_decision",
        "materialization_audit",
        "primary_metrics",
        "role_plan",
        "second_stage_metrics",
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
    effective_names = effective_train_filenames(repo_root, base_role_path)
    if sorted(file_names) != effective_names:
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
    label_summary = validate_label_set(
        label_dir, file_names, evidence["train_label_set"]
    )

    pages: list[dict[str, Any]] = []
    margin = float(data["target_lighter_margin_gray"])
    sample_cap = int(plan["diagnostic"]["max_samples_per_class_per_page"])
    for index, manifest_row in enumerate(manifest_rows, start=1):
        file_name = Path(manifest_row).name
        source_path = repo_path(repo_root, manifest_row)
        page = build_page(
            repo_root=repo_root,
            file_name=file_name,
            source_path=source_path,
            label_dir=label_dir,
            primary_dir=primary_dir,
            second_stage_dir=second_stage_dir,
            primary_row=primary_rows[file_name],
            second_stage_row=second_rows[file_name],
            margin_gray=margin,
            sample_cap=sample_cap,
        )
        pages.append(page)
        if index % 25 == 0 or index == len(manifest_rows):
            print(f"decoded train-only pages {index}/{len(manifest_rows)}", flush=True)
    fold_counts = {fold: sum(page["fold"] == fold for page in pages) for fold in range(5)}
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
            "file": page["file"],
            "fold": page["fold"],
            "gate_features": page["gate_features"],
            "samples_per_class": page["samples_per_class"],
            "positive_pixel_count": page["positive_pixel_count"],
            "preserve_pixel_count": page["preserve_pixel_count"],
            "sample_sha256": page["sample_sha256"],
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
        "feature_count": 13,
        "feature_names": plan["representation"]["channels"],
        "train_page_count": len(pages),
        "fold_page_counts": fold_counts,
        "primary_predictions": primary_prediction_summary,
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
            else "close_exact_dual_input_support_diagnostic"
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"terminal={result['terminal']} output={output_path}",
        flush=True,
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
