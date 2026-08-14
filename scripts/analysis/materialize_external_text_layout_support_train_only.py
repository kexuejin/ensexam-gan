#!/usr/bin/env python3
"""Materialize frozen target-free external text-layout support for train275."""

from __future__ import annotations

import argparse
import csv
from importlib.metadata import PackageNotFoundError, version
import json
import multiprocessing
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.build_sign_separated_residual_patch_index import (  # noqa: E402
    effective_train_filenames,
)
from scripts.analysis import external_text_layout_materialization_runtime as runtime  # noqa: E402
from scripts.analysis.external_text_layout_transformers_runtime_repair import (  # noqa: E402
    RuntimeEquivalenceRepairError,
    apply_runtime_equivalence_repair,
)


PLAN_PATH = Path("docs/external-text-layout-support-prerequisite-v1.json")
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
OUTPUT_ROOT = Path("outputs/external-text-layout-support-materialization-20260813")
FAMILY = "external_printed_text_layout_support_v1"
ACTIVE_ITERATION_ID = "monotonic-residual-erase-support"
PREREGISTRATION_ID = "materially_new_support_successor_preregistration_v4"
DIAGNOSTIC_ID = "external_text_layout_support_train_only_diagnostic"
NPZ_KEYS = runtime.NPZ_KEYS
PAGE_ROW_KEYS = runtime.PAGE_ROW_KEYS
MAX_DETECTOR_RSS_BYTES = runtime.MAX_DETECTOR_RSS_BYTES
MIN_MEMORY_FREE_PERCENT = runtime.MIN_MEMORY_FREE_PERCENT
MAX_SWAP_USED_BYTES = runtime.MAX_SWAP_USED_BYTES
FORBIDDEN_SOURCE_COMPONENTS = {
    "all_labels",
    "development",
    "holdout40",
    "inner_val15",
    "label",
    "labels",
    "reserved_blind",
    "scut115",
    "target",
    "targets",
}
FORBIDDEN_MANIFEST_FIELDS = {
    "character",
    "domain",
    "label",
    "recognition_text",
    "route",
    "split",
    "target",
    "text_content",
}


MaterializationError = runtime.MaterializationError
sha256_file = runtime.sha256_file
sha256_rows = runtime.sha256_rows
read_json = runtime.read_json
fsync_directory = runtime.fsync_directory
atomic_write_json = runtime.atomic_write_json
atomic_write_npz = runtime.atomic_write_npz
assert_no_conflicting_model_processes = runtime.assert_no_conflicting_model_processes
process_tree_rss_bytes = runtime.process_tree_rss_bytes
runtime_health = runtime.runtime_health
enforce_health_limits = runtime.enforce_health_limits


def repo_path(repo_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise MaterializationError(f"path must stay repository-relative: {value}")
    return repo_root / relative


def validate_internal_artifact(
    repo_root: Path, artifact: dict[str, Any], label: str
) -> Path:
    if set(artifact) != {"path", "sha256"}:
        raise MaterializationError(f"{label} artifact contract changed")
    path = repo_path(repo_root, str(artifact["path"]))
    if not path.is_file():
        raise MaterializationError(f"missing {label}: {path}")
    if sha256_file(path) != artifact["sha256"]:
        raise MaterializationError(f"{label} sha256 changed")
    return path


def validate_external_artifact(artifact: dict[str, Any], label: str) -> Path:
    if set(artifact) != {"external_path", "sha256"}:
        raise MaterializationError(f"{label} external artifact contract changed")
    path = Path(str(artifact["external_path"]))
    if not path.is_absolute() or not path.is_file():
        raise MaterializationError(f"missing {label}: {path}")
    if sha256_file(path) != artifact["sha256"]:
        raise MaterializationError(f"{label} sha256 changed")
    return path


def validate_runtime(expected: dict[str, str]) -> dict[str, str]:
    actual = {
        "cv2": cv2.__version__,
        "numpy": np.__version__,
        "paddle": _package_version("paddlepaddle"),
        "paddleocr": _package_version("paddleocr"),
        "paddlex": _package_version("paddlex"),
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
    }
    if actual != expected:
        raise MaterializationError(
            f"registered runtime changed: expected {expected}, got {actual}"
        )
    return actual


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as error:
        raise MaterializationError(f"registered runtime package missing: {name}") from error


def validate_authority(ledger: dict[str, Any]) -> None:
    program = ledger.get("program", {})
    if (
        program.get("product_default") != "artifacts/current-primary"
        or program.get("promotion_state") != "disabled"
        or program.get("reserved_blind_state") != "disabled"
    ):
        raise MaterializationError("quality-loop program authority changed")
    active = ledger.get("active_iteration", {})
    if (
        active.get("id") != ACTIVE_ITERATION_ID
        or active.get("terminal") != "PREREQUISITE_NEEDED"
        or active.get("first_gate") != "scut_inner_val15"
    ):
        raise MaterializationError("active iteration authority changed")
    statuses = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if statuses.get(PREREGISTRATION_ID) != "passed":
        raise MaterializationError("external text-layout preregistration is not passed")
    if statuses.get(DIAGNOSTIC_ID) != "pending":
        raise MaterializationError("external text-layout diagnostic is not pending")
    if statuses.get("independent_hw5k_expert_support_train_only_diagnostic") != "passed":
        raise MaterializationError("prior independent-expert KILL is not recorded")


def validate_plan(plan: dict[str, Any]) -> None:
    exact = {
        "family": FAMILY,
        "iteration_id": ACTIVE_ITERATION_ID,
        "next_boundary_on_pass": (
            "external_text_layout_conditioned_data_training_and_application_preflight_only"
        ),
        "schema_version": 1,
        "state": "preregistered_pending_external_text_layout_materialization",
    }
    for key, expected in exact.items():
        if plan.get(key) != expected:
            raise MaterializationError(f"plan field changed: {key}")
    if plan.get("representation") != {
        "ablation_is_identical_probe_without_external_layout_channels": True,
        "channels": [
            "second_stage_r",
            "second_stage_g",
            "second_stage_b",
            "external_text_occupancy",
            "external_text_confidence",
        ],
        "feature_count": 5,
        "forbidden_features": [
            "ocr_recognition_or_character_content",
            "source_rgb_masks_alpha_reconstruction_stages_or_expert_outputs",
            "domain_split_route_or_caller_metadata",
            "polygon_coordinates_geometry_distances_neighborhoods_or_transforms",
            "detector_threshold_parameter_model_or_runtime_sweep",
            "target_derived_or_quality_derived_features",
            "nonlinear_probe",
        ],
        "single_causal_change": (
            "add_frozen_external_text_detection_occupancy_and_confidence_"
            "to_second_stage_rgb_support_evidence"
        ),
    }:
        raise MaterializationError("registered external text-layout representation changed")
    required_data = {
        "data_root": "data-links/samples/SCUT-HW5K-mixed-20260729",
        "effective_train_count": 275,
        "effective_train_domain_counts": {"hw5k": 253, "scut": 22},
        "effective_train_filename_sha256": (
            "e9ac4d6f700f41ef3a9b7c3f04ce0593f593324a881a0f9fc387901a497f9039"
        ),
        "second_stage_prediction_dir": (
            "outputs/archive/sign-separated-residual-repair-20260810/"
            "train275-frozen-pipeline/pred"
        ),
        "source": "exact_manifest_order_raw_input_pages",
        "split": "train",
        "target_lighter_margin_gray": 2.0,
    }
    data = plan.get("data", {})
    for key, expected in required_data.items():
        if data.get(key) != expected:
            raise MaterializationError(f"data field changed: {key}")
    diagnostic = plan.get("diagnostic", {})
    required_diagnostic = {
        "ablation_features": [
            "second_stage_r",
            "second_stage_g",
            "second_stage_b",
        ],
        "auc_tie_policy": "average_rank",
        "class_balance": "equal_per_page_target_lighter_and_preserve",
        "coordinate_rank": (
            "splitmix64(row_major_index XOR first_64_bits(sha256(utf8_basename))) "
            "then unsigned_hash_y_x"
        ),
        "feature_normalization": (
            "rgb_divided_by_255_occupancy_in_zero_or_one_confidence_in_unit_interval"
        ),
        "fold_assignment": "int(sha256(utf8_basename), 16) % 5",
        "fold_count": 5,
        "full_feature_count": 5,
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
        "target_definition": (
            "target_luma_minus_second_stage_luma_greater_than_2_gray"
        ),
        "target_encoding": {"preserve": -1, "target_lighter": 1},
        "threshold_derivation": "prohibited",
    }
    for key, expected in required_diagnostic.items():
        if diagnostic.get(key) != expected:
            raise MaterializationError(f"diagnostic field changed: {key}")
    if plan.get("acceptance") != {
        "full_auc_ablation_margin_min": 0.03,
        "full_fold_auc_min": 0.55,
        "full_mean_fold_auc_min": 0.65,
        "macro_median_page_auc_min": 0.6,
        "positive_mean_above_preserve_min_folds": 4,
        "required_fold_count": 5,
        "required_terminal_on_pass": "PASS",
    }:
        raise MaterializationError("acceptance contract changed")
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
        raise MaterializationError("a candidate or quality surface was opened")
    if (
        authorization.get("external_text_layout_materialization") is not True
        or authorization.get("external_text_layout_materialization_reads_targets")
        is not False
        or authorization.get("separability_diagnostic_target_decode_roles")
        != ["train"]
    ):
        raise MaterializationError("external text-layout authorization changed")
    spec = plan.get("external_text_layout_materialization", {})
    required_spec = {
        "batch_size": 1,
        "box_thresh": 0.45,
        "device": "cpu",
        "engine": "transformers",
        "input": "exact_train275_raw_source_pages",
        "limit_side_len": 736,
        "limit_type": "min",
        "max_side_limit": 4000,
        "model_dir": (
            "/Users/kexuejin/.paddlex/official_models/"
            "PP-OCRv6_medium_det_safetensors"
        ),
        "model_name": "PP-OCRv6_medium_det",
        "output_root": str(OUTPUT_ROOT),
        "persisted_geometry": (
            "clipped_integer_quadrilaterals_sorted_by_min_y_min_x_max_y_"
            "max_x_flattened_coordinates_score"
        ),
        "persisted_grids": {
            "text_confidence": "float32_pixelwise_max_detection_score",
            "text_occupancy": "uint8_zero_or_one",
        },
        "polygon_rasterization": "cv2.fillPoly_LINE_8_shift_0",
        "recognition": False,
        "target_access": False,
        "thresh": 0.2,
        "unclip_ratio": 1.4,
    }
    if spec != required_spec:
        raise MaterializationError("external text-layout materialization changed")
    if plan.get("planned_implementation") != {
        "audit_output": "outputs/external-text-layout-support-prerequisite-20260813/audit.json",
        "audit_script": "scripts/analysis/audit_external_text_layout_support.py",
        "materialization_output": str(OUTPUT_ROOT),
        "materializer": (
            "scripts/analysis/materialize_external_text_layout_support_train_only.py"
        ),
        "test": "tests/test_external_text_layout_support_prerequisite.py",
    }:
        raise MaterializationError("planned implementation changed")
    leakage = plan.get("leakage_boundary", {})
    if leakage != {
        "diagnostic_scope": "train_role_incremental_support_only",
        "external_model_repo_training": False,
        "external_model_target_access": False,
        "published_training_corpus_overlap_with_scut_or_hw5k": "unverified",
        "quality_or_reserved_blind_access": False,
        "statement": (
            "Even PASS cannot establish product generalization because upstream "
            "corpus overlap is unverified; it can authorize only a separate "
            "leakage-aware preflight."
        ),
    }:
        raise MaterializationError("external model leakage boundary changed")


def assert_source_path(path: Path) -> None:
    for candidate in (path, path.resolve()):
        lowered = {part.lower() for part in candidate.parts}
        if lowered & FORBIDDEN_SOURCE_COMPONENTS:
            raise MaterializationError(f"source path appears forbidden: {path}")


def read_metric_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_file: dict[str, dict[str, str]] = {}
    for row in rows:
        file_name = Path(row.get("file", "")).name
        if not file_name or file_name in by_file:
            raise MaterializationError(f"invalid or duplicate metric row: {file_name}")
        by_file[file_name] = row
    return by_file


def read_manifest(
    repo_root: Path,
    manifest_path: Path,
    primary_rows: dict[str, dict[str, str]],
) -> list[tuple[str, str, Path]]:
    result: list[tuple[str, str, Path]] = []
    seen: set[str] = set()
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value:
            continue
        source = repo_path(repo_root, value)
        assert_source_path(source)
        file_name = source.name
        if file_name in seen or not source.is_file():
            raise MaterializationError(f"invalid or duplicate train source: {value}")
        row = primary_rows.get(file_name)
        if row is None:
            raise MaterializationError(f"source is absent from primary metrics: {file_name}")
        if row.get("image_path") != value:
            raise MaterializationError(f"registered source path changed: {file_name}")
        if sha256_file(source) != row.get("image_sha256"):
            raise MaterializationError(f"registered source content changed: {file_name}")
        seen.add(file_name)
        result.append((file_name, value, source))
    if len(result) != 275 or set(primary_rows) != seen:
        raise MaterializationError("train source population changed")
    return result


def validate_registered_inputs(
    repo_root: Path,
    plan: dict[str, Any],
    ledger: dict[str, Any],
    *,
    require_output_absent: bool,
) -> dict[str, Any]:
    validate_plan(plan)
    validate_authority(ledger)
    evidence = plan.get("evidence", {})
    internal_names = (
        "current_primary_checkpoint",
        "current_primary_config",
        "current_second_stage_checkpoint",
        "independent_expert_kill",
        "primary_metrics",
        "role_plan",
        "second_stage_metrics",
    )
    paths = {
        name: validate_internal_artifact(repo_root, evidence[name], name)
        for name in internal_names
    }
    manifest_path = validate_internal_artifact(
        repo_root, plan["data"]["manifest"], "train manifest"
    )
    role_wrapper = read_json(paths["role_plan"])
    base_role = role_wrapper.get("evidence", {}).get("base_role_contract")
    if not isinstance(base_role, dict):
        raise MaterializationError("role plan lacks base role contract")
    base_role_path = validate_internal_artifact(
        repo_root, base_role, "base role contract"
    )
    primary_rows = read_metric_rows(paths["primary_metrics"])
    sources = read_manifest(repo_root, manifest_path, primary_rows)
    file_names = [name for name, _relative, _path in sources]
    if sorted(file_names) != effective_train_filenames(repo_root, base_role_path):
        raise MaterializationError("train manifest no longer matches frozen roles")
    filename_sha256 = sha256_rows(file_names)
    if filename_sha256 != plan["data"]["effective_train_filename_sha256"]:
        raise MaterializationError("train source filename hash changed")
    external = evidence.get("official_text_detector", {})
    if external.get("license") != "Apache-2.0" or external.get("model_name") != "PP-OCRv6_medium_det":
        raise MaterializationError("external detector identity or license changed")
    model_paths = {
        key: validate_external_artifact(external[key], f"detector {key}")
        for key in (
            "gitattributes",
            "config_json",
            "inference_yml",
            "model_safetensors",
            "preprocessor_config_json",
            "readme_md",
        )
    }
    model_dir = Path(plan["external_text_layout_materialization"]["model_dir"])
    if not model_dir.is_dir() or any(path.parent != model_dir for path in model_paths.values()):
        raise MaterializationError("external detector directory changed")
    model_surface = {path.name for path in model_dir.iterdir() if path.is_file()}
    model_directories = {path.name for path in model_dir.iterdir() if path.is_dir()}
    if model_surface != {
        ".gitattributes",
        "README.md",
        "config.json",
        "inference.yml",
        "model.safetensors",
        "preprocessor_config.json",
    } or model_directories != {".cache"}:
        raise MaterializationError("external detector directory surface changed")
    runtime = validate_runtime(evidence.get("runtime", {}))
    output_root = repo_path(repo_root, str(OUTPUT_ROOT))
    temporary_root = output_root.with_name(f".{output_root.name}.materializing")
    if require_output_absent and output_root.exists():
        raise MaterializationError("external text-layout output already exists")
    return {
        "base_role_path": base_role_path,
        "file_names": file_names,
        "manifest_path": manifest_path,
        "model_paths": model_paths,
        "output_root": output_root,
        "paths": paths,
        "primary_rows": primary_rows,
        "runtime": runtime,
        "sources": sources,
        "temporary_root": temporary_root,
    }


def normalize_detections(
    polygons: Any,
    scores: Any,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    if height <= 0 or width <= 0:
        raise MaterializationError("invalid source dimensions")
    polygon_array = np.asarray(polygons)
    score_array = np.asarray(scores, dtype=np.float32)
    if polygon_array.size == 0:
        polygon_array = np.empty((0, 4, 2), dtype=np.float64)
    if score_array.size == 0:
        score_array = np.empty((0,), dtype=np.float32)
    if polygon_array.ndim != 3 or polygon_array.shape[1:] != (4, 2):
        raise MaterializationError("detector polygons must have shape [N,4,2]")
    if score_array.ndim != 1 or len(score_array) != len(polygon_array):
        raise MaterializationError("detector score count changed")
    if not np.isfinite(polygon_array).all() or not np.isfinite(score_array).all():
        raise MaterializationError("detector output contains non-finite values")
    if np.any(score_array < 0.0) or np.any(score_array > 1.0):
        raise MaterializationError("detector scores escaped unit interval")
    rounded = np.rint(polygon_array).astype(np.int64)
    rounded[:, :, 0] = np.clip(rounded[:, :, 0], 0, width - 1)
    rounded[:, :, 1] = np.clip(rounded[:, :, 1], 0, height - 1)
    clipped = rounded.astype(np.int32)
    order = sorted(
        range(len(clipped)),
        key=lambda index: (
            int(clipped[index, :, 1].min()),
            int(clipped[index, :, 0].min()),
            int(clipped[index, :, 1].max()),
            int(clipped[index, :, 0].max()),
            *[int(value) for value in clipped[index].reshape(-1)],
            float(score_array[index]),
        ),
    )
    indices = np.asarray(order, dtype=np.int64)
    return clipped[indices], score_array[indices].astype(np.float32, copy=False)


def rasterize_layout(
    polygons: np.ndarray,
    scores: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    occupancy = np.zeros((height, width), dtype=np.uint8)
    confidence = np.zeros((height, width), dtype=np.float32)
    for polygon, score in zip(polygons, scores, strict=True):
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1, lineType=cv2.LINE_8, shift=0)
        cv2.fillPoly(occupancy, [polygon], 1, lineType=cv2.LINE_8, shift=0)
        selected = mask.astype(bool)
        confidence[selected] = np.maximum(confidence[selected], np.float32(score))
    return occupancy, confidence


def extract_result(result: Any) -> tuple[Any, Any]:
    try:
        polygons = result["dt_polys"]
        scores = result["dt_scores"]
    except (KeyError, TypeError) as error:
        raise MaterializationError("detector result lacks dt_polys or dt_scores") from error
    return polygons, scores


def create_detector(spec: dict[str, Any]) -> Any:
    try:
        apply_runtime_equivalence_repair()
    except RuntimeEquivalenceRepairError as error:
        raise MaterializationError(
            "Transformers detector runtime equivalence repair failed"
        ) from error
    try:
        from paddleocr import TextDetection
    except ImportError as error:
        raise MaterializationError("PaddleOCR TextDetection import failed") from error
    return TextDetection(
        model_name=spec["model_name"],
        model_dir=spec["model_dir"],
        engine=spec["engine"],
        device=spec["device"],
    )


def predict_one(detector: Any, source_path: Path, spec: dict[str, Any]) -> Any:
    try:
        results = detector.predict(
            input=str(source_path),
            batch_size=spec["batch_size"],
            limit_side_len=spec["limit_side_len"],
            limit_type=spec["limit_type"],
            max_side_limit=spec["max_side_limit"],
            thresh=spec["thresh"],
            box_thresh=spec["box_thresh"],
            unclip_ratio=spec["unclip_ratio"],
        )
    except Exception as error:
        raise MaterializationError("external detector inference failed") from error
    if not isinstance(results, list):
        results = list(results)
    if len(results) != 1:
        raise MaterializationError("detector did not return exactly one page result")
    return results[0]


def build_manifest_payload(
    *,
    repo_root: Path,
    plan_path: Path,
    registered: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "batch_size": 1,
        "box_thresh": 0.45,
        "device": "cpu",
        "encoding": "one_compressed_npz_per_page_with_exact_layout_keys",
        "engine": "transformers",
        "family": FAMILY,
        "limit_side_len": 736,
        "limit_type": "min",
        "max_side_limit": 4000,
        "model_files": {
            key: {"external_path": str(value), "sha256": sha256_file(value)}
            for key, value in registered["model_paths"].items()
        },
        "model_name": "PP-OCRv6_medium_det",
        "output_root": str(OUTPUT_ROOT),
        "pages": rows,
        "pages_directory": "pages",
        "plan": {
            "path": str(plan_path.relative_to(repo_root)),
            "sha256": sha256_file(plan_path),
        },
        "polygon_rasterization": "cv2.fillPoly_LINE_8_shift_0",
        "recognition": False,
        "routing_metadata": False,
        "runtime": registered["runtime"],
        "schema_version": 1,
        "source_manifest": {
            "path": str(registered["manifest_path"].relative_to(repo_root)),
            "sha256": sha256_file(registered["manifest_path"]),
        },
        "target_access": False,
        "terminal": "PASS",
        "thresh": 0.2,
        "train_count": len(rows),
        "unclip_ratio": 1.4,
    }


def write_manifest(
    path: Path,
    *,
    repo_root: Path,
    plan_path: Path,
    registered: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    payload = build_manifest_payload(
        repo_root=repo_root,
        plan_path=plan_path,
        registered=registered,
        rows=rows,
    )
    lowered_keys = {key.lower() for key in payload}
    if lowered_keys & FORBIDDEN_MANIFEST_FIELDS:
        raise MaterializationError("forbidden metadata entered materialization manifest")
    atomic_write_json(path, payload)


def materialize_one(
    *,
    detector: Any,
    file_name: str,
    source_path: Path,
    spec: dict[str, Any],
    page_dir: Path,
) -> dict[str, Any]:
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise MaterializationError(f"source image decode failed: {source_path}")
    height, width = image.shape[:2]
    raw_polygons, raw_scores = extract_result(
        predict_one(detector, source_path, spec)
    )
    polygons, scores = normalize_detections(
        raw_polygons, raw_scores, height=height, width=width
    )
    occupancy, confidence = rasterize_layout(
        polygons, scores, height=height, width=width
    )
    npz_path = page_dir / f"{Path(file_name).stem}.npz"
    atomic_write_npz(
        npz_path,
        polygons=polygons,
        scores=scores,
        confidence=confidence,
        occupancy=occupancy,
    )
    return {
        "confidence_max": float(confidence.max()),
        "confidence_mean": float(confidence.mean()),
        "detection_count": int(len(polygons)),
        "file": file_name,
        "height": int(height),
        "npz_sha256": sha256_file(npz_path),
        "occupancy_pixels": int(occupancy.sum()),
        "source_sha256": sha256_file(source_path),
        "width": int(width),
    }


def validate_page_record(
    row: dict[str, Any],
    *,
    file_name: str,
    source_path: Path,
    npz_path: Path,
) -> dict[str, Any]:
    return runtime.validate_page_record(
        row,
        file_name=file_name,
        source_path=source_path,
        npz_path=npz_path,
        rasterize=rasterize_layout,
    )


def prepare_resume_state(
    *, repo_root: Path, plan_file: Path, registered: dict[str, Any]
) -> tuple[Path, Path, dict[str, dict[str, Any]]]:
    return runtime.prepare_resume_state(
        repo_root=repo_root,
        plan_file=plan_file,
        registered=registered,
        family=FAMILY,
        rasterize=rasterize_layout,
    )


def materialize_page_child(
    spec: dict[str, Any],
    file_name: str,
    source_path_value: str,
    page_dir_value: str,
    record_path_value: str,
) -> None:
    os.setsid()
    detector = create_detector(spec)
    try:
        row = materialize_one(
            detector=detector,
            file_name=file_name,
            source_path=Path(source_path_value),
            spec=spec,
            page_dir=Path(page_dir_value),
        )
        atomic_write_json(Path(record_path_value), row)
    finally:
        close = getattr(detector, "close", None)
        if callable(close):
            close()


def wait_for_page_process(
    process: multiprocessing.Process,
    *,
    health_reader: Callable[[int], dict[str, float | int]] = runtime_health,
) -> dict[str, float | int]:
    return runtime.wait_for_page_process(process, health_reader=health_reader)


def run_isolated_page(
    *,
    spec: dict[str, Any],
    file_name: str,
    source_path: Path,
    page_dir: Path,
    record_path: Path,
) -> dict[str, float | int]:
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=materialize_page_child,
        args=(spec, file_name, str(source_path), str(page_dir), str(record_path)),
        daemon=False,
    )
    process.start()
    return wait_for_page_process(process)


def run_in_process_page(
    *,
    detector_factory: Callable[[dict[str, Any]], Any],
    spec: dict[str, Any],
    file_name: str,
    source_path: Path,
    page_dir: Path,
    record_path: Path,
) -> dict[str, float | int]:
    detector = detector_factory(spec)
    try:
        row = materialize_one(
            detector=detector,
            file_name=file_name,
            source_path=source_path,
            spec=spec,
            page_dir=page_dir,
        )
        atomic_write_json(record_path, row)
    finally:
        close = getattr(detector, "close", None)
        if callable(close):
            close()
    return {
        "minimum_memory_free_percent": 100.0,
        "peak_process_tree_rss_bytes": 0,
        "peak_swap_used_bytes": 0,
    }


def publish_completed_materialization(
    *,
    repo_root: Path,
    plan_file: Path,
    registered: dict[str, Any],
    page_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    temporary_root: Path = registered["temporary_root"]
    output_root: Path = registered["output_root"]
    complete_root = temporary_root / "complete"
    if complete_root.exists():
        shutil.rmtree(complete_root)
    complete_pages = complete_root / "pages"
    complete_pages.mkdir(parents=True)
    for row in rows:
        source = page_dir / f"{Path(row['file']).stem}.npz"
        destination = complete_pages / source.name
        os.link(source, destination)
    fsync_directory(complete_pages)
    write_manifest(
        complete_root / "manifest.json",
        repo_root=repo_root,
        plan_path=plan_file,
        registered=registered,
        rows=rows,
    )
    fsync_directory(complete_root)
    complete_root.replace(output_root)
    fsync_directory(output_root.parent)
    finalize_published_cleanup(repo_root=repo_root, registered=registered)


def published_transaction_paths(output_root: Path) -> tuple[Path, Path]:
    marker = output_root.with_name(f".{output_root.name}.published.json")
    cleanup = output_root.with_name(f".{output_root.name}.cleanup")
    return marker, cleanup


def published_marker_payload(
    *, repo_root: Path, registered: dict[str, Any]
) -> dict[str, Any]:
    output_root: Path = registered["output_root"]
    return {
        "manifest_sha256": sha256_file(output_root / "manifest.json"),
        "output_root": str(output_root.relative_to(repo_root)),
        "schema_version": 1,
    }


def finalize_published_cleanup(
    *, repo_root: Path, registered: dict[str, Any]
) -> None:
    output_root: Path = registered["output_root"]
    temporary_root: Path = registered["temporary_root"]
    marker_path, cleanup_root = published_transaction_paths(output_root)
    expected_marker = published_marker_payload(
        repo_root=repo_root, registered=registered
    )
    if marker_path.exists():
        if read_json(marker_path) != expected_marker:
            raise MaterializationError("published transaction marker changed")
    else:
        atomic_write_json(marker_path, expected_marker)
    if temporary_root.exists():
        if cleanup_root.exists():
            raise MaterializationError("published cleanup states overlap")
        temporary_root.replace(cleanup_root)
        fsync_directory(output_root.parent)
    if cleanup_root.exists():
        shutil.rmtree(cleanup_root)
        fsync_directory(output_root.parent)
    marker_path.unlink()
    fsync_directory(output_root.parent)


def validate_published_materialization(
    *,
    repo_root: Path,
    plan_file: Path,
    registered: dict[str, Any],
) -> list[dict[str, Any]]:
    output_root: Path = registered["output_root"]
    if not output_root.is_dir() or {path.name for path in output_root.iterdir()} != {
        "manifest.json",
        "pages",
    }:
        raise MaterializationError("published materialization surface changed")
    page_dir = output_root / "pages"
    if not page_dir.is_dir():
        raise MaterializationError("published pages directory is missing")
    expected_page_names = {
        f"{Path(file_name).stem}.npz" for file_name in registered["file_names"]
    }
    if {path.name for path in page_dir.iterdir() if path.is_file()} != expected_page_names:
        raise MaterializationError("published page population changed")
    if any(not path.is_file() for path in page_dir.iterdir()):
        raise MaterializationError("published page surface contains a non-file")
    manifest = read_json(output_root / "manifest.json")
    manifest_rows = manifest.get("pages")
    if not isinstance(manifest_rows, list):
        raise MaterializationError("published manifest page rows changed")
    sources = {
        file_name: source_path
        for file_name, _relative, source_path in registered["sources"]
    }
    rows: list[dict[str, Any]] = []
    for index, file_name in enumerate(registered["file_names"]):
        if index >= len(manifest_rows) or not isinstance(manifest_rows[index], dict):
            raise MaterializationError("published manifest page order changed")
        rows.append(
            validate_page_record(
                manifest_rows[index],
                file_name=file_name,
                source_path=sources[file_name],
                npz_path=page_dir / f"{Path(file_name).stem}.npz",
            )
        )
    if len(manifest_rows) != len(rows):
        raise MaterializationError("published manifest page count changed")
    expected_manifest = build_manifest_payload(
        repo_root=repo_root,
        plan_path=plan_file,
        registered=registered,
        rows=rows,
    )
    if manifest != expected_manifest:
        raise MaterializationError("published manifest provenance changed")
    return rows


def materialize(
    *,
    repo_root: Path,
    plan_path: Path = PLAN_PATH,
    ledger_path: Path = LEDGER_PATH,
    detector_factory: Callable[[dict[str, Any]], Any] = create_detector,
    worker_count: int = 1,
) -> dict[str, Any]:
    if worker_count != 1:
        raise MaterializationError(
            "external detector concurrency is fixed at one process after the "
            "2026-08-13 memory-pressure panic"
        )
    plan_file = repo_path(repo_root, str(plan_path))
    ledger_file = repo_path(repo_root, str(ledger_path))
    plan = read_json(plan_file)
    ledger = read_json(ledger_file)
    registered = validate_registered_inputs(
        repo_root, plan, ledger, require_output_absent=False
    )
    output_root: Path = registered["output_root"]
    temporary_root: Path = registered["temporary_root"]
    marker_path, cleanup_root = published_transaction_paths(output_root)
    spec = plan["external_text_layout_materialization"]
    with runtime.exclusive_run_lock(runtime.HOST_USER_RUN_LOCK_PATH):
        if output_root.exists():
            if not temporary_root.exists() and not marker_path.exists():
                raise MaterializationError("external text-layout output already exists")
            rows = validate_published_materialization(
                repo_root=repo_root,
                plan_file=plan_file,
                registered=registered,
            )
            if temporary_root.exists():
                _page_dir, _record_dir, completed = prepare_resume_state(
                    repo_root=repo_root,
                    plan_file=plan_file,
                    registered=registered,
                )
                resumed_rows = [
                    completed.get(file_name) for file_name in registered["file_names"]
                ]
                if resumed_rows != rows:
                    raise MaterializationError(
                        "published materialization disagrees with resumable page records"
                    )
            finalize_published_cleanup(repo_root=repo_root, registered=registered)
            peak_rss = 0
            minimum_free = 100.0
            peak_swap = 0
        else:
            if marker_path.exists() or cleanup_root.exists():
                raise MaterializationError(
                    "published transaction exists without final output"
                )
            rows = []
        if not rows:
            assert_no_conflicting_model_processes()
            enforce_health_limits(runtime_health(os.getpid()))
            page_dir, record_dir, completed = prepare_resume_state(
                repo_root=repo_root,
                plan_file=plan_file,
                registered=registered,
            )
            sources = registered["sources"]
            peak_rss = 0
            minimum_free = 100.0
            peak_swap = 0
            for index, (file_name, _relative, source_path) in enumerate(sources, start=1):
                if file_name in completed:
                    continue
                assert_no_conflicting_model_processes()
                enforce_health_limits(runtime_health(os.getpid()))
                record_path = record_dir / f"{Path(file_name).stem}.json"
                if detector_factory is create_detector:
                    health = run_isolated_page(
                        spec=spec,
                        file_name=file_name,
                        source_path=source_path,
                        page_dir=page_dir,
                        record_path=record_path,
                    )
                else:
                    health = run_in_process_page(
                        detector_factory=detector_factory,
                        spec=spec,
                        file_name=file_name,
                        source_path=source_path,
                        page_dir=page_dir,
                        record_path=record_path,
                    )
                peak_rss = max(peak_rss, int(health["peak_process_tree_rss_bytes"]))
                minimum_free = min(
                    minimum_free, float(health["minimum_memory_free_percent"])
                )
                peak_swap = max(peak_swap, int(health["peak_swap_used_bytes"]))
                row = validate_page_record(
                    read_json(record_path),
                    file_name=file_name,
                    source_path=source_path,
                    npz_path=page_dir / f"{Path(file_name).stem}.npz",
                )
                completed[file_name] = row
                print(
                    f"page={index}/{len(sources)} file={file_name} "
                    f"peak_rss_bytes={int(health['peak_process_tree_rss_bytes'])} "
                    f"minimum_memory_free_percent="
                    f"{float(health['minimum_memory_free_percent']):.1f}",
                    flush=True,
                )
            rows = [completed[name] for name, _relative, _source in sources]
            if len(rows) != len(sources):
                raise MaterializationError("materialization resume population is incomplete")
            publish_completed_materialization(
                repo_root=repo_root,
                plan_file=plan_file,
                registered=registered,
                page_dir=page_dir,
                rows=rows,
            )
    manifest_path = output_root / "manifest.json"
    return {
        "content_sha256": sha256_rows(
            [f"{row['file']} {row['npz_sha256']}" for row in rows]
        ),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "minimum_memory_free_percent": minimum_free,
        "output_root": str(output_root),
        "peak_process_tree_rss_bytes": peak_rss,
        "peak_swap_used_bytes": peak_swap,
        "terminal": "PASS",
        "train_count": len(rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = materialize(
            repo_root=args.repo_root,
            plan_path=args.plan,
            ledger_path=args.ledger,
            worker_count=args.workers,
        )
    except (MaterializationError, OSError, ValueError) as error:
        print(f"terminal=PREREQUISITE_NEEDED reason={error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
