#!/usr/bin/env python3
"""Materialize frozen target-free external text-layout support for train275."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
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


PLAN_PATH = Path("docs/external-text-layout-support-prerequisite-v1.json")
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
OUTPUT_ROOT = Path("outputs/external-text-layout-support-materialization-20260813")
FAMILY = "external_printed_text_layout_support_v1"
ACTIVE_ITERATION_ID = "monotonic-residual-erase-support"
PREREGISTRATION_ID = "materially_new_support_successor_preregistration_v4"
DIAGNOSTIC_ID = "external_text_layout_support_train_only_diagnostic"
NPZ_KEYS = {
    "polygons",
    "scores",
    "text_confidence",
    "text_occupancy",
}
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


class MaterializationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_rows(rows: list[str]) -> str:
    payload = "\n".join(sorted(rows)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MaterializationError(f"expected JSON object: {path}")
    return value


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
    if require_output_absent and (output_root.exists() or temporary_root.exists()):
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


def write_manifest(
    path: Path,
    *,
    repo_root: Path,
    plan_path: Path,
    registered: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    payload = {
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
    lowered_keys = {key.lower() for key in payload}
    if lowered_keys & FORBIDDEN_MANIFEST_FIELDS:
        raise MaterializationError("forbidden metadata entered materialization manifest")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    np.savez_compressed(
        npz_path,
        polygons=polygons,
        scores=scores,
        text_confidence=confidence,
        text_occupancy=occupancy,
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


def materialize_chunk(task: tuple[dict[str, Any], list[tuple[str, str, str]], str]) -> list[dict[str, Any]]:
    spec, source_rows, page_dir_value = task
    page_dir = Path(page_dir_value)
    page_dir.mkdir(parents=True, exist_ok=True)
    detector = create_detector(spec)
    try:
        return [
            materialize_one(
                detector=detector,
                file_name=file_name,
                source_path=Path(source_path),
                spec=spec,
                page_dir=page_dir,
            )
            for file_name, _relative, source_path in source_rows
        ]
    finally:
        close = getattr(detector, "close", None)
        if callable(close):
            close()


def materialize(
    *,
    repo_root: Path,
    plan_path: Path = PLAN_PATH,
    ledger_path: Path = LEDGER_PATH,
    detector_factory: Callable[[dict[str, Any]], Any] = create_detector,
    worker_count: int = 1,
) -> dict[str, Any]:
    if worker_count <= 0:
        raise MaterializationError("worker count must be positive")
    plan_file = repo_path(repo_root, str(plan_path))
    ledger_file = repo_path(repo_root, str(ledger_path))
    plan = read_json(plan_file)
    ledger = read_json(ledger_file)
    registered = validate_registered_inputs(
        repo_root, plan, ledger, require_output_absent=True
    )
    output_root: Path = registered["output_root"]
    temporary_root: Path = registered["temporary_root"]
    spec = plan["external_text_layout_materialization"]
    rows: list[dict[str, Any]] = []
    try:
        worker_root = temporary_root / "workers"
        worker_root.mkdir(parents=True)
        sources = registered["sources"]
        chunk_size = (len(sources) + worker_count - 1) // worker_count
        chunks = [
            sources[start : start + chunk_size]
            for start in range(0, len(sources), chunk_size)
        ]
        if worker_count == 1:
            detector = detector_factory(spec)
            try:
                rows = [
                    materialize_one(
                        detector=detector,
                        file_name=file_name,
                        source_path=source_path,
                        spec=spec,
                        page_dir=worker_root / "worker-000" / "pages",
                    )
                    for file_name, _relative, source_path in sources
                ]
            finally:
                close = getattr(detector, "close", None)
                if callable(close):
                    close()
        else:
            if detector_factory is not create_detector:
                raise MaterializationError(
                    "custom detector factories require worker_count=1"
                )
            tasks = [
                (
                    spec,
                    [
                        (file_name, relative, str(source_path))
                        for file_name, relative, source_path in chunk
                    ],
                    str(worker_root / f"worker-{index:03d}" / "pages"),
                )
                for index, chunk in enumerate(chunks)
            ]
            with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
                for chunk_rows in executor.map(materialize_chunk, tasks):
                    rows.extend(chunk_rows)
        rows_by_name = {row["file"]: row for row in rows}
        if list(rows_by_name) != [name for name, _relative, _source in sources]:
            raise MaterializationError("worker materialization order or identity changed")
        page_dir = temporary_root / "pages"
        page_dir.mkdir(parents=True)
        for file_name, _relative, _source in sources:
            source_page = worker_root / "worker-000" / "pages" / f"{Path(file_name).stem}.npz"
            if not source_page.is_file():
                for worker_page in worker_root.glob("worker-*/pages"):
                    candidate = worker_page / source_page.name
                    if candidate.is_file():
                        source_page = candidate
                        break
            if not source_page.is_file():
                raise MaterializationError(f"missing worker output: {file_name}")
            source_page.replace(page_dir / source_page.name)
        rows = [rows_by_name[name] for name, _relative, _source in sources]
        shutil.rmtree(worker_root)
        write_manifest(
            temporary_root / "manifest.json",
            repo_root=repo_root,
            plan_path=plan_file,
            registered=registered,
            rows=rows,
        )
        temporary_root.replace(output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    manifest_path = output_root / "manifest.json"
    return {
        "content_sha256": sha256_rows(
            [f"{row['file']} {row['npz_sha256']}" for row in rows]
        ),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "output_root": str(output_root),
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
