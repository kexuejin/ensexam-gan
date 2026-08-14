#!/usr/bin/env python3
"""Audit frozen external text-layout support before any model training."""

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
    read_metric_rows,
    validate_label_set,
    validate_prediction_set,
)
from scripts.analysis.materialize_external_text_layout_support_train_only import (  # noqa: E402
    ACTIVE_ITERATION_ID,
    FAMILY,
    LEDGER_PATH,
    NPZ_KEYS,
    OUTPUT_ROOT as MATERIALIZATION_ROOT,
    PLAN_PATH,
    read_json,
    rasterize_layout,
    repo_path,
    sha256_file,
    validate_registered_inputs,
)


OUTPUT_PATH = Path("outputs/external-text-layout-support-prerequisite-20260813/audit.json")
CHANNELS = (
    "second_stage_r",
    "second_stage_g",
    "second_stage_b",
    "external_text_occupancy",
    "external_text_confidence",
)
ABLATION_CHANNELS = CHANNELS[:3]
PAGE_ROW_KEYS = {
    "confidence_max",
    "confidence_mean",
    "detection_count",
    "file",
    "height",
    "npz_sha256",
    "occupancy_pixels",
    "source_sha256",
    "width",
}


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.materializing")
    if path.exists() or temporary.exists():
        raise AuditError(f"refusing to overwrite audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise AuditError(f"image decode failed: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_layout_npz(
    path: Path,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != NPZ_KEYS:
            raise AuditError("external text-layout NPZ keys changed")
        arrays = {key: payload[key] for key in NPZ_KEYS}
    polygons = arrays["polygons"]
    scores = arrays["scores"]
    occupancy = arrays["text_occupancy"]
    confidence = arrays["text_confidence"]
    if polygons.dtype != np.int32 or polygons.ndim != 3 or polygons.shape[1:] != (4, 2):
        raise AuditError("external text-layout polygons changed")
    if scores.dtype != np.float32 or scores.shape != (len(polygons),):
        raise AuditError("external text-layout scores changed")
    if occupancy.dtype != np.uint8 or occupancy.ndim != 2:
        raise AuditError("external text occupancy changed")
    if confidence.dtype != np.float32 or confidence.shape != occupancy.shape:
        raise AuditError("external text confidence changed")
    if expected_shape is not None and occupancy.shape != expected_shape:
        raise AuditError("external text-layout shape changed")
    if not np.isfinite(scores).all() or not np.isfinite(confidence).all():
        raise AuditError("external text-layout contains non-finite values")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise AuditError("external detector scores escaped unit interval")
    if np.any(confidence < 0.0) or np.any(confidence > 1.0):
        raise AuditError("external confidence escaped unit interval")
    if not np.isin(occupancy, [0, 1]).all():
        raise AuditError("external occupancy is not binary")
    height, width = occupancy.shape
    if len(polygons) and (
        np.any(polygons[:, :, 0] < 0)
        or np.any(polygons[:, :, 0] >= width)
        or np.any(polygons[:, :, 1] < 0)
        or np.any(polygons[:, :, 1] >= height)
    ):
        raise AuditError("external polygons escaped page bounds")
    sorted_indices = sorted(
        range(len(polygons)),
        key=lambda index: (
            int(polygons[index, :, 1].min()),
            int(polygons[index, :, 0].min()),
            int(polygons[index, :, 1].max()),
            int(polygons[index, :, 0].max()),
            *[int(value) for value in polygons[index].reshape(-1)],
            float(scores[index]),
        ),
    )
    if sorted_indices != list(range(len(polygons))):
        raise AuditError("external polygons are not in registered order")
    expected_occupancy, expected_confidence = rasterize_layout(
        polygons, scores, height=height, width=width
    )
    if not np.array_equal(occupancy, expected_occupancy):
        raise AuditError("external occupancy does not reproduce from polygons")
    if not np.array_equal(confidence, expected_confidence):
        raise AuditError("external confidence does not reproduce from polygons")
    return arrays


def validate_materialization(
    *,
    repo_root: Path,
    plan: dict[str, Any],
    plan_file: Path,
    registered: dict[str, Any],
) -> tuple[Path, dict[str, dict[str, Any]], dict[str, Any]]:
    root = repo_path(repo_root, str(MATERIALIZATION_ROOT))
    if not root.is_dir() or {path.name for path in root.iterdir()} != {
        "manifest.json",
        "pages",
    }:
        raise AuditError("external text-layout materialization surface changed")
    page_dir = root / "pages"
    if not page_dir.is_dir() or any(path.is_dir() for path in page_dir.iterdir()):
        raise AuditError("external text-layout page directory changed")
    expected_names = sorted(
        f"{Path(name).stem}.npz" for name in registered["file_names"]
    )
    if sorted(path.name for path in page_dir.iterdir()) != expected_names:
        raise AuditError("external text-layout page filenames changed")
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    required = {
        "batch_size": 1,
        "box_thresh": 0.45,
        "device": "cpu",
        "encoding": "one_compressed_npz_per_page_with_exact_layout_keys",
        "engine": "transformers",
        "family": FAMILY,
        "limit_side_len": 736,
        "limit_type": "min",
        "max_side_limit": 4000,
        "model_name": "PP-OCRv6_medium_det",
        "output_root": str(MATERIALIZATION_ROOT),
        "pages_directory": "pages",
        "polygon_rasterization": "cv2.fillPoly_LINE_8_shift_0",
        "recognition": False,
        "routing_metadata": False,
        "runtime": registered["runtime"],
        "schema_version": 1,
        "target_access": False,
        "terminal": "PASS",
        "thresh": 0.2,
        "train_count": len(registered["file_names"]),
        "unclip_ratio": 1.4,
    }
    expected_manifest_keys = set(required) | {
        "model_files",
        "pages",
        "plan",
        "source_manifest",
    }
    if set(manifest) != expected_manifest_keys:
        raise AuditError("external text-layout manifest schema changed")
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise AuditError(f"external text-layout manifest changed: {key}")
    if manifest.get("plan") != {
        "path": str(plan_file.relative_to(repo_root)),
        "sha256": sha256_file(plan_file),
    }:
        raise AuditError("external text-layout plan provenance changed")
    if manifest.get("source_manifest") != {
        "path": str(registered["manifest_path"].relative_to(repo_root)),
        "sha256": sha256_file(registered["manifest_path"]),
    }:
        raise AuditError("external text-layout source manifest changed")
    expected_model_files = {
        key: {"external_path": str(path), "sha256": sha256_file(path)}
        for key, path in registered["model_paths"].items()
    }
    if manifest.get("model_files") != expected_model_files:
        raise AuditError("external text-layout model provenance changed")
    rows = manifest.get("pages")
    if not isinstance(rows, list) or len(rows) != len(registered["file_names"]):
        raise AuditError("external text-layout page count changed")
    sources = {
        name: source for name, _relative, source in registered["sources"]
    }
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != PAGE_ROW_KEYS:
            raise AuditError("external text-layout page schema changed")
        file_name = row.get("file")
        if file_name not in sources or file_name in by_name:
            raise AuditError("external text-layout page identity changed")
        height = row.get("height")
        width = row.get("width")
        if not isinstance(height, int) or not isinstance(width, int) or min(height, width) <= 0:
            raise AuditError(f"invalid layout page dimensions: {file_name}")
        if row.get("source_sha256") != sha256_file(sources[file_name]):
            raise AuditError(f"external layout source hash changed: {file_name}")
        npz_path = page_dir / f"{Path(file_name).stem}.npz"
        if row.get("npz_sha256") != sha256_file(npz_path):
            raise AuditError(f"external layout NPZ hash changed: {file_name}")
        arrays = load_layout_npz(npz_path, expected_shape=(height, width))
        if row.get("detection_count") != len(arrays["polygons"]):
            raise AuditError(f"external layout detection count changed: {file_name}")
        occupancy = arrays["text_occupancy"]
        confidence = arrays["text_confidence"]
        if row.get("occupancy_pixels") != int(occupancy.sum()):
            raise AuditError(f"external layout occupancy count changed: {file_name}")
        if float(row.get("confidence_max", -1.0)) != float(confidence.max()):
            raise AuditError(f"external layout confidence max changed: {file_name}")
        if abs(float(row.get("confidence_mean", -1.0)) - float(confidence.mean())) > 1e-7:
            raise AuditError(f"external layout confidence mean changed: {file_name}")
        by_name[file_name] = row
    if list(by_name) != registered["file_names"]:
        raise AuditError("external text-layout manifest order changed")
    return page_dir, by_name, {
        "manifest_path": str(manifest_path.relative_to(repo_root)),
        "manifest_sha256": sha256_file(manifest_path),
        "page_count": len(rows),
        "target_access": False,
        "recognition": False,
    }


def build_page(
    *,
    file_name: str,
    second_stage_dir: Path,
    label_dir: Path,
    layout_dir: Path,
    layout_row: dict[str, Any],
    second_stage_row: dict[str, str],
    margin_gray: float,
    sample_cap: int,
) -> dict[str, Any]:
    second_stage = read_rgb(second_stage_dir / f"{Path(file_name).stem}.png")
    target = read_rgb(label_dir / file_name)
    if second_stage.shape != target.shape:
        raise AuditError(f"image shape mismatch for {file_name}")
    if second_stage.shape[:2] != (
        int(layout_row["height"]),
        int(layout_row["width"]),
    ):
        raise AuditError(f"layout alignment changed for {file_name}")
    arrays = load_layout_npz(
        layout_dir / f"{Path(file_name).stem}.npz",
        expected_shape=second_stage.shape[:2],
    )
    if (
        metric_float(second_stage_row, "base_edit_threshold", file_name) != 12.0
        or metric_float(second_stage_row, "second_delta_threshold", file_name) != 32.0
        or metric_float(second_stage_row, "dark_threshold", file_name) != 0.0
    ):
        raise AuditError(f"second-stage protocol changed for {file_name}")
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
    second_features = second_float.reshape(-1, 3)[indices] / 255.0
    occupancy = arrays["text_occupancy"].reshape(-1, 1)[indices].astype(np.float32)
    confidence = arrays["text_confidence"].reshape(-1, 1)[indices]
    features = np.column_stack([second_features, occupancy, confidence]).astype(
        np.float32, copy=False
    )
    if features.shape != (len(indices), len(CHANNELS)) or not np.isfinite(features).all():
        raise AuditError(f"external text-layout feature matrix changed: {file_name}")
    sample_digest = hashlib.sha256()
    sample_digest.update(file_name.encode("utf-8"))
    sample_digest.update(positive_indices.astype("<i8").tobytes())
    sample_digest.update(preserve_indices.astype("<i8").tobytes())
    return {
        "ablation_features": features[:, :3],
        "features": features,
        "file": file_name,
        "fold": fold_for_name(file_name),
        "labels": labels,
        "positive_pixel_count": int(positive_mask.sum()),
        "preserve_pixel_count": int((~positive_mask).sum()),
        "sample_sha256": sample_digest.hexdigest(),
        "samples_per_class": len(positive_indices),
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
                "auc": auc_rank(page["labels"], page_scores),
                "file": page["file"],
                "positive_score_mean": float(page_scores[page["labels"] > 0].mean()),
                "preserve_score_mean": float(page_scores[page["labels"] < 0].mean()),
                "samples_per_class": page["samples_per_class"],
            }
        )
        offset += count
    full_auc = auc_rank(test_labels, full_scores)
    ablation_auc = auc_rank(test_labels, ablation_scores)
    return {
        "ablation_auc": ablation_auc,
        "ablation_fit": ablation_fit,
        "auc_margin": full_auc - ablation_auc,
        "fit_page_count": len(train_pages),
        "fold": fold,
        "full_auc": full_auc,
        "full_fit": full_fit,
        "page_auc_median": float(np.median([row["auc"] for row in page_results])),
        "pages": page_results,
        "positive_score_mean": float(full_scores[positive].mean()),
        "preserve_score_mean": float(full_scores[~positive].mean()),
        "test_page_count": len(test_pages),
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
    registered = validate_registered_inputs(
        repo_root, plan, ledger, require_output_absent=False
    )
    layout_dir, layout_rows, materialization = validate_materialization(
        repo_root=repo_root,
        plan=plan,
        plan_file=plan_file,
        registered=registered,
    )
    evidence = plan["evidence"]
    file_names = registered["file_names"]
    second_stage_dir, second_stage_summary = validate_prediction_set(
        repo_root,
        evidence["second_stage_prediction_set"],
        file_names,
        "second-stage prediction set",
    )
    second_rows = read_metric_rows(registered["paths"]["second_stage_metrics"])
    if set(second_rows) != set(file_names):
        raise AuditError("second-stage metric identities changed")
    label_dir = repo_path(repo_root, evidence["train_label_set"]["directory"])
    label_summary = validate_label_set(
        label_dir, file_names, evidence["train_label_set"]
    )
    pages: list[dict[str, Any]] = []
    margin = float(plan["data"]["target_lighter_margin_gray"])
    sample_cap = int(plan["diagnostic"]["max_samples_per_class_per_page"])
    for index, file_name in enumerate(file_names, start=1):
        pages.append(
            build_page(
                file_name=file_name,
                second_stage_dir=second_stage_dir,
                label_dir=label_dir,
                layout_dir=layout_dir,
                layout_row=layout_rows[file_name],
                second_stage_row=second_rows[file_name],
                margin_gray=margin,
                sample_cap=sample_cap,
            )
        )
        if index % 25 == 0 or index == len(file_names):
            print(
                f"decoded train-only external text-layout pages {index}/{len(file_names)}",
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
        "ablation_mean_fold_auc": float(np.mean(ablation_aucs)),
        "full_auc_ablation_margin": float(np.mean(full_aucs) - np.mean(ablation_aucs)),
        "full_mean_fold_auc": float(np.mean(full_aucs)),
        "full_min_fold_auc": min(full_aucs),
        "macro_median_page_auc": float(np.median(page_aucs)),
        "positive_mean_above_preserve_folds": positive_above_preserve,
    }
    contract = plan["acceptance"]
    conditions = {
        "every_fold_auc": aggregates["full_min_fold_auc"] >= contract["full_fold_auc_min"],
        "full_auc_ablation_margin": (
            aggregates["full_auc_ablation_margin"]
            >= contract["full_auc_ablation_margin_min"]
        ),
        "full_mean_fold_auc": (
            aggregates["full_mean_fold_auc"] >= contract["full_mean_fold_auc_min"]
        ),
        "macro_median_page_auc": (
            aggregates["macro_median_page_auc"] >= contract["macro_median_page_auc_min"]
        ),
        "positive_mean_above_preserve": (
            positive_above_preserve
            >= contract["positive_mean_above_preserve_min_folds"]
        ),
        "required_fold_count": len(folds) == contract["required_fold_count"],
    }
    terminal = "PASS" if all(conditions.values()) else "KILL"
    return {
        "ablation_feature_names": list(ABLATION_CHANNELS),
        "acceptance": {
            "conditions": conditions,
            "contract": contract,
            "passed": all(conditions.values()),
        },
        "aggregates": aggregates,
        "candidate_inference_started": False,
        "checkpoint_generated": False,
        "family": FAMILY,
        "feature_count": len(CHANNELS),
        "feature_names": list(CHANNELS),
        "fold_page_counts": fold_counts,
        "folds": folds,
        "full_train_label_set_validated_before_target_decode": True,
        "iteration_id": ACTIVE_ITERATION_ID,
        "ledger_path": str(ledger_path),
        "materialization": materialization,
        "next_boundary": (
            plan["next_boundary_on_pass"]
            if terminal == "PASS"
            else plan["terminal_successors"]["KILL"]
        ),
        "page_samples": [
            {
                key: page[key]
                for key in (
                    "file",
                    "fold",
                    "positive_pixel_count",
                    "preserve_pixel_count",
                    "sample_sha256",
                    "samples_per_class",
                )
            }
            for page in pages
        ],
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_file),
        "promotion_enabled": False,
        "quality_gate_started": False,
        "real_data_access": True,
        "reserved_blind_authorized": False,
        "sample_count": sum(len(page["labels"]) for page in pages),
        "schema_version": 1,
        "second_stage_predictions": second_stage_summary,
        "target_decode_roles": ["train"],
        "target_decoded_page_count": len(pages),
        "terminal": terminal,
        "train_labels": label_summary,
        "train_page_count": len(pages),
        "training_started": False,
        "upstream_training_corpus_overlap": "unverified",
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
    if output_path.exists() or output_path.parent.exists():
        print(
            f"terminal=PREREQUISITE_NEEDED reason=audit output already exists: {output_path.parent}",
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
            "candidate_inference_started": False,
            "checkpoint_generated": False,
            "promotion_enabled": False,
            "quality_gate_started": False,
            "reason": str(error),
            "reserved_blind_authorized": False,
            "schema_version": 1,
            "terminal": "PREREQUISITE_NEEDED",
            "training_started": False,
        }
        return_code = 2
    else:
        return_code = 0 if result["terminal"] == "PASS" else 1
    atomic_write_json(output_path, result)
    print(f"terminal={result['terminal']} output={output_path}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
