#!/usr/bin/env python3
"""Materialize train-only source-chroma primary-edit lift candidates.

This is a target-free source-candidate diagnostic for stroke-only patch
suppression. Candidate generation reads only source images and current-primary
baseline predictions. Target images are read afterward only for train-only
metric scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.eval_hardcase_worst_pages import compute_residual_metrics


DEFAULT_SOURCE_BUCKET = "selector_false_positive_overerase_risk"
DEFAULT_SOURCE_CANDIDATE = "exact129_lam16_relaxed_interval_rejected"
DEFAULT_OUTPUT_BUCKET = "source_chroma_primary_edit_lift_source_candidate"
DEFAULT_OUTPUT_CANDIDATE = "source_chroma_primary_edit_lift_v1"


@dataclass(frozen=True)
class SourceChromaPrimaryEditLiftConfig:
    name: str = "primary_edit_chroma_tiny"
    saturation_min: int = 70
    primary_edit_floor: int = 20
    baseline_dark: int = 242
    local_dark_delta: int = 2
    seed_lift_floor: int = 1
    max_lift: int = 18
    alpha: float = 0.18
    min_component_area: int = 2
    max_component_area: int = 4000
    local_median_kernel: int = 81


CONFIG = SourceChromaPrimaryEditLiftConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-bucket", default=DEFAULT_SOURCE_BUCKET)
    parser.add_argument("--source-candidate", default=DEFAULT_SOURCE_CANDIDATE)
    parser.add_argument("--output-bucket", default=DEFAULT_OUTPUT_BUCKET)
    parser.add_argument("--output-candidate", default=DEFAULT_OUTPUT_CANDIDATE)
    parser.add_argument("--max-rows", type=int, default=60)
    parser.add_argument("--change-threshold", type=int, default=12)
    parser.add_argument("--eval-threshold", type=int, default=12)
    parser.add_argument("--allowed-split", action="append", default=["train", "train160"])
    return parser.parse_args()


def read_rows(path: Path, max_rows: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:max_rows] if max_rows > 0 else rows


def select_rows(
    rows: list[dict[str, str]],
    *,
    source_bucket: str,
    source_candidate: str,
    allowed_splits: set[str],
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row.get("bucket") == source_bucket and row.get("candidate") == source_candidate
    ]
    if not selected:
        raise ValueError("no source rows selected for chroma primary-edit source candidate")
    disallowed = sorted(
        {row.get("split", "") for row in selected if row.get("split", "") not in allowed_splits}
    )
    if disallowed:
        raise ValueError("source rows include split outside train-only authority: " + ", ".join(disallowed))
    return selected


def read_bgr(path_text: str) -> np.ndarray:
    image = cv2.imread(str(Path(path_text)), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path_text)
    return image


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)


def keep_component_area(
    mask: np.ndarray,
    *,
    min_area: int,
    max_area: int,
) -> tuple[np.ndarray, int, int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(mask, dtype=bool)
    kept_components = 0
    rejected_components = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            kept |= labels == label
            kept_components += 1
        else:
            rejected_components += 1
    return kept, kept_components, rejected_components


def source_chroma_primary_edit_lift(
    source: np.ndarray,
    baseline: np.ndarray,
    config: SourceChromaPrimaryEditLiftConfig = CONFIG,
) -> tuple[np.ndarray, dict[str, Any]]:
    if config.local_median_kernel % 2 == 0 or config.local_median_kernel < 3:
        raise ValueError("local_median_kernel must be odd and at least 3")

    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.int16)
    baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY).astype(np.int16)
    source_saturation = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.int16)
    local_bg = cv2.medianBlur(baseline_gray.astype(np.uint8), config.local_median_kernel).astype(np.int16)
    primary_edit = baseline_gray - source_gray
    local_delta = local_bg - baseline_gray
    lift = np.clip(local_delta, 0, config.max_lift)

    seed = (
        (source_saturation >= config.saturation_min)
        & (primary_edit >= config.primary_edit_floor)
        & (baseline_gray < config.baseline_dark)
        & (local_delta > config.local_dark_delta)
        & (lift >= config.seed_lift_floor)
    )
    mask, kept_components, rejected_components = keep_component_area(
        seed,
        min_area=config.min_component_area,
        max_area=config.max_component_area,
    )

    candidate = baseline.astype(np.float32)
    if mask.any():
        candidate[mask] = candidate[mask] + lift[mask, None].astype(np.float32) * config.alpha
    candidate = np.clip(candidate, 0, 255).astype(np.uint8)
    changed = np.any(np.abs(candidate.astype(np.int16) - baseline.astype(np.int16)) > 0, axis=2)
    metrics = {
        "seed_px": int(seed.sum()),
        "mask_px": int(mask.sum()),
        "changed_px": int(changed.sum()),
        "changed_ratio": float(changed.mean()),
        "mean_lift": float(lift[mask].mean()) if mask.any() else 0.0,
        "max_lift": float(lift.max()) if mask.any() else 0.0,
        "mean_primary_edit": float(primary_edit[mask].mean()) if mask.any() else 0.0,
        "mean_source_saturation": float(source_saturation[mask].mean()) if mask.any() else 0.0,
        "kept_components": kept_components,
        "rejected_components": rejected_components,
    }
    return candidate, metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def process_row(
    row: dict[str, str],
    *,
    output_dir: Path,
    output_bucket: str,
    output_candidate: str,
    config: SourceChromaPrimaryEditLiftConfig,
    change_threshold: int,
    eval_threshold: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = read_bgr(row["baseline_pred"])
    source = resize_like(read_bgr(row["source_input"]), baseline)
    candidate, generation_metrics = source_chroma_primary_edit_lift(source, baseline, config)

    candidate_dir = output_dir / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / f"{Path(row['file']).stem}.png"
    if not cv2.imwrite(str(candidate_path), candidate):
        raise OSError(f"failed to write candidate: {candidate_path}")

    target = resize_like(read_bgr(row["target"]), baseline)
    baseline_metrics = compute_residual_metrics(
        source,
        target,
        baseline,
        change_threshold,
        eval_threshold,
    )
    candidate_metrics = compute_residual_metrics(
        source,
        target,
        candidate,
        change_threshold,
        eval_threshold,
    )
    sample_key = f"{row.get('split', '')}/{row.get('file', '')}"
    review_row = {
        "sample_key": sample_key,
        "split": row["split"],
        "file": row["file"],
        "bucket": output_bucket,
        "candidate": f"{output_candidate}_{config.name}",
        "variant": config.name,
        "source_input": row["source_input"],
        "baseline_pred": row["baseline_pred"],
        "candidate_pred": str(candidate_path),
        "target": row["target"],
        "source_bucket": row.get("bucket", ""),
        "source_candidate": row.get("candidate", ""),
        "notes": json.dumps(
            {
                "method": "target_free_source_chroma_primary_edit_lift",
                "target_pixels_read_during_generation": False,
                "config": asdict(config),
            },
            sort_keys=True,
        ),
    }
    diagnostic = {
        **generation_metrics,
        "sample_key": sample_key,
        "split": row["split"],
        "file": row["file"],
        "candidate_pred": str(candidate_path),
        "variant": config.name,
        "baseline_pred": row["baseline_pred"],
        "source_input": row["source_input"],
        "baseline_residual_ratio": float(baseline_metrics["residual_ratio"]),
        "baseline_overerase_ratio": float(baseline_metrics["overerase_ratio"]),
        "candidate_residual_ratio": float(candidate_metrics["residual_ratio"]),
        "candidate_overerase_ratio": float(candidate_metrics["overerase_ratio"]),
        "residual_gain": float(baseline_metrics["residual_ratio"]) - float(candidate_metrics["residual_ratio"]),
        "overerase_delta": float(candidate_metrics["overerase_ratio"]) - float(baseline_metrics["overerase_ratio"]),
    }
    return review_row, diagnostic


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pages": len(rows),
        "baseline_residual_ratio": mean(rows, "baseline_residual_ratio"),
        "baseline_overerase_ratio": mean(rows, "baseline_overerase_ratio"),
        "candidate_residual_ratio": mean(rows, "candidate_residual_ratio"),
        "candidate_overerase_ratio": mean(rows, "candidate_overerase_ratio"),
        "residual_gain": mean(rows, "residual_gain"),
        "overerase_delta": mean(rows, "overerase_delta"),
        "min_page_residual_gain": min(float(row["residual_gain"]) for row in rows),
        "max_page_overerase_delta": max(float(row["overerase_delta"]) for row in rows),
        "changed_px": sum(int(row["changed_px"]) for row in rows),
        "mask_px": sum(int(row["mask_px"]) for row in rows),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = select_rows(
        read_rows(Path(args.review_csv), args.max_rows),
        source_bucket=args.source_bucket,
        source_candidate=args.source_candidate,
        allowed_splits=set(args.allowed_split),
    )

    review_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        review_row, diagnostic = process_row(
            row,
            output_dir=output_dir,
            output_bucket=args.output_bucket,
            output_candidate=args.output_candidate,
            config=CONFIG,
            change_threshold=args.change_threshold,
            eval_threshold=args.eval_threshold,
        )
        review_rows.append(review_row)
        diagnostics.append(diagnostic)

    review_rows_path = output_dir / "review_rows.csv"
    diagnostics_path = output_dir / "diagnostics.csv"
    write_csv(review_rows_path, review_rows)
    write_csv(diagnostics_path, diagnostics)
    summary = {
        "input_csv": args.review_csv,
        "output_bucket": args.output_bucket,
        "output_candidate": args.output_candidate,
        "review_rows": str(review_rows_path),
        "diagnostics": str(diagnostics_path),
        "rows": len(rows),
        "target_pixels_read_during_generation": False,
        "config": asdict(CONFIG),
        "metrics": summarize(diagnostics),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
