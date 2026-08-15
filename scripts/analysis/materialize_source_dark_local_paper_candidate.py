#!/usr/bin/env python3
"""Materialize a target-free source-dark local-paper source candidate.

This is a train-only source-candidate preflight for stroke-only patch
suppression. It reads source images and current-primary baseline predictions,
never reads target pixels during generation, and writes candidate PNGs plus a
review CSV that downstream stroke-only suppression can consume.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_SOURCE_BUCKET = "selector_false_positive_overerase_risk"
DEFAULT_SOURCE_CANDIDATE = "exact129_lam16_relaxed_interval_rejected"
DEFAULT_OUTPUT_BUCKET = "source_dark_local_paper_lift_source_candidate"
DEFAULT_OUTPUT_CANDIDATE = "source_dark_local_paper_lift_v1"


@dataclass(frozen=True)
class LocalPaperLiftConfig:
    source_dark: int = 170
    source_expand_dark: int = 190
    baseline_dark: int = 238
    local_dark_delta: int = 4
    seed_lift_floor: int = 2
    max_lift: int = 80
    alpha: float = 0.85
    dilate: int = 1
    min_component_area: int = 2
    max_component_area: int = 14000
    local_median_kernel: int = 81


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-bucket", default=DEFAULT_SOURCE_BUCKET)
    parser.add_argument("--source-candidate", default=DEFAULT_SOURCE_CANDIDATE)
    parser.add_argument("--output-bucket", default=DEFAULT_OUTPUT_BUCKET)
    parser.add_argument("--output-candidate", default=DEFAULT_OUTPUT_CANDIDATE)
    parser.add_argument("--max-rows", type=int, default=60)
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
        raise ValueError("no source rows selected for local-paper source candidate")
    disallowed = sorted({row.get("split", "") for row in selected if row.get("split", "") not in allowed_splits})
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


def remove_oversized_components(mask: np.ndarray, config: LocalPaperLiftConfig) -> tuple[np.ndarray, int, int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(mask, dtype=bool)
    kept_components = 0
    rejected_components = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < config.min_component_area or area > config.max_component_area:
            rejected_components += 1
            continue
        kept |= labels == label
        kept_components += 1
    return kept, kept_components, rejected_components


def source_dark_local_paper_lift(
    source: np.ndarray,
    baseline: np.ndarray,
    config: LocalPaperLiftConfig = LocalPaperLiftConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.int16)
    baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY).astype(np.int16)
    kernel = config.local_median_kernel
    if kernel % 2 == 0 or kernel < 3:
        raise ValueError("local_median_kernel must be odd and at least 3")
    local_bg = cv2.medianBlur(baseline_gray.astype(np.uint8), kernel).astype(np.int16)
    local_lift = np.clip(local_bg - baseline_gray, 0, config.max_lift)

    seed = (
        (source_gray < config.source_dark)
        & (baseline_gray < config.baseline_dark)
        & ((local_bg - baseline_gray) > config.local_dark_delta)
        & (local_lift >= config.seed_lift_floor)
    )
    if config.dilate > 0:
        dilation_kernel = np.ones((config.dilate * 2 + 1, config.dilate * 2 + 1), np.uint8)
        neighborhood = cv2.dilate(seed.astype(np.uint8), dilation_kernel).astype(bool)
    else:
        neighborhood = seed
    expanded = (
        neighborhood
        & (source_gray < config.source_expand_dark)
        & (baseline_gray < config.baseline_dark)
        & ((local_bg - baseline_gray) > max(1, config.local_dark_delta - 2))
    )
    mask, kept_components, rejected_components = remove_oversized_components(expanded, config)
    lift = np.zeros_like(baseline_gray, dtype=np.float32)
    lift[mask] = local_lift[mask].astype(np.float32) * config.alpha
    candidate = baseline.astype(np.float32) + lift[:, :, None]
    candidate = np.clip(candidate, 0, 255).astype(np.uint8)
    changed = np.any(np.abs(candidate.astype(np.int16) - baseline.astype(np.int16)) > 0, axis=2)
    metrics = {
        "seed_px": int(seed.sum()),
        "mask_px": int(mask.sum()),
        "changed_px": int(changed.sum()),
        "changed_ratio": float(changed.mean()),
        "mean_lift": float(lift[mask].mean()) if mask.any() else 0.0,
        "max_lift": float(lift.max()) if mask.any() else 0.0,
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


def process_row(
    row: dict[str, str],
    *,
    output_dir: Path,
    output_bucket: str,
    output_candidate: str,
    config: LocalPaperLiftConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = read_bgr(row["baseline_pred"])
    source = resize_like(read_bgr(row["source_input"]), baseline)
    candidate, metrics = source_dark_local_paper_lift(source, baseline, config)
    candidate_dir = output_dir / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / f"{Path(row['file']).stem}.png"
    if not cv2.imwrite(str(candidate_path), candidate):
        raise OSError(f"failed to write candidate: {candidate_path}")
    sample_key = f"{row.get('split', '')}/{row.get('file', '')}"
    review_row = {
        "sample_key": sample_key,
        "split": row["split"],
        "file": row["file"],
        "bucket": output_bucket,
        "candidate": output_candidate,
        "source_input": row["source_input"],
        "baseline_pred": row["baseline_pred"],
        "candidate_pred": str(candidate_path),
        "target": row["target"],
        "source_bucket": row.get("bucket", ""),
        "source_candidate": row.get("candidate", ""),
        "source_candidate_pred": row.get("candidate_pred", ""),
        "notes": json.dumps(
            {
                "method": "target_free_source_dark_local_paper_lift",
                "source_notes": row.get("notes", ""),
                "target_pixels_read_during_generation": False,
            },
            sort_keys=True,
        ),
    }
    diagnostic = {
        **metrics,
        "sample_key": sample_key,
        "split": row["split"],
        "file": row["file"],
        "candidate_pred": str(candidate_path),
        "baseline_pred": row["baseline_pred"],
        "source_input": row["source_input"],
    }
    return review_row, diagnostic


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = LocalPaperLiftConfig()
    rows = select_rows(
        read_rows(Path(args.review_csv), args.max_rows),
        source_bucket=args.source_bucket,
        source_candidate=args.source_candidate,
        allowed_splits=set(args.allowed_split),
    )
    review_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    for row in rows:
        review_row, diagnostic = process_row(
            row,
            output_dir=output_dir,
            output_bucket=args.output_bucket,
            output_candidate=args.output_candidate,
            config=config,
        )
        review_rows.append(review_row)
        diagnostics.append(diagnostic)
        selector_rows.append(
            {
                "split": review_row["split"],
                "file": review_row["file"],
                "image_path": review_row["source_input"],
                "baseline_pred_path": review_row["baseline_pred"],
                "candidate_pred_path": review_row["candidate_pred"],
                "changed_px": diagnostic["changed_px"],
                "changed_ratio": diagnostic["changed_ratio"],
                "mask_px": diagnostic["mask_px"],
                "mean_lift": diagnostic["mean_lift"],
                "source_candidate": review_row["source_candidate"],
            }
        )
    review_rows_path = output_dir / "review_rows.csv"
    diagnostics_path = output_dir / "diagnostics.csv"
    selector_rows_path = output_dir / "selector_replay_rows.csv"
    write_csv(review_rows_path, review_rows)
    write_csv(diagnostics_path, diagnostics)
    write_csv(selector_rows_path, selector_rows)
    summary = {
        "config": asdict(config),
        "diagnostics": str(diagnostics_path),
        "input_csv": args.review_csv,
        "output_bucket": args.output_bucket,
        "output_candidate": args.output_candidate,
        "review_rows": str(review_rows_path),
        "rows": len(rows),
        "selector_replay_rows": str(selector_rows_path),
        "target_pixels_read_during_generation": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
