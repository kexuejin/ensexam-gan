#!/usr/bin/env python3
"""Build patch-index rows for target-darker brightening regressions.

This converts offline local target-proxy rows into training windows. The target
failure is the dominant residual-delta t4 mode:

* candidate changed a pixel versus baseline,
* candidate is brighter than baseline,
* target is darker than baseline,
* candidate is farther from target than baseline.

The output is compatible with train_patch_cleanup_erasemap_probe.py
--patch-index-file. Extra source columns are included for auditability and are
ignored by the training dataset.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-comparison-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", action="append", default=[], help="Optional split filter. May be repeated.")
    parser.add_argument("--verdict", action="append", default=["reject"], help="Local verdict filter. May be repeated.")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--candidate-change-threshold", type=float, default=2.0)
    parser.add_argument("--meaningful-regression-threshold", type=float, default=2.0)
    parser.add_argument("--target-darker-threshold", type=float, default=2.0)
    parser.add_argument("--min-component-area", type=int, default=20)
    parser.add_argument("--component-pad", type=int, default=16)
    parser.add_argument("--min-tile-mask-ratio", type=float, default=0.0001)
    parser.add_argument("--max-tiles-per-component", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=512)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.int16)


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape == reference.shape:
        return image
    return cv2.resize(
        image.astype(np.uint8),
        (reference.shape[1], reference.shape[0]),
        interpolation=cv2.INTER_AREA,
    ).astype(np.int16)


def dataset_ticks(total: int, patch_size: int, overlap: int) -> list[int]:
    """Match EnsExamRealDataset-style patch starts."""
    if total <= patch_size:
        return [0]
    step = max(patch_size - overlap, 1)
    count = math.ceil((total - overlap) / step)
    return [index * step for index in range(count)]


def intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    return max(0, x1 - x0) * max(0, y1 - y0)


def failure_mask(row: dict[str, str], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = read_gray(row["baseline_pred"])
    candidate = resize_like(read_gray(row["candidate_pred"]), baseline)
    target = resize_like(read_gray(row["target"]), baseline)

    changed = np.abs(candidate - baseline) > args.candidate_change_threshold
    improvement = np.abs(baseline - target).astype(np.int32) - np.abs(candidate - target).astype(np.int32)
    candidate_brighter = (candidate - baseline) > args.candidate_change_threshold
    target_darker = (baseline - target) > args.target_darker_threshold
    regression = improvement < -args.meaningful_regression_threshold
    mask = changed & candidate_brighter & target_darker & regression
    return mask, candidate - baseline, improvement


def component_box(
    x: int,
    y: int,
    w: int,
    h: int,
    pad: int,
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, x - pad),
        max(0, y - pad),
        min(image_w, x + w + pad),
        min(image_h, y + h + pad),
    )


def file_name(row: dict[str, str]) -> str:
    return Path(row["file"]).with_suffix(".jpg").name


def main() -> None:
    args = parse_args()
    split_filter = set(args.split)
    verdict_filter = set(args.verdict)
    emitted: dict[tuple[str, int, int], dict[str, object]] = {}
    source_components = 0

    for source_index, row in enumerate(read_rows(Path(args.local_comparison_csv)), start=1):
        if split_filter and row["split"] not in split_filter:
            continue
        if verdict_filter and row["local_verdict"] not in verdict_filter:
            continue

        mask, brighten_delta, improvement = failure_mask(row, args)
        image_h, image_w = mask.shape
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for component_id in range(1, count):
            x, y, w, h, area = [int(value) for value in stats[component_id]]
            if area < args.min_component_area:
                continue
            component = labels == component_id
            source_components += 1
            source_box = component_box(x, y, w, h, args.component_pad, image_w, image_h)
            candidates: list[tuple[float, int, int, tuple[int, int, int, int], float]] = []
            for y1 in dataset_ticks(image_h, args.img_size, args.overlap):
                for x1 in dataset_ticks(image_w, args.img_size, args.overlap):
                    tile = (x1, y1, min(x1 + args.img_size, image_w), min(y1 + args.img_size, image_h))
                    inter = intersects(source_box, tile)
                    if inter <= 0:
                        continue
                    x2, y2 = tile[2], tile[3]
                    tile_mask = mask[y1:y2, x1:x2]
                    mask_ratio = float(tile_mask.mean())
                    if mask_ratio < args.min_tile_mask_ratio:
                        continue
                    # Favor large, high-confidence regressions with broad tile coverage.
                    mean_brighten = float(brighten_delta[component].mean())
                    mean_regression = float((-improvement[component]).mean())
                    coverage = inter / max((source_box[2] - source_box[0]) * (source_box[3] - source_box[1]), 1)
                    score = area * mask_ratio * coverage * (mean_brighten + mean_regression)
                    candidates.append((score, x1, y1, tile, mask_ratio))
            candidates.sort(reverse=True)

            for rank, (score, x1, y1, tile, mask_ratio) in enumerate(
                candidates[: args.max_tiles_per_component],
                start=1,
            ):
                key = (file_name(row), x1, y1)
                existing = emitted.get(key)
                if existing is not None and float(existing["rank_score"]) >= score:
                    continue
                emitted[key] = {
                    "rank_score": score,
                    "file": file_name(row),
                    "x1": x1,
                    "y1": y1,
                    "x2": tile[2],
                    "y2": tile[3],
                    "mask_ratio": mask_ratio,
                    "split": row["split"],
                    "local_verdict": row["local_verdict"],
                    "source_index": source_index,
                    "source_rank": rank,
                    "source_component_id": component_id,
                    "source_area": area,
                    "source_x": x,
                    "source_y": y,
                    "source_w": w,
                    "source_h": h,
                    "source_mean_brighten_delta": float(brighten_delta[component].mean()),
                    "source_mean_regression": float((-improvement[component]).mean()),
                    "source_active_mean_gain": float(row.get("active_mean_gain", 0.0)),
                    "source_help_hurt_ratio": float(row.get("help_hurt_ratio", 0.0)),
                }

    rows = sorted(
        emitted.values(),
        key=lambda item: (-float(item["rank_score"]), str(item["file"]), int(item["y1"]), int(item["x1"])),
    )
    if args.top_k > 0:
        rows = rows[: args.top_k]

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank_score",
        "file",
        "x1",
        "y1",
        "x2",
        "y2",
        "mask_ratio",
        "split",
        "local_verdict",
        "source_index",
        "source_rank",
        "source_component_id",
        "source_area",
        "source_x",
        "source_y",
        "source_w",
        "source_h",
        "source_mean_brighten_delta",
        "source_mean_regression",
        "source_active_mean_gain",
        "source_help_hurt_ratio",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"source_components={source_components}")
    print(f"patches={len(rows)}")
    print(f"files={len(set(row['file'] for row in rows))}")
    print(f"output_csv={output}")
    for row in rows[:5]:
        print(row)


if __name__ == "__main__":
    main()
