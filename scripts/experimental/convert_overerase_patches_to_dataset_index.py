#!/usr/bin/env python3
"""Convert over-erasure patch windows into EnsExam dataset patch-index rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument(
        "--max-tiles-per-source",
        type=int,
        default=4,
        help="Maximum dataset tiles emitted for one source over-erasure patch window.",
    )
    return parser.parse_args()


def image_size(path: str) -> tuple[int, int]:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    h, w = image.shape[:2]
    return w, h


def ticks(total: int, patch_size: int, stride: int) -> list[int]:
    if total <= patch_size:
        return [0]
    points = list(range(0, total - patch_size + 1, stride))
    if not points or points[-1] + patch_size < total:
        points.append(total - patch_size)
    return points


def intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    return max(0, x1 - x0) * max(0, y1 - y0)


def optional_float(row: dict[str, str], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return default


def optional_int(row: dict[str, str], *keys: str, default: int = 0) -> int:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return int(float(value))
    return default


def main() -> None:
    args = parse_args()
    stride = max(args.img_size - args.overlap, 1)
    emitted: dict[tuple[str, int, int], dict[str, str | int | float]] = {}

    with Path(args.patches_csv).open(newline="", encoding="utf-8") as f:
        for source_index, row in enumerate(csv.DictReader(f), start=1):
            image_w, image_h = image_size(row["image_path"])
            source_box = (int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"]))
            candidates = []
            for y1 in ticks(image_h, args.img_size, stride):
                for x1 in ticks(image_w, args.img_size, stride):
                    tile = (x1, y1, min(x1 + args.img_size, image_w), min(y1 + args.img_size, image_h))
                    inter = intersects(source_box, tile)
                    if inter <= 0:
                        continue
                    candidates.append((inter, x1, y1, tile))
            candidates.sort(reverse=True)
            for rank, (inter, x1, y1, tile) in enumerate(candidates[: args.max_tiles_per_source], start=1):
                key = (row["file"], x1, y1)
                score = float(row.get("score", 0.0)) * inter / max(
                    (source_box[2] - source_box[0]) * (source_box[3] - source_box[1]),
                    1,
                )
                existing = emitted.get(key)
                if existing is not None and float(existing["rank_score"]) >= score:
                    continue
                emitted[key] = {
                    "rank_score": score,
                    "source_index": source_index,
                    "source_rank": rank,
                    "file": row["file"],
                    "x1": x1,
                    "y1": y1,
                    "x2": tile[2],
                    "y2": tile[3],
                    "source_area": optional_int(row, "source_area"),
                    "source_mean_over_delta": optional_float(row, "source_mean_over_delta"),
                    "source_mean_residual_delta": optional_float(row, "source_mean_residual_delta"),
                    "source_mean_pred_input_delta": optional_float(row, "source_mean_pred_input_delta"),
                    "source_edit_ratio": optional_float(row, "source_edit_ratio"),
                    "source_reason": row.get("source_reason", ""),
                    "source_component_id": optional_int(row, "source_component_id"),
                }

    rows = sorted(emitted.values(), key=lambda row: float(row["rank_score"]), reverse=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        output_csv.write_text("", encoding="utf-8")

    print(f"rows={len(rows)}")
    print(f"output_csv={output_csv}")
    print(f"files={len(set(row['file'] for row in rows))}")


if __name__ == "__main__":
    main()
