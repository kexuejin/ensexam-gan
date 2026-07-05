#!/usr/bin/env python3
"""Convert visible-delta components into EnsExam patch-index rows.

The output patch-index CSV is compatible with
micro_train_region_probe.py --patch-index-file. By default this emits only
visible improvement components and writes regressions to a separate reject CSV
so follow-up probes can train on locally verified wins without reinforcing
known local regressions.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--reject-csv", default="")
    parser.add_argument("--region-type", default="improve", choices=("improve", "regress", "all"))
    parser.add_argument("--reason-contains", default="visible_target_region")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--patch-pad", type=int, default=96)
    parser.add_argument("--max-tiles-per-component", type=int, default=4)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--score-area-weight", type=float, default=1.0)
    parser.add_argument("--score-improvement-weight", type=float, default=0.05)
    return parser.parse_args()


def image_size(path: str) -> tuple[int, int]:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    h, w = image.shape[:2]
    return w, h


def ticks(total: int, patch_size: int, overlap: int) -> list[int]:
    """Match EnsExamRealDataset._build_patch_index coordinates exactly."""
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


def component_box(row: dict[str, str], pad: int, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    x0 = max(0, int(float(row["x"])) - pad)
    y0 = max(0, int(float(row["y"])) - pad)
    x1 = min(image_w, int(float(row["x"])) + int(float(row["w"])) + pad)
    y1 = min(image_h, int(float(row["y"])) + int(float(row["h"])) + pad)
    return x0, y0, x1, y1


def row_matches(row: dict[str, str], region_type: str, reason_contains: str, min_area: int) -> bool:
    if int(float(row["area"])) < min_area:
        return False
    if region_type != "all" and row["region_type"] != region_type:
        return False
    if reason_contains and reason_contains not in row["reason"]:
        return False
    return True


def score_component(row: dict[str, str], inter: int, source_area: int, area_weight: float, improvement_weight: float) -> float:
    area_score = source_area * area_weight
    improvement_score = abs(float(row["mean_improvement"])) * improvement_weight
    coverage = inter / max(source_area, 1)
    return (area_score + improvement_score) * coverage


def main() -> None:
    args = parse_args()
    emitted: dict[tuple[str, int, int], dict[str, str | int | float]] = {}
    reject_rows: list[dict[str, str]] = []

    with Path(args.components_csv).open(newline="", encoding="utf-8") as handle:
        for source_index, row in enumerate(csv.DictReader(handle), start=1):
            if row["region_type"] == "regress":
                reject_rows.append(row)
            if not row_matches(row, args.region_type, args.reason_contains, args.min_area):
                continue

            image_w, image_h = image_size(row["image_path"])
            source_box = component_box(row, args.patch_pad, image_w, image_h)
            source_area = int(float(row["area"]))
            candidates: list[tuple[float, int, int, tuple[int, int, int, int]]] = []
            for y1 in ticks(image_h, args.img_size, args.overlap):
                for x1 in ticks(image_w, args.img_size, args.overlap):
                    tile = (x1, y1, min(x1 + args.img_size, image_w), min(y1 + args.img_size, image_h))
                    inter = intersects(source_box, tile)
                    if inter <= 0:
                        continue
                    score = score_component(
                        row,
                        inter,
                        source_area,
                        args.score_area_weight,
                        args.score_improvement_weight,
                    )
                    candidates.append((score, x1, y1, tile))
            candidates.sort(reverse=True)

            for tile_rank, (score, x1, y1, tile) in enumerate(candidates[: args.max_tiles_per_component], start=1):
                key = (row["file"], x1, y1)
                existing = emitted.get(key)
                if existing is not None and float(existing["rank_score"]) >= score:
                    continue
                emitted[key] = {
                    "rank_score": score,
                    "source_index": source_index,
                    "source_rank": tile_rank,
                    "file": row["file"],
                    "x1": x1,
                    "y1": y1,
                    "x2": tile[2],
                    "y2": tile[3],
                    "source_region_type": row["region_type"],
                    "source_reason": row["reason"],
                    "source_component_id": int(float(row["component_id"])),
                    "source_area": source_area,
                    "source_mean_improvement": float(row["mean_improvement"]),
                    "source_mean_input_label_delta": float(row["mean_input_label_delta"]),
                    "source_mean_abs_candidate_baseline_delta": float(row["mean_abs_candidate_baseline_delta"]),
                    "source_changed_ratio": float(row["changed_ratio"]),
                    "source_x": int(float(row["x"])),
                    "source_y": int(float(row["y"])),
                    "source_w": int(float(row["w"])),
                    "source_h": int(float(row["h"])),
                }

    rows = sorted(emitted.values(), key=lambda item: (-float(item["rank_score"]), str(item["file"]), int(item["y1"]), int(item["x1"])))
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        output_csv.write_text("", encoding="utf-8")

    if args.reject_csv:
        reject_csv = Path(args.reject_csv)
        reject_csv.parent.mkdir(parents=True, exist_ok=True)
        if reject_rows:
            with reject_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(reject_rows[0].keys()))
                writer.writeheader()
                writer.writerows(reject_rows)
        else:
            reject_csv.write_text("", encoding="utf-8")

    print(f"rows={len(rows)}")
    print(f"files={len(set(row['file'] for row in rows))}")
    print(f"output_csv={output_csv}")
    if args.reject_csv:
        print(f"reject_rows={len(reject_rows)}")
        print(f"reject_csv={args.reject_csv}")
    for row in rows[:5]:
        print(row)


if __name__ == "__main__":
    main()
