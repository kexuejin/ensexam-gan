#!/usr/bin/env python3
"""Extract connected over-erasure components from hardcase eval metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="hardcase_worst_metrics.csv from eval_hardcase_worst_pages.py")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-patch-list", default="", help="Optional CSV of expanded patch windows for retraining.")
    parser.add_argument("--change-threshold", type=int, default=12)
    parser.add_argument("--eval-threshold", type=int, default=12)
    parser.add_argument("--min-area", type=int, default=25)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--patch-pad", type=int, default=96)
    return parser.parse_args()


def read_bgr(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def ensure_same_size(image: np.ndarray, target: np.ndarray) -> np.ndarray:
    if image.shape[:2] == target.shape[:2]:
        return image
    h, w = target.shape[:2]
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def build_changed_mask(input_bgr: np.ndarray, label_bgr: np.ndarray, threshold: int) -> np.ndarray:
    delta = cv2.absdiff(input_bgr, label_bgr).mean(axis=2)
    mask = (delta >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask > 0


def patch_window(
    x: int,
    y: int,
    w: int,
    h: int,
    image_w: int,
    image_h: int,
    patch_size: int,
    pad: int,
) -> tuple[int, int, int, int]:
    cx = x + w // 2
    cy = y + h // 2
    size = max(patch_size, w + 2 * pad, h + 2 * pad)
    x0 = max(0, min(cx - size // 2, image_w - size))
    y0 = max(0, min(cy - size // 2, image_h - size))
    x1 = min(image_w, x0 + size)
    y1 = min(image_h, y0 + size)
    return int(x0), int(y0), int(x1), int(y1)


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    component_rows: list[dict[str, str | int | float]] = []
    patch_rows: list[dict[str, str | int | float]] = []

    with metrics_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            input_bgr = read_bgr(row["image_path"])
            label_bgr = ensure_same_size(read_bgr(row["label_path"]), input_bgr)
            pred_bgr = ensure_same_size(read_bgr(row["pred_path"]), input_bgr)

            changed = build_changed_mask(input_bgr, label_bgr, args.change_threshold)
            outside = ~changed
            over_delta = cv2.absdiff(pred_bgr, input_bgr).mean(axis=2)
            residual_delta = cv2.absdiff(pred_bgr, label_bgr).mean(axis=2)
            over_mask = (outside & (over_delta >= args.eval_threshold)).astype(np.uint8)

            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(over_mask, connectivity=8)
            image_h, image_w = over_mask.shape
            for component_id in range(1, count):
                x, y, w, h, area = stats[component_id]
                area = int(area)
                if area < args.min_area:
                    continue
                component = labels == component_id
                x0, y0, x1, y1 = patch_window(
                    int(x),
                    int(y),
                    int(w),
                    int(h),
                    image_w,
                    image_h,
                    args.patch_size,
                    args.patch_pad,
                )
                component_row = {
                    "file": row["file"],
                    "image_path": row["image_path"],
                    "label_path": row["label_path"],
                    "pred_path": row["pred_path"],
                    "component_id": component_id,
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": area,
                    "mean_over_delta": float(over_delta[component].mean()),
                    "mean_residual_delta": float(residual_delta[component].mean()),
                    "patch_x0": x0,
                    "patch_y0": y0,
                    "patch_x1": x1,
                    "patch_y1": y1,
                    "patch_area": (x1 - x0) * (y1 - y0),
                }
                component_rows.append(component_row)
                patch_rows.append({
                    "image_path": row["image_path"],
                    "label_path": row["label_path"],
                    "file": row["file"],
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "source_component_id": component_id,
                    "source_area": area,
                    "source_mean_over_delta": component_row["mean_over_delta"],
                })

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if component_rows:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(component_rows[0].keys()))
            writer.writeheader()
            writer.writerows(component_rows)
    else:
        output_csv.write_text("", encoding="utf-8")

    if args.output_patch_list:
        output_patch_list = Path(args.output_patch_list)
        output_patch_list.parent.mkdir(parents=True, exist_ok=True)
        if patch_rows:
            with output_patch_list.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(patch_rows[0].keys()))
                writer.writeheader()
                writer.writerows(patch_rows)
        else:
            output_patch_list.write_text("", encoding="utf-8")

    total_area = sum(int(row["area"]) for row in component_rows)
    print(f"components={len(component_rows)}")
    print(f"total_area={total_area}")
    print(f"output_csv={output_csv}")
    if args.output_patch_list:
        print(f"output_patch_list={args.output_patch_list}")


if __name__ == "__main__":
    main()
