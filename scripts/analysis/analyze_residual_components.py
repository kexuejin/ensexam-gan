#!/usr/bin/env python3
"""Extract and classify residual handwriting components from hardcase metrics.

The goal is to explain why visually similar handwriting may erase differently.
This uses only local input/label/pred differences from an eval metrics CSV, so
it is cheap and does not require visual-AI inspection or rerunning inference.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="hardcase_worst_metrics.csv from eval_hardcase_worst_pages.py")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--change-threshold", type=int, default=12)
    parser.add_argument("--eval-threshold", type=int, default=12)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--pad", type=int, default=16)
    parser.add_argument("--crops-dir", default="", help="Optional directory for top component crop triplets.")
    parser.add_argument("--max-crops", type=int, default=80)
    return parser.parse_args()


def read_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
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


def crop_window(x: int, y: int, w: int, h: int, image_w: int, image_h: int, pad: int) -> tuple[int, int, int, int]:
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image_w, x + w + pad)
    y1 = min(image_h, y + h + pad)
    return x0, y0, x1, y1


def classify_component(
    mean_input_label_delta: float,
    mean_pred_input_delta: float,
    edit_ratio: float,
    mean_pred_label_delta: float,
    local_changed_density: float,
) -> str:
    """Return a coarse root-cause bucket for a residual component."""
    if mean_input_label_delta < 18:
        return "low_contrast_target"
    if edit_ratio < 0.10 or mean_pred_input_delta < 4:
        return "not_edited_or_mask_missing"
    if edit_ratio < 0.45:
        return "weak_partial_edit"
    if local_changed_density > 0.45:
        return "dense_or_print_overlap_risk"
    if mean_pred_label_delta >= 24:
        return "edited_but_insufficient"
    return "small_residual_after_edit"


def save_crop_triplet(
    crops_dir: Path,
    rank: int,
    file_stem: str,
    reason: str,
    bounds: tuple[int, int, int, int],
    input_bgr: np.ndarray,
    label_bgr: np.ndarray,
    pred_bgr: np.ndarray,
) -> None:
    x0, y0, x1, y1 = bounds
    panels = []
    for title, image in (("input", input_bgr), ("label", label_bgr), ("pred", pred_bgr)):
        crop = image[y0:y1, x0:x1]
        title_bar = np.full((24, crop.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(title_bar, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(np.vstack([title_bar, crop]))
    triplet = np.hstack(panels)
    out = crops_dir / f"{rank:04d}_{file_stem}_{reason}_{x0}_{y0}_{x1}_{y1}.png"
    cv2.imwrite(str(out), triplet)


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    component_rows: list[dict[str, str | int | float]] = []

    with metrics_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            input_bgr = read_bgr(row["image_path"])
            label_bgr = ensure_same_size(read_bgr(row["label_path"]), input_bgr)
            pred_bgr = ensure_same_size(read_bgr(row["pred_path"]), input_bgr)

            changed = build_changed_mask(input_bgr, label_bgr, args.change_threshold)
            input_label_delta = cv2.absdiff(input_bgr, label_bgr).mean(axis=2)
            pred_label_delta = cv2.absdiff(pred_bgr, label_bgr).mean(axis=2)
            pred_input_delta = cv2.absdiff(pred_bgr, input_bgr).mean(axis=2)
            residual_mask = (changed & (pred_label_delta >= args.eval_threshold)).astype(np.uint8)

            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(residual_mask, connectivity=8)
            image_h, image_w = residual_mask.shape
            for component_id in range(1, count):
                x, y, w, h, area = stats[component_id]
                area = int(area)
                if area < args.min_area:
                    continue
                component = labels == component_id
                x0, y0, x1, y1 = crop_window(int(x), int(y), int(w), int(h), image_w, image_h, args.pad)
                local_changed = changed[y0:y1, x0:x1]
                edit_mask = pred_input_delta >= args.eval_threshold

                mean_input_label_delta = float(input_label_delta[component].mean())
                mean_pred_label_delta = float(pred_label_delta[component].mean())
                mean_pred_input_delta = float(pred_input_delta[component].mean())
                edit_ratio = float(edit_mask[component].mean())
                local_changed_density = float(local_changed.mean())
                reason = classify_component(
                    mean_input_label_delta,
                    mean_pred_input_delta,
                    edit_ratio,
                    mean_pred_label_delta,
                    local_changed_density,
                )

                component_rows.append({
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
                    "crop_x0": x0,
                    "crop_y0": y0,
                    "crop_x1": x1,
                    "crop_y1": y1,
                    "mean_input_label_delta": mean_input_label_delta,
                    "mean_pred_label_delta": mean_pred_label_delta,
                    "mean_pred_input_delta": mean_pred_input_delta,
                    "edit_ratio": edit_ratio,
                    "local_changed_density": local_changed_density,
                    "reason": reason,
                })

    component_rows.sort(key=lambda r: int(r["area"]), reverse=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if component_rows:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(component_rows[0].keys()))
            writer.writeheader()
            writer.writerows(component_rows)
    else:
        output_csv.write_text("", encoding="utf-8")

    if args.crops_dir and component_rows:
        crops_dir = Path(args.crops_dir)
        crops_dir.mkdir(parents=True, exist_ok=True)
        image_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for rank, row in enumerate(component_rows[: args.max_crops], start=1):
            key = str(row["image_path"])
            if key not in image_cache:
                input_bgr = read_bgr(row["image_path"])
                label_bgr = ensure_same_size(read_bgr(row["label_path"]), input_bgr)
                pred_bgr = ensure_same_size(read_bgr(row["pred_path"]), input_bgr)
                image_cache[key] = (input_bgr, label_bgr, pred_bgr)
            save_crop_triplet(
                crops_dir,
                rank,
                Path(str(row["file"])).stem,
                str(row["reason"]),
                (int(row["crop_x0"]), int(row["crop_y0"]), int(row["crop_x1"]), int(row["crop_y1"])),
                *image_cache[key],
            )

    by_file: dict[str, Counter[str]] = defaultdict(Counter)
    area_by_reason: Counter[str] = Counter()
    count_by_reason: Counter[str] = Counter()
    for row in component_rows:
        reason = str(row["reason"])
        by_file[str(row["file"])][reason] += int(row["area"])
        area_by_reason[reason] += int(row["area"])
        count_by_reason[reason] += 1

    summary_rows: list[dict[str, str | int]] = []
    for reason, area in area_by_reason.most_common():
        summary_rows.append({
            "scope": "all",
            "file": "",
            "reason": reason,
            "components": count_by_reason[reason],
            "area": area,
        })
    for file, counter in sorted(by_file.items()):
        for reason, area in counter.most_common():
            summary_rows.append({
                "scope": "file",
                "file": file,
                "reason": reason,
                "components": "",
                "area": area,
            })

    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        summary_csv.write_text("", encoding="utf-8")

    print(f"components={len(component_rows)}")
    print(f"output_csv={output_csv}")
    print(f"summary_csv={summary_csv}")
    for reason, area in area_by_reason.most_common():
        print(f"{reason}: components={count_by_reason[reason]} area={area}")


if __name__ == "__main__":
    main()
