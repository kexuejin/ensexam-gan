#!/usr/bin/env python3
"""Analyze candidate-only over-erasure components against a baseline.

This is an offline diagnostic script. It uses labels to find background pixels
where the candidate crosses the over-erasure threshold but the baseline does
not, then writes component summaries and optional crop sheets.
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
    parser.add_argument("--page-choices", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--change-threshold", type=int, default=12)
    parser.add_argument("--eval-threshold", type=int, default=12)
    parser.add_argument("--min-area", type=int, default=25)
    parser.add_argument("--crop-pad", type=int, default=32)
    parser.add_argument("--max-crops", type=int, default=120)
    parser.add_argument(
        "--positive-gain-only",
        action="store_true",
        help="Only analyze pages where candidate improves residual and increases over-erasure.",
    )
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


def label_path_for(image_path: str | Path) -> Path:
    text = str(image_path)
    if "/all_images/" not in text:
        raise ValueError(f"Cannot infer label path: {image_path}")
    return Path(text.replace("/all_images/", "/all_labels/"))


def build_changed_mask(input_bgr: np.ndarray, label_bgr: np.ndarray, threshold: int) -> np.ndarray:
    delta = cv2.absdiff(input_bgr, label_bgr).mean(axis=2)
    mask = (delta >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask > 0


def classify_component(
    area: int,
    mean_candidate_input_delta: float,
    mean_baseline_input_delta: float,
    mean_input_label_delta: float,
    bbox_w: int,
    bbox_h: int,
    local_changed_density: float,
    y: int,
    image_h: int,
) -> str:
    if mean_input_label_delta >= 8:
        return "near_target_boundary_or_low_contrast_label"
    if local_changed_density > 0.08:
        return "near_changed_region_halo"
    if bbox_w >= 80 or bbox_h >= 80 or area >= 500:
        return "large_background_tone_shift"
    if mean_candidate_input_delta - mean_baseline_input_delta < 8:
        return "threshold_crossing_texture_shift"
    if y < image_h * 0.08 or y > image_h * 0.92:
        return "page_edge_artifact"
    return "small_background_edit"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_crop(
    crops_dir: Path,
    rank: int,
    row: dict[str, object],
) -> None:
    image = read_bgr(str(row["image_path"]))
    label = ensure_same_size(read_bgr(label_path_for(str(row["image_path"]))), image)
    baseline = ensure_same_size(read_bgr(str(row["baseline_pred_path"])), image)
    candidate = ensure_same_size(read_bgr(str(row["candidate_pred_path"])), image)
    x0, y0, x1, y1 = [int(row[key]) for key in ("crop_x0", "crop_y0", "crop_x1", "crop_y1")]
    panels = []
    for title, panel in (("input", image), ("label", label), ("baseline", baseline), ("candidate", candidate)):
        crop = panel[y0:y1, x0:x1]
        title_bar = np.full((24, crop.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(title_bar, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(np.vstack([title_bar, crop]))
    reason = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(row["reason"]))
    out = np.hstack(panels)
    file_stem = Path(str(row["file"])).stem
    cv2.imwrite(str(crops_dir / f"{rank:04d}_{row['split']}_{file_stem}_{reason}_{row['area']}.png"), out)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.page_choices).open(newline="", encoding="utf-8") as handle:
        page_rows = list(csv.DictReader(handle))
    if args.positive_gain_only:
        page_rows = [
            row
            for row in page_rows
            if float(row["residual_gain"]) > 0 and float(row["overerase_regret"]) > 0
        ]

    component_rows: list[dict[str, object]] = []
    for page_row in page_rows:
        image = read_bgr(page_row["image_path"])
        label = ensure_same_size(read_bgr(label_path_for(page_row["image_path"])), image)
        baseline = ensure_same_size(read_bgr(page_row["baseline_pred_path"]), image)
        candidate = ensure_same_size(read_bgr(page_row["candidate_pred_path"]), image)

        changed = build_changed_mask(image, label, args.change_threshold)
        outside = ~changed
        baseline_input_delta = cv2.absdiff(baseline, image).mean(axis=2)
        candidate_input_delta = cv2.absdiff(candidate, image).mean(axis=2)
        input_label_delta = cv2.absdiff(image, label).mean(axis=2)
        new_over = outside & (candidate_input_delta >= args.eval_threshold) & (baseline_input_delta < args.eval_threshold)

        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(new_over.astype(np.uint8), connectivity=8)
        image_h, image_w = new_over.shape
        for component_id in range(1, count):
            x, y, w, h, area = stats[component_id]
            area = int(area)
            if area < args.min_area:
                continue
            component = labels == component_id
            x0 = max(0, int(x) - args.crop_pad)
            y0 = max(0, int(y) - args.crop_pad)
            x1 = min(image_w, int(x + w) + args.crop_pad)
            y1 = min(image_h, int(y + h) + args.crop_pad)
            local_changed_density = float(changed[y0:y1, x0:x1].mean())
            mean_candidate_input_delta = float(candidate_input_delta[component].mean())
            mean_baseline_input_delta = float(baseline_input_delta[component].mean())
            mean_input_label_delta = float(input_label_delta[component].mean())
            reason = classify_component(
                area,
                mean_candidate_input_delta,
                mean_baseline_input_delta,
                mean_input_label_delta,
                int(w),
                int(h),
                local_changed_density,
                int(y),
                image_h,
            )
            component_rows.append({
                "split": page_row["split"],
                "file": page_row["file"],
                "residual_gain": float(page_row["residual_gain"]),
                "overerase_regret": float(page_row["overerase_regret"]),
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
                "mean_candidate_input_delta": mean_candidate_input_delta,
                "mean_baseline_input_delta": mean_baseline_input_delta,
                "mean_input_label_delta": mean_input_label_delta,
                "local_changed_density": local_changed_density,
                "reason": reason,
                "image_path": page_row["image_path"],
                "baseline_pred_path": page_row["baseline_pred_path"],
                "candidate_pred_path": page_row["candidate_pred_path"],
            })

    component_rows.sort(key=lambda row: (int(row["area"]), float(row["overerase_regret"])), reverse=True)
    write_csv(output_dir / "new_overerase_components.csv", component_rows)

    by_reason: Counter[str] = Counter()
    area_by_reason: Counter[str] = Counter()
    by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    area_by_split: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in component_rows:
        reason = str(row["reason"])
        by_reason[reason] += 1
        area_by_reason[reason] += int(row["area"])
        by_split[str(row["split"])][reason] += 1
        area_by_split[str(row["split"])][reason] += int(row["area"])

    summary_rows: list[dict[str, object]] = []
    for reason, count in by_reason.most_common():
        summary_rows.append({"scope": "all", "reason": reason, "components": count, "area": area_by_reason[reason]})
    for split in sorted(by_split):
        for reason, count in by_split[split].most_common():
            summary_rows.append({"scope": split, "reason": reason, "components": count, "area": area_by_split[split][reason]})
    write_csv(output_dir / "summary.csv", summary_rows, ["scope", "reason", "components", "area"])

    for rank, row in enumerate(component_rows[: args.max_crops], start=1):
        save_crop(crops_dir, rank, row)

    print(f"pages={len(page_rows)}")
    print(f"components={len(component_rows)}")
    print(f"components_csv={output_dir / 'new_overerase_components.csv'}")
    print(f"summary_csv={output_dir / 'summary.csv'}")
    print(f"crops_dir={crops_dir}")


if __name__ == "__main__":
    main()
