#!/usr/bin/env python3
"""Compare candidate predictions against a baseline using local target deltas.

This is intended to avoid judging gates by aggregate residual/overerase alone.
It extracts connected regions where the candidate is materially closer to the
target than the baseline, and regions where it is materially worse. Optional
crop sheets show input / target / baseline / candidate panels for review.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--crops-dir", default="")
    parser.add_argument("--contact-sheet", default="", help="Optional image path summarizing saved crops.")
    parser.add_argument("--contact-thumb-width", type=int, default=720)
    parser.add_argument("--contact-thumb-height", type=int, default=160)
    parser.add_argument("--max-crops", type=int, default=80)
    parser.add_argument("--change-threshold", type=int, default=12)
    parser.add_argument("--gain-threshold", type=float, default=8.0)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--pad", type=int, default=28)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    path = Path(image_path)
    text = str(path)
    if "/all_images/" in text:
        return Path(text.replace("/all_images/", "/all_labels/"))
    if "image_path" in text:
        return Path(text.replace("image_path", "label_path"))
    raise ValueError(f"Cannot infer label path for {image_path}")


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
    return int(x0), int(y0), int(x1), int(y1)


def classify_region(
    region_type: str,
    changed_ratio: float,
    mean_input_label_delta: float,
    mean_abs_candidate_baseline_delta: float,
) -> str:
    if changed_ratio < 0.25:
        return f"{region_type}_mostly_background"
    if mean_input_label_delta < 18:
        return f"{region_type}_low_contrast_target"
    if mean_abs_candidate_baseline_delta < 6:
        return f"{region_type}_subtle_texture_shift"
    return f"{region_type}_visible_target_region"


def save_crop(
    out_dir: Path,
    rank: int,
    file_stem: str,
    region_type: str,
    reason: str,
    bounds: tuple[int, int, int, int],
    images: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Path:
    x0, y0, x1, y1 = bounds
    panels = []
    for title, image in zip(("input", "target", "baseline", "candidate"), images):
        crop = image[y0:y1, x0:x1]
        title_bar = np.full((24, crop.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(title_bar, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(np.vstack([title_bar, crop]))
    triplet = np.hstack(panels)
    out = out_dir / f"{rank:04d}_{file_stem}_{region_type}_{reason}_{x0}_{y0}_{x1}_{y1}.png"
    cv2.imwrite(str(out), triplet)
    return out


def make_contact_sheet(crop_paths: list[Path], output_path: Path, thumb_width: int, thumb_height: int) -> None:
    if not crop_paths:
        return
    rows = []
    for path in crop_paths:
        image = read_bgr(path)
        h, w = image.shape[:2]
        scale = min(thumb_width / w, thumb_height / h)
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        canvas = np.full((thumb_height, thumb_width, 3), 255, dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        cv2.rectangle(canvas, (0, 0), (thumb_width - 1, thumb_height - 1), (160, 160, 160), 1)
        cv2.putText(
            canvas,
            path.name[:110],
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        rows.append(canvas)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack(rows))


def main() -> None:
    args = parse_args()
    baseline_rows = read_rows(Path(args.baseline_metrics))
    candidate_rows = read_rows(Path(args.candidate_metrics))
    baseline_by_file = {row["file"]: row for row in baseline_rows}

    component_rows: list[dict[str, str | int | float]] = []
    for candidate_row in candidate_rows:
        baseline_row = baseline_by_file[candidate_row["file"]]
        input_bgr = read_bgr(candidate_row["image_path"])
        label_bgr = ensure_same_size(read_bgr(label_path_for(candidate_row["image_path"])), input_bgr)
        baseline_bgr = ensure_same_size(read_bgr(baseline_row["pred_path"]), input_bgr)
        candidate_bgr = ensure_same_size(read_bgr(candidate_row["pred_path"]), input_bgr)

        changed = build_changed_mask(input_bgr, label_bgr, args.change_threshold)
        baseline_label_delta = cv2.absdiff(baseline_bgr, label_bgr).mean(axis=2)
        candidate_label_delta = cv2.absdiff(candidate_bgr, label_bgr).mean(axis=2)
        input_label_delta = cv2.absdiff(input_bgr, label_bgr).mean(axis=2)
        candidate_baseline_delta = cv2.absdiff(candidate_bgr, baseline_bgr).mean(axis=2)
        improvement = baseline_label_delta - candidate_label_delta

        masks = {
            "improve": changed & (improvement >= args.gain_threshold),
            "regress": changed & (improvement <= -args.gain_threshold),
        }
        image_h, image_w = changed.shape
        for region_type, mask in masks.items():
            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            for component_id in range(1, count):
                x, y, w, h, area = stats[component_id]
                area = int(area)
                if area < args.min_area:
                    continue
                component = labels == component_id
                x0, y0, x1, y1 = crop_window(int(x), int(y), int(w), int(h), image_w, image_h, args.pad)
                changed_ratio = float(changed[component].mean())
                mean_improvement = float(improvement[component].mean())
                mean_input_label_delta = float(input_label_delta[component].mean())
                mean_abs_candidate_baseline_delta = float(candidate_baseline_delta[component].mean())
                reason = classify_region(
                    region_type,
                    changed_ratio,
                    mean_input_label_delta,
                    mean_abs_candidate_baseline_delta,
                )
                component_rows.append({
                    "file": candidate_row["file"],
                    "source": candidate_row.get("source", ""),
                    "region_type": region_type,
                    "reason": reason,
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
                    "mean_improvement": mean_improvement,
                    "mean_input_label_delta": mean_input_label_delta,
                    "mean_abs_candidate_baseline_delta": mean_abs_candidate_baseline_delta,
                    "changed_ratio": changed_ratio,
                    "baseline_pred_path": baseline_row["pred_path"],
                    "candidate_pred_path": candidate_row["pred_path"],
                    "image_path": candidate_row["image_path"],
                    "label_path": str(label_path_for(candidate_row["image_path"])),
                })

    component_rows.sort(key=lambda row: (str(row["region_type"]) != "improve", -int(row["area"])))
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if component_rows:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(component_rows[0].keys()))
            writer.writeheader()
            writer.writerows(component_rows)
    else:
        output_csv.write_text("", encoding="utf-8")

    crop_paths: list[Path] = []
    if args.crops_dir and component_rows:
        crops_dir = Path(args.crops_dir)
        crops_dir.mkdir(parents=True, exist_ok=True)
        cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for rank, row in enumerate(component_rows[: args.max_crops], start=1):
            file = str(row["file"])
            if file not in cache:
                input_bgr = read_bgr(row["image_path"])
                cache[file] = (
                    input_bgr,
                    ensure_same_size(read_bgr(row["label_path"]), input_bgr),
                    ensure_same_size(read_bgr(row["baseline_pred_path"]), input_bgr),
                    ensure_same_size(read_bgr(row["candidate_pred_path"]), input_bgr),
                )
            crop_paths.append(
                save_crop(
                    crops_dir,
                    rank,
                    Path(file).stem,
                    str(row["region_type"]),
                    str(row["reason"]),
                    (int(row["crop_x0"]), int(row["crop_y0"]), int(row["crop_x1"]), int(row["crop_y1"])),
                    cache[file],
                )
            )
    if args.contact_sheet and crop_paths:
        make_contact_sheet(
            crop_paths,
            Path(args.contact_sheet),
            thumb_width=args.contact_thumb_width,
            thumb_height=args.contact_thumb_height,
        )

    summary = Counter()
    area = Counter()
    for row in component_rows:
        key = (str(row["region_type"]), str(row["reason"]))
        summary[key] += 1
        area[key] += int(row["area"])
    summary_rows = [
        {
            "region_type": region_type,
            "reason": reason,
            "components": summary[(region_type, reason)],
            "area": area[(region_type, reason)],
        }
        for region_type, reason in sorted(summary)
    ]
    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        summary_csv.write_text("", encoding="utf-8")

    print(f"components={len(component_rows)}")
    print(f"output_csv={output_csv}")
    print(f"summary_csv={summary_csv}")
    if args.contact_sheet and crop_paths:
        print(f"contact_sheet={args.contact_sheet}")
    for row in summary_rows:
        print(f"{row['region_type']} {row['reason']}: components={row['components']} area={row['area']}")


if __name__ == "__main__":
    main()
