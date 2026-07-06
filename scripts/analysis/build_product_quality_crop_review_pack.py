#!/usr/bin/env python3
"""Build crop-level review packs for product-quality candidate comparisons.

The page-level contact sheet is useful for broad regressions, but small
residual strokes, correction-fluid patches, gray tone shifts, and printed-text
damage need zoomed crops. This script creates local side-by-side crop panels
from review CSV rows without requiring visual AI uploads.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from generate_whiteout_tone_harmonization import detect_whiteout_components


PANELS = [
    ("source_input", "input"),
    ("baseline_pred", "baseline"),
    ("candidate_pred", "candidate"),
    ("target", "target"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", default="docs/product-quality-review-pages.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bucket", default="")
    parser.add_argument("--candidate", default="")
    parser.add_argument("--max-crops-per-row", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=280)
    parser.add_argument("--thumb-size", type=int, default=220)
    parser.add_argument("--min-component-area", type=int, default=120)
    parser.add_argument("--diff-threshold", type=int, default=4)
    parser.add_argument(
        "--include-target-residual",
        action="store_true",
        help="Also crop baseline-vs-target residual regions; useful for missed-coverage no-op pages.",
    )
    parser.add_argument("--max-contact-crops", type=int, default=80)
    parser.add_argument(
        "--labels-template",
        default="",
        help="Optional crop-label template CSV path. Defaults to <output-dir>/crop-labels-template.csv.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_image(path_text: str) -> np.ndarray:
    image = cv2.imread(str(Path(path_text)), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path_text)
    return image


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)


Box = tuple[int, int, int, int, int, str]


def component_boxes(mask: np.ndarray, min_area: int, source_type: str) -> list[Box]:
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    boxes = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, x + w - 1, y + h - 1, area, source_type))
    return sorted(boxes, key=lambda item: item[4], reverse=True)


def candidate_diff_boxes(
    baseline: np.ndarray,
    candidate: np.ndarray,
    diff_threshold: int,
    min_area: int,
) -> list[Box]:
    diff = np.abs(candidate.astype(np.int16) - baseline.astype(np.int16)).max(axis=2)
    return component_boxes(diff > diff_threshold, min_area, "candidate_diff")


def target_residual_boxes(
    baseline: np.ndarray,
    target: np.ndarray,
    diff_threshold: int,
    min_area: int,
) -> list[Box]:
    diff = np.abs(baseline.astype(np.int16) - target.astype(np.int16)).max(axis=2)
    gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # Prefer regions where the baseline is darker than target, which is the
    # common residual handwriting / unremoved mark signal.
    residual = (diff > diff_threshold) & ((target_gray.astype(np.int16) - gray.astype(np.int16)) > 8)
    return component_boxes(residual, min_area, "target_residual")


def whiteout_boxes(
    source: np.ndarray,
    baseline: np.ndarray,
    min_area: int,
) -> list[Box]:
    components = detect_whiteout_components(
        source=source,
        baseline=baseline,
        min_area=min_area,
        max_area_ratio=0.08,
        bright_delta=10.0,
    )
    boxes = []
    for component in components:
        ys, xs = np.where(component)
        if len(xs) == 0:
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), int(component.sum()), "whiteout_detector"))
    return sorted(boxes, key=lambda item: item[4], reverse=True)


def expand_box(
    box: Box,
    image_shape: tuple[int, int, int],
    crop_size: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2, _, _ = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = crop_size // 2
    left = max(0, cx - half)
    top = max(0, cy - half)
    right = min(image_shape[1], left + crop_size)
    bottom = min(image_shape[0], top + crop_size)
    left = max(0, right - crop_size)
    top = max(0, bottom - crop_size)
    return left, top, right, bottom


def crop_panel(image: np.ndarray, crop: tuple[int, int, int, int], size: int) -> np.ndarray:
    left, top, right, bottom = crop
    patch = image[top:bottom, left:right]
    h, w = patch.shape[:2]
    canvas = np.full((size, size, 3), 245, np.uint8)
    scale = min(size / max(w, 1), size / max(h, 1))
    resized = cv2.resize(patch, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    y0 = (size - resized.shape[0]) // 2
    x0 = (size - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    bar_h = 30
    out = np.full((image.shape[0] + bar_h, image.shape[1], 3), 255, np.uint8)
    out[bar_h:] = image
    cv2.putText(out, label[:40], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
    return out


def crop_row_image(images: dict[str, np.ndarray], row: dict[str, str], crop: tuple[int, int, int, int], size: int) -> np.ndarray:
    panels = []
    for column, title in PANELS:
        panels.append(add_label(crop_panel(images[column], crop, size), title))
    body = np.concatenate(panels, axis=1)
    caption = np.full((42, body.shape[1], 3), 255, np.uint8)
    text = f"{row.get('split','')} | {row.get('file','')} | {row.get('candidate','')} | crop={crop}"
    cv2.putText(caption, text[:150], (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (20, 20, 20), 1, cv2.LINE_AA)
    return np.concatenate([caption, body], axis=0)


def safe_name(row: dict[str, str], row_index: int, crop_index: int) -> str:
    split = row.get("split", "unknown").replace("/", "_")
    file = row.get("file", "unknown").replace("/", "_")
    bucket = row.get("bucket", "bucket").replace("/", "_")
    candidate = row.get("candidate", "candidate").replace("/", "_")
    return f"{row_index:03d}_{crop_index:02d}_{split}_{file}_{candidate}_{bucket}.png"


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.review_csv))
    if args.bucket:
        rows = [row for row in rows if row.get("bucket") == args.bucket]
    if args.candidate:
        rows = [row for row in rows if row.get("candidate") == args.candidate]

    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, object]] = []
    contact_images: list[np.ndarray] = []
    for row_index, row in enumerate(rows, start=1):
        baseline = load_image(row["baseline_pred"])
        images = {
            "baseline_pred": baseline,
            "source_input": resize_like(load_image(row["source_input"]), baseline),
            "candidate_pred": resize_like(load_image(row["candidate_pred"]), baseline),
            "target": resize_like(load_image(row["target"]), baseline),
        }

        boxes = []
        if row.get("bucket") == "correction_fluid_white_patch":
            boxes.extend(whiteout_boxes(images["source_input"], baseline, args.min_component_area))
        boxes.extend(candidate_diff_boxes(baseline, images["candidate_pred"], args.diff_threshold, args.min_component_area))
        if args.include_target_residual or row.get("bucket") == "coverage_negative_noop":
            boxes.extend(target_residual_boxes(baseline, images["target"], args.diff_threshold, args.min_component_area))

        # Deduplicate boxes with the same center neighborhood.
        seen_centers: set[tuple[int, int]] = set()
        unique_boxes = []
        for box in boxes:
            key = (((box[0] + box[2]) // 2) // 32, ((box[1] + box[3]) // 2) // 32)
            if key in seen_centers:
                continue
            seen_centers.add(key)
            unique_boxes.append(box)

        for crop_index, box in enumerate(unique_boxes[: args.max_crops_per_row], start=1):
            crop = expand_box(box, baseline.shape, args.crop_size)
            crop_image = crop_row_image(images, row, crop, args.thumb_size)
            output_path = crops_dir / safe_name(row, row_index, crop_index)
            cv2.imwrite(str(output_path), crop_image)
            index_row = dict(row)
            index_row.update({
                "crop_index": crop_index,
                "source_box": f"{box[0]},{box[1]},{box[2]},{box[3]}",
                "source_area": box[4],
                "source_type": box[5],
                "crop_box": f"{crop[0]},{crop[1]},{crop[2]},{crop[3]}",
                "crop_review_image": str(output_path),
            })
            index_rows.append(index_row)
            if len(contact_images) < args.max_contact_crops:
                contact_images.append(crop_image)

    if contact_images:
        separator = np.full((10, contact_images[0].shape[1], 3), 230, np.uint8)
        parts: list[np.ndarray] = []
        for image in contact_images:
            parts.extend([image, separator])
        cv2.imwrite(str(output_dir / "contact_sheet.png"), np.concatenate(parts[:-1], axis=0))

    fieldnames = sorted({key for row in index_rows for key in row.keys()})
    with (output_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    labels_path = Path(args.labels_template) if args.labels_template else output_dir / "crop-labels-template.csv"
    label_fieldnames = [
        "split",
        "file",
        "candidate",
        "bucket",
        "crop_index",
        "source_box",
        "source_area",
        "source_type",
        "crop_box",
        "label",
        "flags",
        "reviewer",
        "review_date",
        "comment",
        "crop_review_image",
    ]
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=label_fieldnames)
        writer.writeheader()
        for row in index_rows:
            writer.writerow({
                "split": row.get("split", ""),
                "file": row.get("file", ""),
                "candidate": row.get("candidate", ""),
                "bucket": row.get("bucket", ""),
                "crop_index": row.get("crop_index", ""),
                "source_box": row.get("source_box", ""),
                "source_area": row.get("source_area", ""),
                "source_type": row.get("source_type", ""),
                "crop_box": row.get("crop_box", ""),
                "label": "",
                "flags": "",
                "reviewer": "",
                "review_date": "",
                "comment": "",
                "crop_review_image": row.get("crop_review_image", ""),
            })

    print(f"rows={len(rows)}")
    print(f"crops={len(index_rows)}")
    print(f"crops_dir={crops_dir}")
    print(f"contact_sheet={output_dir / 'contact_sheet.png'}")
    print(f"index_csv={output_dir / 'index.csv'}")
    print(f"labels_template={labels_path}")


if __name__ == "__main__":
    main()
