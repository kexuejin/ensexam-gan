#!/usr/bin/env python3
"""Build review crops from region-component selector outputs.

The review pack is meant to create compact labels for a future learned
patch/region selector. It samples connected edit components from
evaluate_region_component_selector.py outputs, then renders source/baseline/
candidate/target crop panels without relying on visual-AI uploads.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


PANELS = (
    ("source", "input"),
    ("baseline", "baseline"),
    ("candidate", "candidate"),
    ("target", "target"),
)

BUCKET_LIMITS = {
    "high_gain_accept": 20,
    "borderline_review": 20,
    "hard_reject": 20,
    "large_reject": 12,
    "small_accept": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-csv", required=True)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME:BASELINE_METRICS:CANDIDATE_METRICS",
        help="May be repeated. Used to resolve source/baseline/candidate/target image paths.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allowed-split", action="append", default=[])
    parser.add_argument("--max-total", type=int, default=80)
    parser.add_argument("--max-per-page", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=220)
    parser.add_argument("--thumb-size", type=int, default=180)
    parser.add_argument("--min-area", type=int, default=3)
    parser.add_argument("--contact-width", type=int, default=1440)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_split(value: str) -> tuple[str, Path, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Invalid --split {value!r}; expected NAME:BASELINE:CANDIDATE")
    return parts[0], Path(parts[1]), Path(parts[2])


def label_path_for(image_path: str | Path) -> Path:
    parts = list(Path(image_path).parts)
    try:
        idx = parts.index("all_images")
    except ValueError as exc:
        raise ValueError(f"Cannot derive label path from {image_path}") from exc
    parts[idx] = "all_labels"
    return Path(*parts)


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)


def path_index(split_specs: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for split, baseline_metrics, candidate_metrics in map(parse_split, split_specs):
        baseline_by_file = {row["file"]: row for row in read_rows(baseline_metrics)}
        candidate_by_file = {row["file"]: row for row in read_rows(candidate_metrics)}
        for file, baseline in baseline_by_file.items():
            candidate = candidate_by_file[file]
            index[(split, file)] = {
                "source_input": baseline["image_path"],
                "target": str(label_path_for(baseline["image_path"])),
                "baseline_pred": baseline["pred_path"],
                "candidate_pred": candidate["pred_path"],
            }
    return index


def expand_crop(row: dict[str, str], image_shape: tuple[int, int, int], crop_size: int) -> tuple[int, int, int, int]:
    x = int(float(row["x"]))
    y = int(float(row["y"]))
    w = int(float(row["w"]))
    h = int(float(row["h"]))
    cx = x + w // 2
    cy = y + h // 2
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
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    h, w = patch.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    resized = cv2.resize(patch, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    y0 = (size - resized.shape[0]) // 2
    x0 = (size - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    bar = np.full((28, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(bar, label[:32], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return np.concatenate([bar, image], axis=0)


def render_crop(
    images: dict[str, np.ndarray],
    row: dict[str, str],
    crop: tuple[int, int, int, int],
    size: int,
) -> np.ndarray:
    panels = [add_label(crop_panel(images[key], crop, size), title) for key, title in PANELS]
    body = np.concatenate(panels, axis=1)
    header = np.full((48, body.shape[1], 3), 255, dtype=np.uint8)
    text = (
        f"{row['bucket']} | {row['split']}/{row['file']} cid={row['component_id']} "
        f"area={row['area']} gain={float(row['component_gain']):.2f} verdict={row['component_verdict']}"
    )
    cv2.putText(header, text[:170], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
    return np.concatenate([header, body], axis=0)


def bucket(row: dict[str, str]) -> str:
    verdict = row["component_verdict"]
    gain = float(row["component_gain"])
    area = int(float(row["area"]))
    hurt = float(row["component_hurt_ratio"])
    if verdict == "accept" and gain >= 5:
        return "high_gain_accept"
    if verdict == "review":
        return "borderline_review"
    if verdict == "reject" and area >= 100:
        return "large_reject"
    if verdict == "reject" and hurt >= 0.5:
        return "hard_reject"
    if verdict == "accept" and area <= 25:
        return "small_accept"
    return "other"


def priority(row: dict[str, str]) -> float:
    gain = abs(float(row["component_gain"]))
    area = min(int(float(row["area"])), 500)
    hurt = float(row["component_hurt_ratio"])
    help_ratio = float(row["component_help_ratio"])
    return gain * 10.0 + area * 0.2 + hurt * 30.0 + help_ratio * 15.0


def select_rows(rows: list[dict[str, str]], max_total: int, max_per_page: int) -> list[dict[str, str]]:
    for row in rows:
        row["bucket"] = bucket(row)
    by_bucket: dict[str, list[dict[str, str]]] = {name: [] for name in BUCKET_LIMITS}
    for row in rows:
        if int(float(row["area"])) < 3:
            continue
        if row["bucket"] in by_bucket:
            by_bucket[row["bucket"]].append(row)
    for bucket_rows in by_bucket.values():
        bucket_rows.sort(key=priority, reverse=True)

    selected: list[dict[str, str]] = []
    per_page: dict[tuple[str, str], int] = {}
    for bucket_name, limit in BUCKET_LIMITS.items():
        picked = 0
        for row in by_bucket[bucket_name]:
            page_key = (row["split"], row["file"])
            if per_page.get(page_key, 0) >= max_per_page:
                continue
            selected.append(row)
            per_page[page_key] = per_page.get(page_key, 0) + 1
            picked += 1
            if picked >= limit or len(selected) >= max_total:
                break
        if len(selected) >= max_total:
            break
    return selected[:max_total]


def safe_name(row: dict[str, str], rank: int) -> str:
    return (
        f"{rank:03d}_{row['bucket']}_{row['split']}_{Path(row['file']).stem}_"
        f"c{row['component_id']}.png"
    ).replace("/", "_")


def resize_contact_row(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    scale = width / max(image.shape[1], 1)
    return cv2.resize(
        image,
        (width, max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def write_contact_sheet(images: list[np.ndarray], output_path: Path, width: int) -> None:
    if not images:
        return
    sheet_rows = [resize_contact_row(image, width) for image in images]
    separator = np.full((10, width, 3), 230, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for image in sheet_rows:
        parts.extend([image, separator])
    cv2.imwrite(str(output_path), np.concatenate(parts[:-1], axis=0))


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.components_csv))
    allowed_splits = set(args.allowed_split)
    if allowed_splits:
        rows = [row for row in rows if row["split"] in allowed_splits]
    rows = [row for row in rows if int(float(row["area"])) >= args.min_area]
    selected = select_rows(rows, args.max_total, args.max_per_page)
    paths = path_index(args.split)

    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, object]] = []
    contact_images: list[np.ndarray] = []
    contact_images_by_bucket: dict[str, list[np.ndarray]] = {}
    image_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for rank, row in enumerate(selected, start=1):
        key = (row["split"], row["file"])
        if key not in image_cache:
            path_row = paths[key]
            baseline = load_image(path_row["baseline_pred"])
            image_cache[key] = {
                "baseline": baseline,
                "source": resize_like(load_image(path_row["source_input"]), baseline),
                "candidate": resize_like(load_image(path_row["candidate_pred"]), baseline),
                "target": resize_like(load_image(path_row["target"]), baseline),
            }
        crop = expand_crop(row, image_cache[key]["baseline"].shape, args.crop_size)
        rendered = render_crop(image_cache[key], row, crop, args.thumb_size)
        crop_path = crops_dir / safe_name(row, rank)
        cv2.imwrite(str(crop_path), rendered)
        out = dict(row)
        out.update({
            "priority_rank": rank,
            "priority_score": round(priority(row), 3),
            "crop_box": f"{crop[0]},{crop[1]},{crop[2]},{crop[3]}",
            "crop_review_image": str(crop_path),
            "label": "",
            "flags": "",
            "reviewer": "",
            "review_date": "",
            "comment": "",
        })
        index_rows.append(out)
        contact_images.append(rendered)
        contact_images_by_bucket.setdefault(row["bucket"], []).append(rendered)

    fields = [
        "priority_rank",
        "priority_score",
        "bucket",
        "split",
        "file",
        "component_id",
        "x",
        "y",
        "w",
        "h",
        "area",
        "component_gain",
        "component_help_ratio",
        "component_hurt_ratio",
        "component_verdict",
        "baseline_edit_mean",
        "candidate_source_delta_mean",
        "fill_ratio",
        "source_edge_mean",
        "source_texture_mean",
        "crop_box",
        "crop_review_image",
        "label",
        "flags",
        "reviewer",
        "review_date",
        "comment",
    ]
    write_csv(output_dir / "component-review-index.csv", index_rows, fields)
    write_csv(output_dir / "component-labels-template.csv", index_rows, fields)

    write_contact_sheet(contact_images, output_dir / "contact_sheet.png", args.contact_width)
    for bucket_name, images in sorted(contact_images_by_bucket.items()):
        write_contact_sheet(images, output_dir / f"contact_sheet_{bucket_name}.png", args.contact_width)

    counts: dict[str, int] = {}
    for row in index_rows:
        counts[str(row["bucket"])] = counts.get(str(row["bucket"]), 0) + 1
    print(f"input_rows={len(rows)} selected={len(index_rows)} buckets={counts}")
    print(f"index_csv={output_dir / 'component-review-index.csv'}")
    print(f"labels_template={output_dir / 'component-labels-template.csv'}")
    print(f"contact_sheet={output_dir / 'contact_sheet.png'}")
    for bucket_name in sorted(contact_images_by_bucket):
        print(f"contact_sheet_{bucket_name}={output_dir / f'contact_sheet_{bucket_name}.png'}")


if __name__ == "__main__":
    main()
