#!/usr/bin/env python3
"""Compute deterministic residual features for crop-review rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--source-type", action="append", default=[])
    parser.add_argument("--residual-threshold", type=int, default=8)
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


def parse_box(value: str) -> tuple[int, int, int, int]:
    parts = [int(float(part)) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected 4 box values, got {value!r}")
    return parts[0], parts[1], parts[2], parts[3]


def crop(image: np.ndarray, box_text: str) -> np.ndarray:
    left, top, right, bottom = parse_box(box_text)
    return image[top:bottom, left:right]


def edge_density(gray: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    return float((grad[mask] > 24).mean())


def component_fill(mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return 0.0
    largest = max(range(1, count), key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
    area = int(stats[largest, cv2.CC_STAT_AREA])
    width = int(stats[largest, cv2.CC_STAT_WIDTH])
    height = int(stats[largest, cv2.CC_STAT_HEIGHT])
    return area / max(width * height, 1)


def score_handwriting(row: dict[str, object]) -> float:
    return round(
        float(row["source_dark_overlap"]) * 45.0
        + float(row["edge_density"]) * 30.0
        + min(float(row["mean_residual_delta"]), 40.0) * 0.45
        - float(row["largest_component_fill"]) * 18.0
        - max(float(row["residual_ratio"]) - 0.35, 0.0) * 25.0,
        3,
    )


def analyze_row(row: dict[str, str], residual_threshold: int) -> dict[str, object]:
    baseline = load_image(row["baseline_pred"])
    target = resize_like(load_image(row["target"]), baseline)
    source = resize_like(load_image(row["source_input"]), baseline)

    baseline_crop = crop(baseline, row["crop_box"])
    target_crop = crop(target, row["crop_box"])
    source_crop = crop(source, row["crop_box"])

    baseline_gray = cv2.cvtColor(baseline_crop, cv2.COLOR_BGR2GRAY).astype(np.int16)
    target_gray = cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY).astype(np.int16)
    source_gray = cv2.cvtColor(source_crop, cv2.COLOR_BGR2GRAY).astype(np.int16)

    residual_delta = target_gray - baseline_gray
    residual = residual_delta > residual_threshold
    residual_count = int(residual.sum())
    pixels = int(residual.size)
    if residual_count:
        mean_delta = float(residual_delta[residual].mean())
        source_dark_overlap = float(((target_gray - source_gray) > residual_threshold)[residual].mean())
    else:
        mean_delta = 0.0
        source_dark_overlap = 0.0

    output: dict[str, object] = dict(row)
    output.update({
        "residual_px": residual_count,
        "residual_ratio": residual_count / max(pixels, 1),
        "mean_residual_delta": mean_delta,
        "source_dark_overlap": source_dark_overlap,
        "edge_density": edge_density(baseline_gray.astype(np.uint8), residual),
        "largest_component_fill": component_fill(residual),
    })
    output["handwriting_likelihood_score"] = score_handwriting(output)
    return output


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.index_csv))
    if args.source_type:
        allowed = set(args.source_type)
        rows = [row for row in rows if row.get("source_type", "") in allowed]

    analyzed = [analyze_row(row, args.residual_threshold) for row in rows]
    analyzed = sorted(analyzed, key=lambda row: float(row["handwriting_likelihood_score"]), reverse=True)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "handwriting_likelihood_score",
        "split",
        "file",
        "candidate",
        "bucket",
        "source_type",
        "source_area",
        "crop_index",
        "residual_px",
        "residual_ratio",
        "mean_residual_delta",
        "source_dark_overlap",
        "edge_density",
        "largest_component_fill",
        "source_box",
        "crop_box",
        "crop_review_image",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(analyzed)

    print(f"input_rows={len(rows)}")
    print(f"output_rows={len(analyzed)}")
    print(f"output_csv={output_path}")


if __name__ == "__main__":
    main()
