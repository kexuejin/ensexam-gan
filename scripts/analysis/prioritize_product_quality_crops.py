#!/usr/bin/env python3
"""Prioritize crop-review rows for manual product-quality labeling."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


SOURCE_TYPE_WEIGHT = {
    "target_residual": 1000,
    "candidate_diff": 700,
    "whiteout_detector": 600,
}

BUCKET_WEIGHT = {
    "coverage_negative_noop": 500,
    "visible_delta_regress_only": 450,
    "visible_delta_mixed": 350,
    "correction_fluid_white_patch": 325,
    "selector_false_positive_overerase_risk": 300,
    "union_selected_candidate": 250,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-total", type=int, default=24)
    parser.add_argument("--max-per-page", type=int, default=3)
    parser.add_argument("--source-type", action="append", default=[])
    parser.add_argument("--bucket", action="append", default=[])
    parser.add_argument(
        "--feature-csv",
        default="",
        help="Optional residual feature CSV from analyze_crop_residual_features.py.",
    )
    parser.add_argument("--contact-sheet", default="")
    parser.add_argument("--contact-width", type=int, default=1200)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_area(row: dict[str, str]) -> int:
    try:
        return int(float(row.get("source_area", "0")))
    except ValueError:
        return 0


def crop_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row.get("candidate", ""),
        row.get("bucket", ""),
        row.get("split", ""),
        row.get("file", ""),
        row.get("source_type", ""),
        row.get("crop_index", ""),
    )


def read_feature_rows(path_text: str) -> dict[tuple[str, str, str, str, str, str], dict[str, str]]:
    if not path_text:
        return {}
    return {crop_key(row): row for row in read_rows(Path(path_text))}


def feature_score(row: dict[str, str]) -> float:
    try:
        return float(row.get("handwriting_likelihood_score", ""))
    except ValueError:
        return 0.0


def priority_score(row: dict[str, str]) -> float:
    source_type = row.get("source_type", "")
    bucket = row.get("bucket", "")
    area = source_area(row)
    base_score = (
        SOURCE_TYPE_WEIGHT.get(source_type, 0)
        + BUCKET_WEIGHT.get(bucket, 0)
        + min(area, 50000) / 100.0
    )
    return base_score + feature_score(row) * 10.0


def add_priority_header(image: np.ndarray, row: dict[str, object], width: int) -> np.ndarray:
    if image.shape[1] != width:
        scale = width / max(image.shape[1], 1)
        image = cv2.resize(
            image,
            (width, max(1, int(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    header = np.full((58, image.shape[1], 3), 255, np.uint8)
    text = (
        f"rank={row.get('priority_rank')} score={row.get('priority_score')} "
        f"{row.get('split')}/{row.get('file')} {row.get('source_type')} area={row.get('source_area')} "
        f"hw={row.get('handwriting_likelihood_score', '')}"
    )
    cv2.putText(header, text[:170], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    detail = (
        f"residual_px={row.get('residual_px', '')} residual_ratio={row.get('residual_ratio', '')} "
        f"source_overlap={row.get('source_dark_overlap', '')} edge={row.get('edge_density', '')}"
    )
    cv2.putText(header, detail[:170], (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (60, 60, 60), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def write_contact_sheet(rows: list[dict[str, object]], path_text: str, width: int) -> None:
    if not path_text:
        return
    sheet_rows = []
    for row in rows:
        image_path = Path(str(row.get("crop_review_image", "")))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        sheet_rows.append(add_priority_header(image, row, width))
    if not sheet_rows:
        return
    separator = np.full((12, width, 3), 230, np.uint8)
    parts: list[np.ndarray] = []
    for image in sheet_rows:
        parts.extend([image, separator])
    output_path = Path(path_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.concatenate(parts[:-1], axis=0))


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.index_csv))
    if args.source_type:
        allowed = set(args.source_type)
        rows = [row for row in rows if row.get("source_type", "") in allowed]
    if args.bucket:
        allowed = set(args.bucket)
        rows = [row for row in rows if row.get("bucket", "") in allowed]
    feature_by_key = read_feature_rows(args.feature_csv)
    if feature_by_key:
        merged_rows = []
        for row in rows:
            merged = dict(row)
            feature = feature_by_key.get(crop_key(row), {})
            for key in [
                "handwriting_likelihood_score",
                "residual_px",
                "residual_ratio",
                "mean_residual_delta",
                "source_dark_overlap",
                "edge_density",
                "largest_component_fill",
            ]:
                if key in feature:
                    merged[key] = feature[key]
            merged_rows.append(merged)
        rows = merged_rows

    ranked = sorted(rows, key=priority_score, reverse=True)
    selected: list[dict[str, object]] = []
    per_page: dict[tuple[str, str, str, str], int] = {}
    for row in ranked:
        page_key = (row.get("candidate", ""), row.get("bucket", ""), row.get("split", ""), row.get("file", ""))
        if per_page.get(page_key, 0) >= args.max_per_page:
            continue
        out = dict(row)
        out["priority_rank"] = len(selected) + 1
        out["priority_score"] = round(priority_score(row), 3)
        selected.append(out)
        per_page[page_key] = per_page.get(page_key, 0) + 1
        if len(selected) >= args.max_total:
            break

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "priority_rank",
        "priority_score",
        "split",
        "file",
        "candidate",
        "bucket",
        "source_type",
        "source_area",
        "crop_index",
        "source_box",
        "crop_box",
        "crop_review_image",
        "handwriting_likelihood_score",
        "residual_px",
        "residual_ratio",
        "source_dark_overlap",
        "edge_density",
        "label",
        "flags",
        "reviewer",
        "review_date",
        "comment",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            writer.writerow(row)
    write_contact_sheet(selected, args.contact_sheet, args.contact_width)

    print(f"input_rows={len(rows)}")
    print(f"selected={len(selected)}")
    print(f"output_csv={output_path}")
    if args.contact_sheet:
        print(f"contact_sheet={args.contact_sheet}")


if __name__ == "__main__":
    main()
