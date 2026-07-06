#!/usr/bin/env python3
"""Prioritize crop-review rows for manual product-quality labeling."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


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

    print(f"input_rows={len(rows)}")
    print(f"selected={len(selected)}")
    print(f"output_csv={output_path}")


if __name__ == "__main__":
    main()
