#!/usr/bin/env python3
"""Bucket residual crop candidates into training triage groups."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--min-handwriting-score", type=float, default=70.0)
    parser.add_argument("--min-source-overlap", type=float, default=0.9)
    parser.add_argument("--min-edge-density", type=float, default=0.65)
    parser.add_argument("--max-residual-ratio", type=float, default=0.30)
    parser.add_argument("--max-per-page", type=int, default=2)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def float_value(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def triage_bucket(row: dict[str, str], args: argparse.Namespace, page_count: int) -> tuple[str, str]:
    score = float_value(row, "handwriting_likelihood_score")
    source_overlap = float_value(row, "source_dark_overlap")
    edge_density = float_value(row, "edge_density")
    residual_ratio = float_value(row, "residual_ratio")
    if page_count > args.max_per_page:
        return "deprioritized_duplicate_page", "page already has enough higher-ranked crops"
    if residual_ratio > args.max_residual_ratio:
        return "probable_background_or_target_mismatch", "residual region is too broad for handwriting"
    if (
        score >= args.min_handwriting_score
        and source_overlap >= args.min_source_overlap
        and edge_density >= args.min_edge_density
    ):
        return "high_confidence_residual_handwriting", "dark source overlap and edge density are high"
    if source_overlap >= args.min_source_overlap and edge_density >= 0.4:
        return "possible_residual_handwriting", "source overlap is high but edge/score evidence is weaker"
    return "probable_background_or_target_mismatch", "weak handwriting-like evidence"


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.priority_csv))
    output_rows: list[dict[str, object]] = []
    per_page: dict[tuple[str, str, str], int] = {}
    for row in rows:
        page_key = (row.get("split", ""), row.get("file", ""), row.get("bucket", ""))
        per_page[page_key] = per_page.get(page_key, 0) + 1
        bucket, reason = triage_bucket(row, args, per_page[page_key])
        out = dict(row)
        out["triage_bucket"] = bucket
        out["triage_reason"] = reason
        output_rows.append(out)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "triage_bucket",
        "triage_reason",
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
        "handwriting_likelihood_score",
        "residual_px",
        "residual_ratio",
        "source_dark_overlap",
        "edge_density",
        "source_input",
        "baseline_pred",
        "candidate_pred",
        "target",
        "crop_review_image",
        "label",
        "flags",
        "reviewer",
        "review_date",
        "comment",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    counts: dict[str, int] = {}
    for row in output_rows:
        bucket = str(row["triage_bucket"])
        counts[bucket] = counts.get(bucket, 0) + 1
    print(f"rows={len(output_rows)}")
    print(f"output_csv={output_path}")
    for key in sorted(counts):
        print(f"{key}={counts[key]}")


if __name__ == "__main__":
    main()
