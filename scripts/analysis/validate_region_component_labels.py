#!/usr/bin/env python3
"""Validate reviewed labels for region-component selector training."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


DEFAULT_POSITIVE = {"accept", "keep", "safe", "positive", "1", "yes"}
DEFAULT_NEGATIVE = {"reject", "drop", "unsafe", "negative", "0", "no"}
DEFAULT_IGNORE = {"", "review", "unsure", "ambiguous", "skip", "ignore"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-csv", required=True)
    parser.add_argument("--label-csv", action="append", required=True)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--positive-label", action="append", default=[])
    parser.add_argument("--negative-label", action="append", default=[])
    parser.add_argument("--ignore-label", action="append", default=[])
    parser.add_argument("--require-positive", type=int, default=1)
    parser.add_argument("--require-negative", type=int, default=1)
    parser.add_argument("--fail-on-unknown-label", action="store_true")
    parser.add_argument("--fail-on-unmatched", action="store_true")
    parser.add_argument("--fail-on-duplicates", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["split"], row["file"], row["component_id"]


def norm(value: str) -> str:
    return value.strip().lower()


def label_set(values: list[str], defaults: set[str]) -> set[str]:
    return {norm(value) for value in values} if values else set(defaults)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    component_rows = read_rows(Path(args.components_csv))
    component_keys = {key(row) for row in component_rows}
    positive = label_set(args.positive_label, DEFAULT_POSITIVE)
    negative = label_set(args.negative_label, DEFAULT_NEGATIVE)
    ignored = label_set(args.ignore_label, DEFAULT_IGNORE)

    seen: Counter[tuple[str, str, str]] = Counter()
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    rows_out: list[dict[str, object]] = []
    unknown_labels: list[str] = []
    unmatched: list[tuple[str, str, str]] = []

    for label_csv in args.label_csv:
        for row in read_rows(Path(label_csv)):
            row_key = key(row)
            seen[row_key] += 1
            label = norm(row.get(args.label_column, ""))
            if row_key not in component_keys:
                unmatched.append(row_key)
            if label in positive:
                status = "positive"
            elif label in negative:
                status = "negative"
            elif label in ignored:
                status = "ignored"
            else:
                status = "unknown"
                unknown_labels.append(label)
            label_counts[status] += 1
            label_counts[f"raw:{label}"] += 1
            split_counts[row.get("split", "")] += 1
            bucket_counts[row.get("bucket", "")] += 1
            rows_out.append({
                "split": row.get("split", ""),
                "file": row.get("file", ""),
                "component_id": row.get("component_id", ""),
                "label": label,
                "status": status,
                "matched_component": int(row_key in component_keys),
                "duplicate_count": seen[row_key],
                "bucket": row.get("bucket", ""),
            })

    duplicates = [item for item, count in seen.items() if count > 1]
    errors: list[str] = []
    if label_counts["positive"] < args.require_positive:
        errors.append(f"positive labels {label_counts['positive']} < required {args.require_positive}")
    if label_counts["negative"] < args.require_negative:
        errors.append(f"negative labels {label_counts['negative']} < required {args.require_negative}")
    if args.fail_on_unknown_label and unknown_labels:
        errors.append(f"unknown labels: {sorted(set(unknown_labels))}")
    if args.fail_on_unmatched and unmatched:
        sample = ", ".join("/".join(item) for item in unmatched[:5])
        errors.append(f"unmatched component labels={len(unmatched)} sample={sample}")
    if args.fail_on_duplicates and duplicates:
        sample = ", ".join("/".join(item) for item in duplicates[:5])
        errors.append(f"duplicate component labels={len(duplicates)} sample={sample}")

    if args.output_csv:
        write_csv(Path(args.output_csv), rows_out)

    print(f"component_rows={len(component_rows)}")
    print(f"label_rows={len(rows_out)}")
    print(f"positive={label_counts['positive']} negative={label_counts['negative']} ignored={label_counts['ignored']} unknown={label_counts['unknown']}")
    print(f"unmatched={len(unmatched)} duplicates={len(duplicates)}")
    print(f"splits={dict(sorted(split_counts.items()))}")
    print(f"buckets={dict(sorted(bucket_counts.items()))}")
    if args.output_csv:
        print(f"output_csv={args.output_csv}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
