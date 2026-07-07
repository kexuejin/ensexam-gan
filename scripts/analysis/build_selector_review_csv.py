#!/usr/bin/env python3
"""Build product-quality review CSV rows from a materialized selector manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SPLIT_DEFAULTS = {
    "scut115": ("data-links/samples/SCUT-EnsExam/test/all_images", "data-links/samples/SCUT-EnsExam/test/all_labels"),
    "holdout40": ("data-links/samples/SCUT-EnsExam/train/all_images", "data-links/samples/SCUT-EnsExam/train/all_labels"),
    "train160": ("data-links/samples/SCUT-EnsExam/train/all_images", "data-links/samples/SCUT-EnsExam/train/all_labels"),
    "next120": ("data-links/samples/SCUT-EnsExam/train/all_images", "data-links/samples/SCUT-EnsExam/train/all_labels"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        metavar="NAME:SOURCE_DIR:TARGET_DIR:BASELINE_METRICS:CANDIDATE_METRICS",
        help="May be repeated. Source/target dirs default for known split names.",
    )
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--review-pack", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_split(value: str) -> tuple[str, Path, Path, Path, Path]:
    parts = value.split(":", 4)
    if len(parts) != 5 or not all(parts):
        raise ValueError(f"Invalid --split {value!r}; expected NAME:SOURCE:TARGET:BASELINE:CANDIDATE")
    return parts[0], Path(parts[1]), Path(parts[2]), Path(parts[3]), Path(parts[4])


def split_configs(values: list[str]) -> dict[str, tuple[Path, Path, Path, Path]]:
    configs: dict[str, tuple[Path, Path, Path, Path]] = {}
    for value in values:
        name, source_dir, target_dir, baseline_metrics, candidate_metrics = parse_split(value)
        configs[name] = (source_dir, target_dir, baseline_metrics, candidate_metrics)
    return configs


def default_dirs(split: str) -> tuple[Path, Path]:
    if split not in SPLIT_DEFAULTS:
        raise KeyError(f"No default source/target dirs for split {split!r}; provide --split")
    source_dir, target_dir = SPLIT_DEFAULTS[split]
    return Path(source_dir), Path(target_dir)


def metrics_by_file(path: Path) -> dict[str, dict[str, str]]:
    return {row["file"]: row for row in read_rows(path)}


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    args = parse_args()
    selection_rows = read_rows(Path(args.selection_csv))
    selected_rows = [row for row in selection_rows if int(row["selected_candidate"])]
    configs = split_configs(args.split)

    metric_cache: dict[str, tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]] = {}
    review_rows: list[dict[str, str]] = []
    for row in selected_rows:
        split = row["split"]
        if split in configs:
            source_dir, target_dir, baseline_metrics, candidate_metrics = configs[split]
        else:
            source_dir, target_dir = default_dirs(split)
            baseline_metrics = candidate_metrics = Path()

        if split not in metric_cache:
            if not baseline_metrics or not candidate_metrics:
                raise ValueError(f"Metrics CSVs required for split {split!r}; provide --split")
            metric_cache[split] = (metrics_by_file(baseline_metrics), metrics_by_file(candidate_metrics))
        baseline_by_file, candidate_by_file = metric_cache[split]

        file = row["file"]
        baseline = baseline_by_file[file]
        candidate = candidate_by_file[file]
        source_input = require_path(source_dir / file)
        target = require_path(target_dir / file)
        baseline_pred = require_path(Path(baseline["pred_path"]))
        candidate_pred = require_path(Path(candidate["pred_path"]))
        review_rows.append(
            {
                "split": split,
                "file": file,
                "bucket": args.bucket,
                "candidate": args.candidate_name,
                "source_input": str(source_input),
                "baseline_pred": str(baseline_pred),
                "candidate_pred": str(candidate_pred),
                "target": str(target),
                "review_pack": args.review_pack,
                "notes": (
                    f"gain={float(row['gain']):.6f}; "
                    f"over_delta={float(row['over_delta']):.8f}; "
                    f"selected_source={row['selected_source']}"
                ),
            }
        )

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "file",
        "bucket",
        "candidate",
        "source_input",
        "baseline_pred",
        "candidate_pred",
        "target",
        "review_pack",
        "notes",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"rows={len(review_rows)} output_csv={output}")


if __name__ == "__main__":
    main()
