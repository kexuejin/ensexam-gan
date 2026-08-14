#!/usr/bin/env python3
"""Summarize universal-sidecar matched-copy source-guard metrics.

The input is the ``post_freeze_metrics.csv`` emitted by
``scripts/eval/evaluate_prediction_directory.py``.  That CSV contains baseline
metrics copied from the source run plus candidate metrics measured after
freezing predictions.  This guard is intentionally strict: any positive
candidate-minus-baseline residual or overerase delta is a source regression.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {
    "file",
    "baseline_residual_ratio",
    "baseline_overerase_ratio",
    "residual_ratio",
    "overerase_ratio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--post-freeze-metrics",
        required=True,
        help="post_freeze_metrics.csv from evaluate_prediction_directory.py",
    )
    parser.add_argument(
        "--expected-samples-file",
        required=True,
        help="newline-delimited source-guard sample manifest",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_expected_files(path: Path) -> list[str]:
    files = [
        Path(line.strip()).name
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not files:
        raise ValueError(f"{path} has no samples")
    duplicates = sorted(file for file, count in Counter(files).items() if count > 1)
    if duplicates:
        raise ValueError(f"{path} has duplicate samples: {duplicates}")
    return files


def read_rows(path: Path, expected_files: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows")
    missing = REQUIRED_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    row_files = [row["file"].strip() for row in rows]
    if any(not file for file in row_files):
        raise ValueError(f"{path} has a blank file value")
    duplicates = sorted(
        file for file, count in Counter(row_files).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"{path} has duplicate files: {duplicates}")
    missing_files = sorted(set(expected_files).difference(row_files))
    unexpected_files = sorted(set(row_files).difference(expected_files))
    if len(row_files) != len(expected_files) or missing_files or unexpected_files:
        raise ValueError(
            f"{path} does not match expected sample manifest: "
            f"expected={len(expected_files)} actual={len(row_files)} "
            f"missing={missing_files} unexpected={unexpected_files}"
        )
    return rows


def finite_ratio(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid {key} for {row.get('file', '<unknown>')}: {row.get(key)!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"non-finite {key} for {row.get('file', '<unknown>')}: {value}"
        )
    if value < 0.0 or value > 1.0:
        raise ValueError(
            f"out-of-range {key} for {row.get('file', '<unknown>')}: {value}"
        )
    return value


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)


def percentile_linear(values: Iterable[float], q: float) -> float:
    """Return NumPy-style linear percentile without requiring NumPy."""
    sorted_values = sorted(values)
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def summarize_metric(
    rows: list[dict[str, str]],
    baseline_key: str,
    candidate_key: str,
) -> dict[str, float | int]:
    baseline_values = [finite_ratio(row, baseline_key) for row in rows]
    candidate_values = [finite_ratio(row, candidate_key) for row in rows]
    deltas = [
        candidate - baseline
        for baseline, candidate in zip(baseline_values, candidate_values)
    ]
    return {
        "baseline_mean": mean(baseline_values),
        "candidate_mean": mean(candidate_values),
        "delta_mean": mean(deltas),
        "delta_p95": percentile_linear(deltas, 0.95),
        "delta_min": min(deltas),
        "delta_max": max(deltas),
        "nonzero_delta_count": sum(1 for delta in deltas if delta != 0.0),
    }


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    residual_deltas = [
        finite_ratio(row, "residual_ratio")
        - finite_ratio(row, "baseline_residual_ratio")
        for row in rows
    ]
    overerase_deltas = [
        finite_ratio(row, "overerase_ratio")
        - finite_ratio(row, "baseline_overerase_ratio")
        for row in rows
    ]
    nonzero_delta_files = [
        {
            "file": row["file"],
            "delta_residual_ratio": residual_delta,
            "delta_overerase_ratio": overerase_delta,
        }
        for row, residual_delta, overerase_delta in zip(
            rows, residual_deltas, overerase_deltas
        )
        if residual_delta != 0.0 or overerase_delta != 0.0
    ]

    failures: list[str] = []
    if any(delta > 0.0 for delta in residual_deltas):
        failures.append("residual_source_guard_regression")
    if any(delta > 0.0 for delta in overerase_deltas):
        failures.append("overerase_source_guard_regression")

    return {
        "failures": failures,
        "measurable_page_delta": bool(nonzero_delta_files),
        "nonzero_delta_files": nonzero_delta_files,
        "overerase_ratio": summarize_metric(
            rows, "baseline_overerase_ratio", "overerase_ratio"
        ),
        "pages": len(rows),
        "residual_ratio": summarize_metric(
            rows, "baseline_residual_ratio", "residual_ratio"
        ),
        "source_guard_status": "fail" if failures else "pass",
    }


def main() -> int:
    args = parse_args()
    metrics_path = Path(args.post_freeze_metrics)
    expected_samples_path = Path(args.expected_samples_file)
    expected_files = read_expected_files(expected_samples_path)
    summary = summarize(read_rows(metrics_path, expected_files))
    summary["provenance"] = {
        "expected_samples_file": str(expected_samples_path),
        "expected_samples_sha256": sha256_file(expected_samples_path),
        "post_freeze_metrics": str(metrics_path),
        "post_freeze_metrics_sha256": sha256_file(metrics_path),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 1 if summary["source_guard_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
