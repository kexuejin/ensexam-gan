#!/usr/bin/env python3
"""Calibrate metric noise across repeated frozen prediction replays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


POST_COLUMNS = {"file", "residual_ratio", "overerase_ratio"}
INFERENCE_COLUMNS = {
    "file",
    "image_sha256",
    "pred_sha256",
    "primary_config_sha256",
    "primary_weights_sha256",
    "page_overlap",
    "batch_size",
    "copy_input_outside_mask",
    "copy_mask_threshold",
    "copy_mask_threshold_auto",
    "copy_mask_dilate",
}
GLOBAL_PROTOCOL_FIELDS = (
    "primary_config_sha256",
    "primary_weights_sha256",
    "page_overlap",
    "batch_size",
    "copy_input_outside_mask",
    "copy_mask_threshold_auto",
    "copy_mask_dilate",
)
PER_FILE_PROTOCOL_FIELDS = ("image_sha256", "copy_mask_threshold")


@dataclass(frozen=True)
class RunSpec:
    name: str
    post_freeze_metrics: Path
    inference_metrics: Path


@dataclass(frozen=True)
class RunData:
    spec: RunSpec
    post_rows: list[dict[str, str]]
    inference_rows: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME:POST_FREEZE_METRICS:INFERENCE_METRICS",
    )
    parser.add_argument("--expected-samples-file", required=True)
    parser.add_argument("--minimum-gain", type=float, default=0.0005)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_run_spec(value: str) -> RunSpec:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise ValueError(
            "--run must be NAME:POST_FREEZE_METRICS:INFERENCE_METRICS"
        )
    return RunSpec(parts[0].strip(), Path(parts[1]), Path(parts[2]))


def read_expected_files(path: Path) -> list[str]:
    files = [
        Path(line.strip()).name
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not files:
        raise ValueError(f"{path} has no samples")
    duplicates = sorted(name for name, count in Counter(files).items() if count > 1)
    if duplicates:
        raise ValueError(f"{path} has duplicate samples: {duplicates}")
    return files


def read_csv_rows(
    path: Path,
    *,
    required_columns: set[str],
    expected_files: list[str],
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows")
    missing_columns = required_columns.difference(rows[0])
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    row_files = [row["file"].strip() for row in rows]
    if any(not name for name in row_files):
        raise ValueError(f"{path} has a blank file value")
    duplicates = sorted(
        name for name, count in Counter(row_files).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"{path} has duplicate files: {duplicates}")
    if row_files != expected_files:
        missing = sorted(set(expected_files).difference(row_files))
        unexpected = sorted(set(row_files).difference(expected_files))
        raise ValueError(
            f"{path} does not match expected sample order: "
            f"missing={missing} unexpected={unexpected}"
        )
    return rows


def finite_ratio(row: dict[str, str], key: str, path: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid {key} for {row.get('file', '<unknown>')} in {path}"
        ) from exc
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(
            f"invalid {key} for {row.get('file', '<unknown>')} in {path}: {value}"
        )
    return value


def nonblank(row: dict[str, str], key: str, path: Path) -> str:
    value = row[key].strip()
    if not value:
        raise ValueError(f"blank {key} for {row['file']} in {path}")
    return value


def load_run(spec: RunSpec, expected_files: list[str]) -> RunData:
    post_rows = read_csv_rows(
        spec.post_freeze_metrics,
        required_columns=POST_COLUMNS,
        expected_files=expected_files,
    )
    inference_rows = read_csv_rows(
        spec.inference_metrics,
        required_columns=INFERENCE_COLUMNS,
        expected_files=expected_files,
    )
    for post_row, inference_row in zip(post_rows, inference_rows):
        if post_row["file"] != inference_row["file"]:
            raise ValueError(f"run {spec.name} has mismatched post/inference rows")
        finite_ratio(post_row, "residual_ratio", spec.post_freeze_metrics)
        finite_ratio(post_row, "overerase_ratio", spec.post_freeze_metrics)
        for key in INFERENCE_COLUMNS.difference({"file"}):
            nonblank(inference_row, key, spec.inference_metrics)
    return RunData(spec, post_rows, inference_rows)


def protocol_for_run(run: RunData) -> dict[str, str]:
    protocol: dict[str, str] = {}
    for key in GLOBAL_PROTOCOL_FIELDS:
        values = {
            nonblank(row, key, run.spec.inference_metrics)
            for row in run.inference_rows
        }
        if len(values) != 1:
            raise ValueError(f"run {run.spec.name} has non-uniform {key}: {values}")
        protocol[key] = values.pop()
    return protocol


def validate_protocols(runs: list[RunData], expected_files: list[str]) -> dict[str, object]:
    baseline_protocol = protocol_for_run(runs[0])
    baseline_rows = {
        row["file"]: row
        for row in runs[0].inference_rows
    }
    per_file: dict[str, dict[str, str]] = {
        name: {
            key: nonblank(baseline_rows[name], key, runs[0].spec.inference_metrics)
            for key in PER_FILE_PROTOCOL_FIELDS
        }
        for name in expected_files
    }
    for run in runs[1:]:
        protocol = protocol_for_run(run)
        if protocol != baseline_protocol:
            raise ValueError(
                f"run {run.spec.name} protocol differs from {runs[0].spec.name}: "
                f"{protocol} != {baseline_protocol}"
            )
        rows = {row["file"]: row for row in run.inference_rows}
        for name in expected_files:
            for key, expected in per_file[name].items():
                actual = nonblank(rows[name], key, run.spec.inference_metrics)
                if actual != expected:
                    raise ValueError(
                        f"run {run.spec.name} {name} {key} differs: "
                        f"{actual} != {expected}"
                    )
    return {
        **baseline_protocol,
        "per_file_copy_mask_threshold": {
            name: per_file[name]["copy_mask_threshold"] for name in expected_files
        },
    }


def metric_values(run: RunData, key: str) -> list[float]:
    return [finite_ratio(row, key, run.spec.post_freeze_metrics) for row in run.post_rows]


def summarize_metric(
    runs: list[RunData],
    expected_files: list[str],
    key: str,
    minimum_gain: float,
) -> dict[str, object]:
    values_by_run = [metric_values(run, key) for run in runs]
    run_means = [sum(values) / len(values) for values in values_by_run]
    aggregate_stddev = statistics.stdev(run_means)
    page_stddevs = [
        statistics.stdev(values)
        for values in zip(*values_by_run)
    ]
    max_page_index = max(range(len(page_stddevs)), key=page_stddevs.__getitem__)

    baseline_values = values_by_run[0]
    max_delta = 0.0
    max_delta_file = expected_files[0]
    max_delta_run = runs[0].spec.name
    for run, values in zip(runs[1:], values_by_run[1:]):
        for name, baseline, value in zip(expected_files, baseline_values, values):
            delta = abs(value - baseline)
            if delta > max_delta:
                max_delta = delta
                max_delta_file = name
                max_delta_run = run.spec.name

    three_sigma = 3.0 * aggregate_stddev
    return {
        "aggregate_mean_by_run": {
            run.spec.name: value for run, value in zip(runs, run_means)
        },
        "aggregate_replay_stddev": aggregate_stddev,
        "aggregate_three_sigma": three_sigma,
        "calibrated_minimum_gain": max(minimum_gain, three_sigma),
        "max_absolute_delta_from_first_run": max_delta,
        "max_absolute_delta_file": max_delta_file,
        "max_absolute_delta_run": max_delta_run,
        "max_page_replay_stddev": page_stddevs[max_page_index],
        "max_page_replay_stddev_file": expected_files[max_page_index],
    }


def changed_metric_files(runs: list[RunData], expected_files: list[str]) -> list[str]:
    changed: list[str] = []
    baseline_residual = metric_values(runs[0], "residual_ratio")
    baseline_overerase = metric_values(runs[0], "overerase_ratio")
    for index, name in enumerate(expected_files):
        if any(
            metric_values(run, "residual_ratio")[index] != baseline_residual[index]
            or metric_values(run, "overerase_ratio")[index] != baseline_overerase[index]
            for run in runs[1:]
        ):
            changed.append(name)
    return changed


def summarize_prediction_hashes(
    runs: list[RunData], expected_files: list[str]
) -> dict[str, object]:
    differing: list[str] = []
    for index, name in enumerate(expected_files):
        hashes = {
            nonblank(run.inference_rows[index], "pred_sha256", run.spec.inference_metrics)
            for run in runs
        }
        if len(hashes) != 1:
            differing.append(name)
    return {
        "identical_files": len(expected_files) - len(differing),
        "different_files": len(differing),
        "different_file_names": differing,
    }


def summarize(
    runs: list[RunData],
    expected_files: list[str],
    expected_samples_path: Path,
    minimum_gain: float,
) -> dict[str, object]:
    protocol = validate_protocols(runs, expected_files)
    return {
        "status": "passed",
        "passed": True,
        "summary": "Frozen replay inputs and protocol validated; measurement noise calibrated.",
        "run_count": len(runs),
        "pages": len(expected_files),
        "minimum_gain_floor": minimum_gain,
        "nonzero_metric_files": changed_metric_files(runs, expected_files),
        "prediction_hashes": summarize_prediction_hashes(runs, expected_files),
        "protocol": protocol,
        "metrics": {
            key: summarize_metric(runs, expected_files, key, minimum_gain)
            for key in ("residual_ratio", "overerase_ratio")
        },
        "runs": [
            {
                "name": run.spec.name,
                "post_freeze_metrics": str(run.spec.post_freeze_metrics),
                "post_freeze_metrics_sha256": sha256_file(run.spec.post_freeze_metrics),
                "inference_metrics": str(run.spec.inference_metrics),
                "inference_metrics_sha256": sha256_file(run.spec.inference_metrics),
            }
            for run in runs
        ],
        "provenance": {
            "expected_samples_file": str(expected_samples_path),
            "expected_samples_sha256": sha256_file(expected_samples_path),
        },
    }


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.minimum_gain) or args.minimum_gain <= 0.0:
        raise ValueError("--minimum-gain must be finite and positive")
    specs = [parse_run_spec(value) for value in args.run]
    if len(specs) < 3:
        raise ValueError("at least three --run values are required")
    duplicate_names = sorted(
        name for name, count in Counter(spec.name for spec in specs).items() if count > 1
    )
    if duplicate_names:
        raise ValueError(f"duplicate run names: {duplicate_names}")

    expected_samples_path = Path(args.expected_samples_file)
    expected_files = read_expected_files(expected_samples_path)
    runs = [load_run(spec, expected_files) for spec in specs]
    summary = summarize(runs, expected_files, expected_samples_path, args.minimum_gain)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
