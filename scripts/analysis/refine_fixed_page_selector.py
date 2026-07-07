#!/usr/bin/env python3
"""Refine a broad fixed page selector with simple feature conditions."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path


EXCLUDE_FEATURES = {"split", "file", "gain", "over_delta", "win", "safe_win"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME:FEATURES_CSV:BASELINE_METRICS:CANDIDATE_METRICS",
    )
    parser.add_argument("--base-rule", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-losses", type=int, default=3)
    parser.add_argument("--min-selected", type=int, default=8)
    parser.add_argument("--top-features", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=200)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_split(value: str) -> tuple[str, Path, Path, Path]:
    parts = value.split(":", 3)
    if len(parts) != 4 or not all(parts):
        raise ValueError(f"Invalid --split {value!r}; expected NAME:FEATURES:BASELINE:CANDIDATE")
    return parts[0], Path(parts[1]), Path(parts[2]), Path(parts[3])


def base_rule_hit(row: dict[str, str], rule: str) -> bool:
    if rule == "active_gray_p25 >= 123":
        return float(row["active_gray_p25"]) >= 123.0
    raise ValueError(f"Unsupported base rule: {rule}")


def threshold_candidates(values: list[float]) -> list[float]:
    values = sorted(values)
    return sorted({values[round((len(values) - 1) * q / 100)] for q in range(0, 101, 5)})


def numeric_features(row: dict[str, str]) -> list[str]:
    out = []
    for key, value in row.items():
        if key in EXCLUDE_FEATURES:
            continue
        try:
            float(value)
        except ValueError:
            continue
        out.append(key)
    return out


def feature_separation(rows: list[dict[str, str]], feature: str) -> float:
    wins = [float(row[feature]) for row in rows if row["safe_win"] == "True"]
    losses = [float(row[feature]) for row in rows if row["safe_win"] != "True"]
    values = [float(row[feature]) for row in rows]
    if not wins or not losses:
        return 0.0
    spread = max(values) - min(values) or 1.0
    return abs(sum(wins) / len(wins) - sum(losses) / len(losses)) / spread


def selected_by_conditions(rows: list[dict[str, str]], conditions: list[tuple[str, str, float]]) -> list[bool]:
    flags = []
    for row in rows:
        selected = True
        for feature, op, threshold in conditions:
            value = float(row[feature])
            if op == "<=" and value > threshold:
                selected = False
                break
            if op == ">=" and value < threshold:
                selected = False
                break
        flags.append(selected)
    return flags


def evaluate_selection(
    rows: list[dict[str, str]],
    flags: list[bool],
    baseline_by_split: dict[str, dict[str, dict[str, str]]],
    candidate_by_split: dict[str, dict[str, dict[str, str]]],
) -> dict[str, object]:
    selected = [row for row, flag in zip(rows, flags) if flag]
    wins = sum(row["safe_win"] == "True" for row in selected)
    losses = len(selected) - wins
    total_pages = 0
    baseline_residual = 0.0
    selector_residual = 0.0
    baseline_overerase = 0.0
    selector_overerase = 0.0
    for split, baseline in baseline_by_split.items():
        candidate = candidate_by_split[split]
        selected_files = {row["file"] for row in selected if row["split"] == split}
        files = sorted(baseline)
        total_pages += len(files)
        baseline_residual += sum(float(baseline[file]["residual_ratio"]) for file in files)
        baseline_overerase += sum(float(baseline[file]["overerase_ratio"]) for file in files)
        selector_residual += sum(
            float(candidate[file]["residual_ratio"]) if file in selected_files else float(baseline[file]["residual_ratio"])
            for file in files
        )
        selector_overerase += sum(
            float(candidate[file]["overerase_ratio"]) if file in selected_files else float(baseline[file]["overerase_ratio"])
            for file in files
        )
    return {
        "selected": len(selected),
        "wins": wins,
        "losses": losses,
        "coverage": len(selected) / total_pages,
        "residual_gain": (baseline_residual - selector_residual) / total_pages,
        "overerase_delta": (selector_overerase - baseline_overerase) / total_pages,
        "files": " ".join(f"{row['split']}:{row['file']}" for row in selected),
    }


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    baseline_by_split: dict[str, dict[str, dict[str, str]]] = {}
    candidate_by_split: dict[str, dict[str, dict[str, str]]] = {}
    for split, features_csv, baseline_metrics, candidate_metrics in [parse_split(value) for value in args.split]:
        baseline_by_split[split] = {row["file"]: row for row in read_rows(baseline_metrics)}
        candidate_by_split[split] = {row["file"]: row for row in read_rows(candidate_metrics)}
        for row in read_rows(features_csv):
            if not base_rule_hit(row, args.base_rule):
                continue
            out = dict(row)
            out["split"] = split
            out["safe_win"] = str(float(row["gain"]) > 0 and float(row["over_delta"]) <= 0)
            rows.append(out)

    features = numeric_features(rows[0])
    ranked_features = [
        feature for _score, feature in sorted(
            ((feature_separation(rows, feature), feature) for feature in features),
            reverse=True,
        )[: args.top_features]
    ]
    condition_sets: list[list[tuple[str, str, float]]] = []
    for feature in ranked_features:
        for threshold in threshold_candidates([float(row[feature]) for row in rows]):
            for op in ("<=", ">="):
                condition_sets.append([(feature, op, threshold)])
    for left, right in itertools.combinations(ranked_features, 2):
        left_thresholds = threshold_candidates([float(row[left]) for row in rows])
        right_thresholds = threshold_candidates([float(row[right]) for row in rows])
        for left_op, left_threshold, right_op, right_threshold in itertools.product(
            ("<=", ">="),
            left_thresholds,
            ("<=", ">="),
            right_thresholds,
        ):
            condition_sets.append([(left, left_op, left_threshold), (right, right_op, right_threshold)])

    result_rows: list[dict[str, object]] = []
    seen = set()
    for conditions in condition_sets:
        rule = " AND ".join(f"{feature} {op} {threshold:.12g}" for feature, op, threshold in conditions)
        if rule in seen:
            continue
        seen.add(rule)
        metrics = evaluate_selection(
            rows,
            selected_by_conditions(rows, conditions),
            baseline_by_split,
            candidate_by_split,
        )
        if (
            int(metrics["selected"]) >= args.min_selected
            and int(metrics["losses"]) <= args.max_losses
            and float(metrics["residual_gain"]) > 0
            and float(metrics["overerase_delta"]) <= 0
        ):
            result_rows.append({"rule": rule, **metrics})

    result_rows.sort(
        key=lambda row: (
            int(row["losses"]),
            -int(row["selected"]),
            -float(row["residual_gain"]),
        )
    )
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["rule", "selected", "wins", "losses", "coverage", "residual_gain", "overerase_delta", "files"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows[: args.top_n])

    base_metrics = evaluate_selection(rows, [True] * len(rows), baseline_by_split, candidate_by_split)
    print(f"base_rule_selected={base_metrics['selected']} wins={base_metrics['wins']} losses={base_metrics['losses']}")
    print(f"candidate_rules={len(result_rows)} output_csv={output}")
    if result_rows:
        print(result_rows[0])


if __name__ == "__main__":
    main()
