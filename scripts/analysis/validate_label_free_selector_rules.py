#!/usr/bin/env python3
"""Validate label-free selector rules with explicit train/test split separation.

This is an offline analysis tool. It uses target-derived metrics and local proxy
verdicts only to score rules, never as selector inputs. The selected feature
conditions are restricted to fields that are available from source/baseline/
candidate images at inference time.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path


EXCLUDED_FEATURES = {
    "file",
    "split",
    "gain",
    "over_delta",
    "win",
    "safe_win",
}

EXCLUDED_SUBSTRINGS = (
    "help",
    "hurt",
    "label",
    "metric",
    "oracle",
    "residual_gain",
    "target",
    "verdict",
)


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    threshold: float

    def hit(self, row: dict[str, str]) -> bool:
        value = float(row[self.feature])
        if self.op == ">=":
            return value >= self.threshold
        if self.op == "<=":
            return value <= self.threshold
        raise ValueError(f"Unsupported op: {self.op}")

    def text(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:.12g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-comparison-csv", required=True)
    parser.add_argument(
        "--feature-csv",
        action="append",
        required=True,
        metavar="SPLIT:CSV",
        help="May be repeated. SPLIT overrides any split column inside the CSV.",
    )
    parser.add_argument("--train-split", action="append", required=True)
    parser.add_argument("--test-split", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-conditions", type=int, default=2)
    parser.add_argument("--max-rules", type=int, default=100)
    parser.add_argument("--min-train-selected", type=int, default=5)
    parser.add_argument("--min-test-selected", type=int, default=1)
    parser.add_argument("--max-train-reject", type=int, default=0)
    parser.add_argument("--max-train-metric-loss", type=int, default=0)
    parser.add_argument(
        "--quantiles",
        default="0,0.05,0.1,0.2,0.25,0.33,0.5,0.67,0.75,0.8,0.9,0.95,1",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_feature_csv(value: str) -> tuple[str, Path]:
    if ":" not in value:
        return "", Path(value)
    split, path = value.split(":", 1)
    if not split or not path:
        raise ValueError(f"Invalid --feature-csv {value!r}; expected SPLIT:CSV")
    return split, Path(path)


def load_feature_rows(feature_specs: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split, path in map(parse_feature_csv, feature_specs):
        for row in read_rows(path):
            row = dict(row)
            if split:
                row["split"] = split
            elif not row.get("split"):
                raise ValueError(f"Feature CSV requires split when no SPLIT prefix is used: {path}")
            rows.append(row)
    return rows


def feature_fieldnames(feature_specs: list[str]) -> set[str]:
    names: set[str] = set()
    for _split, path in map(parse_feature_csv, feature_specs):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Feature CSV has no header: {path}")
            names.update(reader.fieldnames)
    return names


def is_number(value: str) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def label_free_features(rows: list[dict[str, str]], allowed_keys: set[str]) -> list[str]:
    features: list[str] = []
    for key in rows[0]:
        if key not in allowed_keys:
            continue
        if key in EXCLUDED_FEATURES:
            continue
        if any(token in key for token in EXCLUDED_SUBSTRINGS):
            continue
        if all(is_number(row.get(key, "")) for row in rows):
            features.append(key)
    return sorted(features)


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty values")
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def candidate_conditions(
    rows: list[dict[str, str]],
    features: list[str],
    quantiles: list[float],
) -> list[Condition]:
    conditions: list[Condition] = []
    seen: set[tuple[str, str, float]] = set()
    for feature in features:
        values = sorted(float(row[feature]) for row in rows)
        for q in quantiles:
            threshold = quantile(values, q)
            for op in ("<=", ">="):
                key = (feature, op, threshold)
                if key in seen:
                    continue
                seen.add(key)
                conditions.append(Condition(feature, op, threshold))
    return conditions


def merge_rows(
    local_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    features_by_key = {(row["split"], row["file"]): row for row in feature_rows}
    merged: list[dict[str, str]] = []
    missing: list[tuple[str, str]] = []
    for local in local_rows:
        key = (local["split"], local["file"])
        feature = features_by_key.get(key)
        if feature is None:
            missing.append(key)
            continue
        merged.append({**local, **feature, "split": local["split"], "file": local["file"]})
    if missing:
        sample = ", ".join(f"{split}/{file}" for split, file in missing[:5])
        raise KeyError(f"Missing feature rows for {len(missing)} local rows; sample: {sample}")
    return merged


def selected_flags(rows: list[dict[str, str]], conditions: tuple[Condition, ...]) -> list[bool]:
    return [all(condition.hit(row) for condition in conditions) for row in rows]


def summarize(rows: list[dict[str, str]], flags: list[bool], prefix: str) -> dict[str, object]:
    picked = [row for row, flag in zip(rows, flags) if flag]
    metric_losses = [
        row for row in picked
        if not (float(row["gain"]) > 0.0 and float(row["over_delta"]) <= 0.0)
    ]
    rejects = [row for row in picked if row["local_verdict"] == "reject"]
    reviews = [row for row in picked if row["local_verdict"] == "review"]
    accepts = [row for row in picked if row["local_verdict"] == "accept"]
    split_counts: dict[str, int] = {}
    split_rejects: dict[str, int] = {}
    for row in picked:
        split = row["split"]
        split_counts[split] = split_counts.get(split, 0) + 1
        if row["local_verdict"] == "reject":
            split_rejects[split] = split_rejects.get(split, 0) + 1
    return {
        f"{prefix}_selected": len(picked),
        f"{prefix}_coverage": len(picked) / len(rows) if rows else 0.0,
        f"{prefix}_metric_losses": len(metric_losses),
        f"{prefix}_accept": len(accepts),
        f"{prefix}_review": len(reviews),
        f"{prefix}_reject": len(rejects),
        f"{prefix}_residual_gain": sum(float(row["gain"]) for row in picked) / len(rows) if rows else 0.0,
        f"{prefix}_overerase_delta": sum(float(row["over_delta"]) for row in picked) / len(rows) if rows else 0.0,
        **{f"{prefix}_{split}_selected": count for split, count in sorted(split_counts.items())},
        **{f"{prefix}_{split}_reject": count for split, count in sorted(split_rejects.items())},
    }


def evaluate(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    conditions: tuple[Condition, ...],
) -> dict[str, object]:
    train_flags = selected_flags(train_rows, conditions)
    test_flags = selected_flags(test_rows, conditions)
    return {
        **summarize(train_rows, train_flags, "train"),
        **summarize(test_rows, test_flags, "test"),
        "conditions": len(conditions),
        "rule": " AND ".join(condition.text() for condition in conditions),
    }


def main() -> None:
    args = parse_args()
    local_rows = read_rows(Path(args.local_comparison_csv))
    feature_rows = load_feature_rows(args.feature_csv)
    rows = merge_rows(local_rows, feature_rows)
    features = label_free_features(rows, feature_fieldnames(args.feature_csv))
    train_splits = set(args.train_split)
    test_splits = set(args.test_split)
    train_rows = [row for row in rows if row["split"] in train_splits]
    test_rows = [row for row in rows if row["split"] in test_splits]
    if not train_rows or not test_rows:
        raise ValueError("Both train and test rows are required")

    quantiles = [float(value) for value in args.quantiles.split(",") if value.strip()]
    conditions = candidate_conditions(train_rows, features, quantiles)

    results: list[dict[str, object]] = []
    for size in range(1, args.max_conditions + 1):
        for combo in itertools.combinations(conditions, size):
            if len({condition.feature for condition in combo}) != len(combo):
                continue
            row = evaluate(train_rows, test_rows, combo)
            if int(row["train_selected"]) < args.min_train_selected:
                continue
            if int(row["test_selected"]) < args.min_test_selected:
                continue
            if int(row["train_reject"]) > args.max_train_reject:
                continue
            if int(row["train_metric_losses"]) > args.max_train_metric_loss:
                continue
            results.append(row)

    results.sort(
        key=lambda row: (
            -int(row["test_reject"]),
            -int(row["test_metric_losses"]),
            float(row["test_residual_gain"]),
            int(row["test_selected"]),
            float(row["train_residual_gain"]),
        ),
        reverse=True,
    )
    results = results[: args.max_rules]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "conditions",
        "train_selected",
        "train_coverage",
        "train_metric_losses",
        "train_accept",
        "train_review",
        "train_reject",
        "train_residual_gain",
        "train_overerase_delta",
        "test_selected",
        "test_coverage",
        "test_metric_losses",
        "test_accept",
        "test_review",
        "test_reject",
        "test_residual_gain",
        "test_overerase_delta",
        "rule",
    ]
    extra_fields = sorted({key for row in results for key in row if key not in fields})
    fields = fields[:-1] + extra_fields + ["rule"]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(results, start=1):
            writer.writerow({"rank": rank, **row})

    print(
        f"rows={len(rows)} train={len(train_rows)} test={len(test_rows)} "
        f"features={len(features)} conditions={len(conditions)} rules={len(results)} "
        f"output_csv={output_csv}"
    )
    for row in results[:10]:
        print(row)


if __name__ == "__main__":
    main()
