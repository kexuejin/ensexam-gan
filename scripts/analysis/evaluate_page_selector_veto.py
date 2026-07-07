#!/usr/bin/env python3
"""Evaluate label-free veto rules on top of a page selector.

The base selector is usually a ranker score threshold. This script searches
only inference-time feature conditions on train-selected pages, then reports how
those veto/keep rules transfer to held-out selected pages. Target-derived local
proxy verdicts and metrics are used only for offline scoring.
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
        if self.op == "<=":
            return value <= self.threshold
        if self.op == ">=":
            return value >= self.threshold
        raise ValueError(f"Unsupported op: {self.op}")

    def text(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:.12g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--local-comparison-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--train-split", action="append", required=True)
    parser.add_argument("--test-split", action="append", required=True)
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def is_number(value: str) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def label_free_features(rows: list[dict[str, str]], feature_fieldnames: set[str]) -> list[str]:
    features: list[str] = []
    for key in rows[0]:
        if key not in feature_fieldnames:
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
    predictions: list[dict[str, str]],
    features: list[dict[str, str]],
    comparisons: list[dict[str, str]],
) -> list[dict[str, str]]:
    features_by_key = {(row["split"], row["file"]): row for row in features}
    comparisons_by_key = {(row["split"], row["file"]): row for row in comparisons}
    merged: list[dict[str, str]] = []
    missing: list[tuple[str, str]] = []
    for prediction in predictions:
        key = (prediction["split"], prediction["file"])
        feature = features_by_key.get(key)
        comparison = comparisons_by_key.get(key)
        if feature is None or comparison is None:
            missing.append(key)
            continue
        merged.append({
            **comparison,
            **feature,
            "score": prediction["score"],
            "split": prediction["split"],
            "file": prediction["file"],
        })
    if missing:
        sample = ", ".join(f"{split}/{file}" for split, file in missing[:5])
        raise KeyError(f"Missing rows for {len(missing)} predictions; sample: {sample}")
    return merged


def selected_flags(rows: list[dict[str, str]], conditions: tuple[Condition, ...]) -> list[bool]:
    return [all(condition.hit(row) for condition in conditions) for row in rows]


def summarize(rows: list[dict[str, str]], flags: list[bool], prefix: str) -> dict[str, object]:
    picked = [row for row, flag in zip(rows, flags) if flag]
    metric_losses = [
        row for row in picked
        if not (float(row["gain"]) > 0.0 and float(row["over_delta"]) <= 0.0)
    ]
    verdicts = {"accept": 0, "review": 0, "reject": 0}
    split_counts: dict[str, int] = {}
    split_rejects: dict[str, int] = {}
    for row in picked:
        verdicts[row["local_verdict"]] = verdicts.get(row["local_verdict"], 0) + 1
        split = row["split"]
        split_counts[split] = split_counts.get(split, 0) + 1
        if row["local_verdict"] == "reject":
            split_rejects[split] = split_rejects.get(split, 0) + 1
    return {
        f"{prefix}_selected": len(picked),
        f"{prefix}_coverage": len(picked) / len(rows) if rows else 0.0,
        f"{prefix}_metric_losses": len(metric_losses),
        f"{prefix}_accept": verdicts.get("accept", 0),
        f"{prefix}_review": verdicts.get("review", 0),
        f"{prefix}_reject": verdicts.get("reject", 0),
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
        "keep_rule": " AND ".join(condition.text() for condition in conditions),
    }


def main() -> None:
    args = parse_args()
    predictions = read_rows(Path(args.predictions_csv))
    features = read_rows(Path(args.feature_csv))
    comparisons = read_rows(Path(args.local_comparison_csv))
    rows = merge_rows(predictions, features, comparisons)
    rows = [row for row in rows if float(row["score"]) >= args.score_threshold]
    train_splits = set(args.train_split)
    test_splits = set(args.test_split)
    train_rows = [row for row in rows if row["split"] in train_splits]
    test_rows = [row for row in rows if row["split"] in test_splits]
    if not train_rows or not test_rows:
        raise ValueError("Both train and test selected rows are required")

    feature_fieldnames = set(features[0])
    feature_names = label_free_features(rows, feature_fieldnames)
    quantiles = [float(value) for value in args.quantiles.split(",") if value.strip()]
    candidates = candidate_conditions(train_rows, feature_names, quantiles)

    results: list[dict[str, object]] = []
    for condition_count in range(1, args.max_conditions + 1):
        for conditions in itertools.combinations(candidates, condition_count):
            row = evaluate(train_rows, test_rows, conditions)
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
            int(row["test_reject"]),
            int(row["test_metric_losses"]),
            -int(row["test_selected"]),
            -float(row["test_residual_gain"]),
            int(row["conditions"]),
            str(row["keep_rule"]),
        )
    )
    write_csv(Path(args.output_csv), results[: args.max_rules])
    print(
        f"base_selected train={len(train_rows)} test={len(test_rows)} "
        f"features={len(feature_names)} candidates={len(candidates)} results={len(results)}"
    )
    if results:
        top = results[0]
        print(
            "best "
            f"test_selected={top['test_selected']} "
            f"test_accept={top['test_accept']} "
            f"test_review={top['test_review']} "
            f"test_reject={top['test_reject']} "
            f"test_metric_losses={top['test_metric_losses']} "
            f"rule={top['keep_rule']}"
        )


if __name__ == "__main__":
    main()
