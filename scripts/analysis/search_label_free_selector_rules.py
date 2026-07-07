#!/usr/bin/env python3
"""Search label-free page selector rules against local target-comparison verdicts.

This mines simple AND rules from page features, then scores them with two
separate gates:

* metric safety: selected pages must improve residual and not increase overerase
* local proxy safety: selected pages must avoid target-comparison reject verdicts

The local verdict is target-derived, so this script is an offline analysis aid,
not a production selector generator.
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
    "metric",
    "oracle",
    "residual_gain",
    "target",
    "verdict",
    "label",
)


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    threshold: float

    def rule_text(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:.12g}"

    def hit(self, row: dict[str, str]) -> bool:
        value = float(row[self.feature])
        if self.op == ">=":
            return value >= self.threshold
        if self.op == "<=":
            return value <= self.threshold
        raise ValueError(f"Unsupported operator: {self.op}")


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
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-conditions", type=int, default=2)
    parser.add_argument("--max-rules", type=int, default=50)
    parser.add_argument("--min-selected", type=int, default=1)
    parser.add_argument(
        "--local-mode",
        choices=("no-reject", "accept-only"),
        default="no-reject",
    )
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
    parts = value.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid --feature-csv {value!r}; expected SPLIT:CSV or CSV with split column")
    return parts[0], Path(parts[1])


def load_feature_rows(feature_specs: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split, path in map(parse_feature_csv, feature_specs):
        for row in read_rows(path):
            row = dict(row)
            if split:
                row["split"] = split
            elif not row.get("split"):
                raise ValueError(f"Feature CSV requires split column when no SPLIT prefix is used: {path}")
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
        raise ValueError("empty value list")
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def candidate_conditions(rows: list[dict[str, str]], features: list[str], quantiles: list[float]) -> list[Condition]:
    conditions: list[Condition] = []
    seen: set[tuple[str, str, float]] = set()
    for feature in features:
        values = sorted(float(row[feature]) for row in rows)
        for q in quantiles:
            threshold = quantile(values, q)
            for op in (">=", "<="):
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


def summarize(rows: list[dict[str, str]], selected: list[bool], local_mode: str) -> dict[str, object] | None:
    picked = [row for row, flag in zip(rows, selected) if flag]
    if not picked:
        return None

    metric_losses = [
        row for row in picked
        if not (float(row["gain"]) > 0.0 and float(row["over_delta"]) <= 0.0)
    ]
    if metric_losses:
        return None

    if local_mode == "no-reject":
        local_bad = [row for row in picked if row["local_verdict"] == "reject"]
    else:
        local_bad = [row for row in picked if row["local_verdict"] != "accept"]
    if local_bad:
        return None

    verdict_counts = {"accept": 0, "review": 0, "reject": 0}
    split_counts: dict[str, int] = {}
    for row in picked:
        verdict_counts[row["local_verdict"]] = verdict_counts.get(row["local_verdict"], 0) + 1
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    return {
        "selected": len(picked),
        "coverage": len(picked) / len(rows),
        "residual_gain": sum(float(row["gain"]) for row in picked) / len(rows),
        "overerase_delta": sum(float(row["over_delta"]) for row in picked) / len(rows),
        "accept": verdict_counts.get("accept", 0),
        "review": verdict_counts.get("review", 0),
        "reject": verdict_counts.get("reject", 0),
        **{f"{split}_selected": split_counts.get(split, 0) for split in sorted({row["split"] for row in rows})},
    }


def evaluate_rule(rows: list[dict[str, str]], conditions: tuple[Condition, ...], local_mode: str) -> dict[str, object] | None:
    selected = [all(condition.hit(row) for condition in conditions) for row in rows]
    summary = summarize(rows, selected, local_mode)
    if summary is None:
        return None
    summary["conditions"] = len(conditions)
    summary["rule"] = " AND ".join(condition.rule_text() for condition in conditions)
    return summary


def main() -> None:
    args = parse_args()
    local_rows = read_rows(Path(args.local_comparison_csv))
    feature_rows = load_feature_rows(args.feature_csv)
    rows = merge_rows(local_rows, feature_rows)
    features = label_free_features(rows, feature_fieldnames(args.feature_csv))
    quantiles = [float(value) for value in args.quantiles.split(",") if value.strip()]
    conditions = candidate_conditions(rows, features, quantiles)

    results: list[dict[str, object]] = []
    for size in range(1, args.max_conditions + 1):
        for combo in itertools.combinations(conditions, size):
            if len({condition.feature for condition in combo}) < len(combo):
                continue
            summary = evaluate_rule(rows, combo, args.local_mode)
            if summary is None:
                continue
            if int(summary["selected"]) < args.min_selected:
                continue
            results.append(summary)

    results.sort(
        key=lambda row: (
            int(row["selected"]),
            float(row["residual_gain"]),
            int(row["accept"]),
            -int(row["review"]),
        ),
        reverse=True,
    )
    results = results[: args.max_rules]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "conditions",
        "selected",
        "coverage",
        "residual_gain",
        "overerase_delta",
        "accept",
        "review",
        "reject",
        *[f"{split}_selected" for split in sorted({row["split"] for row in rows})],
        "rule",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(results, start=1):
            writer.writerow({"rank": rank, **row})

    print(
        f"rows={len(rows)} features={len(features)} conditions={len(conditions)} "
        f"rules={len(results)} output_csv={output_csv}"
    )
    for row in results[:10]:
        print(row)


if __name__ == "__main__":
    main()
