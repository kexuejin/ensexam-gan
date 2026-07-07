#!/usr/bin/env python3
"""Compare selector rules against local target-proxy verdicts and metric deltas.

This is an offline validation helper. Rule conditions are evaluated only on
feature CSV columns, while local target-comparison verdicts are used strictly as
evaluation labels.
"""

from __future__ import annotations

import argparse
import csv
import operator
from pathlib import Path


OPS = {
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
    "==": operator.eq,
}


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
    parser.add_argument(
        "--rule",
        action="append",
        required=True,
        metavar="NAME:RULE",
        help='May be repeated, e.g. safe:"active_gray_p25 >= 123".',
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--selected-output-csv", default="")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_feature_csv(value: str) -> tuple[str, Path]:
    split, separator, path = value.partition(":")
    if separator != ":" or not split or not path:
        raise ValueError(f"Invalid --feature-csv {value!r}; expected SPLIT:CSV")
    return split, Path(path)


def read_features(values: list[str]) -> tuple[dict[tuple[str, str], dict[str, str]], set[str]]:
    rows_by_key: dict[tuple[str, str], dict[str, str]] = {}
    feature_names: set[str] = set()
    for split, path in map(parse_feature_csv, values):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Feature CSV has no header: {path}")
            feature_names.update(reader.fieldnames)
            for row in reader:
                row = dict(row)
                row["split"] = split
                rows_by_key[(split, row["file"])] = row
    return rows_by_key, feature_names


def parse_rule(value: str) -> tuple[str, str]:
    name, separator, rule = value.partition(":")
    if separator != ":" or not name or not rule:
        raise ValueError(f"Invalid --rule {value!r}; expected NAME:RULE")
    return name, rule


def selector_hit(row: dict[str, str], rule: str, allowed_features: set[str]) -> bool:
    for condition in (part.strip() for part in rule.split(" AND ")):
        parts = condition.split()
        if len(parts) != 3:
            raise ValueError(f"Unsupported selector condition: {condition!r}")
        feature, op_text, threshold_text = parts
        if feature not in allowed_features:
            raise KeyError(f"Selector feature {feature!r} is not in the feature CSV headers")
        if op_text not in OPS:
            raise ValueError(f"Unsupported selector operator: {op_text!r}")
        if not OPS[op_text](float(row[feature]), float(threshold_text)):
            return False
    return True


def summarize_rule(
    name: str,
    rule: str,
    local_rows: list[dict[str, str]],
    features_by_key: dict[tuple[str, str], dict[str, str]],
    allowed_features: set[str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    verdict_counts = {"accept": 0, "review": 0, "reject": 0}
    split_counts: dict[str, int] = {}
    selected_rows: list[dict[str, object]] = []
    metric_wins = 0
    metric_losses = 0
    residual_gain = 0.0
    overerase_delta = 0.0

    for local in local_rows:
        key = (local["split"], local["file"])
        feature = features_by_key.get(key)
        if feature is None:
            raise KeyError(f"Missing feature row for {local['split']}/{local['file']}")
        if not selector_hit(feature, rule, allowed_features):
            continue

        safe_metric = float(feature["gain"]) > 0.0 and float(feature["over_delta"]) <= 0.0
        metric_wins += int(safe_metric)
        metric_losses += int(not safe_metric)
        residual_gain += float(feature["gain"])
        overerase_delta += float(feature["over_delta"])
        verdict_counts[local["local_verdict"]] = verdict_counts.get(local["local_verdict"], 0) + 1
        split_counts[local["split"]] = split_counts.get(local["split"], 0) + 1
        selected_rows.append({
            "selector": name,
            "rule": rule,
            "split": local["split"],
            "file": local["file"],
            "local_verdict": local["local_verdict"],
            "metric_safe": safe_metric,
            "gain": feature["gain"],
            "over_delta": feature["over_delta"],
        })

    total = len(local_rows)
    selected = len(selected_rows)
    split_names = sorted({row["split"] for row in local_rows})
    summary: dict[str, object] = {
        "selector": name,
        "selected": selected,
        "coverage": selected / total,
        "metric_wins": metric_wins,
        "metric_losses": metric_losses,
        "residual_gain_per_all_pages": residual_gain / total,
        "overerase_delta_per_all_pages": overerase_delta / total,
        "accept": verdict_counts.get("accept", 0),
        "review": verdict_counts.get("review", 0),
        "reject": verdict_counts.get("reject", 0),
        **{f"{split}_selected": split_counts.get(split, 0) for split in split_names},
        "rule": rule,
    }
    return summary, selected_rows


def main() -> None:
    args = parse_args()
    local_rows = read_rows(Path(args.local_comparison_csv))
    features_by_key, allowed_features = read_features(args.feature_csv)

    summary_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for rule_arg in args.rule:
        name, rule = parse_rule(rule_arg)
        summary, selected = summarize_rule(name, rule, local_rows, features_by_key, allowed_features)
        summary_rows.append(summary)
        selected_rows.extend(selected)

    write_csv(Path(args.output_csv), summary_rows)
    if args.selected_output_csv:
        write_csv(Path(args.selected_output_csv), selected_rows)

    for row in summary_rows:
        print(row)
    print(f"output_csv={args.output_csv}")
    if args.selected_output_csv:
        print(f"selected_output_csv={args.selected_output_csv}")


if __name__ == "__main__":
    main()
