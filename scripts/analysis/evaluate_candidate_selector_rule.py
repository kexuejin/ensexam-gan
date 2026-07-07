#!/usr/bin/env python3
"""Evaluate a candidate selector rule on full metric splits.

The rule is evaluated only against label-free page feature CSVs. Baseline and
candidate metric CSVs are used to report residual/overerase impact and
win/loss counts for the selected pages.
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
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME:FEATURES_CSV:BASELINE_METRICS:CANDIDATE_METRICS",
        help="May be repeated for each split.",
    )
    parser.add_argument("--rule", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--selected-output-csv", default="")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_split(value: str) -> tuple[str, Path, Path, Path]:
    parts = value.split(":", 3)
    if len(parts) != 4 or not all(parts):
        raise ValueError(f"Invalid --split {value!r}; expected NAME:FEATURES:BASELINE:CANDIDATE")
    return parts[0], Path(parts[1]), Path(parts[2]), Path(parts[3])


def selector_hit(row: dict[str, str], rule: str) -> bool:
    for condition in (part.strip() for part in rule.split(" AND ")):
        parts = condition.split()
        if len(parts) != 3:
            raise ValueError(f"Unsupported selector condition: {condition!r}")
        feature, op_text, threshold_text = parts
        if feature not in row:
            raise KeyError(f"Selector feature {feature!r} not found")
        if op_text not in OPS:
            raise ValueError(f"Unsupported selector operator: {op_text!r}")
        if not OPS[op_text](float(row[feature]), float(threshold_text)):
            return False
    return True


def summarize_split(
    split: str,
    feature_rows: list[dict[str, str]],
    baseline_by_file: dict[str, dict[str, str]],
    candidate_by_file: dict[str, dict[str, str]],
    rule: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected_rows: list[dict[str, object]] = []
    residual_gain = 0.0
    overerase_delta = 0.0
    wins = 0
    losses = 0
    for feature in feature_rows:
        if not selector_hit(feature, rule):
            continue
        file = feature["file"]
        baseline = baseline_by_file[file]
        candidate = candidate_by_file[file]
        gain = float(baseline["residual_ratio"]) - float(candidate["residual_ratio"])
        over = float(candidate["overerase_ratio"]) - float(baseline["overerase_ratio"])
        safe_win = gain > 0.0 and over <= 0.0
        residual_gain += gain
        overerase_delta += over
        wins += int(safe_win)
        losses += int(not safe_win)
        selected_rows.append(
            {
                "split": split,
                "file": file,
                "metric_safe": safe_win,
                "gain": gain,
                "over_delta": over,
                "rule": rule,
            }
        )

    pages = len(baseline_by_file)
    summary = {
        "split": split,
        "pages": pages,
        "selected": len(selected_rows),
        "coverage": len(selected_rows) / pages,
        "wins": wins,
        "losses": losses,
        "residual_gain_per_split": residual_gain / pages,
        "overerase_delta_per_split": overerase_delta / pages,
        "files": " ".join(str(row["file"]) for row in selected_rows),
        "rule": rule,
    }
    return summary, selected_rows


def main() -> None:
    args = parse_args()
    summary_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    totals = {
        "pages": 0,
        "selected": 0,
        "wins": 0,
        "losses": 0,
        "residual_gain": 0.0,
        "overerase_delta": 0.0,
    }

    for split, features_csv, baseline_metrics, candidate_metrics in map(parse_split, args.split):
        baseline_by_file = {row["file"]: row for row in read_rows(baseline_metrics)}
        candidate_by_file = {row["file"]: row for row in read_rows(candidate_metrics)}
        summary, selected = summarize_split(
            split=split,
            feature_rows=read_rows(features_csv),
            baseline_by_file=baseline_by_file,
            candidate_by_file=candidate_by_file,
            rule=args.rule,
        )
        summary_rows.append(summary)
        selected_rows.extend(selected)
        totals["pages"] += int(summary["pages"])
        totals["selected"] += int(summary["selected"])
        totals["wins"] += int(summary["wins"])
        totals["losses"] += int(summary["losses"])
        totals["residual_gain"] += float(summary["residual_gain_per_split"]) * int(summary["pages"])
        totals["overerase_delta"] += float(summary["overerase_delta_per_split"]) * int(summary["pages"])

    summary_rows.append(
        {
            "split": "all",
            "pages": totals["pages"],
            "selected": totals["selected"],
            "coverage": totals["selected"] / totals["pages"],
            "wins": totals["wins"],
            "losses": totals["losses"],
            "residual_gain_per_split": totals["residual_gain"] / totals["pages"],
            "overerase_delta_per_split": totals["overerase_delta"] / totals["pages"],
            "files": "",
            "rule": args.rule,
        }
    )

    summary_fields = [
        "split",
        "pages",
        "selected",
        "coverage",
        "wins",
        "losses",
        "residual_gain_per_split",
        "overerase_delta_per_split",
        "files",
        "rule",
    ]
    write_csv(Path(args.output_csv), summary_rows, summary_fields)
    if args.selected_output_csv:
        write_csv(Path(args.selected_output_csv), selected_rows)

    for row in summary_rows:
        print(
            f"{row['split']} pages={row['pages']} selected={row['selected']} "
            f"coverage={float(row['coverage']):.3f} wins/losses={row['wins']}/{row['losses']} "
            f"gain={float(row['residual_gain_per_split']):.9f} "
            f"over={float(row['overerase_delta_per_split']):.9f}"
        )
    print(f"output_csv={args.output_csv}")
    if args.selected_output_csv:
        print(f"selected_output_csv={args.selected_output_csv}")


if __name__ == "__main__":
    main()
