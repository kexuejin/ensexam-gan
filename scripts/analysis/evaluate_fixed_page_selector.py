#!/usr/bin/env python3
"""Evaluate a fixed page-level selector rule against baseline/candidate metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME:FEATURES_CSV:BASELINE_METRICS:CANDIDATE_METRICS",
        help="May be repeated.",
    )
    parser.add_argument("--selector-rule", required=True)
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_split(value: str) -> tuple[str, Path, Path, Path]:
    parts = value.split(":", 3)
    if len(parts) != 4 or not all(parts):
        raise ValueError(f"Invalid --split {value!r}; expected NAME:FEATURES:BASELINE:CANDIDATE")
    return parts[0], Path(parts[1]), Path(parts[2]), Path(parts[3])


def selector_hit(row: dict[str, str], rule: str) -> bool:
    if rule == "active_gray_p25 >= 123":
        return float(row["active_gray_p25"]) >= 123.0
    if rule == "active_gray_p25 >= 111.6 AND candidate_delta_max <= 200.133333333":
        return float(row["active_gray_p25"]) >= 111.6 and float(row["candidate_delta_max"]) <= 200.133333333
    raise ValueError(f"Unsupported selector rule: {rule}")


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def summarize_split(
    split_name: str,
    features_csv: Path,
    baseline_metrics: Path,
    candidate_metrics: Path,
    selector_rule: str,
) -> tuple[dict[str, object], list[str]]:
    features_by_file = {row["file"]: row for row in read_rows(features_csv)}
    baseline_by_file = {row["file"]: row for row in read_rows(baseline_metrics)}
    candidate_by_file = {row["file"]: row for row in read_rows(candidate_metrics)}

    files = sorted(baseline_by_file)
    selected = [file for file in files if selector_hit(features_by_file[file], selector_rule)]
    wins = 0
    losses = 0
    for file in selected:
        feature = features_by_file[file]
        is_safe_win = float(feature["gain"]) > 0 and float(feature["over_delta"]) <= 0
        if is_safe_win:
            wins += 1
        else:
            losses += 1

    selector_residual = sum(
        float(candidate_by_file[file]["residual_ratio"]) if file in selected else float(baseline_by_file[file]["residual_ratio"])
        for file in files
    ) / len(files)
    selector_overerase = sum(
        float(candidate_by_file[file]["overerase_ratio"]) if file in selected else float(baseline_by_file[file]["overerase_ratio"])
        for file in files
    ) / len(files)

    summary: dict[str, object] = {
        "split": split_name,
        "pages": len(files),
        "selected": len(selected),
        "coverage": len(selected) / len(files),
        "baseline_residual": mean(list(baseline_by_file.values()), "residual_ratio"),
        "candidate_residual": mean(list(candidate_by_file.values()), "residual_ratio"),
        "selector_residual": selector_residual,
        "residual_gain": mean(list(baseline_by_file.values()), "residual_ratio") - selector_residual,
        "baseline_overerase": mean(list(baseline_by_file.values()), "overerase_ratio"),
        "candidate_overerase": mean(list(candidate_by_file.values()), "overerase_ratio"),
        "selector_overerase": selector_overerase,
        "overerase_delta": selector_overerase - mean(list(baseline_by_file.values()), "overerase_ratio"),
        "wins_selected": wins,
        "losses_selected": losses,
        "selected_files": " ".join(selected),
    }
    return summary, selected


def main() -> None:
    args = parse_args()
    split_specs = [parse_split(value) for value in args.split]
    summaries = []
    total_pages = 0
    total_selected = 0
    weighted_baseline_residual = 0.0
    weighted_selector_residual = 0.0
    weighted_baseline_overerase = 0.0
    weighted_selector_overerase = 0.0
    total_wins = 0
    total_losses = 0

    for split_name, features_csv, baseline_metrics, candidate_metrics in split_specs:
        summary, _selected = summarize_split(
            split_name,
            features_csv,
            baseline_metrics,
            candidate_metrics,
            args.selector_rule,
        )
        summaries.append(summary)
        pages = int(summary["pages"])
        total_pages += pages
        total_selected += int(summary["selected"])
        weighted_baseline_residual += float(summary["baseline_residual"]) * pages
        weighted_selector_residual += float(summary["selector_residual"]) * pages
        weighted_baseline_overerase += float(summary["baseline_overerase"]) * pages
        weighted_selector_overerase += float(summary["selector_overerase"]) * pages
        total_wins += int(summary["wins_selected"])
        total_losses += int(summary["losses_selected"])

    combined = {
        "split": "combined",
        "pages": total_pages,
        "selected": total_selected,
        "coverage": total_selected / total_pages,
        "baseline_residual": weighted_baseline_residual / total_pages,
        "selector_residual": weighted_selector_residual / total_pages,
        "residual_gain": (weighted_baseline_residual - weighted_selector_residual) / total_pages,
        "baseline_overerase": weighted_baseline_overerase / total_pages,
        "selector_overerase": weighted_selector_overerase / total_pages,
        "overerase_delta": (weighted_selector_overerase - weighted_baseline_overerase) / total_pages,
        "wins_selected": total_wins,
        "losses_selected": total_losses,
        "selected_files": "",
    }
    summaries.append(combined)

    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
