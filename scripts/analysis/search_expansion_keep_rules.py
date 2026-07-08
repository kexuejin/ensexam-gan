#!/usr/bin/env python3
"""Search safe expansion rules for ranker-selected page candidates.

The expansion pool is expected to contain pages that are not selected by a
known-safe base selector. This helper searches one-condition threshold rules on
numeric columns from that pool, then reports expansion-only and combined
base-plus-expansion quality using a single local target-proxy CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


OPS = {
    ">=": lambda value, threshold: value >= threshold,
    "<=": lambda value, threshold: value <= threshold,
}

METRIC_RE = re.compile(r"(gain|over_delta)=(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.IGNORECASE)
EXCLUDED_NUMERIC_COLUMNS = {
    "safe26_selected",
    "source_area",
    "crop_index",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion-csv", required=True, help="Ranker-selected non-base candidate CSV.")
    parser.add_argument("--base-selected-csv", required=True, help="Known-safe selector selected pages CSV.")
    parser.add_argument("--local-comparison-csv", required=True, help="Full local target-proxy comparison CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--min-expansion-selected",
        type=int,
        default=1,
        help="Discard threshold rules that select fewer expansion pages.",
    )
    parser.add_argument(
        "--require-expansion-zero-reject",
        action="store_true",
        help="Only keep rules with no expansion-local reject rows.",
    )
    parser.add_argument(
        "--require-zero-metric-losses",
        action="store_true",
        help="Only keep rules whose combined selected pages all improve residual without overerase regression.",
    )
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


def key(row: dict[str, str]) -> tuple[str, str]:
    return row["split"], row["file"]


def fnum(row: dict[str, str], column: str) -> float | None:
    value = row.get(column, "")
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def local_verdict(row: dict[str, str]) -> str:
    verdict = row.get("local_verdict", "")
    if verdict:
        return verdict
    bucket = row.get("bucket", "")
    if "reject" in bucket:
        return "reject"
    if "review" in bucket:
        return "review"
    return "accept"


def metric_pair(row: dict[str, str]) -> tuple[float, float]:
    gain = fnum(row, "gain")
    over_delta = fnum(row, "over_delta")
    if gain is not None and over_delta is not None:
        return gain, over_delta

    found = {match.group(1).lower(): float(match.group(2)) for match in METRIC_RE.finditer(row.get("notes", ""))}
    if "gain" in found and "over_delta" in found:
        return found["gain"], found["over_delta"]

    raise KeyError(f"Cannot find gain/over_delta for {row.get('split')}/{row.get('file')}")


def is_metric_safe(row: dict[str, str]) -> bool:
    gain, over_delta = metric_pair(row)
    return gain > 0.0 and over_delta <= 0.0


def numeric_columns(rows: list[dict[str, str]]) -> list[str]:
    columns: list[str] = []
    if not rows:
        return columns
    for column in rows[0]:
        if column in EXCLUDED_NUMERIC_COLUMNS:
            continue
        values = [fnum(row, column) for row in rows]
        present = [value for value in values if value is not None]
        if len(present) >= 2 and len(set(present)) >= 2:
            columns.append(column)
    return columns


def candidate_rules(rows: list[dict[str, str]]) -> list[tuple[str, str, float]]:
    rules: list[tuple[str, str, float]] = []
    for column in numeric_columns(rows):
        values = sorted({fnum(row, column) for row in rows})
        thresholds = [value for value in values if value is not None]
        for threshold in thresholds:
            rules.append((column, ">=", threshold))
            rules.append((column, "<=", threshold))
    return rules


def selected_by_rule(rows: list[dict[str, str]], column: str, op: str, threshold: float) -> list[dict[str, str]]:
    compare = OPS[op]
    selected = []
    for row in rows:
        value = fnum(row, column)
        if value is not None and compare(value, threshold):
            selected.append(row)
    return selected


def summarize(
    selected_expansion: list[dict[str, str]],
    base_rows: list[dict[str, str]],
    local_by_key: dict[tuple[str, str], dict[str, str]],
    denominator: int,
) -> dict[str, object]:
    base_keys = {key(row) for row in base_rows}
    combined_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in base_rows:
        combined_by_key[key(row)] = row
    for row in selected_expansion:
        combined_by_key[key(row)] = row

    base_verdicts = {"accept": 0, "review": 0, "reject": 0}
    for row in base_rows:
        local = local_by_key.get(key(row), row)
        base_verdicts[local_verdict(local)] = base_verdicts.get(local_verdict(local), 0) + 1

    expansion_verdicts = {"accept": 0, "review": 0, "reject": 0}
    for row in selected_expansion:
        expansion_verdicts[local_verdict(row)] = expansion_verdicts.get(local_verdict(row), 0) + 1

    combined_verdicts = {"accept": 0, "review": 0, "reject": 0}
    added_verdicts = {"accept": 0, "review": 0, "reject": 0}
    metric_losses = 0
    residual_gain = 0.0
    overerase_delta = 0.0
    for row_key, row in combined_by_key.items():
        local = local_by_key.get(row_key, row)
        verdict = local_verdict(local)
        combined_verdicts[verdict] = combined_verdicts.get(verdict, 0) + 1
        if row_key not in base_keys:
            added_verdicts[verdict] = added_verdicts.get(verdict, 0) + 1
        if not is_metric_safe(row):
            metric_losses += 1
        gain, over_delta = metric_pair(row)
        residual_gain += gain
        overerase_delta += over_delta

    return {
        "base_selected": len(base_rows),
        "base_accept": base_verdicts.get("accept", 0),
        "base_review": base_verdicts.get("review", 0),
        "base_reject": base_verdicts.get("reject", 0),
        "expansion_selected": len(selected_expansion),
        "expansion_accept": expansion_verdicts.get("accept", 0),
        "expansion_review": expansion_verdicts.get("review", 0),
        "expansion_reject": expansion_verdicts.get("reject", 0),
        "added_accept": added_verdicts.get("accept", 0),
        "added_review": added_verdicts.get("review", 0),
        "added_reject": added_verdicts.get("reject", 0),
        "combined_selected": len(combined_by_key),
        "combined_accept": combined_verdicts.get("accept", 0),
        "combined_review": combined_verdicts.get("review", 0),
        "combined_reject": combined_verdicts.get("reject", 0),
        "combined_metric_losses": metric_losses,
        "combined_residual_gain": residual_gain / denominator,
        "combined_overerase_delta": overerase_delta / denominator,
    }


def sortable_score(row: dict[str, object]) -> tuple[object, ...]:
    return (
        int(row["combined_reject"]),
        int(row["combined_metric_losses"]),
        int(row["expansion_reject"]),
        -int(row["combined_accept"]),
        int(row["combined_review"]),
        -float(row["combined_residual_gain"]),
        float(row["combined_overerase_delta"]),
        str(row["rule"]),
    )


def main() -> None:
    args = parse_args()
    expansion_rows = read_rows(Path(args.expansion_csv))
    base_rows = read_rows(Path(args.base_selected_csv))
    local_rows = read_rows(Path(args.local_comparison_csv))
    local_by_key = {key(row): row for row in local_rows}
    denominator = len(local_rows)

    summary_rows: list[dict[str, object]] = []
    best_selected: list[dict[str, str]] = []
    for column, op, threshold in candidate_rules(expansion_rows):
        selected = selected_by_rule(expansion_rows, column, op, threshold)
        if len(selected) < args.min_expansion_selected:
            continue
        summary = summarize(selected, base_rows, local_by_key, denominator)
        if args.require_expansion_zero_reject and summary["expansion_reject"]:
            continue
        if args.require_zero_metric_losses and summary["combined_metric_losses"]:
            continue
        rule_text = f"{column} {op} {threshold:.13g}"
        summary["rule"] = rule_text
        summary["files"] = " ".join(f"{row['split']}/{row['file']}" for row in selected)
        summary_rows.append(summary)

    summary_rows.sort(key=sortable_score)
    if summary_rows:
        best_rule = str(summary_rows[0]["rule"])
        column, op, threshold_text = best_rule.split()
        best_selected = selected_by_rule(expansion_rows, column, op, float(threshold_text))

    output_dir = Path(args.output_dir)
    fields = [
        "base_selected",
        "base_accept",
        "base_review",
        "base_reject",
        "expansion_selected",
        "expansion_accept",
        "expansion_review",
        "expansion_reject",
        "added_accept",
        "added_review",
        "added_reject",
        "combined_selected",
        "combined_accept",
        "combined_review",
        "combined_reject",
        "combined_metric_losses",
        "combined_residual_gain",
        "combined_overerase_delta",
        "rule",
        "files",
    ]
    write_csv(output_dir / "expansion_keep_rules.csv", summary_rows, fields)
    write_csv(output_dir / "best_expansion_keep_review.csv", best_selected)

    if summary_rows:
        best = summary_rows[0]
        print(
            "best "
            f"rule={best['rule']} "
            f"base_accept/review/reject={best['base_accept']}/{best['base_review']}/{best['base_reject']} "
            f"expansion={best['expansion_selected']} "
            f"accept/review/reject={best['expansion_accept']}/{best['expansion_review']}/{best['expansion_reject']} "
            f"added_accept/review/reject={best['added_accept']}/{best['added_review']}/{best['added_reject']} "
            f"combined={best['combined_selected']} "
            f"combined_accept/review/reject={best['combined_accept']}/{best['combined_review']}/{best['combined_reject']} "
            f"metric_losses={best['combined_metric_losses']} "
            f"gain={float(best['combined_residual_gain']):.9f} "
            f"over={float(best['combined_overerase_delta']):.9f}"
        )
    else:
        print("best none")
    print(f"rules_csv={output_dir / 'expansion_keep_rules.csv'}")
    print(f"best_review_csv={output_dir / 'best_expansion_keep_review.csv'}")


if __name__ == "__main__":
    main()
