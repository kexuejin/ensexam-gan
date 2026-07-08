#!/usr/bin/env python3
"""Evaluate ranker-score sources as incremental selector expansions.

This helper starts from a current base-plus-expansion selector, then tests
additional ranker prediction score cutoffs as additive sources. It reports the
local target-proxy verdict mix, metric safety, and residual/overerase deltas for
single sources and pairwise unions without committing generated review outputs
to the model logic.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from itertools import combinations
from pathlib import Path


METRIC_RE = re.compile(r"(gain|over_delta)=(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.IGNORECASE)
VERDICTS = ("accept", "review", "reject")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-comparison-csv", required=True)
    parser.add_argument(
        "--current-selected-csv",
        action="append",
        required=True,
        help="Selected page CSV already included in the current selector. May be repeated.",
    )
    parser.add_argument(
        "--ranker-predictions",
        action="append",
        required=True,
        metavar="NAME:CSV",
        help="Ranker prediction CSV with split,file,score columns. May be repeated.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        default=[],
        help="Score cutoff to test. May be repeated. Defaults to a conservative grid.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-union-size", type=int, default=2, help="Currently supports 1 or 2.")
    parser.add_argument(
        "--require-zero-reject",
        action="store_true",
        help="Only include rows whose incremental source has no local reject pages.",
    )
    parser.add_argument(
        "--require-zero-metric-losses",
        action="store_true",
        help="Only include rows whose incremental source has no metric losses.",
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


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition(":")
    if separator != ":" or not name or not path:
        raise ValueError(f"Invalid named path {value!r}; expected NAME:CSV")
    return name, Path(path)


def read_current_keys(paths: list[str]) -> set[tuple[str, str]]:
    current: set[tuple[str, str]] = set()
    for path_text in paths:
        for row in read_rows(Path(path_text)):
            current.add(key(row))
    return current


def read_sources(
    specs: list[str],
    thresholds: list[float],
    current_keys: set[tuple[str, str]],
) -> dict[str, set[tuple[str, str]]]:
    sources: dict[str, set[tuple[str, str]]] = {}
    for spec in specs:
        name, path = parse_named_path(spec)
        rows = read_rows(path)
        for threshold in thresholds:
            selected = {
                key(row)
                for row in rows
                if key(row) not in current_keys and fnum(row, "score") is not None and fnum(row, "score") >= threshold
            }
            if selected:
                sources[f"{name}@{threshold:.2f}"] = selected
    return sources


def summarize_source(
    name: str,
    selected_keys: set[tuple[str, str]],
    local_by_key: dict[tuple[str, str], dict[str, str]],
    denominator: int,
) -> dict[str, object]:
    verdict_counts = {verdict: 0 for verdict in VERDICTS}
    metric_losses = 0
    residual_gain = 0.0
    overerase_delta = 0.0
    split_counts: dict[str, int] = {}
    files: list[str] = []
    missing = 0

    for row_key in sorted(selected_keys):
        row = local_by_key.get(row_key)
        if row is None:
            missing += 1
            continue
        verdict = row.get("local_verdict", "")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        if not is_metric_safe(row):
            metric_losses += 1
        gain, over_delta = metric_pair(row)
        residual_gain += gain
        overerase_delta += over_delta
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        files.append(f"{row['split']}/{row['file']}")

    selected = sum(verdict_counts.values())
    return {
        "source": name,
        "selected": selected,
        "accept": verdict_counts.get("accept", 0),
        "review": verdict_counts.get("review", 0),
        "reject": verdict_counts.get("reject", 0),
        "metric_losses": metric_losses,
        "residual_gain": residual_gain / denominator,
        "overerase_delta": overerase_delta / denominator,
        "missing_local_rows": missing,
        "split_counts": " ".join(f"{split}={count}" for split, count in sorted(split_counts.items())),
        "files": " ".join(files),
    }


def sortable_score(row: dict[str, object]) -> tuple[object, ...]:
    return (
        int(row["reject"]),
        int(row["metric_losses"]),
        -int(row["accept"]),
        int(row["review"]),
        -float(row["residual_gain"]),
        float(row["overerase_delta"]),
        str(row["source"]),
    )


def selected_rows_for(
    selected_keys: set[tuple[str, str]],
    local_by_key: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, str]]:
    return [local_by_key[row_key] for row_key in sorted(selected_keys) if row_key in local_by_key]


def main() -> None:
    args = parse_args()
    if args.max_union_size not in (1, 2):
        raise ValueError("--max-union-size currently supports only 1 or 2")

    thresholds = args.threshold or [0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80, 0.78, 0.75]
    local_rows = read_rows(Path(args.local_comparison_csv))
    local_by_key = {key(row): row for row in local_rows}
    current_keys = read_current_keys(args.current_selected_csv)
    sources = read_sources(args.ranker_predictions, thresholds, current_keys)

    summary_rows: list[dict[str, object]] = []
    evaluated: list[tuple[dict[str, object], set[tuple[str, str]]]] = []
    for source_name, selected_keys in sources.items():
        summary = summarize_source(source_name, selected_keys, local_by_key, len(local_rows))
        evaluated.append((summary, selected_keys))

    if args.max_union_size >= 2:
        for (left_name, left_keys), (right_name, right_keys) in combinations(sources.items(), 2):
            selected_keys = left_keys | right_keys
            summary = summarize_source(f"{left_name}+{right_name}", selected_keys, local_by_key, len(local_rows))
            evaluated.append((summary, selected_keys))

    filtered: list[tuple[dict[str, object], set[tuple[str, str]]]] = []
    for summary, selected_keys in evaluated:
        if args.require_zero_reject and summary["reject"]:
            continue
        if args.require_zero_metric_losses and summary["metric_losses"]:
            continue
        filtered.append((summary, selected_keys))

    filtered.sort(key=lambda item: sortable_score(item[0]))
    summary_rows = [summary for summary, _keys in filtered]
    best_keys = filtered[0][1] if filtered else set()

    output_dir = Path(args.output_dir)
    fields = [
        "source",
        "selected",
        "accept",
        "review",
        "reject",
        "metric_losses",
        "residual_gain",
        "overerase_delta",
        "missing_local_rows",
        "split_counts",
        "files",
    ]
    write_csv(output_dir / "ranker_expansion_sources.csv", summary_rows, fields)
    write_csv(output_dir / "best_ranker_expansion_review.csv", selected_rows_for(best_keys, local_by_key))

    if summary_rows:
        best = summary_rows[0]
        print(
            "best "
            f"source={best['source']} "
            f"selected={best['selected']} "
            f"accept/review/reject={best['accept']}/{best['review']}/{best['reject']} "
            f"metric_losses={best['metric_losses']} "
            f"gain={float(best['residual_gain']):.9f} "
            f"over={float(best['overerase_delta']):.9f}"
        )
    else:
        print("best none")
    print(f"sources_csv={output_dir / 'ranker_expansion_sources.csv'}")
    print(f"best_review_csv={output_dir / 'best_ranker_expansion_review.csv'}")


if __name__ == "__main__":
    main()
