#!/usr/bin/env python3
"""Summarize selector coverage with target-aware quality labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


QUALITY_LABELS = ("clear_win", "slight_win", "noop", "borderline", "slight_loss", "clear_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-csv", required=True, help="Output from score_target_comparison_quality.py.")
    parser.add_argument(
        "--selector",
        action="append",
        required=True,
        metavar="NAME:CSV",
        help="Selector selected-pages CSV. May be repeated.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--denominator", type=int, default=0, help="Coverage denominator. Defaults to quality CSV rows.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, str]:
    return row["split"], row["file"]


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition(":")
    if separator != ":" or not name or not path:
        raise ValueError(f"Invalid selector spec {value!r}; expected NAME:CSV")
    return name, Path(path)


def summarize_selector(
    name: str,
    path: Path,
    quality_by_key: dict[tuple[str, str], dict[str, str]],
    denominator: int,
) -> dict[str, object]:
    selected_rows = read_rows(path)
    quality_counts: Counter[str] = Counter()
    local_counts: Counter[str] = Counter()
    missing: list[str] = []

    for row in selected_rows:
        row_key = key(row)
        quality = quality_by_key.get(row_key)
        if quality is None:
            missing.append(f"{row_key[0]}/{row_key[1]}")
            continue
        quality_counts[quality["target_quality_label"]] += 1
        local_counts[quality.get("local_verdict", "")] += 1

    target_wins = quality_counts["clear_win"] + quality_counts["slight_win"]
    target_losses = quality_counts["clear_loss"] + quality_counts["slight_loss"]
    selected = len(selected_rows)
    row: dict[str, object] = {
        "selector": name,
        "selected": selected,
        "selected_coverage_pct": selected / denominator * 100.0,
        "target_confirmed_wins": target_wins,
        "target_confirmed_win_pct": target_wins / denominator * 100.0,
        "target_borderline": quality_counts["borderline"],
        "target_losses": target_losses,
        "target_loss_pct": target_losses / denominator * 100.0,
        "local_accept": local_counts["accept"],
        "local_review": local_counts["review"],
        "local_reject": local_counts["reject"],
        "missing_quality_rows": " ".join(missing),
    }
    for label in QUALITY_LABELS:
        row[f"target_{label}"] = quality_counts[label]
    return row


def main() -> None:
    args = parse_args()
    quality_rows = read_rows(Path(args.quality_csv))
    quality_by_key = {key(row): row for row in quality_rows}
    denominator = args.denominator or len(quality_rows)

    rows = [
        summarize_selector(name, path, quality_by_key, denominator)
        for name, path in (parse_named_path(value) for value in args.selector)
    ]
    write_csv(Path(args.output_csv), rows)

    summary = {
        "denominator": denominator,
        "selectors": rows,
        "output_csv": args.output_csv,
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
