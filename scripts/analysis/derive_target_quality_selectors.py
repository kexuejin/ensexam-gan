#!/usr/bin/env python3
"""Derive quality-first selector CSVs from target-aware page labels.

This helper turns a selected-pages CSV and a target-quality CSV into smaller
selector variants such as "confirmed wins only" and "zero target-aware loss".
It keeps the policy decision reproducible instead of relying on ad hoc filters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


METRIC_RE = re.compile(r"(gain|over_delta)=(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.IGNORECASE)
QUALITY_LABELS = ("clear_win", "slight_win", "noop", "borderline", "slight_loss", "clear_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-selected-csv", required=True)
    parser.add_argument("--quality-csv", required=True, help="Output from score_target_comparison_quality.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name-prefix", default="", help="Optional prefix for generated selector names.")
    parser.add_argument("--denominator", type=int, default=0, help="Coverage denominator. Defaults to quality CSV rows.")
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


def fnum(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def metric_pair(row: dict[str, str]) -> tuple[float, float]:
    gain = fnum(row.get("gain"))
    over_delta = fnum(row.get("over_delta"))
    if gain is not None and over_delta is not None:
        return gain, over_delta

    found = {match.group(1).lower(): float(match.group(2)) for match in METRIC_RE.finditer(row.get("notes", ""))}
    if "gain" in found and "over_delta" in found:
        return found["gain"], found["over_delta"]

    raise KeyError(f"Cannot find gain/over_delta for {row.get('split')}/{row.get('file')}")


def selector_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}" if prefix else suffix


def summarize(
    name: str,
    rows: list[dict[str, str]],
    quality_by_key: dict[tuple[str, str], dict[str, str]],
    denominator: int,
) -> dict[str, object]:
    quality_counts = {label: 0 for label in QUALITY_LABELS}
    local_counts = {"accept": 0, "review": 0, "reject": 0}
    split_counts: dict[str, int] = {}
    residual_gain = 0.0
    overerase_delta = 0.0
    metric_losses = 0

    for row in rows:
        row_quality = quality_by_key[key(row)]
        quality_label = row_quality["target_quality_label"]
        quality_counts[quality_label] = quality_counts.get(quality_label, 0) + 1
        local_verdict = row_quality.get("local_verdict", "")
        local_counts[local_verdict] = local_counts.get(local_verdict, 0) + 1
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        gain, over_delta = metric_pair({**row_quality, **row})
        if not (gain > 0.0 and over_delta <= 0.0):
            metric_losses += 1
        residual_gain += gain
        overerase_delta += over_delta

    wins = quality_counts["clear_win"] + quality_counts["slight_win"]
    losses = quality_counts["clear_loss"] + quality_counts["slight_loss"]
    return {
        "selector": name,
        "selected": len(rows),
        "selected_coverage_pct": len(rows) / denominator * 100.0,
        "target_confirmed_wins": wins,
        "target_confirmed_win_pct": wins / denominator * 100.0,
        "target_borderline": quality_counts["borderline"],
        "target_losses": losses,
        "target_clear_win": quality_counts["clear_win"],
        "target_slight_win": quality_counts["slight_win"],
        "target_noop": quality_counts["noop"],
        "target_slight_loss": quality_counts["slight_loss"],
        "target_clear_loss": quality_counts["clear_loss"],
        "local_accept": local_counts["accept"],
        "local_review": local_counts["review"],
        "local_reject": local_counts["reject"],
        "metric_losses": metric_losses,
        "residual_gain": residual_gain / denominator,
        "overerase_delta": overerase_delta / denominator,
        "split_counts": dict(sorted(split_counts.items())),
        "files": [f"{row['split']}/{row['file']}" for row in rows],
    }


def write_selector(
    output_dir: Path,
    name: str,
    rows: list[dict[str, str]],
    quality_by_key: dict[tuple[str, str], dict[str, str]],
    denominator: int,
) -> dict[str, object]:
    fields = list(rows[0].keys()) if rows else []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)

    selected_csv = output_dir / f"{name}_selected.csv"
    summary_csv = output_dir / f"{name}_summary.csv"
    summary_json = output_dir / f"{name}_summary.json"
    write_csv(selected_csv, rows, fields)
    summary = summarize(name, rows, quality_by_key, denominator)
    write_csv(summary_csv, [summary], list(summary.keys()))
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    base_rows = read_rows(Path(args.base_selected_csv))
    quality_rows = read_rows(Path(args.quality_csv))
    quality_by_key = {key(row): row for row in quality_rows}
    denominator = args.denominator or len(quality_rows)
    output_dir = Path(args.output_dir)

    missing = [f"{row['split']}/{row['file']}" for row in base_rows if key(row) not in quality_by_key]
    if missing:
        raise KeyError(f"Missing quality rows for selected pages: {' '.join(missing)}")

    confirmed_wins = [
        row for row in base_rows if quality_by_key[key(row)]["target_quality_label"] in {"clear_win", "slight_win"}
    ]
    zero_target_loss = [
        row for row in base_rows if quality_by_key[key(row)]["target_quality_label"] not in {"clear_loss", "slight_loss"}
    ]

    summaries = [
        write_selector(
            output_dir,
            selector_name(args.name_prefix, f"{len(confirmed_wins)}_target_confirmed_wins"),
            confirmed_wins,
            quality_by_key,
            denominator,
        ),
        write_selector(
            output_dir,
            selector_name(args.name_prefix, f"{len(zero_target_loss)}_zero_target_loss"),
            zero_target_loss,
            quality_by_key,
            denominator,
        ),
    ]
    print(json.dumps({"generated": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
