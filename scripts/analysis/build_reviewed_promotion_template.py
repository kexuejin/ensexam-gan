#!/usr/bin/env python3
"""Build a reviewed-promotion template from target-quality triage rows.

The promotion overlay accepts explicit reviewed pages. This helper creates a
ranked CSV for that human/local review step without promoting anything by
itself.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--bucket", action="append", default=["ratio_noise_review"])
    parser.add_argument("--risk-px-tier", action="append", default=["low_abs_risk", "moderate_abs_risk"])
    parser.add_argument("--min-gain", type=float, default=0.002)
    parser.add_argument("--min-help-hurt", type=float, default=1.75)
    parser.add_argument("--max-risk-px", type=float, default=1500.0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, str], field: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(field, ""))
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def priority_score(row: dict[str, str]) -> float:
    gain = fnum(row, "mean_error_gain")
    help_hurt = fnum(row, "help_hurt_ratio")
    residual_help_hurt = min(fnum(row, "residual_help_hurt_ratio"), 5000.0) / 5000.0
    risk_px = fnum(row, "risk_px_total")
    risk_penalty = risk_px / 1500.0
    return gain * 100.0 + help_hurt + residual_help_hurt - risk_penalty


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.triage_csv))
    buckets = set(args.bucket)
    risk_tiers = set(args.risk_px_tier)
    selected = [
        row
        for row in rows
        if row.get("triage_bucket") in buckets
        and row.get("risk_px_tier") in risk_tiers
        and fnum(row, "mean_error_gain") >= args.min_gain
        and fnum(row, "help_hurt_ratio") >= args.min_help_hurt
        and fnum(row, "risk_px_total") <= args.max_risk_px
    ]
    selected.sort(key=priority_score, reverse=True)

    output_rows = []
    for index, row in enumerate(selected, start=1):
        output_rows.append({
            "rank": index,
            "split": row["split"],
            "file": row["file"],
            "page": f"{row['split']}/{row['file']}",
            "recommendation": "review_for_promotion",
            "review_decision": "",
            "reviewer": "",
            "review_date": "",
            "comment": "",
            "triage_bucket": row.get("triage_bucket", ""),
            "risk_trigger": row.get("risk_trigger", ""),
            "risk_px_tier": row.get("risk_px_tier", ""),
            "risk_px_total": row.get("risk_px_total", ""),
            "mean_error_gain": row.get("mean_error_gain", ""),
            "help_hurt_ratio": row.get("help_hurt_ratio", ""),
            "residual_help_hurt_ratio": row.get("residual_help_hurt_ratio", ""),
            "overerase_changed_ratio": row.get("overerase_changed_ratio", ""),
            "target_dark_damage_changed_ratio": row.get("target_dark_damage_changed_ratio", ""),
        })

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys()) if output_rows else [
        "rank",
        "split",
        "file",
        "page",
        "recommendation",
        "review_decision",
        "reviewer",
        "review_date",
        "comment",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"rows={len(output_rows)}")
    print(f"output_csv={output_path}")
    for row in output_rows[:20]:
        print(
            row["rank"],
            row["page"],
            row["risk_trigger"],
            row["risk_px_tier"],
            row["mean_error_gain"],
            row["help_hurt_ratio"],
            row["risk_px_total"],
        )


if __name__ == "__main__":
    main()
