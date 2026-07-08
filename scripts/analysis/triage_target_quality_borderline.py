#!/usr/bin/env python3
"""Triage target-aware borderline selector pages into review buckets.

The target-aware scorer intentionally marks mixed pages as borderline when
risk ratios exceed conservative thresholds. This helper does not override that
label. It makes the next quality pass reproducible by separating high-yield
ratio-noise candidates from pages that need visual review first.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


LOSS_LABELS = {"clear_loss", "slight_loss"}
WIN_LABELS = {"clear_win", "slight_win"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-csv", required=True, help="Output from score_target_comparison_quality.py.")
    parser.add_argument("--selected-csv", required=True, help="Selector selected-pages CSV to triage.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", default="target_quality_borderline_triage")
    parser.add_argument("--min-strong-gain", type=float, default=0.002)
    parser.add_argument("--min-strong-help-hurt", type=float, default=1.75)
    parser.add_argument("--min-strong-residual-help-hurt", type=float, default=500.0)
    parser.add_argument("--max-low-overerase-changed-ratio", type=float, default=0.035)
    parser.add_argument("--max-low-dark-damage-changed-ratio", type=float, default=0.25)
    parser.add_argument("--min-weak-gain", type=float, default=0.0005)
    parser.add_argument("--min-weak-help-hurt", type=float, default=1.35)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for field in row:
                if field not in fieldnames:
                    fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, str]:
    return row["split"], row["file"]


def fnum(row: dict[str, str], field: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(field, ""))
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def risk_trigger(row: dict[str, str], args: argparse.Namespace) -> str:
    overerase_high = fnum(row, "overerase_changed_ratio") > args.max_low_overerase_changed_ratio
    dark_damage_high = fnum(row, "target_dark_damage_changed_ratio") > args.max_low_dark_damage_changed_ratio
    if overerase_high and dark_damage_high:
        return "overerase_and_dark_damage"
    if overerase_high:
        return "overerase_ratio"
    if dark_damage_high:
        return "dark_damage_ratio"
    return "low_ratio_risk"


def risk_px_tier(row: dict[str, str]) -> str:
    risk_px = fnum(row, "overerase_hurt_px") + fnum(row, "target_dark_damage_px")
    if risk_px <= 750:
        return "low_abs_risk"
    if risk_px <= 1500:
        return "moderate_abs_risk"
    return "high_abs_risk"


def classify(row: dict[str, str], args: argparse.Namespace) -> tuple[str, str]:
    local_verdict = row.get("local_verdict", "")
    gain = fnum(row, "mean_error_gain")
    active_gain = fnum(row, "active_mean_gain")
    help_hurt = fnum(row, "help_hurt_ratio")
    residual_help_hurt = fnum(row, "residual_help_hurt_ratio")
    overerase_changed = fnum(row, "overerase_changed_ratio")
    dark_damage_changed = fnum(row, "target_dark_damage_changed_ratio")
    overerase_px = fnum(row, "overerase_hurt_px")
    dark_damage_px = fnum(row, "target_dark_damage_px")

    strong_metric_win = (
        gain >= args.min_strong_gain
        and active_gain > 0.0
        and help_hurt >= args.min_strong_help_hurt
        and residual_help_hurt >= args.min_strong_residual_help_hurt
    )
    low_ratio_risk = (
        overerase_changed <= args.max_low_overerase_changed_ratio
        and dark_damage_changed <= args.max_low_dark_damage_changed_ratio
    )
    weak_metric_win = gain >= args.min_weak_gain and help_hurt >= args.min_weak_help_hurt and active_gain > 0.0

    if local_verdict == "accept" and strong_metric_win and low_ratio_risk:
        return (
            "auto_win_candidate",
            (
                "local accept, strong target metric win, and low changed-area risk ratios; "
                "candidate for promotion after spot-check"
            ),
        )
    if local_verdict == "accept" and strong_metric_win:
        return (
            "ratio_noise_review",
            (
                "local accept and strong target metric win, but one changed-area risk ratio is high; "
                "inspect whether absolute risk pixels are visual noise"
            ),
        )
    if local_verdict == "accept" and weak_metric_win:
        return "accept_weak_win_review", "local accept with weak positive target metrics; visual review before promotion"
    if local_verdict == "review" and strong_metric_win:
        return "manual_review_strong_metric", "local review but strong target metrics; high-priority manual review"
    if local_verdict == "review":
        return "manual_review_mixed_metric", "local review with mixed or weak target metrics; keep manual"
    if overerase_px + dark_damage_px > 1000:
        return "risk_pixel_review", "high absolute risk-pixel count; inspect before any promotion"
    return "keep_borderline", "mixed target metrics; keep borderline"


def main() -> None:
    args = parse_args()
    quality_rows = read_rows(Path(args.quality_csv))
    selected_rows = read_rows(Path(args.selected_csv))
    quality_by_key = {key(row): row for row in quality_rows}

    missing = [f"{row['split']}/{row['file']}" for row in selected_rows if key(row) not in quality_by_key]
    if missing:
        raise KeyError(f"Missing quality rows for selected pages: {' '.join(missing)}")

    triage_rows: list[dict[str, object]] = []
    for selected in selected_rows:
        quality = quality_by_key[key(selected)]
        label = quality["target_quality_label"]
        if label in WIN_LABELS or label in LOSS_LABELS:
            continue
        if label != "borderline":
            continue
        bucket, reason = classify(quality, args)
        out: dict[str, object] = {**selected, **quality}
        out["triage_bucket"] = bucket
        out["triage_reason"] = reason
        out["risk_px_total"] = fnum(quality, "overerase_hurt_px") + fnum(quality, "target_dark_damage_px")
        out["risk_trigger"] = risk_trigger(quality, args)
        out["risk_px_tier"] = risk_px_tier(quality)
        out["triage_subbucket"] = f"{bucket}__{out['risk_trigger']}__{out['risk_px_tier']}"
        triage_rows.append(out)

    triage_rows.sort(
        key=lambda row: (
            str(row["triage_bucket"]),
            -fnum(row, "mean_error_gain"),
            fnum(row, "risk_px_total"),
            row["split"],
            row["file"],
        )
    )
    counts = Counter(str(row["triage_bucket"]) for row in triage_rows)
    subbucket_counts = Counter(str(row["triage_subbucket"]) for row in triage_rows)
    local_counts = Counter(str(row.get("local_verdict", "")) for row in triage_rows)
    summary = {
        "name": args.name,
        "selected_rows": len(selected_rows),
        "quality_rows": len(quality_rows),
        "borderline_rows": len(triage_rows),
        "bucket_counts": dict(sorted(counts.items())),
        "subbucket_counts": dict(sorted(subbucket_counts.items())),
        "local_counts": dict(sorted(local_counts.items())),
        "thresholds": {
            "min_strong_gain": args.min_strong_gain,
            "min_strong_help_hurt": args.min_strong_help_hurt,
            "min_strong_residual_help_hurt": args.min_strong_residual_help_hurt,
            "max_low_overerase_changed_ratio": args.max_low_overerase_changed_ratio,
            "max_low_dark_damage_changed_ratio": args.max_low_dark_damage_changed_ratio,
            "min_weak_gain": args.min_weak_gain,
            "min_weak_help_hurt": args.min_weak_help_hurt,
        },
    }

    output_dir = Path(args.output_dir)
    write_csv(output_dir / f"{args.name}.csv", triage_rows)
    write_csv(output_dir / f"{args.name}_summary.csv", [summary], list(summary.keys()))
    (output_dir / f"{args.name}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
