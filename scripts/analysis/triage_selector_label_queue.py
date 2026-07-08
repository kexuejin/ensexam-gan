#!/usr/bin/env python3
"""Auto-triage selector labeling queues into promotion and review buckets.

The output is a prioritization aid, not a final visual label source. It uses
local proxy metrics and review-queue fields to reduce the number of pages that
need close manual inspection before expanding selector coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


TRIAGE_VALUES = (
    "promote_candidate",
    "borderline_review",
    "keep_review",
    "reject_candidate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-csv", required=True, help="Labeling queue CSV with metric and verdict columns.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-promote-help-hurt", type=float, default=1.35)
    parser.add_argument("--min-borderline-help-hurt", type=float, default=1.08)
    parser.add_argument("--min-promote-residual-help-hurt", type=float, default=100.0)
    parser.add_argument("--max-promote-changed-ratio", type=float, default=0.002)
    parser.add_argument("--max-promote-risk-changed-ratio", type=float, default=0.0)
    parser.add_argument("--max-promote-hurt-target-darker-ratio", type=float, default=0.97)
    parser.add_argument(
        "--accept-verdict-promotes",
        action="store_true",
        help="Treat metric-safe local accept rows as promote candidates unless risk guards fail.",
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict[str, str], column: str, default: float = 0.0) -> float:
    value = row.get(column, "")
    if value == "":
        return default
    try:
        number = float(value)
    except ValueError:
        return default
    return number if math.isfinite(number) else default


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def triage_row(row: dict[str, str], args: argparse.Namespace) -> tuple[str, str, str]:
    group = row.get("label_queue_group", "")
    verdict = row.get("local_verdict", "")
    metric_safe = truthy(row.get("metric_safe", ""))
    gain = fnum(row, "gain")
    over_delta = fnum(row, "over_delta")
    changed_ratio = fnum(row, "changed_ratio")
    help_hurt = fnum(row, "help_hurt_ratio")
    residual_help_hurt = fnum(row, "residual_help_hurt_ratio")
    risk_changed_ratio = fnum(row, "risk_changed_ratio")
    hurt_target_darker_ratio = fnum(row, "hurt_target_darker_ratio")
    hurt_base_ok_ratio = fnum(row, "hurt_base_ok_ratio")

    if verdict == "reject" or not metric_safe or gain <= 0.0 or over_delta > 0.0:
        return (
            "reject_candidate",
            "high",
            f"unsafe metric/verdict; verdict={verdict}; metric_safe={metric_safe}; gain={gain:.9f}; over_delta={over_delta:.9f}",
        )

    risk_reasons: list[str] = []
    if risk_changed_ratio > args.max_promote_risk_changed_ratio:
        risk_reasons.append(f"risk_changed_ratio={risk_changed_ratio:.9f}")
    if changed_ratio > args.max_promote_changed_ratio:
        risk_reasons.append(f"changed_ratio={changed_ratio:.9f}")
    if hurt_target_darker_ratio > args.max_promote_hurt_target_darker_ratio:
        risk_reasons.append(f"hurt_target_darker_ratio={hurt_target_darker_ratio:.6f}")
    if hurt_base_ok_ratio > 0.50 and help_hurt < 1.50:
        risk_reasons.append(f"hurt_base_ok_ratio={hurt_base_ok_ratio:.6f}")

    strong_signal = (
        help_hurt >= args.min_promote_help_hurt
        and residual_help_hurt >= args.min_promote_residual_help_hurt
        and not risk_reasons
    )
    local_accept_signal = args.accept_verdict_promotes and verdict == "accept" and not risk_reasons

    if strong_signal or local_accept_signal:
        return (
            "promote_candidate",
            "low",
            f"{group}; metric-safe {verdict}; gain={gain:.9f}; help_hurt={help_hurt:.3f}; residual_help_hurt={residual_help_hurt:.3f}",
        )

    if help_hurt >= args.min_borderline_help_hurt and residual_help_hurt > 0:
        reason = (
            f"{group}; metric-safe but needs visual check; gain={gain:.9f}; "
            f"help_hurt={help_hurt:.3f}; residual_help_hurt={residual_help_hurt:.3f}"
        )
        if risk_reasons:
            reason += "; risk=" + ",".join(risk_reasons)
        return "borderline_review", "medium", reason

    reason = (
        f"{group}; weak visual proxy; gain={gain:.9f}; help_hurt={help_hurt:.3f}; "
        f"residual_help_hurt={residual_help_hurt:.3f}"
    )
    if risk_reasons:
        reason += "; risk=" + ",".join(risk_reasons)
    return "keep_review", "high", reason


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.queue_csv))
    output_dir = Path(args.output_dir)

    output_rows: list[dict[str, object]] = []
    for row in rows:
        triage, priority, reason = triage_row(row, args)
        out: dict[str, object] = dict(row)
        out["auto_triage_label"] = triage
        out["auto_review_priority"] = priority
        out["auto_triage_reason"] = reason
        output_rows.append(out)

    fieldnames = list(output_rows[0].keys()) if output_rows else []
    write_csv(output_dir / "auto_triage_labels.csv", output_rows, fieldnames)

    by_label: dict[str, list[dict[str, object]]] = {label: [] for label in TRIAGE_VALUES}
    for row in output_rows:
        by_label[str(row["auto_triage_label"])].append(row)

    for label in TRIAGE_VALUES:
        write_csv(output_dir / f"{label}.csv", by_label[label], fieldnames)

    counts = Counter(str(row["auto_triage_label"]) for row in output_rows)
    by_group_label = Counter(
        (str(row.get("label_queue_group", "")), str(row["auto_triage_label"])) for row in output_rows
    )
    summary = {
        "rows": len(output_rows),
        "counts": {label: counts.get(label, 0) for label in TRIAGE_VALUES},
        "by_group_label": {f"{group}:{label}": count for (group, label), count in sorted(by_group_label.items())},
        "outputs": {label: str(output_dir / f"{label}.csv") for label in TRIAGE_VALUES},
    }
    (output_dir / "auto_triage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
