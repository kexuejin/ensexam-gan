#!/usr/bin/env python3
"""Suggest page-level product-quality labels for manual review queues.

The suggestions are triage hints only. They are derived from metric deltas and
page features, so they must not be merged as final visual labels without human
review.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTPUT_FIELDS = [
    "split",
    "file",
    "candidate",
    "bucket",
    "label",
    "flags",
    "reviewer",
    "review_date",
    "comment",
    "auto_suggest_label",
    "auto_confidence",
    "auto_review_priority",
    "auto_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-template", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument(
        "--features-csv",
        action="append",
        default=[],
        help="Feature CSV from analyze_residual_delta_selector_features.py. May be repeated.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--high-activity-gate", type=float, default=0.015)
    parser.add_argument("--strong-win-gain", type=float, default=0.01)
    parser.add_argument("--weak-win-gain", type=float, default=0.002)
    parser.add_argument("--loss-gain", type=float, default=-0.002)
    parser.add_argument("--overerase-risk", type=float, default=0.0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, str]:
    return row["split"], row["file"]


def read_features(paths: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        for row in read_rows(Path(path)):
            split = row.get("split", "")
            if split == "main":
                # Single-split files are ambiguous when mixed with joint output;
                # prefer explicit split feature files for label suggestions.
                continue
            out[(split, row["file"])] = row
    return out


def fnum(row: dict[str, str] | None, key_name: str, default: float = 0.0) -> float:
    if not row:
        return default
    value = row.get(key_name, "")
    if value == "":
        return default
    return float(value)


def suggestion(
    label_row: dict[str, str],
    feature: dict[str, str] | None,
    args: argparse.Namespace,
) -> tuple[str, str, str, str]:
    bucket = label_row["bucket"]
    gain = fnum(feature, "gain")
    over_delta = fnum(feature, "over_delta")
    gate_ratio = fnum(feature, "gate_ratio")
    candidate_delta_max = fnum(feature, "candidate_delta_max")

    if feature is None:
        return "needs_review", "low", "high", "missing page features"

    risk_parts: list[str] = []
    if over_delta > args.overerase_risk:
        risk_parts.append(f"over_delta={over_delta:.9f}")
    if gate_ratio >= args.high_activity_gate:
        risk_parts.append(f"gate_ratio={gate_ratio:.6f}")
    if bucket.endswith("_risk"):
        risk_parts.append(f"bucket={bucket}")

    if bucket == "residual_delta_joint_selector" and gain > 0 and over_delta <= 0:
        return (
            "slight_win",
            "medium",
            "medium",
            f"safe selector hit; gain={gain:.9f}; candidate_delta_max={candidate_delta_max:.3f}",
        )

    if gain >= args.strong_win_gain and over_delta <= 0 and gate_ratio < args.high_activity_gate:
        return "slight_win", "medium", "medium", f"strong metric win; gain={gain:.9f}; over_delta={over_delta:.9f}"

    if gain >= args.weak_win_gain and over_delta <= 0:
        priority = "high" if risk_parts else "medium"
        confidence = "low" if risk_parts else "medium"
        reason = f"weak metric win; gain={gain:.9f}; over_delta={over_delta:.9f}"
        if risk_parts:
            reason += "; risk=" + ",".join(risk_parts)
        return "slight_win", confidence, priority, reason

    if gain <= args.loss_gain or over_delta > args.overerase_risk:
        reason = f"metric loss/risk; gain={gain:.9f}; over_delta={over_delta:.9f}"
        if risk_parts:
            reason += "; risk=" + ",".join(risk_parts)
        return "slight_loss", "medium", "high", reason

    reason = f"near tie; gain={gain:.9f}; over_delta={over_delta:.9f}"
    if risk_parts:
        reason += "; risk=" + ",".join(risk_parts)
        return "noop", "low", "high", reason
    return "noop", "medium", "low", reason


def main() -> None:
    args = parse_args()
    labels = read_rows(Path(args.labels_template))
    review_by_key = {(
        row["split"],
        row["file"],
        row["candidate"],
        row["bucket"],
    ): row for row in read_rows(Path(args.review_csv))}
    features_by_key = read_features(args.features_csv)

    output_rows: list[dict[str, str]] = []
    for row in labels:
        out = dict(row)
        review = review_by_key.get((row["split"], row["file"], row["candidate"], row["bucket"]), {})
        feature = features_by_key.get(key(row))
        suggested, confidence, priority, reason = suggestion(row, feature, args)
        out["auto_suggest_label"] = suggested
        out["auto_confidence"] = confidence
        out["auto_review_priority"] = priority
        out["auto_reason"] = reason
        if review.get("notes"):
            out["auto_reason"] += f"; notes={review['notes']}"
        output_rows.append(out)

    write_csv(Path(args.output_csv), output_rows)

    counts: dict[tuple[str, str, str], int] = {}
    for row in output_rows:
        item = (row["auto_suggest_label"], row["auto_confidence"], row["auto_review_priority"])
        counts[item] = counts.get(item, 0) + 1
    print(f"rows={len(output_rows)}")
    for item, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
        print(f"{item[0]},{item[1]},{item[2]}={count}")
    print(f"output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
