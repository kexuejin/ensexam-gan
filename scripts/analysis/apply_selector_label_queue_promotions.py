#!/usr/bin/env python3
"""Apply reviewed selector label-queue promotions to a selected-pages CSV.

The post-125 selector auto-triage output is a review queue, not an automatic
promotion source. This helper builds a reproducible selected-pages overlay from
rows that have an explicit positive manual decision and pass metric-safety
checks. Blank templates promote nothing by design.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


POSITIVE_REVIEW_DECISIONS = ("promote", "accept", "approve", "approved", "yes", "y", "true", "1")
DEFAULT_DECISION_COLUMNS = ("manual_label", "review_decision")
TRUTHY_VALUES = {"1", "true", "yes", "y"}
REJECT_VALUES = {"reject", "rejected", "no", "n", "false", "0", "keep_review", "keep", "skip"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-selected-csv", required=True, help="Current selected-pages CSV, e.g. the 125 default.")
    parser.add_argument(
        "--reviewed-labels-csv",
        required=True,
        help="Auto-triage / label-queue CSV with split,file and manual_label/review_decision columns.",
    )
    parser.add_argument("--output-csv", required=True, help="Output selected-pages CSV after reviewed additions.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON path.")
    parser.add_argument(
        "--decision-column",
        action="append",
        default=[],
        help=(
            "Decision column to inspect. May be repeated. Defaults to manual_label and review_decision "
            "when those columns exist."
        ),
    )
    parser.add_argument(
        "--accepted-decision",
        action="append",
        default=[],
        help="Accepted reviewed decision value. Defaults to promote/accept/approve/approved/yes/y/true/1.",
    )
    parser.add_argument(
        "--require-auto-triage-label",
        action="append",
        default=["promote_candidate"],
        help=(
            "Required auto_triage_label for new additions. May be repeated. Defaults to promote_candidate; "
            "pass additional labels after visual review to widen the allowed queue."
        ),
    )
    parser.add_argument(
        "--allow-any-auto-triage-label",
        action="store_true",
        help="Do not restrict by auto_triage_label; explicit positive manual decisions and metric gates still apply.",
    )
    parser.add_argument("--promotion-note", default="reviewed selector label-queue promotion")
    parser.add_argument("--expect-accepted", type=int, default=-1, help="Fail unless this many reviewed rows pass all gates.")
    parser.add_argument("--expect-added", type=int, default=-1, help="Fail unless this many new pages are added.")
    parser.add_argument("--allow-skipped-reviewed", action="store_true", help="Do not fail when positive rows fail gates.")
    return parser.parse_args()


def normalize(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def page_key(row: dict[str, str]) -> tuple[str, str]:
    split = row.get("split", "").strip()
    file = row.get("file", "").strip()
    page = row.get("page", "").strip()
    if split and file:
        return split, file
    if page.count("/") == 1:
        return tuple(page.split("/", 1))  # type: ignore[return-value]
    if file.count("/") == 1:
        return tuple(file.split("/", 1))  # type: ignore[return-value]
    return "", ""


def page_text(key: tuple[str, str]) -> str:
    return f"{key[0]}/{key[1]}" if key[0] and key[1] else ""


def fnum(row: dict[str, str], field: str, default: float = 0.0) -> float:
    value = row.get(field, "")
    if value == "":
        return default
    try:
        number = float(value)
    except ValueError:
        return default
    return number if math.isfinite(number) else default


def truthy(value: str) -> bool:
    return normalize(value) in TRUTHY_VALUES


def metric_safe(row: dict[str, str]) -> bool:
    if row.get("metric_safe", ""):
        return truthy(row["metric_safe"])
    return fnum(row, "gain") > 0.0 and fnum(row, "over_delta") <= 0.0


def ordered_fields(*row_groups: Iterable[dict[str, str]]) -> list[str]:
    fields: list[str] = []
    for rows in row_groups:
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    return fields


def selected_decision(row: dict[str, str], decision_columns: list[str]) -> tuple[str, str, list[dict[str, str]]]:
    values = [(column, normalize(row.get(column, ""))) for column in decision_columns if row.get(column, "").strip()]
    if not values:
        return "", "", []
    unique_values = {value for _, value in values}
    if len(unique_values) > 1:
        return "", "", [{"skip_reason": "conflicting_decisions", "decisions": ";".join(f"{c}={v}" for c, v in values)}]
    column, value = values[0]
    return column, value, []


def add_promotion_metadata(row: dict[str, str], args: argparse.Namespace, source_row: dict[str, str]) -> dict[str, str]:
    out = dict(row)
    prior_notes = out.get("notes", "")
    note_parts = [args.promotion_note]
    if source_row.get("auto_triage_label"):
        note_parts.append(f"auto_triage_label={source_row['auto_triage_label']}")
    if source_row.get("manual_label"):
        note_parts.append(f"manual_label={source_row['manual_label']}")
    if source_row.get("review_decision"):
        note_parts.append(f"review_decision={source_row['review_decision']}")
    if prior_notes:
        note_parts.append(prior_notes)
    out["notes"] = "; ".join(note_parts)
    out["selector_reviewed_promotion"] = "1"
    out["selector_reviewed_promotion_source"] = args.reviewed_labels_csv
    return out


def main() -> None:
    args = parse_args()
    base_rows = read_rows(Path(args.base_selected_csv))
    label_rows = read_rows(Path(args.reviewed_labels_csv))
    decision_columns = args.decision_column or [column for column in DEFAULT_DECISION_COLUMNS if any(column in row for row in label_rows)]
    if not decision_columns:
        raise ValueError("No decision columns found; pass --decision-column or add manual_label/review_decision.")

    accepted_decisions = {normalize(value) for value in (args.accepted_decision or list(POSITIVE_REVIEW_DECISIONS))}
    required_labels = {normalize(value) for value in args.require_auto_triage_label}

    base_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in base_rows:
        key = page_key(row)
        if not page_text(key):
            raise ValueError(f"Base selected row needs split/file or page: {row}")
        if key in base_by_key:
            raise ValueError(f"Base selected CSV has duplicate page: {page_text(key)}")
        base_by_key[key] = row

    reviewed_by_key: dict[tuple[str, str], dict[str, str]] = {}
    accepted_rows: list[dict[str, str]] = []
    added_rows: list[dict[str, str]] = []
    already_selected: list[str] = []
    skipped: list[dict[str, str]] = []
    blank_or_negative = Counter()

    for row_index, row in enumerate(label_rows, start=2):
        key = page_key(row)
        page = page_text(key)
        decision_column, decision, decision_skips = selected_decision(row, decision_columns)
        if decision_skips:
            skipped.extend({"row": str(row_index), "file": page, **item} for item in decision_skips)
            continue
        if not decision:
            blank_or_negative["blank_decision"] += 1
            continue
        if decision not in accepted_decisions:
            blank_or_negative["negative_or_unaccepted_decision"] += 1
            if decision in REJECT_VALUES:
                blank_or_negative[f"decision_{decision}"] += 1
            continue
        if not page:
            skipped.append({"row": str(row_index), "file": "", "decision": decision, "skip_reason": "missing_page"})
            continue
        if key in reviewed_by_key:
            raise ValueError(f"Reviewed labels CSV has duplicate accepted page: {page}")

        triage_label = normalize(row.get("auto_triage_label", ""))
        if not args.allow_any_auto_triage_label:
            if not triage_label:
                skipped.append({
                    "row": str(row_index),
                    "file": page,
                    "decision": row.get(decision_column, decision),
                    "skip_reason": "missing_auto_triage_label",
                })
                continue
            if triage_label not in required_labels:
                skipped.append({
                    "row": str(row_index),
                    "file": page,
                    "decision": row.get(decision_column, decision),
                    "auto_triage_label": row.get("auto_triage_label", ""),
                    "skip_reason": "auto_triage_label_not_allowed",
                })
                continue
        if row.get("local_verdict", "").strip().lower() == "reject":
            skipped.append({"row": str(row_index), "file": page, "skip_reason": "local_verdict_reject"})
            continue
        if not metric_safe(row):
            skipped.append({"row": str(row_index), "file": page, "skip_reason": "metric_safe_gate_failed"})
            continue
        if fnum(row, "gain") <= 0.0:
            skipped.append({"row": str(row_index), "file": page, "skip_reason": "gain_not_positive", "gain": row.get("gain", "")})
            continue
        if fnum(row, "over_delta") > 0.0:
            skipped.append({"row": str(row_index), "file": page, "skip_reason": "over_delta_positive", "over_delta": row.get("over_delta", "")})
            continue

        accepted_rows.append(row)
        reviewed_by_key[key] = row
        if key in base_by_key:
            already_selected.append(page)
        else:
            added_rows.append(add_promotion_metadata(row, args, row))

    if skipped and not args.allow_skipped_reviewed:
        raise ValueError(f"Positive reviewed rows failed promotion gates: {skipped}")
    if args.expect_accepted >= 0 and len(accepted_rows) != args.expect_accepted:
        raise ValueError(f"Expected {args.expect_accepted} accepted reviewed rows, got {len(accepted_rows)}")
    if args.expect_added >= 0 and len(added_rows) != args.expect_added:
        raise ValueError(f"Expected {args.expect_added} added pages, got {len(added_rows)}")

    output_rows = [dict(row) for row in base_rows] + added_rows
    fieldnames = ordered_fields(base_rows, added_rows)
    output_csv = Path(args.output_csv)
    write_csv(output_csv, output_rows, fieldnames)

    summary = {
        "base_selected_csv": args.base_selected_csv,
        "reviewed_labels_csv": args.reviewed_labels_csv,
        "output_csv": str(output_csv),
        "decision_columns": decision_columns,
        "accepted_decisions": sorted(accepted_decisions),
        "required_auto_triage_labels": [] if args.allow_any_auto_triage_label else sorted(required_labels),
        "base_count": len(base_rows),
        "reviewed_label_rows": len(label_rows),
        "accepted_reviewed_count": len(accepted_rows),
        "already_selected_count": len(already_selected),
        "already_selected_files": sorted(already_selected),
        "added_count": len(added_rows),
        "added_files": sorted(page_text(page_key(row)) for row in added_rows),
        "output_count": len(output_rows),
        "blank_or_negative_decisions": dict(blank_or_negative),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "promotion_note": args.promotion_note,
        "status": "reviewed selector overlay; not a product default until visual labels are confirmed",
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
