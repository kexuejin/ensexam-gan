#!/usr/bin/env python3
"""Apply reviewed sustainable-quality-goal labels to overlay CSVs.

The quality-goal review queue intentionally keeps manual labels outside the
source metrics until a reviewer has made explicit decisions. This helper merges
those reviewed decisions into reproducible overlay files:

1. target-borderline review decisions update a target-quality CSV copy;
2. post-125 review decisions update a post-125 label-queue CSV copy.

Blank review labels change nothing. Conflicting duplicate decisions fail by
default so selector gates cannot accidentally advance from ambiguous evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path("outputs/balanced007_ranker_expansion_source_eval_20260708")
DEFAULT_TARGET_QUALITY_CSV = ROOT / "target_quality_scoring_20260708/full435_target_quality_v2.csv"
DEFAULT_POST125_CSV = ROOT / "labeling_queue_post125_coverage_20260708/auto_triage/auto_triage_labels.csv"

TARGET_STREAMS = {"release_target_borderline", "target_borderline", "selected_target_borderline"}
POST125_STREAMS = {"post125_unresolved", "post125", "post_125_unresolved"}

TARGET_CONFIRM_DECISIONS = {
    "confirm_win",
    "confirmed_win",
    "clear_win",
    "slight_win",
    "accept",
    "accepted",
    "approve",
    "approved",
    "promote",
    "yes",
    "y",
    "true",
    "1",
}
TARGET_KEEP_DECISIONS = {
    "keep_borderline",
    "borderline",
    "keep_review",
    "review",
    "defer",
    "skip",
    "noop",
    "no_change",
}
TARGET_LOSS_DECISIONS = {
    "reject_selector_page",
    "reject",
    "rejected",
    "repair_needed",
    "repair",
    "slight_loss",
    "clear_loss",
    "loss",
}
POST125_ALLOWED_DECISIONS = {
    "promote",
    "accept",
    "approve",
    "approved",
    "yes",
    "y",
    "true",
    "1",
    "keep_review",
    "keep",
    "review",
    "defer",
    "skip",
    "reject",
    "rejected",
    "no",
    "n",
    "false",
    "0",
}
QUALITY_LABELS = {"clear_win", "slight_win", "noop", "borderline", "slight_loss", "clear_loss"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", action="append", required=True, help="Reviewed queue CSV. May be repeated.")
    parser.add_argument("--target-quality-csv", default=str(DEFAULT_TARGET_QUALITY_CSV))
    parser.add_argument("--post125-csv", default=str(DEFAULT_POST125_CSV))
    parser.add_argument("--output-target-quality-csv", required=True)
    parser.add_argument("--output-post125-csv", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--target-confirmed-label", default="slight_win", choices=sorted(QUALITY_LABELS))
    parser.add_argument("--target-loss-label", default="slight_loss", choices=sorted(QUALITY_LABELS))
    parser.add_argument("--expect-target-confirmed", type=int, default=-1)
    parser.add_argument("--expect-target-loss", type=int, default=-1)
    parser.add_argument("--expect-post125-labeled", type=int, default=-1)
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow a reviewed label to overwrite an existing source label.")
    return parser.parse_args()


def normalize(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ordered_fields(*row_groups: Iterable[dict[str, str]]) -> list[str]:
    fields: list[str] = []
    for rows in row_groups:
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    return fields


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = ordered_fields(rows)
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
        split, file = page.split("/", 1)
        return split, file
    if file.count("/") == 1:
        split, file = file.split("/", 1)
        return split, file
    return "", ""


def page_text(key: tuple[str, str]) -> str:
    return f"{key[0]}/{key[1]}" if key[0] and key[1] else ""


def review_label(row: dict[str, str]) -> str:
    return normalize(row.get("manual_label", "") or row.get("review_decision", "") or row.get("label", ""))


def collect_review_decisions(paths: list[Path]) -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], dict[str, str]], Counter[str]]:
    target: dict[tuple[str, str], dict[str, str]] = {}
    post125: dict[tuple[str, str], dict[str, str]] = {}
    skipped = Counter()
    conflicts: list[str] = []

    def add_decision(dest: dict[tuple[str, str], dict[str, str]], key: tuple[str, str], row: dict[str, str], label: str) -> None:
        existing = dest.get(key)
        if existing is not None and review_label(existing) != label:
            conflicts.append(f"{page_text(key)}: {review_label(existing)} vs {label}")
            return
        dest[key] = row

    for path in paths:
        for row_index, row in enumerate(read_rows(path), start=2):
            key = page_key(row)
            page = page_text(key)
            if not page:
                skipped["missing_page"] += 1
                continue
            label = review_label(row)
            if not label:
                skipped["blank_label"] += 1
                continue
            stream = normalize(row.get("review_stream", ""))
            if stream in TARGET_STREAMS:
                add_decision(target, key, row, label)
            elif stream in POST125_STREAMS:
                add_decision(post125, key, row, label)
            else:
                skipped[f"unknown_stream_row_{row_index}"] += 1
    if conflicts:
        raise ValueError(f"Conflicting duplicate review decisions: {conflicts}")
    return target, post125, skipped


def append_note(existing: str, note: str) -> str:
    return f"{note}; {existing}" if existing else note


def apply_target_reviews(
    rows: list[dict[str, str]],
    reviews: dict[tuple[str, str], dict[str, str]],
    confirmed_label: str,
    loss_label: str,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    counts = Counter()
    missing_reviews = set(reviews)
    output: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []
    for row in rows:
        key = page_key(row)
        review = reviews.get(key)
        out = dict(row)
        if review is None:
            output.append(out)
            continue
        missing_reviews.discard(key)
        decision = review_label(review)
        old_label = out.get("target_quality_label", "")
        out["target_review_manual_label"] = decision
        out["target_review_manual_notes"] = review.get("manual_notes", "")
        out["target_review_source"] = review.get("review_stream", "")
        if decision in TARGET_CONFIRM_DECISIONS:
            out["target_quality_label"] = confirmed_label
            out["target_quality_promoted_from"] = old_label
            out["target_quality_reason"] = append_note(
                out.get("target_quality_reason", ""),
                f"quality-goal reviewed {decision}; previous={old_label}",
            )
            counts["target_confirmed"] += 1
        elif decision in TARGET_LOSS_DECISIONS:
            out["target_quality_label"] = loss_label
            out["target_quality_promoted_from"] = old_label
            out["target_quality_reason"] = append_note(
                out.get("target_quality_reason", ""),
                f"quality-goal reviewed {decision}; previous={old_label}",
            )
            counts["target_loss"] += 1
        elif decision in TARGET_KEEP_DECISIONS:
            counts["target_kept_borderline"] += 1
        else:
            invalid.append({"file": page_text(key), "decision": decision, "stream": review.get("review_stream", "")})
        output.append(out)
    if invalid:
        raise ValueError(f"Unsupported target review labels: {invalid}")
    if missing_reviews:
        raise ValueError(f"Target review rows not found in target-quality CSV: {[page_text(key) for key in sorted(missing_reviews)]}")
    return output, dict(counts)


def apply_post125_reviews(
    rows: list[dict[str, str]],
    reviews: dict[tuple[str, str], dict[str, str]],
    allow_overwrite: bool,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    counts = Counter()
    missing_reviews = set(reviews)
    output: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []
    conflicts: list[dict[str, str]] = []
    for row in rows:
        key = page_key(row)
        review = reviews.get(key)
        out = dict(row)
        if review is None:
            output.append(out)
            continue
        missing_reviews.discard(key)
        decision = review_label(review)
        if decision not in POST125_ALLOWED_DECISIONS:
            invalid.append({"file": page_text(key), "decision": decision})
            output.append(out)
            continue
        existing = normalize(out.get("manual_label", ""))
        if existing and existing != decision and not allow_overwrite:
            conflicts.append({"file": page_text(key), "existing": existing, "review_decision": decision})
            output.append(out)
            continue
        out["manual_label"] = decision
        if review.get("manual_notes"):
            out["manual_notes"] = append_note(out.get("manual_notes", ""), review["manual_notes"])
        out["quality_goal_review_rank"] = review.get("review_rank", "")
        out["quality_goal_review_source"] = review.get("review_stream", "")
        counts["post125_labeled"] += 1
        counts[f"post125_{decision}"] += 1
        output.append(out)
    if invalid:
        raise ValueError(f"Unsupported post-125 review labels: {invalid}")
    if conflicts:
        raise ValueError(f"Post-125 source labels conflict with review labels: {conflicts}")
    if missing_reviews:
        raise ValueError(f"Post-125 review rows not found in label queue CSV: {[page_text(key) for key in sorted(missing_reviews)]}")
    return output, dict(counts)


def main() -> None:
    args = parse_args()
    review_paths = [Path(path) for path in args.review_csv]
    target_reviews, post125_reviews, skipped = collect_review_decisions(review_paths)

    target_rows = read_rows(Path(args.target_quality_csv))
    post125_rows = read_rows(Path(args.post125_csv))
    target_output, target_counts = apply_target_reviews(
        target_rows,
        target_reviews,
        args.target_confirmed_label,
        args.target_loss_label,
    )
    post125_output, post125_counts = apply_post125_reviews(post125_rows, post125_reviews, args.allow_overwrite)

    target_confirmed = int(target_counts.get("target_confirmed", 0))
    target_loss = int(target_counts.get("target_loss", 0))
    post125_labeled = int(post125_counts.get("post125_labeled", 0))
    if args.expect_target_confirmed >= 0 and target_confirmed != args.expect_target_confirmed:
        raise ValueError(f"Expected {args.expect_target_confirmed} target confirmations, got {target_confirmed}")
    if args.expect_target_loss >= 0 and target_loss != args.expect_target_loss:
        raise ValueError(f"Expected {args.expect_target_loss} target losses, got {target_loss}")
    if args.expect_post125_labeled >= 0 and post125_labeled != args.expect_post125_labeled:
        raise ValueError(f"Expected {args.expect_post125_labeled} post-125 labels, got {post125_labeled}")

    target_fields = ordered_fields(target_rows, target_output)
    post125_fields = ordered_fields(post125_rows, post125_output)
    write_csv(Path(args.output_target_quality_csv), target_output, target_fields)
    write_csv(Path(args.output_post125_csv), post125_output, post125_fields)

    summary = {
        "review_csvs": [str(path) for path in review_paths],
        "target_quality_csv": args.target_quality_csv,
        "post125_csv": args.post125_csv,
        "output_target_quality_csv": args.output_target_quality_csv,
        "output_post125_csv": args.output_post125_csv,
        "skipped_review_rows": dict(skipped),
        "target_review_rows": len(target_reviews),
        "post125_review_rows": len(post125_reviews),
        "target_counts": target_counts,
        "post125_counts": post125_counts,
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
