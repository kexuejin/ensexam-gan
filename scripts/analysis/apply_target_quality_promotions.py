#!/usr/bin/env python3
"""Apply reviewed target-quality promotions to a quality CSV.

This keeps target-aware scoring immutable and records promotions as a
reproducible overlay. The intended first use is promoting high-confidence
borderline rows from triage_target_quality_borderline.py after local review.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


QUALITY_LABELS = ("clear_win", "slight_win", "noop", "borderline", "slight_loss", "clear_loss")
LOSS_LABELS = {"clear_loss", "slight_loss"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-csv", required=True, help="Source target-quality CSV.")
    parser.add_argument("--triage-csv", required=True, help="Output from triage_target_quality_borderline.py.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument(
        "--promote-bucket",
        action="append",
        default=[],
        help="Triage bucket to promote. May be repeated. Defaults to auto_win_candidate.",
    )
    parser.add_argument("--promoted-label", default="slight_win", choices=QUALITY_LABELS)
    parser.add_argument("--require-local-verdict", default="accept")
    parser.add_argument("--require-source-label", default="borderline", choices=QUALITY_LABELS)
    parser.add_argument("--promotion-note", default="reviewed promotion from target-quality triage")
    parser.add_argument("--expect-promoted", type=int, default=-1, help="Fail unless exactly this many rows promote.")
    parser.add_argument("--allow-skipped", action="store_true", help="Allow matched triage rows that fail promotion gates.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, str]:
    return row["split"], row["file"]


def label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row.get("target_quality_label", "") for row in rows)
    return {label: counts.get(label, 0) for label in QUALITY_LABELS}


def main() -> None:
    args = parse_args()
    promote_buckets = set(args.promote_bucket or ["auto_win_candidate"])
    quality_rows = read_rows(Path(args.quality_csv))
    triage_rows = read_rows(Path(args.triage_csv))
    triage_by_key = {key(row): row for row in triage_rows if row.get("triage_bucket") in promote_buckets}

    before_counts = label_counts(quality_rows)
    promoted: list[str] = []
    skipped: list[dict[str, str]] = []
    output_rows: list[dict[str, str]] = []
    for row in quality_rows:
        row_key = key(row)
        triage = triage_by_key.get(row_key)
        out = dict(row)
        if triage is None:
            output_rows.append(out)
            continue

        source_label = row.get("target_quality_label", "")
        local_verdict = row.get("local_verdict", "")
        if source_label != args.require_source_label or local_verdict != args.require_local_verdict:
            skipped.append({
                "file": f"{row_key[0]}/{row_key[1]}",
                "source_label": source_label,
                "local_verdict": local_verdict,
                "triage_bucket": triage.get("triage_bucket", ""),
            })
            output_rows.append(out)
            continue

        out["target_quality_label"] = args.promoted_label
        reason = out.get("target_quality_reason", "")
        out["target_quality_reason"] = (
            f"{args.promotion_note}; bucket={triage.get('triage_bucket', '')}; previous={source_label}; {reason}"
        )
        out["target_quality_promoted_from"] = source_label
        out["target_quality_promotion_bucket"] = triage.get("triage_bucket", "")
        promoted.append(f"{row_key[0]}/{row_key[1]}")
        output_rows.append(out)

    after_counts = label_counts(output_rows)
    before_losses = sum(before_counts[label] for label in LOSS_LABELS)
    after_losses = sum(after_counts[label] for label in LOSS_LABELS)
    if skipped and not args.allow_skipped:
        raise ValueError(f"Matched triage rows failed promotion gates: {skipped}")
    if args.expect_promoted >= 0 and len(promoted) != args.expect_promoted:
        raise ValueError(f"Expected {args.expect_promoted} promotions, got {len(promoted)}")
    if after_losses != before_losses:
        raise ValueError(f"Promotion changed loss count: before={before_losses} after={after_losses}")

    output_csv = Path(args.output_csv)
    write_csv(output_csv, output_rows)
    summary = {
        "quality_csv": args.quality_csv,
        "triage_csv": args.triage_csv,
        "output_csv": str(output_csv),
        "promote_buckets": sorted(promote_buckets),
        "promoted_label": args.promoted_label,
        "promoted_count": len(promoted),
        "promoted_files": promoted,
        "skipped": skipped,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "before_losses": before_losses,
        "after_losses": after_losses,
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
