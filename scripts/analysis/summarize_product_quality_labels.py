#!/usr/bin/env python3
"""Summarize page-level product-quality labels by candidate and bucket."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


VALID_LABELS = {
    "",
    "clear_win",
    "slight_win",
    "noop",
    "slight_loss",
    "clear_loss",
}

LABEL_COLUMNS = ["clear_win", "slight_win", "noop", "slight_loss", "clear_loss", "unlabeled"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", default="docs/product-quality-labels.csv")
    parser.add_argument("--review-csv", default="docs/product-quality-review-pages.csv")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalized_label(value: str) -> str:
    label = value.strip()
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label {label!r}; expected one of {sorted(VALID_LABELS)}")
    return label or "unlabeled"


def keyed(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], dict[str, str]]:
    out = {}
    for row in rows:
        key = (row["split"], row["file"], row["candidate"], row["bucket"])
        out[key] = row
    return out


def metric_row(candidate: str, bucket: str, counts: Counter[str]) -> dict[str, object]:
    pages = sum(counts.values())
    wins = counts["clear_win"] + counts["slight_win"]
    losses = counts["slight_loss"] + counts["clear_loss"]
    labeled = pages - counts["unlabeled"]
    row: dict[str, object] = {
        "candidate": candidate,
        "bucket": bucket,
        "pages": pages,
        "labeled": labeled,
        "wins": wins,
        "losses": losses,
        "net_wins": wins - losses,
        "win_rate": wins / labeled if labeled else "",
        "clear_loss_rate": counts["clear_loss"] / labeled if labeled else "",
    }
    for label in LABEL_COLUMNS:
        row[label] = counts[label]
    return row


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    labels = read_rows(Path(args.labels_csv))
    review_by_key = keyed(read_rows(Path(args.review_csv)))

    by_candidate: dict[str, Counter[str]] = defaultdict(Counter)
    by_bucket: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    pending: list[dict[str, object]] = []

    for row in labels:
        key = (row["split"], row["file"], row["candidate"], row["bucket"])
        review = review_by_key.get(key, {})
        label = normalized_label(row.get("label", ""))
        candidate = row["candidate"]
        bucket = row["bucket"]
        by_candidate[candidate][label] += 1
        by_bucket[(candidate, bucket)][label] += 1
        if label == "unlabeled":
            pending_row: dict[str, object] = dict(row)
            pending_row["source_input"] = review.get("source_input", "")
            pending_row["baseline_pred"] = review.get("baseline_pred", "")
            pending_row["candidate_pred"] = review.get("candidate_pred", "")
            pending_row["target"] = review.get("target", "")
            pending_row["review_pack"] = review.get("review_pack", "")
            pending.append(pending_row)

    candidate_rows = [metric_row(candidate, "ALL", counts) for candidate, counts in sorted(by_candidate.items())]
    bucket_rows = [
        metric_row(candidate, bucket, counts)
        for (candidate, bucket), counts in sorted(by_bucket.items())
    ]

    write_csv(output_dir / "candidate_summary.csv", candidate_rows)
    write_csv(output_dir / "bucket_summary.csv", bucket_rows)
    write_csv(output_dir / "pending_labels.csv", pending)

    print(f"labels={len(labels)}")
    print(f"pending={len(pending)}")
    print(f"candidate_summary={output_dir / 'candidate_summary.csv'}")
    print(f"bucket_summary={output_dir / 'bucket_summary.csv'}")
    print(f"pending_labels={output_dir / 'pending_labels.csv'}")


if __name__ == "__main__":
    main()
