#!/usr/bin/env python3
"""Summarize crop-level product-quality labels by candidate and bucket."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", required=True)
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


def metric_row(candidate: str, bucket: str, counts: Counter[str]) -> dict[str, object]:
    crops = sum(counts.values())
    labeled = crops - counts["unlabeled"]
    wins = counts["clear_win"] + counts["slight_win"]
    losses = counts["slight_loss"] + counts["clear_loss"]
    row: dict[str, object] = {
        "candidate": candidate,
        "bucket": bucket,
        "crops": crops,
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
    rows = read_rows(Path(args.labels_csv))

    by_candidate: dict[str, Counter[str]] = defaultdict(Counter)
    by_bucket: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    by_page: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    pending: list[dict[str, object]] = []

    for row in rows:
        label = normalized_label(row.get("label", ""))
        candidate = row["candidate"]
        bucket = row["bucket"]
        page_key = (candidate, bucket, row["split"], row["file"])
        by_candidate[candidate][label] += 1
        by_bucket[(candidate, bucket)][label] += 1
        by_page[page_key][label] += 1
        if label == "unlabeled":
            pending.append(dict(row))

    candidate_rows = [metric_row(candidate, "ALL", counts) for candidate, counts in sorted(by_candidate.items())]
    bucket_rows = [
        metric_row(candidate, bucket, counts)
        for (candidate, bucket), counts in sorted(by_bucket.items())
    ]
    page_rows = [
        {
            **metric_row(candidate, bucket, counts),
            "split": split,
            "file": file,
        }
        for (candidate, bucket, split, file), counts in sorted(by_page.items())
    ]

    write_csv(output_dir / "candidate_summary.csv", candidate_rows)
    write_csv(output_dir / "bucket_summary.csv", bucket_rows)
    write_csv(output_dir / "page_summary.csv", page_rows)
    write_csv(output_dir / "pending_crop_labels.csv", pending)

    print(f"labels={len(rows)}")
    print(f"pending={len(pending)}")
    print(f"candidate_summary={output_dir / 'candidate_summary.csv'}")
    print(f"bucket_summary={output_dir / 'bucket_summary.csv'}")
    print(f"page_summary={output_dir / 'page_summary.csv'}")
    print(f"pending_crop_labels={output_dir / 'pending_crop_labels.csv'}")


if __name__ == "__main__":
    main()
