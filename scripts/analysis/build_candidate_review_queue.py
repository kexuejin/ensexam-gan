#!/usr/bin/env python3
"""Build deterministic page-level review queues from metric CSV pairs.

This is a generic counterpart to residual-delta-specific review builders. It
selects metric wins, metric losses/overerase risks, and high-activity pages from
baseline-vs-candidate metric CSVs, then writes page review rows plus a blank
label template for local visual review pack generation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REVIEW_FIELDS = [
    "split",
    "file",
    "bucket",
    "candidate",
    "source_input",
    "baseline_pred",
    "candidate_pred",
    "target",
    "review_pack",
    "notes",
]

LABEL_FIELDS = [
    "split",
    "file",
    "candidate",
    "bucket",
    "label",
    "flags",
    "reviewer",
    "review_date",
    "comment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME:BASELINE_METRICS:CANDIDATE_METRICS",
        help="May be repeated for each split to include in the review queue.",
    )
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--review-pack", default="")
    parser.add_argument("--max-win", type=int, default=8)
    parser.add_argument("--max-loss", type=int, default=8)
    parser.add_argument("--max-high-activity", type=int, default=8)
    return parser.parse_args()


def parse_split(value: str) -> tuple[str, Path, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Invalid --split {value!r}; expected NAME:BASELINE:CANDIDATE")
    return parts[0], Path(parts[1]), Path(parts[2])


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def label_path_for(image_path: str) -> str:
    if "/all_images/" not in image_path:
        raise ValueError(f"Cannot infer target path from {image_path}")
    return image_path.replace("/all_images/", "/all_labels/")


def fnum(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def metric_tuple(baseline: dict[str, str], candidate: dict[str, str]) -> tuple[float, float, float]:
    gain = fnum(baseline, "residual_ratio") - fnum(candidate, "residual_ratio")
    over_delta = fnum(candidate, "overerase_ratio") - fnum(baseline, "overerase_ratio")
    gate = fnum(candidate, "gate_ratio", 0.0)
    return gain, over_delta, gate


def add_unique(
    rows: list[dict[str, str]],
    seen: set[tuple[str, str, str]],
    row: dict[str, str],
) -> None:
    key = (row["split"], row["file"], row["bucket"])
    if key in seen:
        return
    seen.add(key)
    rows.append(row)


def make_review_row(
    split: str,
    baseline: dict[str, str],
    candidate: dict[str, str],
    candidate_name: str,
    bucket: str,
    review_pack: str,
    notes: str,
) -> dict[str, str]:
    return {
        "split": split,
        "file": candidate["file"],
        "bucket": bucket,
        "candidate": candidate_name,
        "source_input": candidate["image_path"],
        "baseline_pred": baseline["pred_path"],
        "candidate_pred": candidate["pred_path"],
        "target": label_path_for(candidate["image_path"]),
        "review_pack": review_pack,
        "notes": notes,
    }


def build_split_rows(
    split: str,
    baseline_metrics: Path,
    candidate_metrics: Path,
    candidate_name: str,
    review_pack: str,
    max_win: int,
    max_loss: int,
    max_high_activity: int,
) -> list[dict[str, str]]:
    baseline_by_file = {row["file"]: row for row in read_rows(baseline_metrics)}
    candidate_by_file = {row["file"]: row for row in read_rows(candidate_metrics)}
    metric_rows = []
    for file, candidate in candidate_by_file.items():
        baseline = baseline_by_file[file]
        gain, over_delta, gate = metric_tuple(baseline, candidate)
        metric_rows.append((file, baseline, candidate, gain, over_delta, gate))

    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    wins = sorted(
        [row for row in metric_rows if row[3] > 0.0 and row[4] <= 0.0],
        key=lambda row: row[3],
        reverse=True,
    )
    for file, baseline, candidate, gain, over_delta, gate in wins[:max_win]:
        add_unique(
            selected,
            seen,
            make_review_row(
                split,
                baseline,
                candidate,
                candidate_name,
                f"{candidate_name}_metric_win",
                review_pack,
                f"gain={gain:.9f}; over_delta={over_delta:.9f}; gate={gate:.9f}",
            ),
        )

    losses = sorted(
        [row for row in metric_rows if row[3] <= 0.0 or row[4] > 0.0],
        key=lambda row: (row[3], -row[4]),
    )
    for file, baseline, candidate, gain, over_delta, gate in losses[:max_loss]:
        add_unique(
            selected,
            seen,
            make_review_row(
                split,
                baseline,
                candidate,
                candidate_name,
                f"{candidate_name}_metric_loss_or_overrisk",
                review_pack,
                f"gain={gain:.9f}; over_delta={over_delta:.9f}; gate={gate:.9f}",
            ),
        )

    high_activity = sorted(metric_rows, key=lambda row: row[5], reverse=True)
    for file, baseline, candidate, gain, over_delta, gate in high_activity[:max_high_activity]:
        add_unique(
            selected,
            seen,
            make_review_row(
                split,
                baseline,
                candidate,
                candidate_name,
                f"{candidate_name}_high_activity_risk",
                review_pack,
                f"gain={gain:.9f}; over_delta={over_delta:.9f}; gate={gate:.9f}",
            ),
        )

    return selected


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    review_pack = args.review_pack or str(output_dir / "page_pack")
    rows: list[dict[str, str]] = []
    for split_arg in args.split:
        split, baseline_metrics, candidate_metrics = parse_split(split_arg)
        rows.extend(
            build_split_rows(
                split=split,
                baseline_metrics=baseline_metrics,
                candidate_metrics=candidate_metrics,
                candidate_name=args.candidate_name,
                review_pack=review_pack,
                max_win=args.max_win,
                max_loss=args.max_loss,
                max_high_activity=args.max_high_activity,
            )
        )

    write_csv(output_dir / "review-pages.csv", REVIEW_FIELDS, rows)
    label_rows = [
        {
            "split": row["split"],
            "file": row["file"],
            "candidate": row["candidate"],
            "bucket": row["bucket"],
            "label": "",
            "flags": "",
            "reviewer": "",
            "review_date": "",
            "comment": "",
        }
        for row in rows
    ]
    write_csv(output_dir / "labels-template.csv", LABEL_FIELDS, label_rows)

    split_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        bucket_counts[row["bucket"]] = bucket_counts.get(row["bucket"], 0) + 1
    print(f"rows={len(rows)} splits={split_counts} buckets={bucket_counts}")
    print(f"review_csv={output_dir / 'review-pages.csv'}")
    print(f"labels_template={output_dir / 'labels-template.csv'}")


if __name__ == "__main__":
    main()
