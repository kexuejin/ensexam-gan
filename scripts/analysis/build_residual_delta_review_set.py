#!/usr/bin/env python3
"""Build page-level manual review rows for residual-delta cleanup candidates.

This script turns metric/feature evidence into a deterministic labeling queue.
It does not create visual labels. The output is intended for local contact-sheet
generation followed by human labeling.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME:BASELINE_METRICS:CANDIDATE_METRICS:FEATURES_CSV",
        help="May be repeated. FEATURES_CSV is from analyze_residual_delta_selector_features.py.",
    )
    parser.add_argument("--candidate-name", default="residual_delta_bias3_scale008")
    parser.add_argument("--output-review-csv", required=True)
    parser.add_argument("--output-labels-template", required=True)
    parser.add_argument("--review-pack", default="")
    parser.add_argument("--max-win", type=int, default=24)
    parser.add_argument("--max-loss", type=int, default=24)
    parser.add_argument("--max-selector", type=int, default=24)
    parser.add_argument("--max-high-activity", type=int, default=24)
    parser.add_argument("--selector-rule", default="active_gray_p25 >= 111.6 AND candidate_delta_max <= 200.133333333")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_split(value: str) -> tuple[str, Path, Path, Path]:
    parts = value.split(":", 3)
    if len(parts) != 4 or not all(parts):
        raise ValueError(f"Invalid --split {value!r}; expected NAME:BASELINE:CANDIDATE:FEATURES")
    return parts[0], Path(parts[1]), Path(parts[2]), Path(parts[3])


def label_path_for(image_path: str) -> str:
    if "/all_images/" not in image_path:
        raise ValueError(f"Cannot infer target path from {image_path}")
    return image_path.replace("/all_images/", "/all_labels/")


def fnum(row: dict[str, str], key: str) -> float:
    return float(row[key])


def selector_hit(row: dict[str, str], rule_name: str) -> bool:
    if rule_name != "active_gray_p25 >= 111.6 AND candidate_delta_max <= 200.133333333":
        raise ValueError(f"Unsupported selector rule {rule_name!r}")
    return fnum(row, "active_gray_p25") >= 111.6 and fnum(row, "candidate_delta_max") <= 200.133333333


def review_row(
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


def add_unique(
    rows: list[dict[str, str]],
    seen: set[tuple[str, str, str, str]],
    row: dict[str, str],
) -> None:
    key = (row["split"], row["file"], row["candidate"], row["bucket"])
    if key in seen:
        return
    seen.add(key)
    rows.append(row)


def split_rows(
    split: str,
    baseline_metrics: Path,
    candidate_metrics: Path,
    features_csv: Path,
    candidate_name: str,
    review_pack: str,
    selector_rule: str,
    max_win: int,
    max_loss: int,
    max_selector: int,
    max_high_activity: int,
) -> list[dict[str, str]]:
    baseline_by_file = {row["file"]: row for row in read_rows(baseline_metrics)}
    candidate_by_file = {row["file"]: row for row in read_rows(candidate_metrics)}
    features = [
        row for row in read_rows(features_csv)
        if row.get("split", split) in {split, "main", ""}
    ]
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def paired(feature: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
        file = feature["file"]
        return baseline_by_file[file], candidate_by_file[file]

    wins = sorted(
        [row for row in features if fnum(row, "gain") > 0 and fnum(row, "over_delta") <= 0],
        key=lambda row: fnum(row, "gain"),
        reverse=True,
    )
    for feature in wins[:max_win]:
        baseline, candidate = paired(feature)
        add_unique(
            rows,
            seen,
            review_row(
                split,
                baseline,
                candidate,
                candidate_name,
                "residual_delta_metric_win",
                review_pack,
                f"metric win gain={fnum(feature, 'gain'):.9f}; over_delta={fnum(feature, 'over_delta'):.9f}",
            ),
        )

    losses = sorted(
        [row for row in features if fnum(row, "gain") <= 0 or fnum(row, "over_delta") > 0],
        key=lambda row: (fnum(row, "gain"), -fnum(row, "over_delta")),
    )
    for feature in losses[:max_loss]:
        baseline, candidate = paired(feature)
        add_unique(
            rows,
            seen,
            review_row(
                split,
                baseline,
                candidate,
                candidate_name,
                "residual_delta_metric_loss",
                review_pack,
                f"metric loss gain={fnum(feature, 'gain'):.9f}; over_delta={fnum(feature, 'over_delta'):.9f}",
            ),
        )

    selector_rows = sorted(
        [row for row in features if selector_hit(row, selector_rule)],
        key=lambda row: fnum(row, "gain"),
        reverse=True,
    )
    for feature in selector_rows[:max_selector]:
        baseline, candidate = paired(feature)
        add_unique(
            rows,
            seen,
            review_row(
                split,
                baseline,
                candidate,
                candidate_name,
                "residual_delta_joint_selector",
                review_pack,
                f"joint safe selector hit; gain={fnum(feature, 'gain'):.9f}; rule={selector_rule}",
            ),
        )

    high_activity = sorted(
        features,
        key=lambda row: (fnum(row, "gate_ratio"), fnum(row, "candidate_delta_max")),
        reverse=True,
    )
    for feature in high_activity[:max_high_activity]:
        baseline, candidate = paired(feature)
        add_unique(
            rows,
            seen,
            review_row(
                split,
                baseline,
                candidate,
                candidate_name,
                "residual_delta_high_activity_risk",
                review_pack,
                (
                    f"high activity gate={fnum(feature, 'gate_ratio'):.9f}; "
                    f"candidate_delta_max={fnum(feature, 'candidate_delta_max'):.3f}; "
                    f"gain={fnum(feature, 'gain'):.9f}"
                ),
            ),
        )

    return rows


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    for split in args.split:
        split_name, baseline_metrics, candidate_metrics, features_csv = parse_split(split)
        rows.extend(
            split_rows(
                split=split_name,
                baseline_metrics=baseline_metrics,
                candidate_metrics=candidate_metrics,
                features_csv=features_csv,
                candidate_name=args.candidate_name,
                review_pack=args.review_pack,
                selector_rule=args.selector_rule,
                max_win=args.max_win,
                max_loss=args.max_loss,
                max_selector=args.max_selector,
                max_high_activity=args.max_high_activity,
            )
        )

    write_csv(Path(args.output_review_csv), REVIEW_FIELDS, rows)
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
    write_csv(Path(args.output_labels_template), LABEL_FIELDS, label_rows)

    bucket_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for row in rows:
        bucket_counts[row["bucket"]] = bucket_counts.get(row["bucket"], 0) + 1
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    print(f"rows={len(rows)} splits={split_counts} buckets={bucket_counts}")
    print(f"review_csv={args.output_review_csv}")
    print(f"labels_template={args.output_labels_template}")


if __name__ == "__main__":
    main()
