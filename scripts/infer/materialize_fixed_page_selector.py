#!/usr/bin/env python3
"""Materialize baseline/candidate predictions selected by a fixed page rule."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME:FEATURES_CSV:BASELINE_METRICS:CANDIDATE_METRICS",
        help="May be repeated. Outputs are written under <output-dir>/<NAME>/pred.",
    )
    parser.add_argument("--selector-rule", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_split(value: str) -> tuple[str, Path, Path, Path]:
    parts = value.split(":", 3)
    if len(parts) != 4 or not all(parts):
        raise ValueError(f"Invalid --split {value!r}; expected NAME:FEATURES:BASELINE:CANDIDATE")
    return parts[0], Path(parts[1]), Path(parts[2]), Path(parts[3])


def selector_hit(row: dict[str, str], rule: str) -> bool:
    if rule == "active_gray_p25 >= 123":
        return float(row["active_gray_p25"]) >= 123.0
    if rule == "active_gray_p25 >= 123 AND active_baseline_edit_p95 <= 149 AND candidate_delta_mean >= 0.0182428157494":
        return (
            float(row["active_gray_p25"]) >= 123.0
            and float(row["active_baseline_edit_p95"]) <= 149.0
            and float(row["candidate_delta_mean"]) >= 0.0182428157494
        )
    if rule == "active_gray_p25 >= 111.6 AND candidate_delta_max <= 200.133333333":
        return float(row["active_gray_p25"]) >= 111.6 and float(row["candidate_delta_max"]) <= 200.133333333
    raise ValueError(f"Unsupported selector rule: {rule}")


def copy_prediction(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def materialize_split(
    split_name: str,
    features_csv: Path,
    baseline_metrics: Path,
    candidate_metrics: Path,
    selector_rule: str,
    output_dir: Path,
) -> list[dict[str, object]]:
    features_by_file = {row["file"]: row for row in read_rows(features_csv)}
    baseline_rows = read_rows(baseline_metrics)
    candidate_by_file = {row["file"]: row for row in read_rows(candidate_metrics)}
    split_pred_dir = output_dir / split_name / "pred"
    rows: list[dict[str, object]] = []

    for baseline in baseline_rows:
        file = baseline["file"]
        feature = features_by_file[file]
        candidate = candidate_by_file[file]
        selected = selector_hit(feature, selector_rule)
        source_path = Path(candidate["pred_path"] if selected else baseline["pred_path"])
        output_name = f"{Path(file).stem}.png"
        output_path = split_pred_dir / output_name
        copy_prediction(source_path, output_path)
        gain = float(feature["gain"])
        over_delta = float(feature["over_delta"])
        rows.append({
            "split": split_name,
            "file": file,
            "selected_candidate": int(selected),
            "selected_source": "candidate" if selected else "baseline",
            "safe_metric_win": int(gain > 0 and over_delta <= 0),
            "gain": gain,
            "over_delta": over_delta,
            "source_pred_path": str(source_path),
            "output_pred_path": str(output_path),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    all_rows: list[dict[str, object]] = []
    for split in args.split:
        all_rows.extend(materialize_split(*parse_split(split), args.selector_rule, output_dir))
    write_csv(output_dir / "selection.csv", all_rows)

    selected = sum(int(row["selected_candidate"]) for row in all_rows)
    selected_wins = sum(int(row["selected_candidate"]) and int(row["safe_metric_win"]) for row in all_rows)
    selected_losses = selected - selected_wins
    print(f"rows={len(all_rows)} selected={selected} selected_wins={selected_wins} selected_losses={selected_losses}")
    print(f"selection_csv={output_dir / 'selection.csv'}")


if __name__ == "__main__":
    main()
