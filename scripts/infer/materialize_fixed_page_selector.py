#!/usr/bin/env python3
"""Materialize baseline/candidate predictions selected by a fixed page rule or selected-pages CSV."""

from __future__ import annotations

import argparse
import csv
import operator
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
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--selector-rule", help="Feature rule selecting candidate pages.")
    selector.add_argument("--selected-csv", help="CSV with split,file rows selecting candidate pages.")
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


def page_key(split_name: str, file: str) -> tuple[str, str]:
    return split_name, file


def selected_row_key(row: dict[str, str]) -> tuple[str, str]:
    split = row.get("split", "")
    file = row.get("file", "")
    if not split and row.get("page", "").count("/") == 1:
        split, file = row["page"].split("/", 1)
    elif not split and file.count("/") == 1:
        split, file = file.split("/", 1)
    if not split or not file:
        raise ValueError(f"Selected CSV row needs split/file or page: {row}")
    return page_key(split, file)


def read_selected_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_rows(path):
        key = selected_row_key(row)
        if key in rows:
            split, file = key
            raise ValueError(f"Selected CSV has duplicate page: {split}/{file}")
        rows[key] = row
    return rows


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def selected_metric_safe(row: dict[str, str], gain: float, over_delta: float) -> bool:
    if row.get("metric_safe", ""):
        return parse_bool(row["metric_safe"])
    return gain > 0 and over_delta <= 0


OPS = {
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
    "==": operator.eq,
}


def selector_hit(row: dict[str, str], rule: str) -> bool:
    for condition in (part.strip() for part in rule.split(" AND ")):
        parts = condition.split()
        if len(parts) != 3:
            raise ValueError(f"Unsupported selector condition: {condition!r}")
        feature, op_text, threshold_text = parts
        if op_text not in OPS:
            raise ValueError(f"Unsupported selector operator: {op_text!r}")
        if feature not in row:
            raise KeyError(f"Selector feature {feature!r} not found")
        if not OPS[op_text](float(row[feature]), float(threshold_text)):
            return False
    return True


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
    selector_rule: str | None,
    selected_rows: dict[tuple[str, str], dict[str, str]] | None,
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
        selected_row = selected_rows.get(page_key(split_name, file)) if selected_rows is not None else None
        if selected_rows is not None:
            selected = selected_row is not None
        elif selector_rule is not None:
            selected = selector_hit(feature, selector_rule)
        else:
            raise ValueError("Either selector_rule or selected_rows is required")

        gain = float((selected_row or feature)["gain"])
        over_delta = float((selected_row or feature)["over_delta"])
        if selected and selected_row is not None and selected_row.get("candidate_pred"):
            source_path = Path(selected_row["candidate_pred"])
        else:
            source_path = Path(candidate["pred_path"] if selected else baseline["pred_path"])
        output_name = f"{Path(file).stem}.png"
        output_path = split_pred_dir / output_name
        copy_prediction(source_path, output_path)
        if selected_row is not None:
            safe_metric_win = selected_metric_safe(selected_row, gain, over_delta)
        else:
            safe_metric_win = gain > 0 and over_delta <= 0
        rows.append({
            "split": split_name,
            "file": file,
            "selected_candidate": int(selected),
            "selected_source": "candidate" if selected else "baseline",
            "safe_metric_win": int(safe_metric_win),
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
    selected_rows = read_selected_rows(Path(args.selected_csv)) if args.selected_csv else None
    all_rows: list[dict[str, object]] = []
    for split in args.split:
        all_rows.extend(materialize_split(*parse_split(split), args.selector_rule, selected_rows, output_dir))
    if args.selected_csv:
        seen_selected = {page_key(str(row["split"]), str(row["file"])) for row in all_rows if int(row["selected_candidate"])}
        missing = sorted(set(selected_rows or {}) - seen_selected)
        if missing:
            missing_text = " ".join(f"{split}/{file}" for split, file in missing[:20])
            raise ValueError(f"Selected CSV contains pages absent from split metrics: {missing_text}")
    write_csv(output_dir / "selection.csv", all_rows)

    selected = sum(int(row["selected_candidate"]) for row in all_rows)
    selected_wins = sum(int(row["selected_candidate"]) and int(row["safe_metric_win"]) for row in all_rows)
    selected_losses = selected - selected_wins
    print(f"rows={len(all_rows)} selected={selected} selected_wins={selected_wins} selected_losses={selected_losses}")
    print(f"selection_csv={output_dir / 'selection.csv'}")


if __name__ == "__main__":
    main()
