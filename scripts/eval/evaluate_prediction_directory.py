#!/usr/bin/env python3
"""Evaluate an existing prediction directory against SCUT target images.

This is for post-processing/materialization outputs that already wrote
``pred/*.png`` files and should be scored without rerunning a model.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_hardcase_worst_pages import (  # noqa: E402
    compute_residual_metrics,
    ensure_same_size,
    label_path_for,
    read_bgr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--change-threshold", type=int, default=12)
    parser.add_argument("--eval-threshold", type=int, default=12)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prediction_path(pred_dir: Path, file: str) -> Path:
    stem = Path(file).stem
    for name in (f"{stem}.png", f"{stem}.clean.png", file):
        candidate = pred_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No prediction found for {file} in {pred_dir}")


def mean(rows: list[dict[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def evaluate_row(row: dict[str, str], pred_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    file = row["file"]
    image_path = Path(row["image_path"])
    pred_path = prediction_path(pred_dir, file)
    input_bgr = read_bgr(image_path)
    label_bgr = ensure_same_size(read_bgr(label_path_for(image_path)), input_bgr)
    pred_bgr = ensure_same_size(read_bgr(pred_path), input_bgr)
    metrics = compute_residual_metrics(
        input_bgr,
        label_bgr,
        pred_bgr,
        change_threshold=args.change_threshold,
        eval_threshold=args.eval_threshold,
    )
    baseline_residual = float(row.get("residual_ratio", 0.0))
    baseline_overerase = float(row.get("overerase_ratio", 0.0))
    return {
        "file": file,
        "image_path": str(image_path),
        "pred_path": str(pred_path),
        "baseline_pred_path": row.get("pred_path", ""),
        "baseline_residual_ratio": baseline_residual,
        "baseline_overerase_ratio": baseline_overerase,
        **metrics,
        "delta_residual_ratio": baseline_residual - float(metrics["residual_ratio"]),
        "delta_overerase_ratio": float(metrics["overerase_ratio"]) - baseline_overerase,
    }


def main() -> None:
    args = parse_args()
    rows = [
        evaluate_row(row, Path(args.pred_dir), args)
        for row in read_rows(Path(args.baseline_metrics))
    ]
    if not rows:
        raise ValueError("baseline metrics CSV has no rows")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    baseline_residual = mean(rows, "baseline_residual_ratio")
    baseline_overerase = mean(rows, "baseline_overerase_ratio")
    residual = mean(rows, "residual_ratio")
    overerase = mean(rows, "overerase_ratio")
    print(
        f"pages={len(rows)} "
        f"baseline_residual={baseline_residual:.6f} residual={residual:.6f} "
        f"residual_gain={baseline_residual - residual:.6f} "
        f"baseline_overerase={baseline_overerase:.6f} overerase={overerase:.6f} "
        f"overerase_delta={overerase - baseline_overerase:.6f}"
    )
    print(f"output_csv={output_csv}")


if __name__ == "__main__":
    main()
