#!/usr/bin/env python3
"""Diagnose selected metric-loss pages with local target-distance deltas."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--candidate-delta-threshold", type=float, default=2.0)
    parser.add_argument("--baseline-edit-threshold", type=float, default=12.0)
    parser.add_argument("--hurt-threshold", type=float, default=1.0)
    parser.add_argument("--help-threshold", type=float, default=1.0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_bgr(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else 0.0


def diagnose_row(row: dict[str, str], args: argparse.Namespace) -> dict[str, str | int | float]:
    source = read_bgr(row["source_input"])
    baseline = resize_like(read_bgr(row["baseline_pred"]), source)
    candidate = resize_like(read_bgr(row["candidate_pred"]), source)
    target = resize_like(read_bgr(row["target"]), source)

    candidate_delta = cv2.absdiff(candidate, baseline).mean(axis=2)
    baseline_edit = cv2.absdiff(baseline, source).mean(axis=2)
    active = (candidate_delta >= args.candidate_delta_threshold) & (baseline_edit >= args.baseline_edit_threshold)

    baseline_target = cv2.absdiff(baseline, target).mean(axis=2)
    candidate_target = cv2.absdiff(candidate, target).mean(axis=2)
    improvement = baseline_target.astype(np.float32) - candidate_target.astype(np.float32)
    active_improvement = improvement[active]

    help_ratio = float((active_improvement > args.help_threshold).mean()) if active_improvement.size else 0.0
    hurt_ratio = float((active_improvement < -args.hurt_threshold).mean()) if active_improvement.size else 0.0
    neutral_ratio = float((np.abs(active_improvement) <= args.help_threshold).mean()) if active_improvement.size else 0.0
    if hurt_ratio >= 0.25:
        verdict = "likely_true_loss"
    elif hurt_ratio <= 0.10 and help_ratio >= 0.20:
        verdict = "review_accept_candidate"
    else:
        verdict = "ambiguous_review"

    return {
        **row,
        "diag_active_px": int(active.sum()),
        "diag_active_improve_mean": float(active_improvement.mean()) if active_improvement.size else 0.0,
        "diag_active_improve_p25": percentile(active_improvement, 25),
        "diag_active_improve_p50": percentile(active_improvement, 50),
        "diag_active_improve_p75": percentile(active_improvement, 75),
        "diag_active_help_ratio": help_ratio,
        "diag_active_hurt_ratio": hurt_ratio,
        "diag_active_neutral_ratio": neutral_ratio,
        "diag_candidate_delta_mean_active": float(candidate_delta[active].mean()) if active.any() else 0.0,
        "diag_candidate_delta_p95_active": percentile(candidate_delta[active], 95),
        "diag_heuristic_verdict": verdict,
    }


def main() -> None:
    args = parse_args()
    rows = [diagnose_row(row, args) for row in read_rows(Path(args.review_csv))]
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        verdict = str(row["diag_heuristic_verdict"])
        counts[verdict] = counts.get(verdict, 0) + 1
    print(f"rows={len(rows)} verdicts={counts} output_csv={output}")


if __name__ == "__main__":
    main()
