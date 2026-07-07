#!/usr/bin/env python3
"""Compare selected candidate predictions against baseline using local targets.

This is a deterministic review aid for page-level selectors. It does not replace
human visual review, but it catches obvious help/hurt/risk patterns without
uploading review images.
"""

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
    parser.add_argument("--candidate-change-threshold", type=float, default=2.0)
    parser.add_argument("--meaningful-improvement-threshold", type=float, default=2.0)
    parser.add_argument("--baseline-ok-threshold", type=float, default=3.0)
    parser.add_argument("--risk-change-threshold", type=float, default=4.0)
    parser.add_argument("--accept-help-hurt-ratio", type=float, default=1.5)
    parser.add_argument("--max-risk-changed-ratio", type=float, default=0.35)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_bgr(path: str) -> np.ndarray:
    image = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int16)


def verdict(
    help_px: int,
    hurt_px: int,
    risk_changed_ratio: float,
    accept_help_hurt_ratio: float,
    max_risk_changed_ratio: float,
) -> str:
    if help_px > hurt_px * accept_help_hurt_ratio and risk_changed_ratio < max_risk_changed_ratio:
        return "accept"
    if help_px > hurt_px:
        return "review"
    return "reject"


def compare_row(row: dict[str, str], args: argparse.Namespace) -> dict[str, str | int | float]:
    source = read_bgr(row["source_input"])
    baseline = read_bgr(row["baseline_pred"])
    candidate = resize_like(read_bgr(row["candidate_pred"]), baseline)
    target = resize_like(read_bgr(row["target"]), baseline)
    source = resize_like(source, baseline)

    source_gray = gray(source)
    baseline_gray = gray(baseline)
    candidate_gray = gray(candidate)
    target_gray = gray(target)

    baseline_error = np.abs(baseline_gray - target_gray)
    candidate_error = np.abs(candidate_gray - target_gray)
    improvement = baseline_error.astype(np.int32) - candidate_error.astype(np.int32)

    changed = np.abs(candidate_gray - baseline_gray) > args.candidate_change_threshold
    target_changed = np.abs(target_gray - baseline_gray) > args.candidate_change_threshold
    active = changed | target_changed
    if not active.any():
        active = np.ones_like(changed, dtype=bool)

    help_px = int((changed & (improvement > args.meaningful_improvement_threshold)).sum())
    hurt_px = int((changed & (improvement < -args.meaningful_improvement_threshold)).sum())
    neutral_px = int((changed & (np.abs(improvement) <= args.meaningful_improvement_threshold)).sum())
    changed_px = int(changed.sum())

    baseline_already_ok = baseline_error <= args.baseline_ok_threshold
    risk_px = int(
        (
            changed
            & baseline_already_ok
            & (np.abs(candidate_gray - baseline_gray) > args.risk_change_threshold)
        ).sum()
    )
    risk_changed_ratio = risk_px / max(changed_px, 1)

    residual_mask = (target_gray - baseline_gray) > 8
    residual_help_px = int(
        (
            residual_mask
            & changed
            & (candidate_gray > baseline_gray)
            & (np.abs(candidate_gray - target_gray) < np.abs(baseline_gray - target_gray))
        ).sum()
    )
    residual_hurt_px = int(
        (
            residual_mask
            & changed
            & (np.abs(candidate_gray - target_gray) > np.abs(baseline_gray - target_gray))
        ).sum()
    )

    source_changed_error = np.abs(source_gray - target_gray)
    source_active_error = float(source_changed_error[active].mean()) if active.any() else 0.0

    return {
        **row,
        "changed_ratio": changed_px / changed.size,
        "active_mean_source_err": source_active_error,
        "active_mean_baseline_err": float(baseline_error[active].mean()),
        "active_mean_candidate_err": float(candidate_error[active].mean()),
        "active_mean_gain": float(improvement[active].mean()),
        "help_px": help_px,
        "hurt_px": hurt_px,
        "neutral_px": neutral_px,
        "help_hurt_ratio": (help_px + 1) / (hurt_px + 1),
        "risk_px": risk_px,
        "risk_changed_ratio": risk_changed_ratio,
        "residual_help_px": residual_help_px,
        "residual_hurt_px": residual_hurt_px,
        "residual_help_hurt_ratio": (residual_help_px + 1) / (residual_hurt_px + 1),
        "local_verdict": verdict(
            help_px,
            hurt_px,
            risk_changed_ratio,
            args.accept_help_hurt_ratio,
            args.max_risk_changed_ratio,
        ),
    }


def main() -> None:
    args = parse_args()
    rows = [compare_row(row, args) for row in read_rows(Path(args.review_csv))]
    if not rows:
        raise ValueError("review CSV has no rows")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["local_verdict"])
        counts[label] = counts.get(label, 0) + 1
    print(f"rows={len(rows)} verdicts={counts} output_csv={output_csv}")


if __name__ == "__main__":
    main()
