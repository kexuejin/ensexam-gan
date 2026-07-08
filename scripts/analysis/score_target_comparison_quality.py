#!/usr/bin/env python3
"""Score candidate page quality against local targets.

This target-aware scorer compares baseline and candidate predictions against the
available target image. It is intended to move most page-quality decisions into
local, reproducible code while leaving only borderline cases for human review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


QUALITY_LABELS = ("clear_win", "slight_win", "noop", "borderline", "slight_loss", "clear_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--change-threshold", type=float, default=2.0)
    parser.add_argument("--meaningful-threshold", type=float, default=2.0)
    parser.add_argument("--target-dark-threshold", type=float, default=210.0)
    parser.add_argument("--target-residual-threshold", type=float, default=8.0)
    parser.add_argument("--clear-win-help-hurt", type=float, default=1.8)
    parser.add_argument("--slight-win-help-hurt", type=float, default=1.25)
    parser.add_argument("--max-clear-win-overerase-ratio", type=float, default=0.05)
    parser.add_argument("--max-slight-win-overerase-ratio", type=float, default=0.12)
    parser.add_argument("--max-clear-win-damage-ratio", type=float, default=0.15)
    parser.add_argument("--max-slight-win-damage-ratio", type=float, default=0.30)
    parser.add_argument("--max-clear-win-overerase-changed-ratio", type=float, default=0.02)
    parser.add_argument("--max-slight-win-overerase-changed-ratio", type=float, default=0.08)
    parser.add_argument("--max-clear-win-damage-changed-ratio", type=float, default=0.10)
    parser.add_argument("--max-slight-win-damage-changed-ratio", type=float, default=0.20)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_gray(path: str) -> np.ndarray:
    image = cv2.imread(str(Path(path)), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.int16)


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape == reference.shape:
        return image
    resized = cv2.resize(image.astype(np.uint8), (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    return resized.astype(np.int16)


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def classify(metrics: dict[str, float], args: argparse.Namespace) -> tuple[str, str]:
    help_hurt = metrics["help_hurt_ratio"]
    residual_help_hurt = metrics["residual_help_hurt_ratio"]
    overerase_ratio = metrics["overerase_hurt_ratio"]
    target_dark_damage_ratio = metrics["target_dark_damage_ratio"]
    overerase_changed_ratio = metrics["overerase_changed_ratio"]
    target_dark_damage_changed_ratio = metrics["target_dark_damage_changed_ratio"]
    active_mean_gain = metrics["active_mean_gain"]
    global_gain = metrics["mean_error_gain"]
    changed_ratio = metrics["changed_ratio"]

    risk_reasons: list[str] = []
    if overerase_changed_ratio > args.max_slight_win_overerase_changed_ratio:
        risk_reasons.append(f"overerase_changed_ratio={overerase_changed_ratio:.4f}")
    if target_dark_damage_changed_ratio > args.max_slight_win_damage_changed_ratio:
        risk_reasons.append(f"target_dark_damage_changed_ratio={target_dark_damage_changed_ratio:.4f}")
    if help_hurt < 1.0:
        risk_reasons.append(f"help_hurt_ratio={help_hurt:.4f}")
    if active_mean_gain < 0.0:
        risk_reasons.append(f"active_mean_gain={active_mean_gain:.4f}")

    if global_gain < 0.0 or help_hurt < 0.85 or active_mean_gain < -0.5:
        return "clear_loss", "; ".join(risk_reasons) or f"global_gain={global_gain:.9f}"

    if risk_reasons:
        if help_hurt >= args.slight_win_help_hurt and residual_help_hurt > 1.0 and global_gain > 0.0:
            return "borderline", "; ".join(risk_reasons)
        return "slight_loss", "; ".join(risk_reasons)

    if changed_ratio == 0.0 or abs(global_gain) < 1e-9:
        return "noop", "candidate is effectively unchanged"

    if (
        help_hurt >= args.clear_win_help_hurt
        and residual_help_hurt >= 50.0
        and overerase_ratio <= args.max_clear_win_overerase_ratio
        and overerase_changed_ratio <= args.max_clear_win_overerase_changed_ratio
        and target_dark_damage_ratio <= args.max_clear_win_damage_ratio
        and target_dark_damage_changed_ratio <= args.max_clear_win_damage_changed_ratio
    ):
        return "clear_win", (
            f"strong help; help_hurt={help_hurt:.4f}; residual_help_hurt={residual_help_hurt:.4f}; "
            f"global_gain={global_gain:.9f}"
        )

    if help_hurt >= args.slight_win_help_hurt and residual_help_hurt > 1.0:
        return "slight_win", (
            f"metric win; help_hurt={help_hurt:.4f}; residual_help_hurt={residual_help_hurt:.4f}; "
            f"global_gain={global_gain:.9f}"
        )

    return "borderline", (
        f"weak or mixed win; help_hurt={help_hurt:.4f}; residual_help_hurt={residual_help_hurt:.4f}; "
        f"global_gain={global_gain:.9f}"
    )


def score_row(row: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    source = read_gray(row["source_input"])
    baseline = read_gray(row["baseline_pred"])
    candidate = resize_like(read_gray(row["candidate_pred"]), baseline)
    target = resize_like(read_gray(row["target"]), baseline)
    source = resize_like(source, baseline)

    baseline_error = np.abs(baseline - target)
    candidate_error = np.abs(candidate - target)
    improvement = baseline_error.astype(np.int32) - candidate_error.astype(np.int32)
    candidate_delta = candidate - baseline

    changed = np.abs(candidate_delta) > args.change_threshold
    target_changed = np.abs(target - baseline) > args.change_threshold
    active = changed | target_changed
    if not active.any():
        active = np.ones_like(changed, dtype=bool)

    help_mask = changed & (improvement > args.meaningful_threshold)
    hurt_mask = changed & (improvement < -args.meaningful_threshold)
    neutral_mask = changed & (np.abs(improvement) <= args.meaningful_threshold)
    target_residual_mask = (target - baseline) > args.target_residual_threshold
    residual_help = target_residual_mask & changed & (improvement > args.meaningful_threshold)
    residual_hurt = target_residual_mask & changed & (improvement < -args.meaningful_threshold)

    target_dark_mask = target < args.target_dark_threshold
    candidate_too_light = candidate > target + args.meaningful_threshold
    baseline_close_to_target = baseline_error <= args.meaningful_threshold
    overerase_hurt = hurt_mask & candidate_too_light & baseline_close_to_target
    target_dark_damage = hurt_mask & target_dark_mask

    changed_px = int(changed.sum())
    help_px = int(help_mask.sum())
    hurt_px = int(hurt_mask.sum())
    residual_help_px = int(residual_help.sum())
    residual_hurt_px = int(residual_hurt.sum())
    overerase_hurt_px = int(overerase_hurt.sum())
    target_dark_damage_px = int(target_dark_damage.sum())

    metrics = {
        "changed_ratio": safe_ratio(changed_px, changed.size),
        "mean_error_gain": float(baseline_error.mean() - candidate_error.mean()),
        "active_mean_gain": float(improvement[active].mean()),
        "source_to_target_error": float(np.abs(source - target).mean()),
        "baseline_to_target_error": float(baseline_error.mean()),
        "candidate_to_target_error": float(candidate_error.mean()),
        "help_px": float(help_px),
        "hurt_px": float(hurt_px),
        "neutral_px": float(int(neutral_mask.sum())),
        "help_hurt_ratio": safe_ratio(help_px + 1, hurt_px + 1),
        "residual_help_px": float(residual_help_px),
        "residual_hurt_px": float(residual_hurt_px),
        "residual_help_hurt_ratio": safe_ratio(residual_help_px + 1, residual_hurt_px + 1),
        "overerase_hurt_px": float(overerase_hurt_px),
        "overerase_hurt_ratio": safe_ratio(overerase_hurt_px, hurt_px),
        "target_dark_damage_px": float(target_dark_damage_px),
        "target_dark_damage_ratio": safe_ratio(target_dark_damage_px, hurt_px),
        "overerase_changed_ratio": safe_ratio(overerase_hurt_px, changed_px),
        "target_dark_damage_changed_ratio": safe_ratio(target_dark_damage_px, changed_px),
    }
    label, reason = classify(metrics, args)

    out: dict[str, object] = dict(row)
    out.update(metrics)
    out["target_quality_label"] = label
    out["target_quality_reason"] = reason
    return out


def main() -> None:
    args = parse_args()
    rows = [score_row(row, args) for row in read_rows(Path(args.review_csv))]
    if not rows:
        raise ValueError("review CSV has no rows")

    output_csv = Path(args.output_csv)
    write_csv(output_csv, rows)

    counts = Counter(str(row["target_quality_label"]) for row in rows)
    summary = {
        "rows": len(rows),
        "counts": {label: counts.get(label, 0) for label in QUALITY_LABELS},
        "output_csv": str(output_csv),
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
