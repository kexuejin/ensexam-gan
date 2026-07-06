#!/usr/bin/env python3
"""Analyze whether mask-confidence features separate useful candidates.

This is a read-only diagnostic for saved hybrid-gate candidate outputs. It
re-runs the primary model to recover page-level `ms`/`mb` masks, loads saved
candidate images, and writes per-page features plus residual/overerase deltas
against a baseline metrics CSV.

Use this as diagnostic evidence only. A feature threshold discovered here still
needs joint SCUT115 + holdout40 replay before it can become a product selector.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVAL_SCRIPT_DIR = ROOT / "scripts" / "eval"
if str(EVAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPT_DIR))

from eval_hardcase_worst_pages import (  # noqa: E402
    compute_residual_metrics,
    ensure_same_size,
    label_path_for,
    load_generator,
    pick_device,
    read_bgr,
    read_sample_paths,
)
from utils.page_inference import infer_full_page  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--page-overlap", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--change-threshold", type=int, default=12)
    parser.add_argument("--eval-threshold", type=int, default=12)
    parser.add_argument("--edit-threshold", type=float, default=12.0)
    parser.add_argument("--mask-thresholds", default="8,16,32,64,98,128,160,192")
    return parser.parse_args()


def read_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["file"]: row for row in rows}


def candidate_path(candidate_dir: Path, image_path: Path) -> Path:
    for name in (f"{image_path.stem}.png", image_path.name):
        path = candidate_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No candidate image for {image_path.name} in {candidate_dir}")


def mask_features(mask: np.ndarray, edit_mask: np.ndarray, prefix: str, thresholds: list[int]) -> dict[str, object]:
    features: dict[str, object] = {
        f"{prefix}_mean": f"{float(mask.mean()):.6f}",
        f"{prefix}_p95": f"{float(np.percentile(mask, 95)):.6f}",
        f"{prefix}_edit_mean": f"{float(mask[edit_mask].mean()) if edit_mask.any() else 0.0:.6f}",
    }
    edit_count = int(edit_mask.sum())
    for threshold in thresholds:
        mask_on = mask >= threshold
        overlap = mask_on & edit_mask
        features[f"{prefix}_cov{threshold}"] = f"{float(mask_on.mean()):.12f}"
        features[f"{prefix}_edit_overlap{threshold}"] = f"{float(overlap.sum() / max(edit_count, 1)):.12f}"
        features[f"{prefix}_edit_outside{threshold}"] = int((edit_mask & ~mask_on).sum())
    return features


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [int(value.strip()) for value in args.mask_thresholds.split(",") if value.strip()]

    device = pick_device(args.device)
    generator = load_generator(args.config, args.weights, device)
    baseline_by_file = read_metrics(Path(args.baseline_metrics))
    candidate_dir = Path(args.candidate_dir)

    rows: list[dict[str, object]] = []
    for image_path in read_sample_paths(Path(args.samples_file)):
        input_bgr = read_bgr(image_path)
        label_bgr = ensure_same_size(read_bgr(label_path_for(image_path)), input_bgr)
        candidate_bgr = ensure_same_size(read_bgr(candidate_path(candidate_dir, image_path)), input_bgr)
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
        outputs = infer_full_page(
            generator,
            input_rgb,
            device,
            overlap=args.page_overlap,
            batch_size=args.batch_size,
        )

        candidate_metrics = compute_residual_metrics(
            input_bgr,
            label_bgr,
            candidate_bgr,
            change_threshold=args.change_threshold,
            eval_threshold=args.eval_threshold,
        )
        baseline = baseline_by_file[image_path.name]
        residual_gain = float(baseline["residual_ratio"]) - float(candidate_metrics["residual_ratio"])
        overerase_regret = float(candidate_metrics["overerase_ratio"]) - float(baseline["overerase_ratio"])
        primary_delta = cv2.absdiff(candidate_bgr, input_bgr).mean(axis=2)
        edit_mask = primary_delta >= args.edit_threshold

        row: dict[str, object] = {
            "file": image_path.name,
            "image_path": str(image_path),
            "candidate_path": str(candidate_path(candidate_dir, image_path)),
            "baseline_residual_ratio": baseline["residual_ratio"],
            "candidate_residual_ratio": f"{float(candidate_metrics['residual_ratio']):.12f}",
            "residual_gain": f"{residual_gain:.12f}",
            "baseline_overerase_ratio": baseline["overerase_ratio"],
            "candidate_overerase_ratio": f"{float(candidate_metrics['overerase_ratio']):.12f}",
            "overerase_regret": f"{overerase_regret:.12f}",
            "primary_edit_px": int(edit_mask.sum()),
            "primary_edit_ratio": f"{float(edit_mask.mean()):.12f}",
            "primary_mean_edit_delta": f"{float(primary_delta.mean()):.6f}",
            "primary_p95_edit_delta": f"{float(np.percentile(primary_delta, 95)):.6f}",
        }
        row.update(mask_features(outputs["mb"], edit_mask, "mb", thresholds))
        row.update(mask_features(outputs["ms"], edit_mask, "ms", thresholds))
        rows.append(row)

    detail_path = output_dir / "mask_confidence_features.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "summary.csv"
    positive = [row for row in rows if float(row["residual_gain"]) > 0]
    safe = [row for row in positive if float(row["overerase_regret"]) <= 0]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "pages",
            "positive_gain_pages",
            "safe_positive_pages",
            "avg_residual_gain",
            "avg_overerase_regret",
            "max_residual_gain",
            "max_overerase_regret",
        ])
        writer.writeheader()
        writer.writerow({
            "pages": len(rows),
            "positive_gain_pages": len(positive),
            "safe_positive_pages": len(safe),
            "avg_residual_gain": f"{np.mean([float(row['residual_gain']) for row in rows]):.12f}",
            "avg_overerase_regret": f"{np.mean([float(row['overerase_regret']) for row in rows]):.12f}",
            "max_residual_gain": f"{max(float(row['residual_gain']) for row in rows):.12f}",
            "max_overerase_regret": f"{max(float(row['overerase_regret']) for row in rows):.12f}",
        })

    print(f"features={detail_path}")
    print(f"summary={summary_path}")
    print(f"pages={len(rows)} positive_gain={len(positive)} safe_positive={len(safe)}")


if __name__ == "__main__":
    main()
