#!/usr/bin/env python3
"""Sweep hardcase copy-mask postprocess settings with cached page inference.

Unlike batch_eval_hardcase_checkpoints.py, this script runs the generator once
per page and then evaluates many mask/threshold/dilation combinations offline.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_hardcase_worst_pages import (  # noqa: E402
    compute_residual_metrics,
    copy_input_outside_mask,
    ensure_same_size,
    label_path_for,
    load_generator,
    pick_device,
    read_bgr,
    read_sample_paths,
)
from utils.page_inference import infer_full_page  # noqa: E402


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_mask_list(value: str) -> list[str]:
    masks = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(masks) - {"ms", "mb"})
    if invalid:
        raise ValueError(f"Unsupported mask names: {invalid}; expected ms and/or mb")
    return masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--name", default="candidate")
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--baseline-pred-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--per-page-csv", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--page-overlap", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mask-threshold", type=int, default=12)
    parser.add_argument("--change-threshold", type=int, default=12)
    parser.add_argument("--copy-masks", default="mb", help="Comma-separated list: mb,ms")
    parser.add_argument("--copy-thresholds", default="4,8,16,32,48,70,96,128,160")
    parser.add_argument("--copy-dilates", default="0,1,2")
    parser.add_argument("--score-overerase-penalty", type=float, default=8.0)
    return parser.parse_args()


def average(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and row[key] != ""]
    return sum(values) / max(len(values), 1)


def load_samples(samples_file: Path, baseline_dir: Path | None, change_threshold: int, eval_threshold: int):
    samples = []
    for image_path in read_sample_paths(samples_file):
        label_path = label_path_for(image_path)
        input_bgr = read_bgr(image_path)
        label_bgr = ensure_same_size(read_bgr(label_path), input_bgr)
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
        baseline_metrics = None
        baseline_path = None
        if baseline_dir is not None:
            candidate = baseline_dir / image_path.name
            if not candidate.exists():
                candidate = baseline_dir / f"{image_path.stem}.png"
            if candidate.exists():
                baseline_path = candidate
                baseline_bgr = ensure_same_size(read_bgr(candidate), input_bgr)
                baseline_metrics = compute_residual_metrics(
                    input_bgr,
                    label_bgr,
                    baseline_bgr,
                    change_threshold=change_threshold,
                    eval_threshold=eval_threshold,
                )
        samples.append({
            "image_path": image_path,
            "label_path": label_path,
            "input_bgr": input_bgr,
            "label_bgr": label_bgr,
            "input_rgb": input_rgb,
            "baseline_path": baseline_path,
            "baseline_metrics": baseline_metrics,
        })
    return samples


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    per_page_csv = Path(args.per_page_csv) if args.per_page_csv else output_dir / "per_page_metrics.csv"

    masks = parse_mask_list(args.copy_masks)
    thresholds = parse_int_list(args.copy_thresholds)
    dilates = parse_int_list(args.copy_dilates)
    device = pick_device(args.device)
    baseline_dir = Path(args.baseline_pred_dir) if args.baseline_pred_dir else None
    samples = load_samples(
        Path(args.samples_file),
        baseline_dir,
        change_threshold=args.change_threshold,
        eval_threshold=args.mask_threshold,
    )
    generator = load_generator(args.config, args.weights, device)

    page_outputs = []
    for index, sample in enumerate(samples, start=1):
        outputs = infer_full_page(
            generator,
            sample["input_rgb"],
            device,
            overlap=args.page_overlap,
            batch_size=args.batch_size,
        )
        page_outputs.append((sample, outputs))
        print(f"inferred [{index}/{len(samples)}] {sample['image_path'].name}", flush=True)

    per_page_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for mask_name in masks:
        for dilate in dilates:
            for threshold in thresholds:
                rows = []
                for sample, outputs in page_outputs:
                    pred_bgr = cv2.cvtColor(outputs["icomp"], cv2.COLOR_RGB2BGR)
                    pred_bgr = copy_input_outside_mask(
                        pred_bgr,
                        sample["input_bgr"],
                        outputs[mask_name],
                        threshold=threshold,
                        dilate=dilate,
                    )
                    metrics = compute_residual_metrics(
                        sample["input_bgr"],
                        sample["label_bgr"],
                        pred_bgr,
                        change_threshold=args.change_threshold,
                        eval_threshold=args.mask_threshold,
                    )
                    row = {
                        "name": args.name,
                        "file": sample["image_path"].name,
                        "mask": mask_name,
                        "threshold": threshold,
                        "dilate": dilate,
                        **metrics,
                    }
                    baseline_metrics = sample["baseline_metrics"]
                    if baseline_metrics is not None:
                        row.update({
                            "baseline_pred_path": str(sample["baseline_path"]),
                            "baseline_residual_ratio": baseline_metrics["residual_ratio"],
                            "baseline_overerase_ratio": baseline_metrics["overerase_ratio"],
                            "delta_residual_ratio": float(baseline_metrics["residual_ratio"]) - float(metrics["residual_ratio"]),
                            "delta_overerase_ratio": float(metrics["overerase_ratio"]) - float(baseline_metrics["overerase_ratio"]),
                        })
                    rows.append(row)
                    per_page_rows.append(row)

                residual = average(rows, "residual_ratio")
                overerase = average(rows, "overerase_ratio")
                baseline_residual = average(rows, "baseline_residual_ratio") if "baseline_residual_ratio" in rows[0] else 0.0
                baseline_overerase = average(rows, "baseline_overerase_ratio") if "baseline_overerase_ratio" in rows[0] else 0.0
                score = (baseline_residual - residual) - args.score_overerase_penalty * max(overerase - baseline_overerase, 0.0)
                summary_rows.append({
                    "name": args.name,
                    "config": args.config,
                    "weights": args.weights,
                    "mask": mask_name,
                    "threshold": threshold,
                    "dilate": dilate,
                    "pages": len(rows),
                    "residual_ratio": residual,
                    "overerase_ratio": overerase,
                    "baseline_residual_ratio": baseline_residual,
                    "baseline_overerase_ratio": baseline_overerase,
                    "residual_gain": baseline_residual - residual,
                    "overerase_delta": overerase - baseline_overerase,
                    "score": score,
                })

    summary_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with per_page_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in per_page_rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(per_page_rows)

    print(f"summary_csv: {summary_csv}", flush=True)
    for row in summary_rows[:10]:
        print(
            f"{row['mask']}_t{row['threshold']}_d{row['dilate']} "
            f"residual={float(row['residual_ratio']):.6f} "
            f"overerase={float(row['overerase_ratio']):.6f} "
            f"score={float(row['score']):+.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
