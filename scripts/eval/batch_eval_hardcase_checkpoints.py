#!/usr/bin/env python3
"""Batch-evaluate EnsExam-GAN checkpoints on selected hard SCUT pages.

This is a faster search helper than invoking eval_hardcase_worst_pages.py once
per checkpoint. It caches sample images and baseline metrics, skips visual
sheet generation by default, and writes one summary CSV for candidate ranking.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_hardcase_worst_pages import (  # noqa: E402
    auto_copy_mask_threshold,
    compute_residual_metrics,
    copy_input_outside_mask,
    ensure_same_size,
    hstack,
    label_path_for,
    load_generator,
    panel,
    parse_threshold_map,
    pick_device,
    read_bgr,
    read_sample_paths,
    save_sheet,
)
from utils.page_inference import infer_full_page  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-csv", required=True, help="CSV with name,config,weights[,output_dir].")
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--baseline-pred-dir", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--page-overlap", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mask-threshold", type=int, default=12)
    parser.add_argument("--change-threshold", type=int, default=12)
    parser.add_argument("--copy-input-outside-mask", choices=("none", "ms", "mb"), default="none")
    parser.add_argument("--copy-mask-threshold", type=int, default=32)
    parser.add_argument("--copy-mask-threshold-map", default="")
    parser.add_argument("--copy-mask-threshold-auto", choices=("none", "mb_cov8_step"), default="none")
    parser.add_argument("--copy-mask-dilate", type=int, default=0)
    parser.add_argument("--score-overerase-penalty", type=float, default=8.0)
    parser.add_argument("--save-images", action="store_true", help="Also save pred images, crop comparisons, and sheet.")
    return parser.parse_args()


def read_items(path: Path) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open()))
    required = {"name", "config", "weights"}
    if not rows:
        raise ValueError(f"items CSV is empty: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"items CSV missing required columns: {sorted(missing)}")
    return rows


def load_samples(samples_file: Path, baseline_dir: Path | None, change_threshold: int, eval_threshold: int):
    loaded = []
    for image_path in read_sample_paths(samples_file):
        label_path = label_path_for(image_path)
        input_bgr = read_bgr(image_path)
        label_bgr = ensure_same_size(read_bgr(label_path), input_bgr)
        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
        baseline_bgr = None
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
        loaded.append({
            "image_path": image_path,
            "label_path": label_path,
            "input_bgr": input_bgr,
            "label_bgr": label_bgr,
            "input_rgb": input_rgb,
            "baseline_bgr": baseline_bgr,
            "baseline_path": baseline_path,
            "baseline_metrics": baseline_metrics,
        })
    return loaded


def average(rows: list[dict[str, str | float | int]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and row[key] != ""]
    return sum(values) / max(len(values), 1)


def evaluate_item(
    item: dict[str, str],
    samples,
    args: argparse.Namespace,
    device: torch.device,
    threshold_map: dict[str, int],
) -> tuple[dict[str, str | float | int], list[dict[str, str | float | int]]]:
    name = item["name"]
    output_dir = Path(item.get("output_dir") or Path(args.output_root) / name)
    pred_dir = output_dir / "pred"
    crop_dir = output_dir / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        pred_dir.mkdir(parents=True, exist_ok=True)
        crop_dir.mkdir(parents=True, exist_ok=True)

    generator = load_generator(item["config"], item["weights"], device)
    rows: list[dict[str, str | float | int]] = []
    crop_rows = []

    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            image_path: Path = sample["image_path"]
            input_bgr = sample["input_bgr"]
            label_bgr = sample["label_bgr"]
            outputs = infer_full_page(
                generator,
                sample["input_rgb"],
                device,
                overlap=args.page_overlap,
                batch_size=args.batch_size,
            )
            pred_bgr = cv2.cvtColor(outputs["icomp"], cv2.COLOR_RGB2BGR)
            copy_mask_cov8 = 0.0
            if args.copy_input_outside_mask != "none":
                selected_mask = outputs[args.copy_input_outside_mask]
                if image_path.name in threshold_map:
                    copy_mask_threshold = threshold_map[image_path.name]
                else:
                    copy_mask_threshold, copy_mask_cov8 = auto_copy_mask_threshold(
                        args.copy_input_outside_mask,
                        selected_mask,
                        args.copy_mask_threshold_auto,
                        args.copy_mask_threshold,
                    )
                pred_bgr = copy_input_outside_mask(
                    pred_bgr,
                    input_bgr,
                    selected_mask,
                    threshold=copy_mask_threshold,
                    dilate=args.copy_mask_dilate,
                )
            else:
                copy_mask_threshold = args.copy_mask_threshold

            pred_path = pred_dir / f"{image_path.stem}.png"
            if args.save_images:
                cv2.imwrite(str(pred_path), pred_bgr)
            metrics = compute_residual_metrics(
                input_bgr,
                label_bgr,
                pred_bgr,
                change_threshold=args.change_threshold,
                eval_threshold=args.mask_threshold,
            )
            row: dict[str, str | float | int] = {
                "candidate": name,
                "file": image_path.name,
                "image_path": str(image_path),
                "label_path": str(sample["label_path"]),
                "pred_path": str(pred_path) if args.save_images else "",
                "copy_input_outside_mask": args.copy_input_outside_mask,
                "copy_mask_threshold": copy_mask_threshold,
                "copy_mask_threshold_map": args.copy_mask_threshold_map,
                "copy_mask_threshold_auto": args.copy_mask_threshold_auto,
                "copy_mask_cov8": copy_mask_cov8,
                "copy_mask_dilate": args.copy_mask_dilate,
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

            if args.save_images:
                panels = [
                    panel(input_bgr, f"{image_path.stem} input"),
                    panel(label_bgr, "target"),
                ]
                if sample["baseline_bgr"] is not None:
                    panels.append(panel(sample["baseline_bgr"], "baseline"))
                panels.append(panel(pred_bgr, name))
                crop_rows.append(hstack(panels))

            print(f"{name} [{index}/{len(samples)}] {image_path.name} residual={metrics['residual_ratio']:.4f} over={metrics['overerase_ratio']:.4f}", flush=True)

    metrics_csv = output_dir / "hardcase_worst_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.save_images and crop_rows:
        save_sheet(crop_rows, output_dir / "hardcase_worst_sheet.png")

    residual = average(rows, "residual_ratio")
    overerase = average(rows, "overerase_ratio")
    baseline_residual = average(rows, "baseline_residual_ratio") if "baseline_residual_ratio" in rows[0] else 0.0
    baseline_overerase = average(rows, "baseline_overerase_ratio") if "baseline_overerase_ratio" in rows[0] else 0.0
    score = (baseline_residual - residual) - args.score_overerase_penalty * max(overerase - baseline_overerase, 0.0)
    summary = {
        "name": name,
        "config": item["config"],
        "weights": item["weights"],
        "output_dir": str(output_dir),
        "metrics_csv": str(metrics_csv),
        "pages": len(rows),
        "residual_ratio": residual,
        "overerase_ratio": overerase,
        "baseline_residual_ratio": baseline_residual,
        "baseline_overerase_ratio": baseline_overerase,
        "residual_gain": baseline_residual - residual,
        "overerase_delta": overerase - baseline_overerase,
        "score": score,
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    threshold_map = parse_threshold_map(args.copy_mask_threshold_map)
    baseline_dir = Path(args.baseline_pred_dir) if args.baseline_pred_dir else None
    samples = load_samples(Path(args.samples_file), baseline_dir, args.change_threshold, args.mask_threshold)
    items = read_items(Path(args.items_csv))

    summaries = []
    for item in items:
        summary, _ = evaluate_item(item, samples, args, device, threshold_map)
        summaries.append(summary)
        print(
            f"summary {summary['name']}: residual={summary['residual_ratio']:.6f} "
            f"overerase={summary['overerase_ratio']:.6f} score={summary['score']:+.6f}",
            flush=True,
        )

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0].keys()) if summaries else []
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(summaries, key=lambda row: float(row["score"]), reverse=True))
    print(f"summary_csv: {summary_path}")


if __name__ == "__main__":
    main()
