#!/usr/bin/env python3
"""Run EnsExam-GAN output through a gated second-stage residual repair model.

The intended production-safe path is:

1. Produce the primary EnsExam-GAN page output, optionally restricted by its
   predicted mask.
2. Run a small erasemap cleanup model on that primary output.
3. Blend second-stage pixels back only where the primary model already edited
   the input and the second stage differs enough from the primary output.
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
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
EVAL_SCRIPT_DIR = ROOT / "scripts" / "eval"
if str(EVAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPT_DIR))

from eval_hardcase_worst_pages import (  # noqa: E402
    auto_copy_mask_threshold,
    compute_residual_metrics,
    copy_input_outside_mask,
    ensure_same_size,
    label_path_for,
    load_generator,
    pick_device,
    read_bgr,
    read_sample_paths,
)
from utils.page_inference import infer_full_page as infer_ensexam_page  # noqa: E402
from patch_cleanup_erasemap import (  # noqa: E402
    infer_full_page as infer_cleanup_page,
    load_model as load_cleanup_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primary-pred-dir", default="")
    parser.add_argument("--primary-config", default="")
    parser.add_argument("--primary-weights", default="")
    parser.add_argument("--cleanup-checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--primary-page-overlap", type=int, default=32)
    parser.add_argument("--primary-batch-size", type=int, default=8)
    parser.add_argument("--primary-copy-mask", choices=("none", "ms", "mb"), default="mb")
    parser.add_argument("--primary-copy-threshold", type=int, default=70)
    parser.add_argument("--primary-copy-threshold-auto", choices=("none", "mb_cov8_step"), default="mb_cov8_step")
    parser.add_argument("--primary-copy-dilate", type=int, default=0)
    parser.add_argument("--cleanup-tile-size", type=int, default=160)
    parser.add_argument("--cleanup-stride", type=int, default=160)
    parser.add_argument("--cleanup-alpha-threshold", type=float, default=0.3)
    parser.add_argument("--base-edit-threshold", type=float, default=12.0)
    parser.add_argument("--second-delta-threshold", type=float, default=32.0)
    parser.add_argument("--dark-threshold", type=int, default=0)
    parser.add_argument("--change-threshold", type=int, default=12)
    parser.add_argument("--eval-threshold", type=int, default=12)
    parser.add_argument("--save-primary", action="store_true")
    parser.add_argument("--save-second-stage", action="store_true")
    return parser.parse_args()


def primary_pred_path(primary_dir: Path, image_path: Path) -> Path:
    for name in (f"{image_path.stem}.png", f"{image_path.stem}.clean.png", image_path.name):
        candidate = primary_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No primary prediction found for {image_path.name} in {primary_dir}")


def run_primary(args: argparse.Namespace, image_path: Path, input_bgr: np.ndarray, generator, device) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if args.primary_pred_dir:
        pred = ensure_same_size(read_bgr(primary_pred_path(Path(args.primary_pred_dir), image_path)), input_bgr)
        return pred, {}
    if not args.primary_config or not args.primary_weights:
        raise ValueError("Provide either --primary-pred-dir or both --primary-config and --primary-weights")
    input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
    outputs = infer_ensexam_page(
        generator,
        input_rgb,
        device,
        overlap=args.primary_page_overlap,
        batch_size=args.primary_batch_size,
    )
    pred_bgr = cv2.cvtColor(outputs["icomp"], cv2.COLOR_RGB2BGR)
    if args.primary_copy_mask != "none":
        selected_mask = outputs[args.primary_copy_mask]
        threshold, _cov8 = auto_copy_mask_threshold(
            args.primary_copy_mask,
            selected_mask,
            args.primary_copy_threshold_auto,
            args.primary_copy_threshold,
        )
        pred_bgr = copy_input_outside_mask(
            pred_bgr,
            input_bgr,
            selected_mask,
            threshold=threshold,
            dilate=args.primary_copy_dilate,
        )
    return pred_bgr, outputs


def gated_blend(
    input_bgr: np.ndarray,
    primary_bgr: np.ndarray,
    second_bgr: np.ndarray,
    base_edit_threshold: float,
    second_delta_threshold: float,
    dark_threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    base_edit = cv2.absdiff(primary_bgr, input_bgr).mean(axis=2)
    second_delta = cv2.absdiff(second_bgr, primary_bgr).mean(axis=2)
    gate = (base_edit >= base_edit_threshold) & (second_delta >= second_delta_threshold)
    if dark_threshold > 0:
        gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
        gate &= gray <= dark_threshold
    merged = primary_bgr.copy()
    merged[gate] = second_bgr[gate]
    return merged, gate


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "pred"
    primary_dir = output_dir / "primary"
    second_dir = output_dir / "second_stage"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if args.save_primary:
        primary_dir.mkdir(parents=True, exist_ok=True)
    if args.save_second_stage:
        second_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    generator = None
    if not args.primary_pred_dir:
        generator = load_generator(args.primary_config, args.primary_weights, device)
    cleanup_model = load_cleanup_model(Path(args.cleanup_checkpoint), device)

    rows: list[dict[str, object]] = []
    for index, image_path in enumerate(read_sample_paths(Path(args.samples_file)), start=1):
        input_bgr = read_bgr(image_path)
        primary_bgr, _primary_outputs = run_primary(args, image_path, input_bgr, generator, device)
        second_bgr = infer_cleanup_page(
            cleanup_model,
            primary_bgr,
            device,
            tile_size=args.cleanup_tile_size,
            stride=args.cleanup_stride,
            alpha_threshold=args.cleanup_alpha_threshold,
        )
        merged_bgr, gate = gated_blend(
            input_bgr,
            primary_bgr,
            second_bgr,
            base_edit_threshold=args.base_edit_threshold,
            second_delta_threshold=args.second_delta_threshold,
            dark_threshold=args.dark_threshold,
        )

        pred_path = pred_dir / f"{image_path.stem}.png"
        cv2.imwrite(str(pred_path), merged_bgr)
        if args.save_primary:
            cv2.imwrite(str(primary_dir / f"{image_path.stem}.png"), primary_bgr)
        if args.save_second_stage:
            cv2.imwrite(str(second_dir / f"{image_path.stem}.png"), second_bgr)

        row: dict[str, object] = {
            "file": image_path.name,
            "image_path": str(image_path),
            "pred_path": str(pred_path),
            "base_edit_threshold": args.base_edit_threshold,
            "second_delta_threshold": args.second_delta_threshold,
            "dark_threshold": args.dark_threshold,
            "gate_ratio": float(gate.mean()),
        }
        try:
            label_bgr = ensure_same_size(read_bgr(label_path_for(image_path)), input_bgr)
            row.update(compute_residual_metrics(
                input_bgr,
                label_bgr,
                merged_bgr,
                change_threshold=args.change_threshold,
                eval_threshold=args.eval_threshold,
            ))
        except Exception as exc:  # noqa: BLE001 - labels are optional for product inference.
            row["metrics_error"] = str(exc)
        rows.append(row)
        print(
            f"{index} {image_path.name} -> {pred_path} gate={float(gate.mean()):.6f}",
            flush=True,
        )

    metrics_csv = output_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    if rows and "residual_ratio" in rows[0]:
        residual = sum(float(row["residual_ratio"]) for row in rows) / len(rows)
        overerase = sum(float(row["overerase_ratio"]) for row in rows) / len(rows)
        print(f"summary residual={residual:.6f} overerase={overerase:.6f}", flush=True)
    print(f"metrics_csv: {metrics_csv}", flush=True)


if __name__ == "__main__":
    main()
