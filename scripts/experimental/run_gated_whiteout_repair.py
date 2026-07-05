#!/usr/bin/env python3
"""Run gated correction-fluid / whiteout background repair.

This is intended as a third-stage postprocess after handwriting removal:

1. Read an already-cleaned page prediction.
2. Detect local bright paper-tone anomalies from the prediction only.
3. Enable repair only when the detected bright region overlaps an area that
   the previous stages already edited from the input page.
4. Inpaint only the gated bright anomaly mask and write an audit CSV.

The detection gate does not use target/label images. Labels are optional and
only used to report residual/over-erase metrics for offline validation.
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
    compute_residual_metrics,
    ensure_same_size,
    label_path_for,
    read_bgr,
    read_sample_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--input-pred-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-median-ksize", type=int, default=101)
    parser.add_argument("--bright-delta-threshold", type=int, default=10)
    parser.add_argument("--bright-gray-threshold", type=int, default=215)
    parser.add_argument("--max-saturation", type=int, default=45)
    parser.add_argument("--component-area-threshold", type=int, default=120)
    parser.add_argument("--max-component-width-ratio", type=float, default=0.55)
    parser.add_argument("--max-component-height-ratio", type=float, default=0.20)
    parser.add_argument("--mask-dilate", type=int, default=3)
    parser.add_argument("--inpaint-radius", type=int, default=3)
    parser.add_argument("--min-mask-ratio", type=float, default=0.001)
    parser.add_argument("--min-edit-ratio-in-mask", type=float, default=0.04)
    parser.add_argument("--min-edit-px-in-mask", type=int, default=0)
    parser.add_argument("--edit-threshold", type=float, default=12.0)
    parser.add_argument("--min-edit-dark-px-in-mask", type=int, default=0)
    parser.add_argument("--dark-gray-threshold", type=int, default=210)
    parser.add_argument("--change-threshold", type=int, default=12)
    parser.add_argument("--eval-threshold", type=int, default=12)
    parser.add_argument("--save-mask", action="store_true")
    return parser.parse_args()


def pred_path_for(pred_dir: Path, image_path: Path) -> Path:
    for name in (f"{image_path.stem}.png", f"{image_path.stem}.clean.png", image_path.name):
        candidate = pred_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No prediction found for {image_path.name} in {pred_dir}")


def make_odd(value: int) -> int:
    if value <= 1:
        return 1
    return value if value % 2 else value + 1


def detect_bright_anomaly_mask(
    pred_bgr: np.ndarray,
    *,
    local_median_ksize: int,
    bright_delta_threshold: int,
    bright_gray_threshold: int,
    max_saturation: int,
    component_area_threshold: int,
    max_component_width_ratio: float,
    max_component_height_ratio: float,
    mask_dilate: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    gray = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, make_odd(local_median_ksize))
    delta = gray.astype(np.int16) - bg.astype(np.int16)
    saturation = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    raw = (
        (delta >= bright_delta_threshold)
        & (gray >= bright_gray_threshold)
        & (saturation <= max_saturation)
    )

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(raw.astype(np.uint8), 8)
    mask = np.zeros(gray.shape, np.uint8)
    component_areas: list[int] = []
    component_deltas: list[float] = []
    image_h, image_w = gray.shape
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < component_area_threshold:
            continue
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if width > image_w * max_component_width_ratio:
            continue
        if height > image_h * max_component_height_ratio:
            continue
        component = labels == index
        mask[component] = 255
        component_areas.append(area)
        component_deltas.append(float(delta[component].mean()))

    raw_component_area = int(sum(component_areas))
    max_component_area = int(max(component_areas)) if component_areas else 0
    max_component_delta = float(max(component_deltas)) if component_deltas else 0.0
    if mask_dilate > 0 and mask.any():
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (mask_dilate * 2 + 1, mask_dilate * 2 + 1),
        )
        mask = cv2.dilate(mask, kernel)

    features: dict[str, float | int] = {
        "raw_component_area": raw_component_area,
        "raw_component_ratio": raw_component_area / max(image_h * image_w, 1),
        "component_count": len(component_areas),
        "max_component_area": max_component_area,
        "max_component_ratio": max_component_area / max(image_h * image_w, 1),
        "max_component_mean_delta": max_component_delta,
        "mask_px": int((mask > 0).sum()),
        "mask_ratio": float((mask > 0).mean()),
    }
    return mask > 0, features


def should_apply_whiteout_repair(
    input_bgr: np.ndarray,
    pred_bgr: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[bool, dict[str, float | int]]:
    if not mask.any():
        return False, {
            "edit_mean_in_mask": 0.0,
            "edit_px_in_mask": 0,
            "edit_ratio_in_mask": 0.0,
            "edit_dark_px_in_mask": 0,
        }

    edit_delta = cv2.absdiff(input_bgr, pred_bgr).mean(axis=2)
    input_gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
    edit_pixels = edit_delta[mask] >= args.edit_threshold
    edit_dark = (edit_delta >= args.edit_threshold) & (input_gray <= args.dark_gray_threshold) & mask
    features: dict[str, float | int] = {
        "edit_mean_in_mask": float(edit_delta[mask].mean()),
        "edit_px_in_mask": int(edit_pixels.sum()),
        "edit_ratio_in_mask": float(edit_pixels.mean()),
        "edit_dark_px_in_mask": int(edit_dark.sum()),
    }
    apply = (
        float(mask.mean()) >= args.min_mask_ratio
        and int(features["edit_px_in_mask"]) >= args.min_edit_px_in_mask
        and float(features["edit_ratio_in_mask"]) >= args.min_edit_ratio_in_mask
        and int(features["edit_dark_px_in_mask"]) >= args.min_edit_dark_px_in_mask
    )
    return apply, features


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "pred"
    mask_dir = output_dir / "mask"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if args.save_mask:
        mask_dir.mkdir(parents=True, exist_ok=True)

    input_pred_dir = Path(args.input_pred_dir)
    rows: list[dict[str, object]] = []
    for index, image_path in enumerate(read_sample_paths(Path(args.samples_file)), start=1):
        input_bgr = read_bgr(image_path)
        base_bgr = ensure_same_size(read_bgr(pred_path_for(input_pred_dir, image_path)), input_bgr)
        mask, mask_features = detect_bright_anomaly_mask(
            base_bgr,
            local_median_ksize=args.local_median_ksize,
            bright_delta_threshold=args.bright_delta_threshold,
            bright_gray_threshold=args.bright_gray_threshold,
            max_saturation=args.max_saturation,
            component_area_threshold=args.component_area_threshold,
            max_component_width_ratio=args.max_component_width_ratio,
            max_component_height_ratio=args.max_component_height_ratio,
            mask_dilate=args.mask_dilate,
        )
        apply_repair, gate_features = should_apply_whiteout_repair(input_bgr, base_bgr, mask, args)
        if apply_repair:
            repaired_bgr = cv2.inpaint(
                base_bgr,
                (mask.astype(np.uint8) * 255),
                args.inpaint_radius,
                cv2.INPAINT_TELEA,
            )
        else:
            repaired_bgr = base_bgr.copy()

        pred_path = pred_dir / f"{image_path.stem}.png"
        cv2.imwrite(str(pred_path), repaired_bgr)
        if args.save_mask:
            cv2.imwrite(str(mask_dir / f"{image_path.stem}.png"), mask.astype(np.uint8) * 255)

        row: dict[str, object] = {
            "file": image_path.name,
            "image_path": str(image_path),
            "input_pred_path": str(pred_path_for(input_pred_dir, image_path)),
            "pred_path": str(pred_path),
            "whiteout_applied": int(apply_repair),
            **mask_features,
            **gate_features,
        }
        try:
            label_bgr = ensure_same_size(read_bgr(label_path_for(image_path)), input_bgr)
            base_metrics = compute_residual_metrics(
                input_bgr,
                label_bgr,
                base_bgr,
                change_threshold=args.change_threshold,
                eval_threshold=args.eval_threshold,
            )
            repaired_metrics = compute_residual_metrics(
                input_bgr,
                label_bgr,
                repaired_bgr,
                change_threshold=args.change_threshold,
                eval_threshold=args.eval_threshold,
            )
            row.update({
                "base_residual_ratio": base_metrics["residual_ratio"],
                "base_overerase_ratio": base_metrics["overerase_ratio"],
                "residual_ratio": repaired_metrics["residual_ratio"],
                "overerase_ratio": repaired_metrics["overerase_ratio"],
                "residual_gain": float(base_metrics["residual_ratio"]) - float(repaired_metrics["residual_ratio"]),
                "overerase_delta": float(repaired_metrics["overerase_ratio"]) - float(base_metrics["overerase_ratio"]),
            })
        except Exception as exc:  # noqa: BLE001 - labels are optional for product inference.
            row["metrics_error"] = str(exc)
        rows.append(row)
        print(
            f"{index} {image_path.name} -> {pred_path} "
            f"applied={int(apply_repair)} mask={float(mask_features['mask_ratio']):.6f} "
            f"edit_ratio={float(gate_features['edit_ratio_in_mask']):.6f}",
            flush=True,
        )

    metrics_csv = output_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)

    applied = sum(int(row["whiteout_applied"]) for row in rows)
    if rows and "residual_ratio" in rows[0]:
        residual = sum(float(row["residual_ratio"]) for row in rows) / len(rows)
        overerase = sum(float(row["overerase_ratio"]) for row in rows) / len(rows)
        base_residual = sum(float(row["base_residual_ratio"]) for row in rows) / len(rows)
        base_overerase = sum(float(row["base_overerase_ratio"]) for row in rows) / len(rows)
        print(
            "summary "
            f"applied={applied}/{len(rows)} "
            f"residual={base_residual:.6f}->{residual:.6f} "
            f"overerase={base_overerase:.6f}->{overerase:.6f} "
            f"gain={base_residual - residual:+.6f} "
            f"overerase_delta={overerase - base_overerase:+.6f}",
            flush=True,
        )
    else:
        print(f"summary applied={applied}/{len(rows)}", flush=True)
    print(f"metrics_csv: {metrics_csv}", flush=True)


if __name__ == "__main__":
    main()
