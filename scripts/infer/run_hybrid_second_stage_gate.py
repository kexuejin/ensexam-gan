#!/usr/bin/env python3
"""Run a page-level hybrid gate between baseline and candidate second-stage outputs.

This is a productization bridge for candidate checkpoints that reduce residual
ink but slightly increase over-erasure when enabled on every page. The script
computes inference-time safety features from the candidate primary output and
uses them to choose either:

* baseline final prediction from --baseline-pred-dir, or
* candidate primary + second-stage residual repair.

No labels are required for the gate decision. Labels are used only when present
to write offline evaluation metrics.
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
from patch_cleanup_erasemap import (  # noqa: E402
    infer_full_page as infer_cleanup_page,
    load_model as load_cleanup_model,
)
from utils.page_inference import infer_full_page as infer_ensexam_page  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-pred-dir", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--candidate-weights", required=True)
    parser.add_argument("--cleanup-checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--page-overlap", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-copy-mask", choices=("none", "ms", "mb"), default="mb")
    parser.add_argument("--candidate-copy-threshold", type=int, default=98)
    parser.add_argument("--candidate-copy-threshold-auto", choices=("none", "mb_cov8_step"), default="none")
    parser.add_argument("--candidate-copy-dilate", type=int, default=0)
    parser.add_argument("--cleanup-tile-size", type=int, default=160)
    parser.add_argument("--cleanup-stride", type=int, default=160)
    parser.add_argument("--cleanup-alpha-threshold", type=float, default=0.3)
    parser.add_argument("--base-edit-threshold", type=float, default=12.0)
    parser.add_argument("--second-delta-threshold", type=float, default=32.0)
    parser.add_argument("--dark-threshold", type=int, default=0)
    parser.add_argument(
        "--max-brighten-delta",
        type=float,
        default=1e9,
        help=(
            "Optional inference-time safety cap for pixels where the second stage "
            "brightens the primary prediction. The default disables this guard."
        ),
    )
    parser.add_argument("--change-threshold", type=int, default=12)
    parser.add_argument("--eval-threshold", type=int, default=12)
    parser.add_argument("--min-copy-mask-cov8", type=float, default=0.18436555)
    parser.add_argument("--max-copy-mask-cov8", type=float, default=1.0)
    parser.add_argument("--min-primary-edit-px", type=int, default=0)
    parser.add_argument("--max-primary-edit-px", type=int, default=107112)
    parser.add_argument("--min-primary-p95-edit-delta", type=float, default=0.0)
    parser.add_argument("--max-primary-p95-edit-delta", type=float, default=1e9)
    parser.add_argument("--min-second-stage-gate-ratio", type=float, default=0.0)
    parser.add_argument("--max-second-stage-gate-ratio", type=float, default=1.0)
    parser.add_argument(
        "--candidate-interval-rule",
        action="append",
        default=[],
        metavar="MIN_COV8,MAX_COV8,MIN_EDIT_PX,MAX_EDIT_PX,MIN_P95,MAX_P95,MIN_GATE,MAX_GATE",
        help=(
            "Optional candidate selector interval. May be repeated; candidate is used when "
            "any interval matches. Overrides the individual min/max selector arguments."
        ),
    )
    parser.add_argument("--save-candidate", action="store_true")
    return parser.parse_args()


IntervalBox = tuple[float, float, float, float, float, float, float, float]


def parse_interval_box(value: str) -> IntervalBox:
    parts = value.split(",")
    if len(parts) != 8 or not all(parts):
        raise ValueError(
            f"Invalid --candidate-interval-rule {value!r}; expected "
            "MIN_COV8,MAX_COV8,MIN_EDIT_PX,MAX_EDIT_PX,MIN_P95,MAX_P95,MIN_GATE,MAX_GATE"
        )
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def prediction_path(pred_dir: Path, image_path: Path) -> Path:
    for name in (f"{image_path.stem}.png", f"{image_path.stem}.clean.png", image_path.name):
        candidate = pred_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No prediction found for {image_path.name} in {pred_dir}")


def gated_blend(
    input_bgr: np.ndarray,
    primary_bgr: np.ndarray,
    second_bgr: np.ndarray,
    base_edit_threshold: float,
    second_delta_threshold: float,
    dark_threshold: int,
    max_brighten_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    base_edit = cv2.absdiff(primary_bgr, input_bgr).mean(axis=2)
    second_delta = cv2.absdiff(second_bgr, primary_bgr).mean(axis=2)
    gate = (base_edit >= base_edit_threshold) & (second_delta >= second_delta_threshold)
    if dark_threshold > 0:
        gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
        gate &= gray <= dark_threshold
    if max_brighten_delta < 1e9:
        primary_gray = cv2.cvtColor(primary_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        second_gray = cv2.cvtColor(second_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        gate &= (second_gray - primary_gray) <= max_brighten_delta
    merged = primary_bgr.copy()
    merged[gate] = second_bgr[gate]
    return merged, gate


def candidate_primary(
    args: argparse.Namespace,
    generator,
    input_bgr: np.ndarray,
    device,
) -> tuple[np.ndarray, dict[str, float]]:
    input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
    outputs = infer_ensexam_page(
        generator,
        input_rgb,
        device,
        overlap=args.page_overlap,
        batch_size=args.batch_size,
    )
    pred_bgr = cv2.cvtColor(outputs["icomp"], cv2.COLOR_RGB2BGR)
    copy_mask_cov8 = 0.0
    copy_threshold = args.candidate_copy_threshold
    if args.candidate_copy_mask != "none":
        selected_mask = outputs[args.candidate_copy_mask]
        copy_mask_cov8 = float((selected_mask >= 8).mean())
        copy_threshold, _auto_cov8 = auto_copy_mask_threshold(
            args.candidate_copy_mask,
            selected_mask,
            args.candidate_copy_threshold_auto,
            args.candidate_copy_threshold,
        )
        pred_bgr = copy_input_outside_mask(
            pred_bgr,
            input_bgr,
            selected_mask,
            threshold=copy_threshold,
            dilate=args.candidate_copy_dilate,
        )
    primary_delta = cv2.absdiff(pred_bgr, input_bgr).mean(axis=2)
    edit_mask = primary_delta >= args.base_edit_threshold
    features = {
        "copy_mask_cov8": copy_mask_cov8,
        "candidate_copy_threshold": float(copy_threshold),
        "primary_edit_ratio": float(edit_mask.mean()),
        "primary_edit_px": int(edit_mask.sum()),
        "primary_mean_edit_delta": float(primary_delta.mean()),
        "primary_p95_edit_delta": float(np.percentile(primary_delta, 95)),
    }
    return pred_bgr, features


def choose_candidate(
    features: dict[str, float],
    second_stage_gate_ratio: float,
    min_cov8: float,
    max_cov8: float,
    min_edit_px: int,
    max_edit_px: int,
    min_p95_edit_delta: float,
    max_p95_edit_delta: float,
    min_second_stage_gate_ratio: float,
    max_second_stage_gate_ratio: float,
) -> bool:
    return (
        min_cov8 <= features["copy_mask_cov8"] <= max_cov8
        and min_edit_px <= features["primary_edit_px"] <= max_edit_px
        and min_p95_edit_delta
        <= features["primary_p95_edit_delta"]
        <= max_p95_edit_delta
        and min_second_stage_gate_ratio
        <= second_stage_gate_ratio
        <= max_second_stage_gate_ratio
    )


def choose_candidate_from_interval_boxes(
    features: dict[str, float],
    second_stage_gate_ratio: float,
    boxes: list[IntervalBox],
) -> bool:
    for box in boxes:
        min_cov8, max_cov8, min_edit_px, max_edit_px, min_p95, max_p95, min_gate, max_gate = box
        if choose_candidate(
            features,
            second_stage_gate_ratio,
            min_cov8=min_cov8,
            max_cov8=max_cov8,
            min_edit_px=int(min_edit_px),
            max_edit_px=int(max_edit_px),
            min_p95_edit_delta=min_p95,
            max_p95_edit_delta=max_p95,
            min_second_stage_gate_ratio=min_gate,
            max_second_stage_gate_ratio=max_gate,
        ):
            return True
    return False


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "pred"
    candidate_dir = output_dir / "candidate"
    primary_dir = output_dir / "candidate_primary"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if args.save_candidate:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        primary_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    generator = load_generator(args.candidate_config, args.candidate_weights, device)
    cleanup_model = load_cleanup_model(Path(args.cleanup_checkpoint), device)
    baseline_dir = Path(args.baseline_pred_dir)
    candidate_interval_boxes = [parse_interval_box(value) for value in args.candidate_interval_rule]

    rows: list[dict[str, object]] = []
    for index, image_path in enumerate(read_sample_paths(Path(args.samples_file)), start=1):
        input_bgr = read_bgr(image_path)
        baseline_bgr = ensure_same_size(read_bgr(prediction_path(baseline_dir, image_path)), input_bgr)
        primary_bgr, features = candidate_primary(args, generator, input_bgr, device)
        second_bgr = infer_cleanup_page(
            cleanup_model,
            primary_bgr,
            device,
            tile_size=args.cleanup_tile_size,
            stride=args.cleanup_stride,
            alpha_threshold=args.cleanup_alpha_threshold,
        )
        candidate_bgr, second_stage_gate = gated_blend(
            input_bgr,
            primary_bgr,
            second_bgr,
            base_edit_threshold=args.base_edit_threshold,
            second_delta_threshold=args.second_delta_threshold,
            dark_threshold=args.dark_threshold,
            max_brighten_delta=args.max_brighten_delta,
        )
        second_stage_gate_ratio = float(second_stage_gate.mean())
        if candidate_interval_boxes:
            use_candidate = choose_candidate_from_interval_boxes(
                features,
                second_stage_gate_ratio,
                candidate_interval_boxes,
            )
        else:
            use_candidate = choose_candidate(
                features,
                second_stage_gate_ratio,
                min_cov8=args.min_copy_mask_cov8,
                max_cov8=args.max_copy_mask_cov8,
                min_edit_px=args.min_primary_edit_px,
                max_edit_px=args.max_primary_edit_px,
                min_p95_edit_delta=args.min_primary_p95_edit_delta,
                max_p95_edit_delta=args.max_primary_p95_edit_delta,
                min_second_stage_gate_ratio=args.min_second_stage_gate_ratio,
                max_second_stage_gate_ratio=args.max_second_stage_gate_ratio,
            )
        final_bgr = candidate_bgr if use_candidate else baseline_bgr

        pred_path = pred_dir / f"{image_path.stem}.png"
        cv2.imwrite(str(pred_path), final_bgr)
        if args.save_candidate:
            cv2.imwrite(str(candidate_dir / f"{image_path.stem}.png"), candidate_bgr)
            cv2.imwrite(str(primary_dir / f"{image_path.stem}.png"), primary_bgr)

        row: dict[str, object] = {
            "file": image_path.name,
            "image_path": str(image_path),
            "pred_path": str(pred_path),
            "source": "candidate" if use_candidate else "baseline",
            "use_candidate": int(use_candidate),
            "min_copy_mask_cov8": args.min_copy_mask_cov8,
            "max_copy_mask_cov8": args.max_copy_mask_cov8,
            "min_primary_edit_px": args.min_primary_edit_px,
            "max_primary_edit_px": args.max_primary_edit_px,
            "min_primary_p95_edit_delta": args.min_primary_p95_edit_delta,
            "max_primary_p95_edit_delta": args.max_primary_p95_edit_delta,
            "min_second_stage_gate_ratio": args.min_second_stage_gate_ratio,
            "max_second_stage_gate_ratio": args.max_second_stage_gate_ratio,
            "max_brighten_delta": args.max_brighten_delta,
            "candidate_interval_rules": ";".join(args.candidate_interval_rule),
            "second_stage_gate_ratio": second_stage_gate_ratio,
            **features,
        }
        try:
            label_bgr = ensure_same_size(read_bgr(label_path_for(image_path)), input_bgr)
            row.update(compute_residual_metrics(
                input_bgr,
                label_bgr,
                final_bgr,
                change_threshold=args.change_threshold,
                eval_threshold=args.eval_threshold,
            ))
        except Exception as exc:  # noqa: BLE001 - labels are optional for product inference.
            row["metrics_error"] = str(exc)
        rows.append(row)
        print(
            f"{index} {image_path.name} source={row['source']} "
            f"cov8={features['copy_mask_cov8']:.6f} edit_px={features['primary_edit_px']} -> {pred_path}",
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
        selected = sum(int(row["use_candidate"]) for row in rows)
        print(f"summary residual={residual:.6f} overerase={overerase:.6f} selected={selected}/{len(rows)}", flush=True)
    print(f"metrics_csv: {metrics_csv}", flush=True)


if __name__ == "__main__":
    main()
