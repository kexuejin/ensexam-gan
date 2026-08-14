#!/usr/bin/env python3
"""Apply a monotonic residual-erase candidate after the frozen pipeline."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer.monotonic_residual_erase import (  # noqa: E402
    MonotonicResidualEraseCleanupNet,
    load_monotonic_residual_erase_model,
)
from scripts.infer.patch_cleanup_erasemap import resolve_device  # noqa: E402
from scripts.infer.run_sign_separated_residual_candidate import (  # noqa: E402
    find_baseline_prediction,
    infer_candidate_components,
    read_bgr,
    read_sample_paths,
    sha256_file,
)


def apply_monotonic_candidate_gate(
    baseline_bgr: np.ndarray,
    candidate_bgr: np.ndarray,
    edit_probability: np.ndarray,
    *,
    edit_probability_threshold: float,
    minimum_delta_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if baseline_bgr.shape != candidate_bgr.shape:
        raise ValueError("baseline/candidate shape mismatch")
    if edit_probability.shape != baseline_bgr.shape[:2]:
        raise ValueError("edit probability shape mismatch")
    signed_delta = candidate_bgr.astype(np.int16) - baseline_bgr.astype(np.int16)
    if bool((signed_delta < 0).any()):
        raise ValueError("monotonic candidate darkened the frozen baseline")
    mean_delta = signed_delta.mean(axis=2)
    gate = (
        (edit_probability >= edit_probability_threshold)
        & (mean_delta >= minimum_delta_threshold)
    )
    merged = baseline_bgr.copy()
    merged[gate] = candidate_bgr[gate]
    return merged, gate, mean_delta


def load_candidate(
    checkpoint: Path,
    device: torch.device,
) -> MonotonicResidualEraseCleanupNet:
    model = load_monotonic_residual_erase_model(checkpoint, device)
    if not isinstance(model, MonotonicResidualEraseCleanupNet):
        raise TypeError("checkpoint is not a monotonic residual-erase candidate")
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-file", type=Path, required=True)
    parser.add_argument("--baseline-pred-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=160)
    parser.add_argument("--edit-probability-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-delta-threshold", type=float, default=12.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.tile_size <= 0 or args.stride <= 0 or args.stride > args.tile_size:
        raise ValueError("tile/stride configuration is invalid")
    if not 0.0 <= args.edit_probability_threshold <= 1.0:
        raise ValueError("edit probability threshold must be in [0, 1]")
    if args.minimum_delta_threshold <= 0.0:
        raise ValueError("minimum delta threshold must be positive")

    device = resolve_device(args.device)
    model = load_candidate(args.checkpoint, device)
    maximum_delta = model.residual_delta_bound * 255.0
    if args.minimum_delta_threshold > maximum_delta:
        raise ValueError(
            "minimum delta threshold exceeds the candidate output bound: "
            f"{args.minimum_delta_threshold} > {maximum_delta}"
        )
    samples = read_sample_paths(args.samples_file)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prediction_dir = args.output_dir / "pred"
    prediction_dir.mkdir()
    checkpoint_hash = sha256_file(args.checkpoint)
    rows: list[dict[str, object]] = []
    for index, sample in enumerate(samples, start=1):
        baseline_path = find_baseline_prediction(args.baseline_pred_dir, sample)
        baseline = read_bgr(baseline_path)
        candidate, edit_probability = infer_candidate_components(
            model,
            baseline,
            device,
            tile_size=args.tile_size,
            stride=args.stride,
        )
        merged, gate, mean_delta = apply_monotonic_candidate_gate(
            baseline,
            candidate,
            edit_probability,
            edit_probability_threshold=args.edit_probability_threshold,
            minimum_delta_threshold=args.minimum_delta_threshold,
        )
        output_path = prediction_dir / f"{sample.stem}.png"
        if not cv2.imwrite(str(output_path), merged):
            raise RuntimeError(f"could not write candidate prediction: {output_path}")
        rows.append(
            {
                "file": sample.name,
                "baseline_path": str(baseline_path),
                "baseline_sha256": sha256_file(baseline_path),
                "pred_path": str(output_path),
                "pred_sha256": sha256_file(output_path),
                "checkpoint_sha256": checkpoint_hash,
                "tile_size": args.tile_size,
                "stride": args.stride,
                "edit_probability_threshold": args.edit_probability_threshold,
                "minimum_delta_threshold": args.minimum_delta_threshold,
                "maximum_delta_bound": maximum_delta,
                "applied_ratio": float(gate.mean()),
                "candidate_delta_mean": float(mean_delta.mean()),
                "candidate_delta_max": float(mean_delta.max()),
                "darkened_channel_count": 0,
                "labels_opened": 0,
            }
        )
        print(
            f"{index}/{len(samples)} {sample.name} "
            f"applied={float(gate.mean()):.6f}",
            flush=True,
        )

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"metrics_csv: {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
