#!/usr/bin/env python3
"""Apply an external-text-layout conditioned monotonic candidate."""

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
from scripts.infer.run_monotonic_residual_erase_candidate import (  # noqa: E402
    apply_monotonic_candidate_gate,
)
from scripts.infer.run_sign_separated_residual_candidate import (  # noqa: E402
    find_baseline_prediction,
    read_bgr,
    read_sample_paths,
    sha256_file,
    tile_starts,
)
from scripts.train.train_external_text_layout_conditioned_monotonic import (  # noqa: E402
    CONDITIONED_INPUT_CHANNELS,
    find_layout_npz,
    load_layout_grids,
)


def pad_tile(array: np.ndarray, tile_size: int, *, constant_value: float) -> np.ndarray:
    height, width = array.shape[:2]
    if height == tile_size and width == tile_size:
        return array
    padded = np.full(
        (tile_size, tile_size, *array.shape[2:]),
        constant_value,
        dtype=array.dtype,
    )
    padded[:height, :width] = array
    return padded


def to_conditioned_tensor(
    rgb: np.ndarray,
    occupancy: np.ndarray,
    confidence: np.ndarray,
) -> torch.Tensor:
    rgb_tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    layout_tensor = np.stack(
        [occupancy.astype(np.float32), confidence.astype(np.float32)]
    )
    return torch.from_numpy(np.concatenate([rgb_tensor, layout_tensor], axis=0))


def infer_conditioned_candidate_components(
    model: MonotonicResidualEraseCleanupNet,
    baseline_bgr: np.ndarray,
    occupancy: np.ndarray,
    confidence: np.ndarray,
    device: torch.device,
    *,
    tile_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    if model.input_channels != CONDITIONED_INPUT_CHANNELS:
        raise ValueError("conditioned candidate requires a 5-channel model")
    height, width = baseline_bgr.shape[:2]
    if occupancy.shape != (height, width) or confidence.shape != (height, width):
        raise ValueError("layout grid shape must match baseline")
    baseline_rgb = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2RGB)
    candidate_accum = np.zeros((height, width, 3), dtype=np.float32)
    probability_accum = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    with torch.no_grad():
        for y1 in tile_starts(height, tile_size, stride):
            for x1 in tile_starts(width, tile_size, stride):
                rgb_tile = baseline_rgb[y1 : y1 + tile_size, x1 : x1 + tile_size]
                occupancy_tile = occupancy[y1 : y1 + tile_size, x1 : x1 + tile_size]
                confidence_tile = confidence[y1 : y1 + tile_size, x1 : x1 + tile_size]
                tile_height, tile_width = rgb_tile.shape[:2]
                if tile_height != tile_size or tile_width != tile_size:
                    rgb_tile = pad_tile(rgb_tile, tile_size, constant_value=255)
                    occupancy_tile = pad_tile(
                        occupancy_tile,
                        tile_size,
                        constant_value=0,
                    )
                    confidence_tile = pad_tile(
                        confidence_tile,
                        tile_size,
                        constant_value=0.0,
                    )
                tensor = to_conditioned_tensor(
                    rgb_tile,
                    occupancy_tile,
                    confidence_tile,
                ).unsqueeze(0).to(device)
                components = model.forward_components(tensor)
                candidate = (
                    components["candidate"]
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    .clip(0.0, 1.0)
                    * 255.0
                )
                edit_probability = (
                    components["edit_alpha"].squeeze(0).squeeze(0).cpu().numpy()
                )
                candidate_accum[
                    y1 : y1 + tile_height, x1 : x1 + tile_width
                ] += candidate[:tile_height, :tile_width]
                probability_accum[
                    y1 : y1 + tile_height, x1 : x1 + tile_width
                ] += edit_probability[:tile_height, :tile_width]
                weight[y1 : y1 + tile_height, x1 : x1 + tile_width] += 1.0

    denominator = np.clip(weight, 1e-6, None)
    candidate_rgb = np.clip(
        np.rint(candidate_accum / denominator[:, :, None]), 0.0, 255.0
    ).astype(np.uint8)
    edit_probability = probability_accum / denominator
    return cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2BGR), edit_probability


def load_candidate(
    checkpoint: Path,
    device: torch.device,
) -> MonotonicResidualEraseCleanupNet:
    model = load_monotonic_residual_erase_model(checkpoint, device)
    if not isinstance(model, MonotonicResidualEraseCleanupNet):
        raise TypeError("checkpoint is not a monotonic residual-erase candidate")
    if model.input_channels != CONDITIONED_INPUT_CHANNELS:
        raise ValueError("checkpoint is not external-text-layout conditioned")
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-file", type=Path, required=True)
    parser.add_argument("--baseline-pred-dir", type=Path, required=True)
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu"
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
        layout_path = find_layout_npz(args.layout_dir, sample.name)
        occupancy, confidence = load_layout_grids(
            layout_path,
            expected_shape=baseline.shape[:2],
        )
        candidate, edit_probability = infer_conditioned_candidate_components(
            model,
            baseline,
            occupancy,
            confidence,
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
                "layout_path": str(layout_path),
                "layout_sha256": sha256_file(layout_path),
                "pred_path": str(output_path),
                "pred_sha256": sha256_file(output_path),
                "checkpoint_sha256": checkpoint_hash,
                "input_channels": CONDITIONED_INPUT_CHANNELS,
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
