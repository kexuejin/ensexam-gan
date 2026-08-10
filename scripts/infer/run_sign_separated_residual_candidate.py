#!/usr/bin/env python3
"""Apply a sign-separated candidate after the frozen current pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer.patch_cleanup_erasemap import (  # noqa: E402
    SignSeparatedResidualDeltaCleanupNet,
    load_model,
    resolve_device,
)


FORBIDDEN_SAMPLE_COMPONENTS = {
    "label",
    "labels",
    "target",
    "targets",
    "all_labels",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sample_paths(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[Path] = []
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        sample = Path(value)
        components = {part.lower() for part in sample.parts}
        if components & FORBIDDEN_SAMPLE_COMPONENTS:
            raise ValueError(f"sample path exposes a target or label: {sample}")
        output_name = f"{sample.stem}.png"
        if output_name in names:
            raise ValueError(f"sample list has colliding output name: {output_name}")
        names.add(output_name)
        rows.append(sample)
    if not rows:
        raise ValueError("sample list is empty")
    return rows


def find_baseline_prediction(directory: Path, sample: Path) -> Path:
    for name in (f"{sample.stem}.png", f"{sample.stem}.clean.png", sample.name):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no frozen pipeline prediction for {sample.name} in {directory}"
    )


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final = max(0, length - tile_size)
    if not starts or starts[-1] != final:
        starts.append(final)
    return starts


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(
        rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    )


def infer_candidate_components(
    model: SignSeparatedResidualDeltaCleanupNet,
    baseline_bgr: np.ndarray,
    device: torch.device,
    *,
    tile_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = baseline_bgr.shape[:2]
    baseline_rgb = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2RGB)
    candidate_accum = np.zeros((height, width, 3), dtype=np.float32)
    probability_accum = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    with torch.no_grad():
        for y1 in tile_starts(height, tile_size, stride):
            for x1 in tile_starts(width, tile_size, stride):
                tile = baseline_rgb[y1 : y1 + tile_size, x1 : x1 + tile_size]
                tile_height, tile_width = tile.shape[:2]
                if tile_height != tile_size or tile_width != tile_size:
                    padded = np.full(
                        (tile_size, tile_size, 3), 255, dtype=np.uint8
                    )
                    padded[:tile_height, :tile_width] = tile
                    tile = padded
                tensor = to_tensor(tile).unsqueeze(0).to(device)
                components = model.forward_components(tensor)
                candidate = (
                    components["candidate"]
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    .clip(0.0, 1.0)
                    * 255.0
                ).astype(np.float32)
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
        candidate_accum / denominator[:, :, None], 0.0, 255.0
    ).astype(np.uint8)
    edit_probability = probability_accum / denominator
    return cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2BGR), edit_probability


def apply_candidate_gate(
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
    delta = cv2.absdiff(candidate_bgr, baseline_bgr).mean(axis=2)
    gate = (
        (edit_probability >= edit_probability_threshold)
        & (delta >= minimum_delta_threshold)
    )
    merged = baseline_bgr.copy()
    merged[gate] = candidate_bgr[gate]
    return merged, gate, delta


def load_candidate(
    checkpoint: Path,
    device: torch.device,
) -> SignSeparatedResidualDeltaCleanupNet:
    model = load_model(checkpoint, device)
    if not isinstance(model, SignSeparatedResidualDeltaCleanupNet):
        raise TypeError("checkpoint is not a sign-separated residual candidate")
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
        merged, gate, delta = apply_candidate_gate(
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
                "candidate_delta_mean": float(delta.mean()),
                "candidate_delta_max": float(delta.max()),
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
