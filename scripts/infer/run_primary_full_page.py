#!/usr/bin/env python3
"""Run a frozen EnsExam-GAN primary checkpoint over full source-image pages.

This is deliberately an inference-only entry point for blind evaluation. It
accepts source images plus a primary configuration/checkpoint, never derives
or opens label paths, and always writes label-free metadata alongside the
predictions. Labels belong exclusively to the separate post-freeze scorer.
"""

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

from config_loader import load_config  # noqa: E402
from networks.generator import Generator  # noqa: E402
from utils.page_inference import infer_full_page  # noqa: E402
from utils.path_utils import normalize_path  # noqa: E402


FORBIDDEN_SAMPLE_COMPONENTS = {"label", "labels", "target", "targets", "all_labels"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-file", required=True, help="Frozen list of source-image paths only.")
    parser.add_argument("--output-dir", required=True, help="New directory for frozen predictions.")
    parser.add_argument("--primary-config", required=True, help="EnsExam-GAN configuration used to construct Generator.")
    parser.add_argument("--primary-weights", required=True, help="Frozen EnsExam-GAN checkpoint.")
    parser.add_argument(
        "--image-dir",
        default="",
        help="Optional directory used only to resolve bare filenames in --samples-file.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--page-overlap", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--copy-input-outside-mask",
        choices=("none", "ms", "mb"),
        default="none",
        help="Optional fixed global postprocess; the blind primary default is pure model output.",
    )
    parser.add_argument("--copy-mask-threshold", type=int, default=70)
    parser.add_argument(
        "--copy-mask-threshold-auto",
        choices=("none", "mb_cov8_step"),
        default="none",
        help="Optional fixed predicted-mask heuristic; never uses labels or targets.",
    )
    parser.add_argument("--copy-mask-dilate", type=int, default=0)
    parser.add_argument(
        "--skip-label-metrics",
        action="store_true",
        required=True,
        help="Required protocol marker. This script never reads labels in any mode.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_generator(config_path: str, weights_path: str, device: torch.device) -> Generator:
    config = load_config(normalize_path(config_path))
    generator = Generator(cfg=config["model"]).to(device)
    checkpoint = torch.load(normalize_path(weights_path), map_location=device, weights_only=False)
    if "G_state_dict" in checkpoint:
        generator.load_state_dict(checkpoint["G_state_dict"])
    elif "state_dict" in checkpoint:
        generator.load_state_dict(checkpoint["state_dict"])
    else:
        generator.load_state_dict(checkpoint)
    generator.eval()
    return generator


def assert_source_image_path(path: Path) -> None:
    """Reject direct or symlinked target/label paths before opening any image."""
    for candidate in (path, path.resolve()):
        components = {part.lower() for part in candidate.parts}
        if components & FORBIDDEN_SAMPLE_COMPONENTS:
            raise ValueError(f"samples-file appears to include a target/label path: {path}")


def read_sample_paths(samples_file: Path, image_dir: Path | None = None) -> list[Path]:
    if not samples_file.is_file():
        raise FileNotFoundError(samples_file)
    paths: list[Path] = []
    seen: set[Path] = set()
    for line in samples_file.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        path = Path(value)
        if image_dir is not None and len(path.parts) == 1:
            path = image_dir / path
        assert_source_image_path(path)
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(f"samples-file contains duplicate entries: {path}")
        seen.add(resolved)
        paths.append(path)
    if not paths:
        raise ValueError("samples-file contains no source images")
    return paths


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def copy_input_outside_mask(
    prediction_bgr: np.ndarray,
    input_bgr: np.ndarray,
    mask_u8: np.ndarray,
    *,
    threshold: int,
    dilate: int,
) -> np.ndarray:
    edit_mask = mask_u8 >= threshold
    if dilate > 0:
        kernel = np.ones((3, 3), np.uint8)
        edit_mask = cv2.dilate(edit_mask.astype(np.uint8), kernel, iterations=dilate) > 0
    restricted = input_bgr.copy()
    restricted[edit_mask] = prediction_bgr[edit_mask]
    return restricted


def auto_copy_mask_threshold(mask_name: str, mask_u8: np.ndarray, mode: str, default: int) -> tuple[int, float]:
    if mode == "none":
        return default, 0.0
    if mode != "mb_cov8_step":
        raise ValueError(f"Unsupported copy-mask threshold mode: {mode}")
    if mask_name != "mb":
        raise ValueError("--copy-mask-threshold-auto mb_cov8_step requires --copy-input-outside-mask mb")
    coverage = float((mask_u8 >= 8).mean())
    if coverage <= 0.129:
        return 8, coverage
    if coverage <= 0.421:
        return 76, coverage
    return 160, coverage


def prediction_name(image_path: Path) -> str:
    return f"{image_path.stem}.png"


def main() -> None:
    args = parse_args()
    if args.copy_mask_threshold_auto != "none" and args.copy_input_outside_mask != "mb":
        raise ValueError("--copy-mask-threshold-auto requires --copy-input-outside-mask mb")
    if args.page_overlap < 0:
        raise ValueError("--page-overlap must be non-negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.copy_mask_dilate < 0:
        raise ValueError("--copy-mask-dilate must be non-negative")

    image_dir = Path(args.image_dir) if args.image_dir else None
    sample_paths = read_sample_paths(Path(args.samples_file), image_dir)
    output_names = [prediction_name(path) for path in sample_paths]
    if len(set(output_names)) != len(output_names):
        raise ValueError("samples-file has colliding prediction filenames")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    pred_dir = output_dir / "pred"
    pred_dir.mkdir()

    config_path = Path(normalize_path(args.primary_config))
    weights_path = Path(normalize_path(args.primary_weights))
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    config_sha256 = sha256_file(config_path)
    weights_sha256 = sha256_file(weights_path)

    device = pick_device(args.device)
    generator = load_generator(str(config_path), str(weights_path), device)
    rows: list[dict[str, object]] = []
    for index, image_path in enumerate(sample_paths, start=1):
        input_bgr = read_bgr(image_path)
        outputs = infer_full_page(
            generator,
            cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB),
            device,
            overlap=args.page_overlap,
            batch_size=args.batch_size,
        )
        prediction_bgr = cv2.cvtColor(outputs["icomp"], cv2.COLOR_RGB2BGR)
        copy_mask_cov8 = 0.0
        copy_mask_threshold = args.copy_mask_threshold
        if args.copy_input_outside_mask != "none":
            selected_mask = outputs[args.copy_input_outside_mask]
            copy_mask_threshold, copy_mask_cov8 = auto_copy_mask_threshold(
                args.copy_input_outside_mask,
                selected_mask,
                args.copy_mask_threshold_auto,
                args.copy_mask_threshold,
            )
            prediction_bgr = copy_input_outside_mask(
                prediction_bgr,
                input_bgr,
                selected_mask,
                threshold=copy_mask_threshold,
                dilate=args.copy_mask_dilate,
            )

        pred_path = pred_dir / prediction_name(image_path)
        if not cv2.imwrite(str(pred_path), prediction_bgr):
            raise RuntimeError(f"Could not write prediction: {pred_path}")
        rows.append({
            "file": image_path.name,
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            "pred_path": str(pred_path),
            "pred_sha256": sha256_file(pred_path),
            "metrics_skipped": "1",
            "primary_config_sha256": config_sha256,
            "primary_weights_sha256": weights_sha256,
            "page_overlap": args.page_overlap,
            "batch_size": args.batch_size,
            "copy_input_outside_mask": args.copy_input_outside_mask,
            "copy_mask_threshold": copy_mask_threshold,
            "copy_mask_threshold_auto": args.copy_mask_threshold_auto,
            "copy_mask_dilate": args.copy_mask_dilate,
            "copy_mask_cov8": copy_mask_cov8,
        })
        print(f"{index} {image_path.name} -> {pred_path}", flush=True)

    metrics_csv = output_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("summary label_metrics=skipped", flush=True)
    print(f"metrics_csv: {metrics_csv}", flush=True)


if __name__ == "__main__":
    main()
