#!/usr/bin/env python3
"""Train the registered external-text-layout conditioned monotonic probe.

This trainer is intentionally single-purpose. It consumes recovered
second-stage RGB predictions plus frozen external text-layout occupancy and
confidence grids, learns only bounded nonnegative RGB deltas, and exposes no
validation, resume, model-type, or target-routing alternatives.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer.monotonic_residual_erase import (  # noqa: E402
    MODEL_TYPE,
    MonotonicResidualEraseCleanupNet,
)
from scripts.infer.patch_cleanup_erasemap import resolve_device  # noqa: E402


MASK_SOURCE = "target_luma_delta"
CONDITIONED_INPUT_CHANNELS = 5
RESIDUAL_DELTA_BOUND = 0.08
LOSS_TERM_NAMES = (
    "loss",
    "support_positive_bce",
    "support_preserve_bce",
    "bright_magnitude_l1",
    "preserve_delta_l1",
)


def find_prediction(directory: Path, file_name: str) -> Path:
    stem = Path(file_name).stem
    for name in (f"{stem}.png", f"{stem}.clean.png", file_name):
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No prediction for {file_name} in {directory}")


def find_layout_npz(directory: Path, file_name: str) -> Path:
    path = directory / f"{Path(file_name).stem}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"No external text-layout NPZ for {file_name}")
    return path


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_layout_grids(
    path: Path,
    *,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if not {"text_occupancy", "text_confidence"} <= set(payload.files):
            raise ValueError("external text-layout NPZ lacks registered grids")
        occupancy = payload["text_occupancy"]
        confidence = payload["text_confidence"]
    if occupancy.dtype != np.uint8 or occupancy.shape != expected_shape:
        raise ValueError("external text occupancy changed")
    if confidence.dtype != np.float32 or confidence.shape != expected_shape:
        raise ValueError("external text confidence changed")
    if not np.isin(occupancy, [0, 1]).all():
        raise ValueError("external text occupancy must be binary")
    if (
        not np.isfinite(confidence).all()
        or np.any(confidence < 0.0)
        or np.any(confidence > 1.0)
    ):
        raise ValueError("external text confidence must stay in [0, 1]")
    return occupancy.astype(np.float32), confidence.astype(np.float32)


def pad_to_size(
    array: np.ndarray,
    size: int,
    *,
    constant_value: float | None = None,
) -> np.ndarray:
    height, width = array.shape[:2]
    pad_height = max(size - height, 0)
    pad_width = max(size - width, 0)
    if pad_height == 0 and pad_width == 0:
        return array
    border_type = cv2.BORDER_REPLICATE if constant_value is None else cv2.BORDER_CONSTANT
    kwargs = {} if constant_value is None else {"value": constant_value}
    return cv2.copyMakeBorder(
        array,
        0,
        pad_height,
        0,
        pad_width,
        border_type,
        **kwargs,
    )


class ExternalTextLayoutConditionedPatchDataset(Dataset):
    """Load registered train-only RGB/layout/target patches."""

    def __init__(
        self,
        *,
        data_root: Path,
        split: str,
        input_dir: Path,
        layout_dir: Path,
        patch_index_file: Path,
        tile_size: int,
    ):
        if split != "train":
            raise ValueError("conditioned monotonic training only accepts split=train")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        self.label_dir = data_root / split / "all_labels"
        self.input_dir = input_dir
        self.layout_dir = layout_dir
        self.tile_size = tile_size
        with patch_index_file.open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise RuntimeError(f"empty patch index: {patch_index_file}")
        required = {"file", "x1", "y1", "x2", "y2"}
        missing = required - set(self.rows[0])
        if missing:
            raise ValueError(f"patch index lacks columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        file_name = row["file"]
        if Path(file_name).name != file_name:
            raise ValueError(f"invalid patch filename: {file_name}")
        x1, y1, x2, y2 = (
            int(row["x1"]),
            int(row["y1"]),
            int(row["x2"]),
            int(row["y2"]),
        )
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError(f"invalid patch coordinates for {file_name}")
        if x2 - x1 > self.tile_size or y2 - y1 > self.tile_size:
            raise ValueError(f"patch exceeds tile size for {file_name}")

        input_image = read_rgb(find_prediction(self.input_dir, file_name))
        target_image = read_rgb(self.label_dir / file_name)
        if input_image.shape != target_image.shape:
            raise ValueError(
                f"input/target shape mismatch for {file_name}: "
                f"{input_image.shape} != {target_image.shape}"
            )
        height, width = input_image.shape[:2]
        if x2 > width or y2 > height:
            raise ValueError(f"patch lies outside image for {file_name}")
        occupancy, confidence = load_layout_grids(
            find_layout_npz(self.layout_dir, file_name),
            expected_shape=(height, width),
        )

        inp = pad_to_size(input_image[y1:y2, x1:x2], self.tile_size)
        target = pad_to_size(target_image[y1:y2, x1:x2], self.tile_size)
        occupancy_patch = pad_to_size(
            occupancy[y1:y2, x1:x2],
            self.tile_size,
            constant_value=0.0,
        )
        confidence_patch = pad_to_size(
            confidence[y1:y2, x1:x2],
            self.tile_size,
            constant_value=0.0,
        )
        rgb_tensor = torch.from_numpy(
            inp.transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        layout_tensor = torch.from_numpy(
            np.stack([occupancy_patch, confidence_patch]).astype(np.float32)
        )
        target_tensor = torch.from_numpy(
            target.transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        return torch.cat([rgb_tensor, layout_tensor], dim=0), target_tensor


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.expand_as(values)
    denominator = weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
    per_sample = (values * weight).sum(dim=(1, 2, 3)) / denominator
    return per_sample.mean()


def compute_conditioned_loss_terms(
    model: torch.nn.Module,
    features: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    if not isinstance(model, MonotonicResidualEraseCleanupNet):
        raise TypeError("conditioned loss requires monotonic residual-erase model")
    if model.input_channels != CONDITIONED_INPUT_CHANNELS:
        raise ValueError("conditioned model must use five input channels")
    if features.ndim != 4 or features.shape[1] != CONDITIONED_INPUT_CHANNELS:
        raise ValueError("conditioned features must be NCHW with 5 channels")
    baseline_rgb = features[:, :3]
    if target.shape != baseline_rgb.shape:
        raise ValueError("target RGB must match the baseline RGB shape")

    components = model.forward_components(features)
    input_luma = baseline_rgb.mean(dim=1, keepdim=True)
    target_luma = target.mean(dim=1, keepdim=True)
    target_delta = target_luma - input_luma
    margin = args.luminance_margin_gray / 255.0
    positive_mask = target_delta > margin
    preserve_mask = ~positive_mask

    positive_bce = masked_mean(
        F.binary_cross_entropy_with_logits(
            components["edit_logits"],
            torch.ones_like(components["edit_logits"]),
            reduction="none",
        ),
        positive_mask.float(),
    ) * args.support_positive_weight
    preserve_bce = masked_mean(
        F.binary_cross_entropy_with_logits(
            components["edit_logits"],
            torch.zeros_like(components["edit_logits"]),
            reduction="none",
        ),
        preserve_mask.float(),
    ) * args.support_preserve_weight
    desired_magnitude = target_delta.clamp(0.0, model.residual_delta_bound)
    magnitude_l1 = masked_mean(
        torch.abs(components["bright_magnitude"] - desired_magnitude),
        positive_mask.float(),
    ) * args.magnitude_weight
    preserve_delta_l1 = masked_mean(
        components["signed_delta"].abs().mean(dim=1, keepdim=True),
        preserve_mask.float(),
    ) * args.preserve_delta_weight
    loss = positive_bce + preserve_bce + magnitude_l1 + preserve_delta_l1
    return {
        "loss": loss,
        "support_positive_bce": positive_bce,
        "support_preserve_bce": preserve_bce,
        "bright_magnitude_l1": magnitude_l1,
        "preserve_delta_l1": preserve_delta_l1,
    }


def build_model(residual_delta_bound: float) -> MonotonicResidualEraseCleanupNet:
    model = MonotonicResidualEraseCleanupNet(
        residual_delta_bound=residual_delta_bound,
        input_channels=CONDITIONED_INPUT_CHANNELS,
    )
    model.reset_output_to_identity()
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--patch-index-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--residual-delta-bound", type=float, default=RESIDUAL_DELTA_BOUND
    )
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu"
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--luminance-margin-gray", type=float, default=2.0)
    parser.add_argument("--support-positive-weight", type=float, default=1.0)
    parser.add_argument("--support-preserve-weight", type=float, default=1.0)
    parser.add_argument("--magnitude-weight", type=float, default=1.0)
    parser.add_argument("--preserve-delta-weight", type=float, default=1.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "residual_delta_bound": args.residual_delta_bound,
        "tile_size": args.tile_size,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "log_every": args.log_every,
        "luminance_margin_gray": args.luminance_margin_gray,
        "support_positive_weight": args.support_positive_weight,
        "support_preserve_weight": args.support_preserve_weight,
        "magnitude_weight": args.magnitude_weight,
        "preserve_delta_weight": args.preserve_delta_weight,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"registered training values must be positive: {invalid}")
    if args.residual_delta_bound > 1.0:
        raise ValueError("residual_delta_bound must be at most 1.0")
    if args.save_every < 0:
        raise ValueError("save_every must be nonnegative")
    if args.output_dir.exists():
        raise FileExistsError(
            f"registered training output must be absent: {args.output_dir}"
        )


def checkpoint_payload(
    model: MonotonicResidualEraseCleanupNet,
    args: argparse.Namespace,
    step: int,
) -> dict[str, object]:
    return {
        "model": model.state_dict(),
        "args": {
            **vars(args),
            "input_channels": CONDITIONED_INPUT_CHANNELS,
            "layout_source": "external_text_occupancy_confidence",
            "mask_source": MASK_SOURCE,
            "model_type": MODEL_TYPE,
            "validation_enabled": False,
        },
        "step": step,
    }


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dataset = ExternalTextLayoutConditionedPatchDataset(
        data_root=args.data_root,
        split=args.split,
        input_dir=args.input_dir,
        layout_dir=args.layout_dir,
        patch_index_file=args.patch_index_file,
        tile_size=args.tile_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = build_model(args.residual_delta_bound).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    history_path = args.output_dir / "conditioned_monotonic_loss_history.csv"
    with history_path.open("x", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["step", *LOSS_TERM_NAMES])

    print(
        f"device={device} patches={len(dataset)} batch_size={args.batch_size}",
        flush=True,
    )
    started = time.time()
    data_iterator = iter(loader)
    for step in range(1, args.max_steps + 1):
        try:
            features, target = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            features, target = next(data_iterator)
        features = features.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        terms = compute_conditioned_loss_terms(model, features, target, args)
        terms["loss"].backward()
        optimizer.step()

        values = [float(terms[name].detach().cpu()) for name in LOSS_TERM_NAMES]
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [step, *(f"{value:.8f}" for value in values)]
            )
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            metrics = " ".join(
                f"{name}={value:.6f}"
                for name, value in zip(LOSS_TERM_NAMES, values)
            )
            print(
                f"step={step}/{args.max_steps} {metrics} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
        if args.save_every > 0 and step % args.save_every == 0:
            torch.save(
                checkpoint_payload(model, args, step),
                args.output_dir / f"conditioned_monotonic_step{step:04d}.pt",
            )

    final_path = args.output_dir / "external_text_layout_conditioned_monotonic.pt"
    torch.save(checkpoint_payload(model, args, args.max_steps), final_path)
    print(f"saved={final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
