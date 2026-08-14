#!/usr/bin/env python3
"""Run the registered monotonic residual-erase training probe.

This trainer is intentionally single-purpose. It reads frozen pipeline
predictions and train-only targets, learns only bounded nonnegative luminance
deltas, starts from exact identity, and exposes no validation or resume path.
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


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def pad_to_size(array: np.ndarray, size: int) -> np.ndarray:
    height, width = array.shape[:2]
    pad_height = max(size - height, 0)
    pad_width = max(size - width, 0)
    if pad_height == 0 and pad_width == 0:
        return array
    return cv2.copyMakeBorder(
        array,
        0,
        pad_height,
        0,
        pad_width,
        cv2.BORDER_REPLICATE,
    )


class MonotonicTargetDifferencePatchDataset(Dataset):
    """Load registered train-only prediction/target patches."""

    def __init__(
        self,
        *,
        data_root: Path,
        split: str,
        input_dir: Path,
        patch_index_file: Path,
        tile_size: int,
    ):
        if split != "train":
            raise ValueError("monotonic training only accepts split=train")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        self.label_dir = data_root / split / "all_labels"
        self.input_dir = input_dir
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

        inp = pad_to_size(input_image[y1:y2, x1:x2], self.tile_size)
        target = pad_to_size(target_image[y1:y2, x1:x2], self.tile_size)
        inp_tensor = torch.from_numpy(
            inp.transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        target_tensor = torch.from_numpy(
            target.transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        return inp_tensor, target_tensor


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.expand_as(values)
    denominator = weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
    per_sample = (values * weight).sum(dim=(1, 2, 3)) / denominator
    return per_sample.mean()


def compute_monotonic_loss_terms(
    model: torch.nn.Module,
    inp: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    if not isinstance(model, MonotonicResidualEraseCleanupNet):
        raise TypeError("monotonic loss requires monotonic residual-erase model")
    components = model.forward_components(inp)
    input_luma = inp.mean(dim=1, keepdim=True)
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
    desired_magnitude = target_delta.clamp(
        0.0, model.residual_delta_bound
    )
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
        residual_delta_bound=residual_delta_bound
    )
    model.reset_output_to_identity()
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--patch-index-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--residual-delta-bound", type=float, default=RESIDUAL_DELTA_BOUND
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
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
            "model_type": MODEL_TYPE,
            "mask_source": MASK_SOURCE,
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
    dataset = MonotonicTargetDifferencePatchDataset(
        data_root=args.data_root,
        split=args.split,
        input_dir=args.input_dir,
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
    history_path = args.output_dir / "monotonic_loss_history.csv"
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
            inp, target = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            inp, target = next(data_iterator)
        inp = inp.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        terms = compute_monotonic_loss_terms(model, inp, target, args)
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
                args.output_dir / f"monotonic_step{step:04d}.pt",
            )

    final_path = args.output_dir / "monotonic_residual_erase_probe.pt"
    torch.save(checkpoint_payload(model, args, args.max_steps), final_path)
    print(f"saved={final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
