#!/usr/bin/env python3
"""Bounded training probe for the second-stage erasemap cleanup model.

This trains ``EraseMapCleanupNet`` on explicit-mask patches without updating
EnsExam-GAN. It is intended to test whether a separate cleanup candidate can
improve residual coverage while keeping paper/background edits constrained.
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
INFER_DIR = ROOT / "scripts" / "infer"
if str(INFER_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_DIR))

from patch_cleanup_erasemap import EraseMapCleanupNet, resolve_device  # noqa: E402


def find_image(directory: Path, file_name: str) -> Path:
    stem = Path(file_name).stem
    for name in (f"{stem}.png", f"{stem}.clean.png", file_name):
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No image for {file_name} in {directory}")


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: Path, threshold: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > threshold).astype(np.float32)


def pad_to_size(array: np.ndarray, size: int, value: int | float) -> np.ndarray:
    h, w = array.shape[:2]
    pad_h = max(size - h, 0)
    pad_w = max(size - w, 0)
    if pad_h == 0 and pad_w == 0:
        return array
    border_type = cv2.BORDER_REPLICATE if array.ndim == 3 else cv2.BORDER_CONSTANT
    return cv2.copyMakeBorder(array, 0, pad_h, 0, pad_w, border_type, value=value)


class CleanupPatchDataset(Dataset):
    def __init__(
        self,
        *,
        data_root: Path,
        split: str,
        input_dir: Path,
        patch_index_file: Path,
        tile_size: int,
        mask_threshold: int,
    ):
        self.data_root = data_root
        self.split = split
        self.input_dir = input_dir
        self.tile_size = tile_size
        self.mask_threshold = mask_threshold
        with patch_index_file.open(newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise RuntimeError(f"empty patch index: {patch_index_file}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        file_name = row["file"]
        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])
        label_path = self.data_root / self.split / "all_labels" / file_name
        mask_path = self.data_root / self.split / "all_masks" / f"{Path(file_name).stem}.png"
        input_path = find_image(self.input_dir, file_name)

        inp = read_rgb(input_path)[y1:y2, x1:x2]
        target = read_rgb(label_path)[y1:y2, x1:x2]
        mask = read_mask(mask_path, self.mask_threshold)[y1:y2, x1:x2]

        inp = pad_to_size(inp, self.tile_size, 255)
        target = pad_to_size(target, self.tile_size, 255)
        mask = pad_to_size(mask, self.tile_size, 0.0)

        inp_t = torch.from_numpy(inp.transpose(2, 0, 1).astype(np.float32) / 255.0)
        target_t = torch.from_numpy(target.transpose(2, 0, 1).astype(np.float32) / 255.0)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return inp_t, target_t, mask_t


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.expand_as(pred)
    denom = weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
    per_sample = (torch.abs(pred - target) * weight).sum(dim=(1, 2, 3)) / denom
    return per_sample.mean()


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def compute_loss_terms(
    model: EraseMapCleanupNet,
    inp: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    pred, alpha, clean = model(inp)
    inside = masked_l1(pred, target, mask) * args.inside_weight
    outside = masked_l1(pred, inp, 1.0 - mask) * args.outside_weight
    clean_inside = masked_l1(clean, target, mask) * args.clean_inside_weight
    alpha_weight = 1.0 + mask * (args.alpha_positive_weight - 1.0)
    alpha_bce = (
        F.binary_cross_entropy(alpha.clamp(1e-6, 1.0 - 1e-6), mask, weight=alpha_weight)
        * args.alpha_bce_weight
    )
    alpha_dice = dice_loss(alpha, mask) * args.alpha_dice_weight
    alpha_sparse = alpha.mean() * args.alpha_sparsity_weight
    loss = inside + outside + clean_inside + alpha_bce + alpha_dice + alpha_sparse
    return {
        "loss": loss,
        "inside_l1": inside,
        "outside_l1": outside,
        "clean_inside_l1": clean_inside,
        "alpha_bce": alpha_bce,
        "alpha_dice": alpha_dice,
        "alpha_sparsity": alpha_sparse,
    }


def tensor_terms_to_floats(terms: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in terms.items()}


def evaluate_loss(
    model: EraseMapCleanupNet,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch_index, (inp, target, mask) in enumerate(loader, start=1):
            inp = inp.to(device)
            target = target.to(device)
            mask = mask.to(device)
            batch_size = int(inp.shape[0])
            terms = tensor_terms_to_floats(compute_loss_terms(model, inp, target, mask, args))
            for key, value in terms.items():
                sums[key] = sums.get(key, 0.0) + value * batch_size
            count += batch_size
            if max_batches > 0 and batch_index >= max_batches:
                break
    model.train()
    if count == 0:
        raise RuntimeError("validation loader produced no batches")
    return {key: value / count for key, value in sums.items()}


def initialize_identity_safe(model: EraseMapCleanupNet, alpha_init_bias: float) -> None:
    """Make the untrained cleanup branch an identity mapping by default."""
    final_alpha = model.alpha_head[-2]
    torch.nn.init.zeros_(final_alpha.weight)
    torch.nn.init.constant_(final_alpha.bias, alpha_init_bias)


def load_initial_state(model: EraseMapCleanupNet, checkpoint: str, device: torch.device) -> None:
    if not checkpoint:
        return
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--patch-index-file", required=True)
    parser.add_argument("--val-data-root", default="")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--val-input-dir", default="")
    parser.add_argument("--val-patch-index-file", default="")
    parser.add_argument("--val-every", type=int, default=0)
    parser.add_argument("--val-max-batches", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--mask-threshold", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--inside-weight", type=float, default=4.0)
    parser.add_argument("--outside-weight", type=float, default=1.0)
    parser.add_argument("--clean-inside-weight", type=float, default=2.0)
    parser.add_argument("--alpha-bce-weight", type=float, default=0.5)
    parser.add_argument("--alpha-positive-weight", type=float, default=8.0)
    parser.add_argument("--alpha-dice-weight", type=float, default=0.25)
    parser.add_argument("--alpha-sparsity-weight", type=float, default=0.02)
    parser.add_argument("--alpha-init-bias", type=float, default=-6.0)
    return parser.parse_args()


def build_val_loader(args: argparse.Namespace) -> DataLoader | None:
    val_paths = [args.val_data_root, args.val_input_dir, args.val_patch_index_file]
    if not any(val_paths):
        return None
    if not all(val_paths):
        raise ValueError("Provide all of --val-data-root, --val-input-dir, and --val-patch-index-file")
    if args.val_every <= 0:
        raise ValueError("--val-every must be > 0 when validation data is provided")
    dataset = CleanupPatchDataset(
        data_root=Path(args.val_data_root),
        split=args.val_split,
        input_dir=Path(args.val_input_dir),
        patch_index_file=Path(args.val_patch_index_file),
        tile_size=args.tile_size,
        mask_threshold=args.mask_threshold,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"device={device}", flush=True)
    dataset = CleanupPatchDataset(
        data_root=Path(args.data_root),
        split=args.split,
        input_dir=Path(args.input_dir),
        patch_index_file=Path(args.patch_index_file),
        tile_size=args.tile_size,
        mask_threshold=args.mask_threshold,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"patches={len(dataset)} batch_size={args.batch_size}", flush=True)
    val_loader = build_val_loader(args)
    if val_loader is not None:
        print(f"val_patches={len(val_loader.dataset)} val_every={args.val_every}", flush=True)

    model = EraseMapCleanupNet().to(device)
    if args.init_checkpoint:
        load_initial_state(model, args.init_checkpoint, device)
    else:
        initialize_identity_safe(model, args.alpha_init_bias)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history_path = output_dir / "cleanup_loss_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "step",
            "loss",
            "inside_l1",
            "outside_l1",
            "clean_inside_l1",
            "alpha_bce",
            "alpha_dice",
            "alpha_sparsity",
        ])
    val_history_path = output_dir / "cleanup_val_loss_history.csv"
    if val_loader is not None:
        with val_history_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "step",
                "loss",
                "inside_l1",
                "outside_l1",
                "clean_inside_l1",
                "alpha_bce",
                "alpha_dice",
                "alpha_sparsity",
            ])

    start = time.time()
    data_iter = iter(loader)
    best_val_loss: float | None = None
    for step in range(1, args.max_steps + 1):
        try:
            inp, target, mask = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inp, target, mask = next(data_iter)
        inp = inp.to(device)
        target = target.to(device)
        mask = mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        terms = compute_loss_terms(model, inp, target, mask, args)
        terms["loss"].backward()
        optimizer.step()

        train_terms = tensor_terms_to_floats(terms)
        row = [
            step,
            train_terms["loss"],
            train_terms["inside_l1"],
            train_terms["outside_l1"],
            train_terms["clean_inside_l1"],
            train_terms["alpha_bce"],
            train_terms["alpha_dice"],
            train_terms["alpha_sparsity"],
        ]
        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row[0], *[f"{value:.8f}" for value in row[1:]]])
        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            print(
                f"step={step}/{args.max_steps} loss={row[1]:.6f} "
                f"inside={row[2]:.6f} outside={row[3]:.6f} "
                f"clean_inside={row[4]:.6f} alpha_bce={row[5]:.6f} "
                f"alpha_dice={row[6]:.6f} alpha_sparse={row[7]:.6f} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
        if val_loader is not None and (step % args.val_every == 0 or step == args.max_steps):
            val_terms = evaluate_loss(model, val_loader, device, args, args.val_max_batches)
            val_row = [
                step,
                val_terms["loss"],
                val_terms["inside_l1"],
                val_terms["outside_l1"],
                val_terms["clean_inside_l1"],
                val_terms["alpha_bce"],
                val_terms["alpha_dice"],
                val_terms["alpha_sparsity"],
            ]
            with val_history_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([val_row[0], *[f"{value:.8f}" for value in val_row[1:]]])
            print(
                f"val step={step}/{args.max_steps} loss={val_row[1]:.6f} "
                f"inside={val_row[2]:.6f} outside={val_row[3]:.6f} "
                f"clean_inside={val_row[4]:.6f} alpha_bce={val_row[5]:.6f} "
                f"alpha_dice={val_row[6]:.6f} alpha_sparse={val_row[7]:.6f}",
                flush=True,
            )
            if best_val_loss is None or val_terms["loss"] < best_val_loss:
                best_val_loss = val_terms["loss"]
                best_path = output_dir / "cleanup_best.pt"
                torch.save(
                    {"model": model.state_dict(), "args": vars(args), "step": step, "val_loss": best_val_loss},
                    best_path,
                )
                print(f"saved_best={best_path} val_loss={best_val_loss:.6f}", flush=True)
        if args.save_every > 0 and (step % args.save_every == 0):
            torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, output_dir / f"cleanup_step{step:04d}.pt")

    final_path = output_dir / "cleanup_probe.pt"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.max_steps}, final_path)
    print(f"saved={final_path}", flush=True)


if __name__ == "__main__":
    main()
