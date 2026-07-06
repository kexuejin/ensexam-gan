#!/usr/bin/env python3
"""Second-stage erasemap cleanup model and sliding-window page inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EraseMapCleanupNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(64, 96)
        self.up2 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.alpha_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        self.clean_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        alpha = self.alpha_head(d1)
        clean_candidate = self.clean_head(d1)
        pred = torch.clamp(x * (1.0 - alpha) + clean_candidate * alpha, 0.0, 1.0)
        return pred, alpha, clean_candidate


class ResidualDeltaCleanupNet(nn.Module):
    """Cleanup model that predicts a bounded residual delta instead of a full image."""

    def __init__(self, residual_delta_scale: float = 0.25):
        super().__init__()
        self.residual_delta_scale = residual_delta_scale
        self.enc1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(64, 96)
        self.up2 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.alpha_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        self.delta_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        alpha = self.alpha_head(d1)
        bounded_delta = self.delta_head(d1) * self.residual_delta_scale
        clean_candidate = torch.clamp(x + bounded_delta, 0.0, 1.0)
        pred = torch.clamp(x + alpha * bounded_delta, 0.0, 1.0)
        return pred, alpha, clean_candidate


def build_model(model_type: str, residual_delta_scale: float = 0.25) -> nn.Module:
    if model_type == "erasemap":
        return EraseMapCleanupNet()
    if model_type == "residual_delta":
        return ResidualDeltaCleanupNet(residual_delta_scale=residual_delta_scale)
    raise ValueError(f"Unsupported cleanup model type: {model_type}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    state = torch.load(checkpoint_path, map_location="cpu")
    args = state.get("args", {}) if isinstance(state, dict) else {}
    model_type = args.get("model_type", "erasemap")
    residual_delta_scale = float(args.get("residual_delta_scale", 0.25))
    model = build_model(model_type, residual_delta_scale=residual_delta_scale).to(device)
    model.load_state_dict(state["model"] if isinstance(state, dict) else state)
    model.eval()
    return model


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)


def infer_full_page(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile_size: int,
    stride: int,
    alpha_threshold: float,
) -> np.ndarray:
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    accum = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)

    ys = list(range(0, max(1, h - tile_size + 1), stride))
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    if not ys or ys[-1] != max(0, h - tile_size):
        ys.append(max(0, h - tile_size))
    if not xs or xs[-1] != max(0, w - tile_size):
        xs.append(max(0, w - tile_size))

    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                tile = rgb[y0 : y0 + tile_size, x0 : x0 + tile_size]
                th, tw = tile.shape[:2]
                if th != tile_size or tw != tile_size:
                    pad = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
                    pad[:th, :tw] = tile
                    tile = pad
                x = to_tensor(tile).unsqueeze(0).to(device)
                pred, alpha, clean_candidate = model(x)
                if alpha_threshold > 0.0:
                    alpha = torch.where(alpha >= alpha_threshold, alpha, torch.zeros_like(alpha))
                    pred = torch.clamp(x * (1.0 - alpha) + clean_candidate * alpha, 0.0, 1.0)
                pred_np = (
                    pred.squeeze(0)
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    .clip(0, 1)
                    * 255.0
                ).astype(np.float32)
                accum[y0 : y0 + th, x0 : x0 + tw] += pred_np[:th, :tw]
                weight[y0 : y0 + th, x0 : x0 + tw] += 1.0

    merged = np.clip(accum / np.clip(weight, 1e-6, None), 0, 255).astype(np.uint8)
    return cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--stride", type=int, default=80)
    parser.add_argument("--alpha-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = load_model(Path(args.checkpoint), device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_paths = [
        Path(line.strip())
        for line in Path(args.sample_list).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for idx, image_path in enumerate(sample_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        pred = infer_full_page(model, image, device, args.tile_size, args.stride, args.alpha_threshold)
        output_path = output_dir / f"{image_path.stem}.clean.png"
        cv2.imwrite(str(output_path), pred)
        print(f"{idx}/{len(sample_paths)} {image_path} -> {output_path}")


if __name__ == "__main__":
    main()
