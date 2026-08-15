#!/usr/bin/env python3
"""Synthetic-only preserve-or-brighten cleanup representation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from scripts.infer.patch_cleanup_erasemap import ConvBlock


MODEL_TYPE = "monotonic_residual_erase"


class MonotonicResidualEraseCleanupNet(nn.Module):
    """Identity-initialized cleanup that can only brighten selected pixels."""

    def __init__(
        self,
        residual_delta_bound: float = 0.08,
        *,
        input_channels: int = 3,
    ):
        super().__init__()
        if residual_delta_bound <= 0.0 or residual_delta_bound > 1.0:
            raise ValueError("residual_delta_bound must be in (0, 1]")
        if input_channels < 3:
            raise ValueError("input_channels must be at least 3")
        self.residual_delta_bound = float(residual_delta_bound)
        self.input_channels = int(input_channels)
        self.enc1 = ConvBlock(self.input_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(64, 96)
        self.up2 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.edit_support_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
        self.bright_magnitude_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
        self.reset_output_to_identity()

    def reset_output_to_identity(self) -> None:
        for head in (self.edit_support_head, self.bright_magnitude_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(
                "monotonic residual-erase input must be NCHW with "
                f"{self.input_channels} channels"
            )
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        return self.dec1(torch.cat([self.up1(d2), e1], dim=1))

    def forward_components(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        baseline = x[:, :3]
        feature = self._decode(x)
        edit_logits = self.edit_support_head(feature)
        edit_alpha = torch.sigmoid(edit_logits)
        magnitude_raw = self.bright_magnitude_head(feature)
        magnitude_folded = torch.where(
            magnitude_raw >= 0,
            magnitude_raw,
            -magnitude_raw,
        )
        bright_magnitude = self.residual_delta_bound * torch.tanh(
            magnitude_folded
        )
        clean_delta = bright_magnitude.expand_as(baseline)
        signed_delta = edit_alpha * clean_delta
        clean_candidate = torch.clamp(baseline + clean_delta, 0.0, 1.0)
        candidate = torch.clamp(baseline + signed_delta, 0.0, 1.0)
        return {
            "candidate": candidate,
            "edit_alpha": edit_alpha,
            "clean_candidate": clean_candidate,
            "edit_logits": edit_logits,
            "bright_magnitude": bright_magnitude,
            "signed_delta": signed_delta,
        }

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        components = self.forward_components(x)
        return (
            components["candidate"],
            components["edit_alpha"],
            components["clean_candidate"],
        )


def build_monotonic_residual_erase_model(
    residual_delta_bound: float = 0.08,
    *,
    input_channels: int = 3,
) -> MonotonicResidualEraseCleanupNet:
    return MonotonicResidualEraseCleanupNet(
        residual_delta_bound=residual_delta_bound,
        input_channels=input_channels,
    )


def load_monotonic_residual_erase_model(
    checkpoint_path: Path,
    device: torch.device,
) -> MonotonicResidualEraseCleanupNet:
    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("monotonic checkpoint must contain args and model")
    args = state.get("args", {})
    if args.get("model_type") != MODEL_TYPE:
        raise ValueError("checkpoint model_type is not monotonic_residual_erase")
    model = build_monotonic_residual_erase_model(
        residual_delta_bound=float(args.get("residual_delta_bound", 0.08)),
        input_channels=int(args.get("input_channels", 3)),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model
