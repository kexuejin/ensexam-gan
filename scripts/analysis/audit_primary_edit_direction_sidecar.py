#!/usr/bin/env python3
"""Synthetic prerequisite audit for the primary-edit-direction sidecar."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from networks.generator import Generator, UniversalResidualAdapterSidecar  # noqa: E402
from train import apply_generator_trainable_patterns  # noqa: E402


def direction_model_config() -> dict:
    return {
        "coarse_in_channels": 3,
        "refine_in_channels": 7,
        "cbam_reduction": 16,
        "universal_residual_adapter_sidecar": {
            "enabled": True,
            "adapter_count": 3,
            "hidden_channels": 16,
            "residual_bound": 12.0 / 255.0,
            "residual_parameterization": "primary_edit_direction",
        },
    }


def run_audit(seed: int = 20260808) -> dict:
    torch.manual_seed(seed)
    public_parameters = set(inspect.signature(Generator.forward).parameters)
    if public_parameters != {
        "self",
        "Iin",
        "return_reconstruction_feature",
        "return_universal_sidecar_telemetry",
    }:
        raise ValueError("Generator.forward public interface changed")

    baseline_generator = Generator().eval()
    direction_generator = Generator(direction_model_config()).eval()
    incompatible = direction_generator.load_state_dict(
        baseline_generator.state_dict(), strict=False
    )
    if incompatible.unexpected_keys or not incompatible.missing_keys:
        raise ValueError("baseline checkpoint compatibility failed")
    if any(
        not key.startswith("universal_residual_adapter_sidecar.")
        for key in incompatible.missing_keys
    ):
        raise ValueError("non-sidecar missing key in baseline load")
    summary = apply_generator_trainable_patterns(
        direction_generator,
        [r"^universal_residual_adapter_sidecar\."],
    )
    trainable = [
        name
        for name, parameter in direction_generator.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any(
        not name.startswith("universal_residual_adapter_sidecar.")
        for name in trainable
    ):
        raise ValueError("direction sidecar trainable scope reaches base generator")

    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        baseline_outputs = baseline_generator(image)
        direction_outputs, telemetry = direction_generator(
            image, return_universal_sidecar_telemetry=True
        )
    if not all(
        torch.equal(expected, actual)
        for expected, actual in zip(baseline_outputs, direction_outputs)
    ):
        raise ValueError("direction sidecar zero initialization is not exact")
    if float(telemetry["ura_fallback_code"]) != 0.0:
        raise ValueError("zero-init direction sidecar unexpectedly fell back")

    sidecar = UniversalResidualAdapterSidecar(
        feature_channels=16,
        residual_bound=12.0 / 255.0,
        residual_parameterization="primary_edit_direction",
    ).train()
    feature = torch.randn(2, 16, 8, 8)
    input_image = torch.zeros(2, 3, 8, 8)
    primary = torch.empty_like(input_image)
    primary[:, 0] = 0.4
    primary[:, 1] = -0.2
    primary[:, 2] = 0.1
    target = primary + 0.25 * torch.sign(primary)
    candidate, _ = sidecar(feature, primary, input_image)
    if not torch.equal(candidate.detach(), primary):
        raise ValueError("synthetic direction sidecar is not zero-output initialized")
    torch.nn.functional.mse_loss(candidate, target).backward()
    final_bias_gradient_sums = [
        float(adapter[-1].bias.grad.detach().abs().sum())
        for adapter in sidecar.adapters
    ]
    if not all(value > 0 for value in final_bias_gradient_sums):
        raise ValueError("direction-sidecar final projections are gradient-dead")

    sidecar.eval()
    with torch.no_grad():
        sidecar.global_residual_scale.fill_(10.0)
        for adapter in sidecar.adapters:
            adapter[-1].bias.fill_(100.0)
        candidate, _ = sidecar(feature, primary, input_image)
    delta = candidate - primary
    primary_edit = primary - input_image
    opposed_channels = int((delta * primary_edit < -1e-8).sum())
    residual_abs_max = float(delta.abs().max())
    if opposed_channels:
        raise ValueError(f"direction sidecar emitted {opposed_channels} opposed channels")
    if residual_abs_max > sidecar.residual_bound + 1e-7:
        raise ValueError("direction sidecar exceeded its residual bound")

    with torch.no_grad():
        zero_edit_candidate, _ = sidecar(feature, primary, primary.clone())
    if not torch.equal(zero_edit_candidate, primary):
        raise ValueError("zero primary edit must remain a no-op")

    return {
        "status": "pass",
        "terminal": "PASS",
        "exact_zero_init": True,
        "gradient_live_final_bias_count": sum(
            value > 0 for value in final_bias_gradient_sums
        ),
        "opposed_channel_count": opposed_channels,
        "residual_abs_max": residual_abs_max,
        "residual_bound": sidecar.residual_bound,
        "zero_primary_edit_noop": True,
        "trainable_tensors": summary["trainable_tensors"],
        "frozen_tensors": summary["frozen_tensors"],
        "sidecar_missing_keys": len(incompatible.missing_keys),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = run_audit()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
