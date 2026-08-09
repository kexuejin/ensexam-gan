#!/usr/bin/env python3
"""Fail-closed synthetic prerequisite for the D5 folded-direction sidecar."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from networks.generator import Generator, UniversalResidualAdapterSidecar  # noqa: E402
from train import apply_generator_trainable_patterns  # noqa: E402


FOLDED_MODE = "primary_edit_direction_folded"
EXPECTED_PUBLIC_PARAMETERS = {
    "self",
    "Iin",
    "return_reconstruction_feature",
    "return_universal_sidecar_telemetry",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def folded_model_config(base_model_config: dict) -> dict:
    config = dict(base_model_config)
    config["universal_residual_adapter_sidecar"] = {
        "enabled": True,
        "adapter_count": 3,
        "hidden_channels": 16,
        "residual_bound": 12.0 / 255.0,
        "residual_parameterization": FOLDED_MODE,
    }
    return config


def resolve_frozen_artifact(
    repo_root: Path,
    ledger: dict,
    artifact_name: str,
) -> tuple[Path, str]:
    entry = ledger.get("baseline", {}).get(artifact_name)
    if not isinstance(entry, dict):
        raise ValueError(f"ledger baseline.{artifact_name} must be a mapping")
    relative_path = entry.get("path")
    expected_hash = entry.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
        raise ValueError(f"ledger baseline.{artifact_name} is incomplete")
    artifact_path = repo_root / relative_path
    if not artifact_path.is_file():
        raise ValueError(f"frozen {artifact_name} is missing: {relative_path}")
    actual_hash = sha256_file(artifact_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"frozen {artifact_name} hash mismatch: "
            f"expected={expected_hash} actual={actual_hash}"
        )
    return artifact_path, actual_hash


def synthetic_two_step_probe(raw_sign: int) -> dict:
    torch.manual_seed(20260809)
    sidecar = UniversalResidualAdapterSidecar(
        feature_channels=16,
        residual_bound=12.0 / 255.0,
        residual_parameterization=FOLDED_MODE,
    ).train()
    optimizer = torch.optim.SGD(sidecar.parameters(), lr=0.25)
    feature = torch.ones(1, 16, 4, 4)
    input_image = torch.zeros(1, 3, 4, 4)
    baseline = torch.empty_like(input_image)
    baseline[:, 0] = 0.4
    baseline[:, 1] = -0.2
    baseline[:, 2] = 0.1
    primary_direction = baseline / baseline.abs().amax(dim=1, keepdim=True)
    target = baseline + raw_sign * 0.2 * primary_direction

    candidate, _ = sidecar(feature, baseline, input_image)
    if not torch.equal(candidate.detach(), baseline):
        raise ValueError(f"raw_sign={raw_sign} did not start at exact zero output")
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(candidate, target).backward()
    first_bias_gradients = [
        float(adapter[-1].bias.grad.detach().abs().sum())
        for adapter in sidecar.adapters
    ]
    if not all(value > 0.0 for value in first_bias_gradients):
        raise ValueError(f"raw_sign={raw_sign} zero-branch projection gradient died")
    first_scale_gradient = float(sidecar.global_residual_scale.grad.detach().abs())
    if first_scale_gradient != 0.0:
        raise ValueError("global scale must have zero gradient before magnitude exists")
    optimizer.step()

    with torch.no_grad():
        gate_weights = torch.softmax(sidecar.gate(feature), dim=1)
        adapter_residuals = torch.stack(
            [adapter(feature) for adapter in sidecar.adapters], dim=1
        )
        mixed_after_first = (
            gate_weights.view(1, sidecar.adapter_count, 1, 1, 1)
            * adapter_residuals
        ).sum(dim=1)
    if not torch.all(torch.sign(mixed_after_first) == raw_sign):
        raise ValueError(f"raw_sign={raw_sign} did not survive the first update")
    folded_support = int((mixed_after_first.abs() > 0).sum())
    if folded_support == 0:
        raise ValueError(f"raw_sign={raw_sign} produced no folded support")

    candidate, _ = sidecar(feature, baseline, input_image)
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(candidate, target).backward()
    second_bias_gradients = [
        float(adapter[-1].bias.grad.detach().abs().sum())
        for adapter in sidecar.adapters
    ]
    if not all(value > 0.0 for value in second_bias_gradients):
        raise ValueError(f"raw_sign={raw_sign} second-step projection gradient died")
    second_scale_gradient = float(sidecar.global_residual_scale.grad.detach().abs())
    if second_scale_gradient <= 0.0:
        raise ValueError(f"raw_sign={raw_sign} second-step scale gradient died")
    optimizer.step()
    final_scale = float(sidecar.global_residual_scale.detach())
    if final_scale == sidecar.initial_global_residual_scale:
        raise ValueError(f"raw_sign={raw_sign} global scale did not move")

    return {
        "raw_sign": raw_sign,
        "first_projection_gradient_min": min(first_bias_gradients),
        "first_scale_gradient_abs": first_scale_gradient,
        "folded_support_count": folded_support,
        "mixed_residual_abs_max_after_first": float(mixed_after_first.abs().max()),
        "second_projection_gradient_min": min(second_bias_gradients),
        "second_scale_gradient_abs": second_scale_gradient,
        "final_global_residual_scale": final_scale,
    }


def run_audit(
    repo_root: Path = ROOT,
    ledger_path: Path | None = None,
    seed: int = 20260809,
) -> dict:
    torch.manual_seed(seed)
    public_parameters = set(inspect.signature(Generator.forward).parameters)
    if public_parameters != EXPECTED_PUBLIC_PARAMETERS:
        raise ValueError("Generator.forward public interface changed")

    ledger_path = ledger_path or (
        repo_root / "docs/current-primary-quality-loop-ledger.json"
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    config_path, config_hash = resolve_frozen_artifact(
        repo_root, ledger, "config"
    )
    checkpoint_path, checkpoint_hash = resolve_frozen_artifact(
        repo_root, ledger, "checkpoint"
    )
    baseline_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_model_config = baseline_config.get("model")
    if not isinstance(base_model_config, dict):
        raise ValueError("current-primary config.model must be a mapping")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    generator_state = checkpoint.get("G_state_dict")
    if not isinstance(generator_state, dict):
        raise ValueError("current-primary checkpoint lacks G_state_dict")

    image = torch.randn(1, 3, 32, 32)
    baseline_generator = Generator(cfg=base_model_config).eval()
    baseline_generator.load_state_dict(generator_state, strict=True)
    with torch.no_grad():
        baseline_outputs = tuple(tensor.clone() for tensor in baseline_generator(image))
    del baseline_generator

    folded_generator = Generator(folded_model_config(base_model_config)).eval()
    incompatible = folded_generator.load_state_dict(generator_state, strict=False)
    if incompatible.unexpected_keys or not incompatible.missing_keys:
        raise ValueError("current-primary sidecar compatibility failed")
    if any(
        not key.startswith("universal_residual_adapter_sidecar.")
        for key in incompatible.missing_keys
    ):
        raise ValueError("current-primary load has a non-sidecar missing key")
    sidecar = folded_generator.universal_residual_adapter_sidecar
    zero_projection_tensors = [
        tensor
        for adapter in sidecar.adapters
        for tensor in (adapter[-1].weight, adapter[-1].bias)
    ]
    if not all(bool((tensor.detach() == 0).all()) for tensor in zero_projection_tensors):
        raise ValueError("folded sidecar final projection is not exactly zero initialized")
    expected_scale = sidecar.global_residual_scale.detach().new_tensor(
        sidecar.initial_global_residual_scale
    )
    if not torch.equal(sidecar.global_residual_scale.detach(), expected_scale):
        raise ValueError("folded sidecar global scale initialization changed")

    trainable_summary = apply_generator_trainable_patterns(
        folded_generator,
        [r"^universal_residual_adapter_sidecar\."],
    )
    trainable_names = [
        name
        for name, parameter in folded_generator.named_parameters()
        if parameter.requires_grad
    ]
    frozen_base_names = [
        name
        for name, parameter in folded_generator.named_parameters()
        if not name.startswith("universal_residual_adapter_sidecar.")
        and not parameter.requires_grad
    ]
    base_names = [
        name
        for name, _ in folded_generator.named_parameters()
        if not name.startswith("universal_residual_adapter_sidecar.")
    ]
    if not trainable_names or any(
        not name.startswith("universal_residual_adapter_sidecar.")
        for name in trainable_names
    ):
        raise ValueError("folded sidecar trainable scope reaches base generator")
    if len(frozen_base_names) != len(base_names):
        raise ValueError("not every base generator tensor is frozen")
    with torch.no_grad():
        folded_outputs, telemetry = folded_generator(
            image,
            return_universal_sidecar_telemetry=True,
        )
    if not all(
        torch.equal(expected, actual)
        for expected, actual in zip(baseline_outputs, folded_outputs)
    ):
        raise ValueError("folded sidecar zero initialization is not exact")
    if float(telemetry["ura_fallback_code"]) != 0.0:
        raise ValueError("zero-init folded sidecar unexpectedly fell back")
    del checkpoint, generator_state, folded_generator, baseline_outputs, folded_outputs

    probes = [synthetic_two_step_probe(raw_sign) for raw_sign in (1, -1)]

    forced_sidecar = UniversalResidualAdapterSidecar(
        feature_channels=16,
        residual_bound=0.02,
        residual_parameterization=FOLDED_MODE,
    ).eval()
    feature = torch.zeros(1, 16, 4, 4)
    input_image = torch.zeros(1, 3, 4, 4)
    primary = torch.empty_like(input_image)
    primary[:, 0] = 0.4
    primary[:, 1] = -0.2
    primary[:, 2] = 0.1
    deltas = []
    with torch.no_grad():
        forced_sidecar.global_residual_scale.fill_(10.0)
        for raw_value in (2.0, -2.0):
            for adapter in forced_sidecar.adapters:
                adapter[-1].bias.fill_(raw_value)
            candidate, telemetry = forced_sidecar(feature, primary, input_image)
            if float(telemetry["ura_fallback_code"]) != 0.0:
                raise ValueError(f"forced raw_value={raw_value} unexpectedly fell back")
            deltas.append(candidate - primary)
    if not torch.allclose(deltas[0], deltas[1], atol=1e-7, rtol=0.0):
        raise ValueError("positive and negative raw magnitudes did not fold equally")
    primary_edit = primary - input_image
    opposed_channels = int((deltas[1] * primary_edit < -1e-8).sum())
    residual_abs_max = float(deltas[1].abs().max())
    if opposed_channels:
        raise ValueError(f"folded sidecar emitted {opposed_channels} opposed channels")
    if residual_abs_max > forced_sidecar.residual_bound + 1e-7:
        raise ValueError("folded sidecar exceeded its residual bound")
    with torch.no_grad():
        zero_edit_candidate, _ = forced_sidecar(feature, primary, primary.clone())
    if not torch.equal(zero_edit_candidate, primary):
        raise ValueError("zero primary edit must remain a no-op")

    return {
        "status": "pass",
        "terminal": "PASS",
        "mode": FOLDED_MODE,
        "current_primary_config_sha256": config_hash,
        "current_primary_checkpoint_sha256": checkpoint_hash,
        "strict_current_primary_load": True,
        "sidecar_only_missing_key_count": len(incompatible.missing_keys),
        "exact_zero_init": True,
        "base_parameter_tensors": len(base_names),
        "frozen_base_parameter_tensors": len(frozen_base_names),
        "trainable_tensors": trainable_summary["trainable_tensors"],
        "positive_negative_fold_equal": True,
        "opposed_channel_count": opposed_channels,
        "residual_abs_max": residual_abs_max,
        "residual_bound": forced_sidecar.residual_bound,
        "zero_primary_edit_noop": True,
        "public_interface_unchanged": True,
        "two_step_probes": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = run_audit(ledger_path=args.ledger)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
