#!/usr/bin/env python3
"""Synthetic structure audit for the universal residual adapter sidecar."""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from networks.generator import Generator  # noqa: E402


FORBIDDEN_ROUTING_TOKENS = ("domain", "source", "caller", "route", "expert", "path")


def compliant_model_config() -> dict[str, Any]:
    return {
        "coarse_in_channels": 3,
        "refine_in_channels": 7,
        "cbam_reduction": 16,
        "universal_residual_adapter_sidecar": {
            "enabled": True,
            "adapter_count": 3,
            "hidden_channels": 16,
            "residual_bound": 12.0 / 255.0,
        },
    }


def _contains_forbidden_token(value: str) -> bool:
    normalized = value.lower()
    return any(token in normalized for token in FORBIDDEN_ROUTING_TOKENS)


def _assert_no_forbidden_model_config_keys(model_cfg: dict[str, Any]) -> None:
    sidecar_cfg = model_cfg.get("universal_residual_adapter_sidecar", {})
    if isinstance(sidecar_cfg, bool):
        return
    if not isinstance(sidecar_cfg, dict):
        raise ValueError("sidecar config must be a mapping or boolean")
    bad_keys = sorted(
        key for key in sidecar_cfg if _contains_forbidden_token(str(key))
    )
    if bad_keys:
        raise ValueError(f"sidecar config contains routing-like keys: {bad_keys}")


def _assert_no_public_routing_args() -> None:
    public_parameters = inspect.signature(Generator.forward).parameters
    bad_args = sorted(
        name for name in public_parameters if _contains_forbidden_token(name)
    )
    if bad_args:
        raise ValueError(f"Generator.forward exposes routing-like args: {bad_args}")


def _assert_trainable_patterns_sidecar_only(
    patterns: list[str] | None,
    parameter_names: list[str],
) -> None:
    if patterns is None:
        return
    for pattern in patterns:
        compiled = re.compile(pattern)
        matched = [name for name in parameter_names if compiled.search(name)]
        if not matched:
            raise ValueError(f"trainable pattern misses sidecar params: {pattern}")
        if any(
            not name.startswith("universal_residual_adapter_sidecar.")
            for name in matched
        ):
            raise ValueError(f"trainable pattern reaches base params: {pattern}")


def _assert_numeric_domain_free_telemetry(telemetry: dict[str, Any]) -> None:
    for key, value in telemetry.items():
        if _contains_forbidden_token(key):
            raise ValueError(f"telemetry key is routing-like: {key}")
        if isinstance(value, torch.Tensor):
            if value.numel() != 1 or not torch.isfinite(value.detach()).all():
                raise ValueError(f"telemetry tensor must be finite scalar: {key}")
            continue
        if not isinstance(value, (int, float)):
            raise ValueError(f"telemetry value must be numeric: {key}")


def run_audit(
    *,
    model_cfg: dict[str, Any] | None = None,
    trainable_patterns: list[str] | None = None,
    seed: int = 123,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    _assert_no_public_routing_args()

    default_generator = Generator().eval()
    disabled_cfg = compliant_model_config()
    disabled_cfg["universal_residual_adapter_sidecar"] = {"enabled": False}
    disabled_generator = Generator(disabled_cfg).eval()
    default_keys = tuple(default_generator.state_dict())
    disabled_keys = tuple(disabled_generator.state_dict())
    if default_keys != disabled_keys:
        raise ValueError("disabled sidecar changes default generator state_dict keys")
    if any("universal_residual_adapter_sidecar" in key for key in default_keys):
        raise ValueError("default Generator() unexpectedly registers sidecar params")

    enabled_cfg = model_cfg or compliant_model_config()
    _assert_no_forbidden_model_config_keys(enabled_cfg)
    enabled_generator = Generator(enabled_cfg).eval()
    _assert_trainable_patterns_sidecar_only(
        trainable_patterns,
        [name for name, _ in enabled_generator.named_parameters()],
    )
    incompatible = enabled_generator.load_state_dict(
        default_generator.state_dict(),
        strict=False,
    )
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(f"unexpected state_dict keys loading baseline: {unexpected}")
    if not missing or not all(
        key.startswith("universal_residual_adapter_sidecar.") for key in missing
    ):
        raise ValueError(f"missing keys are not sidecar-only: {sorted(missing)}")

    synthetic = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        baseline_outputs = default_generator(synthetic)
        enabled_outputs, telemetry = enabled_generator(
            synthetic,
            return_universal_sidecar_telemetry=True,
        )
    if len(baseline_outputs) != 7 or len(enabled_outputs) != 7:
        raise ValueError("generator output arity changed")
    if not torch.equal(baseline_outputs[-1], enabled_outputs[-1]):
        raise ValueError("zero-init sidecar output does not match baseline")
    _assert_numeric_domain_free_telemetry(telemetry)

    _, reconstruction_feature = enabled_generator(
        synthetic,
        return_reconstruction_feature=True,
    )
    gate_weights = torch.softmax(
        enabled_generator.universal_residual_adapter_sidecar.gate(
            reconstruction_feature
        ),
        dim=1,
    )
    if not torch.isfinite(gate_weights).all():
        raise ValueError("sidecar gate weights are not finite")
    if torch.any(gate_weights < 0):
        raise ValueError("sidecar gate weights must be non-negative")
    if not torch.allclose(
        gate_weights.sum(dim=1),
        torch.ones(gate_weights.shape[0]),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("sidecar gate weights do not sum to one")

    return {
        "status": "pass",
        "default_state_keys": len(default_keys),
        "sidecar_missing_keys": len(missing),
        "gate_weight_shape": list(gate_weights.shape),
        "telemetry_keys": sorted(telemetry),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", help="Optional path for the audit summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_audit()
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
