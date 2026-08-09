#!/usr/bin/env python3
"""Synthetic prerequisite audit for sign-separated residual repair."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer.patch_cleanup_erasemap import (  # noqa: E402
    EraseMapCleanupNet,
    ResidualDeltaCleanupNet,
    SignSeparatedResidualDeltaCleanupNet,
    build_model,
    infer_full_page,
    load_model,
)


MODEL_TYPE = "sign_separated_residual_delta"
RESIDUAL_BOUND = 0.08
DIRECTION_MARGIN = 2.0 / 255.0


def masked_l1(
    value: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype)
    denominator = weight.sum().clamp_min(1.0)
    return (torch.abs(value - target) * weight).sum() / denominator


def compute_synthetic_loss_terms(
    model: SignSeparatedResidualDeltaCleanupNet,
    input_image: torch.Tensor,
    target: torch.Tensor,
    *,
    direction_margin: float = DIRECTION_MARGIN,
) -> dict[str, torch.Tensor]:
    if input_image.shape != target.shape:
        raise ValueError("input_image and target must have matching shapes")
    components = model.forward_components(input_image)
    target_luma_delta = (target - input_image).mean(dim=1, keepdim=True)
    bright_mask = target_luma_delta > direction_margin
    dark_mask = target_luma_delta < -direction_margin
    identity_mask = ~(bright_mask | dark_mask)

    route_target = torch.full_like(
        target_luma_delta[:, 0],
        model.IDENTITY_ROUTE,
        dtype=torch.long,
    )
    route_target = torch.where(
        bright_mask[:, 0],
        torch.full_like(route_target, model.BRIGHTEN_ROUTE),
        route_target,
    )
    route_target = torch.where(
        dark_mask[:, 0],
        torch.full_like(route_target, model.DARKEN_ROUTE),
        route_target,
    )
    route_loss = F.cross_entropy(components["route_logits"], route_target)
    bright_target = target_luma_delta.clamp(0.0, model.residual_delta_bound)
    dark_target = (-target_luma_delta).clamp(0.0, model.residual_delta_bound)
    bright_loss = masked_l1(
        components["bright_magnitude"], bright_target, bright_mask
    )
    dark_loss = masked_l1(
        components["dark_magnitude"], dark_target, dark_mask
    )
    signed_luma_delta = components["signed_delta"].mean(dim=1, keepdim=True)
    identity_loss = masked_l1(
        signed_luma_delta,
        torch.zeros_like(signed_luma_delta),
        identity_mask,
    )
    return {
        "loss": route_loss + bright_loss + dark_loss + identity_loss,
        "route_loss": route_loss,
        "bright_loss": bright_loss,
        "dark_loss": dark_loss,
        "identity_loss": identity_loss,
        "bright_mask_count": bright_mask.sum(),
        "dark_mask_count": dark_mask.sum(),
        "identity_mask_count": identity_mask.sum(),
    }


def final_bias_gradient(
    model: SignSeparatedResidualDeltaCleanupNet,
    head_name: str,
) -> float:
    gradient = getattr(model, head_name)[-1].bias.grad
    return 0.0 if gradient is None else float(gradient.detach().abs().sum())


def forced_route_case(route_index: int) -> dict[str, object]:
    torch.manual_seed(20260809)
    model = SignSeparatedResidualDeltaCleanupNet(
        residual_delta_bound=RESIDUAL_BOUND
    ).eval()
    with torch.no_grad():
        model.route_head[-1].bias.fill_(-100.0)
        model.route_head[-1].bias[route_index] = 100.0
        model.bright_magnitude_head[-1].bias.fill_(-100.0)
        model.dark_magnitude_head[-1].bias.fill_(100.0)
        input_image = torch.full((1, 3, 8, 8), 0.5)
        components = model.forward_components(input_image)
        candidate = components["candidate"]
    delta = candidate - input_image
    return {
        "route_index": route_index,
        "route_probability_mean": [
            float(value)
            for value in components["route_prob"].mean(dim=(0, 2, 3))
        ],
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "delta_abs_max": float(delta.abs().max()),
        "negative_pixel_count": int((delta < -1e-8).sum()),
        "positive_pixel_count": int((delta > 1e-8).sum()),
    }


def gradient_case(direction: int) -> dict[str, float | int]:
    torch.manual_seed(20260809)
    model = SignSeparatedResidualDeltaCleanupNet(
        residual_delta_bound=RESIDUAL_BOUND
    ).train()
    input_image = torch.full((1, 3, 8, 8), 0.5)
    target = input_image + direction * 0.05
    terms = compute_synthetic_loss_terms(model, input_image, target)
    terms["loss"].backward()
    return {
        "direction": direction,
        "bright_gradient_abs": final_bias_gradient(
            model, "bright_magnitude_head"
        ),
        "dark_gradient_abs": final_bias_gradient(model, "dark_magnitude_head"),
        "route_gradient_abs": float(
            model.route_head[-1].bias.grad.detach().abs().sum()
        ),
        "bright_mask_count": int(terms["bright_mask_count"]),
        "dark_mask_count": int(terms["dark_mask_count"]),
        "identity_mask_count": int(terms["identity_mask_count"]),
    }


def two_step_case(direction: int) -> dict[str, float | int]:
    torch.manual_seed(20260809)
    model = SignSeparatedResidualDeltaCleanupNet(
        residual_delta_bound=RESIDUAL_BOUND
    ).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    input_image = torch.full((1, 3, 8, 8), 0.5)
    target = input_image + direction * 0.05
    for _ in range(2):
        optimizer.zero_grad()
        terms = compute_synthetic_loss_terms(model, input_image, target)
        terms["loss"].backward()
        optimizer.step()
    with torch.no_grad():
        candidate, _, _ = model(input_image)
    delta = candidate - input_image
    return {
        "direction": direction,
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "delta_abs_max": float(delta.abs().max()),
        "opposed_pixel_count": int((delta * direction < -1e-8).sum()),
    }


def identity_two_step_case() -> dict[str, float | bool]:
    torch.manual_seed(20260809)
    model = SignSeparatedResidualDeltaCleanupNet(
        residual_delta_bound=RESIDUAL_BOUND
    ).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    input_image = torch.full((1, 3, 8, 8), 0.5)
    for _ in range(2):
        optimizer.zero_grad()
        terms = compute_synthetic_loss_terms(model, input_image, input_image)
        terms["loss"].backward()
        optimizer.step()
    with torch.no_grad():
        components = model.forward_components(input_image)
    return {
        "exact_identity": bool(torch.equal(components["candidate"], input_image)),
        "bright_magnitude_abs_max": float(
            components["bright_magnitude"].abs().max()
        ),
        "dark_magnitude_abs_max": float(
            components["dark_magnitude"].abs().max()
        ),
    }


def run_audit(seed: int = 20260809) -> dict[str, object]:
    torch.manual_seed(seed)
    input_image = torch.rand(2, 3, 16, 16) * 0.8 + 0.1
    legacy = {}
    for model_type, expected_class in (
        ("erasemap", EraseMapCleanupNet),
        ("residual_delta", ResidualDeltaCleanupNet),
    ):
        model = build_model(model_type, residual_delta_scale=0.08).eval()
        with torch.no_grad():
            outputs = model(input_image)
        if not isinstance(model, expected_class):
            raise ValueError(f"legacy builder changed for {model_type}")
        if sum(parameter.numel() for parameter in model.parameters()) != 384612:
            raise ValueError(f"legacy parameter count changed for {model_type}")
        if len(model.state_dict()) != 32:
            raise ValueError(f"legacy state shape changed for {model_type}")
        if [tuple(value.shape) for value in outputs] != [
            (2, 3, 16, 16),
            (2, 1, 16, 16),
            (2, 3, 16, 16),
        ]:
            raise ValueError(f"legacy forward shape changed for {model_type}")
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy.pt"
            torch.save(
                {
                    "args": {
                        "model_type": model_type,
                        "residual_delta_scale": 0.08,
                    },
                    "model": model.state_dict(),
                },
                checkpoint_path,
            )
            restored = load_model(checkpoint_path, torch.device("cpu"))
            with torch.no_grad():
                restored_outputs = restored(input_image)
        checkpoint_roundtrip_exact = all(
            torch.equal(expected, actual)
            for expected, actual in zip(outputs, restored_outputs)
        )
        if not checkpoint_roundtrip_exact:
            raise ValueError(f"legacy checkpoint load changed for {model_type}")
        legacy[model_type] = {
            "parameter_count": 384612,
            "state_tensor_count": 32,
            "checkpoint_roundtrip_exact": True,
        }

    training_script = ROOT / "scripts/train/train_patch_cleanup_erasemap_probe.py"
    training_cli_enabled = MODEL_TYPE in training_script.read_text(encoding="utf-8")
    if training_cli_enabled:
        raise ValueError("synthetic-only model is enabled in the training CLI")

    model = SignSeparatedResidualDeltaCleanupNet(
        residual_delta_bound=RESIDUAL_BOUND
    ).eval()
    zero_magnitude_projection_init = all(
        bool((tensor.detach() == 0).all())
        for head in (model.bright_magnitude_head, model.dark_magnitude_head)
        for tensor in (head[-1].weight, head[-1].bias)
    )
    if not zero_magnitude_projection_init:
        raise ValueError("magnitude final projections are not exactly zero")
    with torch.no_grad():
        candidate, _, clean_candidate = model(input_image)
    exact_identity = bool(torch.equal(candidate, input_image))
    if not exact_identity or not torch.equal(clean_candidate, input_image):
        raise ValueError("sign-separated model is not exact identity initialized")
    if any(
        "global" in name and "scale" in name
        for name, _ in model.named_parameters()
    ):
        raise ValueError("sign-separated model contains a global scale")
    if set(inspect.signature(model.forward).parameters) != {"x"}:
        raise ValueError("sign-separated forward public interface changed")
    expected_infer_parameters = {
        "model",
        "image",
        "device",
        "tile_size",
        "stride",
        "alpha_threshold",
    }
    if set(inspect.signature(infer_full_page).parameters) != expected_infer_parameters:
        raise ValueError("cleanup inference public interface changed")

    forced_cases = [forced_route_case(index) for index in (0, 1, 2)]
    identity_forced, brighten_forced, darken_forced = forced_cases
    if identity_forced["delta_abs_max"] != 0.0:
        raise ValueError("forced identity route is not a no-op")
    if brighten_forced["negative_pixel_count"] != 0:
        raise ValueError("forced brighten route emitted a negative delta")
    if darken_forced["positive_pixel_count"] != 0:
        raise ValueError("forced darken route emitted a positive delta")
    if any(case["delta_abs_max"] > RESIDUAL_BOUND + 1e-7 for case in forced_cases):
        raise ValueError("forced route exceeded residual bound")

    gradient_cases = [gradient_case(direction) for direction in (1, -1)]
    for case in gradient_cases:
        matching_key = (
            "bright_gradient_abs" if case["direction"] > 0 else "dark_gradient_abs"
        )
        opposite_key = (
            "dark_gradient_abs" if case["direction"] > 0 else "bright_gradient_abs"
        )
        if case[matching_key] <= 0.0 or case["route_gradient_abs"] <= 0.0:
            raise ValueError(f"matching branch gradient died: {case}")
        if case[opposite_key] != 0.0:
            raise ValueError(f"opposite branch gradient is not isolated: {case}")

    two_step_cases = [two_step_case(direction) for direction in (1, -1)]
    opposed_pixel_count = sum(
        int(case["opposed_pixel_count"]) for case in two_step_cases
    )
    if opposed_pixel_count:
        raise ValueError("two-step update emitted an opposing pixel")
    if any(case["delta_abs_max"] <= 0.0 for case in two_step_cases):
        raise ValueError("two-step update produced no movement")
    if any(
        case["delta_abs_max"] > RESIDUAL_BOUND + 1e-7
        for case in two_step_cases
    ):
        raise ValueError("two-step update exceeded residual bound")

    identity_case = identity_two_step_case()
    if not identity_case["exact_identity"]:
        raise ValueError("identity-target optimization moved the candidate")
    if (
        identity_case["bright_magnitude_abs_max"] != 0.0
        or identity_case["dark_magnitude_abs_max"] != 0.0
    ):
        raise ValueError("identity-target optimization activated a magnitude")

    serialization_model = SignSeparatedResidualDeltaCleanupNet(
        residual_delta_bound=RESIDUAL_BOUND
    ).eval()
    with torch.no_grad():
        serialization_model.route_head[-1].bias.copy_(
            torch.tensor([0.0, 1.0, -1.0])
        )
        serialization_model.bright_magnitude_head[-1].bias.fill_(0.25)
        serialization_expected = serialization_model(input_image)[0]
    serialization_delta_abs_max = float(
        (serialization_expected - input_image).abs().max()
    )
    if serialization_delta_abs_max <= 0.0:
        raise ValueError("serialization fixture did not create nonzero output")
    with TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "model.pt"
        torch.save(
            {
                "args": {
                    "model_type": MODEL_TYPE,
                    "residual_delta_bound": RESIDUAL_BOUND,
                },
                "model": serialization_model.state_dict(),
            },
            checkpoint_path,
        )
        restored = load_model(checkpoint_path, torch.device("cpu"))
        with torch.no_grad():
            restored_candidate = restored(input_image)[0]
    serialization_exact = bool(
        torch.equal(restored_candidate, serialization_expected)
    )
    if not serialization_exact:
        raise ValueError("checkpoint roundtrip changed output")

    return {
        "status": "pass",
        "terminal": "PASS",
        "model_type": MODEL_TYPE,
        "residual_delta_bound": RESIDUAL_BOUND,
        "exact_identity_init": exact_identity,
        "zero_magnitude_projection_init": zero_magnitude_projection_init,
        "has_global_scale": False,
        "training_cli_enabled": training_cli_enabled,
        "legacy_models": legacy,
        "forced_route_cases": forced_cases,
        "gradient_cases": gradient_cases,
        "two_step_cases": two_step_cases,
        "identity_two_step_case": identity_case,
        "opposed_pixel_count": opposed_pixel_count,
        "serialization_exact": serialization_exact,
        "serialization_delta_abs_max": serialization_delta_abs_max,
        "forward_parameters": sorted(inspect.signature(model.forward).parameters),
        "inference_parameters": sorted(expected_infer_parameters),
        "state_tensor_count": len(model.state_dict()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
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
