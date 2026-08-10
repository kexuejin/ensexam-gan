#!/usr/bin/env python3
"""Synthetic prerequisite audit for monotonic residual erasure."""

from __future__ import annotations

import argparse
import hashlib
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
from scripts.infer.monotonic_residual_erase import (  # noqa: E402
    MonotonicResidualEraseCleanupNet,
    load_monotonic_residual_erase_model,
)


MODEL_TYPE = "monotonic_residual_erase"
RESIDUAL_BOUND = 0.08
DIRECTION_MARGIN = 2.0 / 255.0
HISTORICAL_TRAINER_SHA256 = (
    "ce45f17c7d377aa665c9583215baead7ca555858cfe291ac089072ca8e51dc16"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def masked_l1(
    value: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype)
    denominator = weight.sum().clamp_min(1.0)
    return (torch.abs(value - target) * weight).sum() / denominator


def compute_synthetic_loss_terms(
    model: MonotonicResidualEraseCleanupNet,
    input_image: torch.Tensor,
    target: torch.Tensor,
    *,
    direction_margin: float = DIRECTION_MARGIN,
) -> dict[str, torch.Tensor]:
    if input_image.shape != target.shape:
        raise ValueError("input_image and target must have matching shapes")
    components = model.forward_components(input_image)
    target_luma_delta = (target - input_image).mean(dim=1, keepdim=True)
    edit_mask = target_luma_delta > direction_margin
    preserve_mask = ~edit_mask
    support_target = edit_mask.to(dtype=input_image.dtype)
    support_loss = F.binary_cross_entropy_with_logits(
        components["edit_logits"],
        support_target,
    )
    bright_target = target_luma_delta.clamp(
        0.0,
        model.residual_delta_bound,
    )
    magnitude_loss = masked_l1(
        components["bright_magnitude"],
        bright_target,
        edit_mask,
    )
    predicted_luma_delta = components["signed_delta"].mean(
        dim=1,
        keepdim=True,
    )
    preserve_loss = masked_l1(
        predicted_luma_delta,
        torch.zeros_like(predicted_luma_delta),
        preserve_mask,
    )
    return {
        "loss": support_loss + magnitude_loss + preserve_loss,
        "support_loss": support_loss,
        "magnitude_loss": magnitude_loss,
        "preserve_loss": preserve_loss,
        "edit_mask_count": edit_mask.sum(),
        "preserve_mask_count": preserve_mask.sum(),
    }


def final_bias_gradient(
    model: MonotonicResidualEraseCleanupNet,
    head_name: str,
) -> float:
    gradient = getattr(model, head_name)[-1].bias.grad
    return 0.0 if gradient is None else float(gradient.detach().abs().sum())


def gradient_case(target_delta: float) -> dict[str, float | int]:
    torch.manual_seed(20260810)
    model = MonotonicResidualEraseCleanupNet(RESIDUAL_BOUND).train()
    input_image = torch.full((1, 3, 8, 8), 0.5)
    target = input_image + target_delta
    terms = compute_synthetic_loss_terms(model, input_image, target)
    terms["loss"].backward()
    return {
        "target_delta": target_delta,
        "support_gradient_abs": final_bias_gradient(
            model,
            "edit_support_head",
        ),
        "magnitude_gradient_abs": final_bias_gradient(
            model,
            "bright_magnitude_head",
        ),
        "edit_mask_count": int(terms["edit_mask_count"]),
        "preserve_mask_count": int(terms["preserve_mask_count"]),
    }


def optimization_case(target_delta: float) -> dict[str, float | int | bool]:
    torch.manual_seed(20260810)
    model = MonotonicResidualEraseCleanupNet(RESIDUAL_BOUND).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    input_image = torch.full((1, 3, 8, 8), 0.5)
    target = input_image + target_delta
    for _step in range(2):
        optimizer.zero_grad(set_to_none=True)
        terms = compute_synthetic_loss_terms(model, input_image, target)
        terms["loss"].backward()
        optimizer.step()
    with torch.no_grad():
        components = model.forward_components(input_image)
    delta = components["candidate"] - input_image
    return {
        "target_delta": target_delta,
        "exact_noop": bool(torch.equal(components["candidate"], input_image)),
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "delta_abs_max": float(delta.abs().max()),
        "negative_pixel_count": int((delta < -1e-8).sum()),
        "bright_magnitude_abs_max": float(
            components["bright_magnitude"].abs().max()
        ),
    }


def audit_existing_models(input_image: torch.Tensor) -> dict[str, object]:
    specs = {
        "erasemap": (EraseMapCleanupNet, 384612, 32),
        "residual_delta": (ResidualDeltaCleanupNet, 384612, 32),
        "sign_separated_residual_delta": (
            SignSeparatedResidualDeltaCleanupNet,
            389253,
            36,
        ),
    }
    result = {}
    for model_type, (expected_class, parameter_count, state_count) in specs.items():
        model = build_model(
            model_type,
            residual_delta_scale=RESIDUAL_BOUND,
            residual_delta_bound=RESIDUAL_BOUND,
        ).eval()
        if not isinstance(model, expected_class):
            raise ValueError(f"existing builder changed for {model_type}")
        if sum(parameter.numel() for parameter in model.parameters()) != parameter_count:
            raise ValueError(f"existing parameter count changed for {model_type}")
        if len(model.state_dict()) != state_count:
            raise ValueError(f"existing state shape changed for {model_type}")
        with torch.no_grad():
            expected = model(input_image)
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "existing.pt"
            torch.save(
                {
                    "args": {
                        "model_type": model_type,
                        "residual_delta_scale": RESIDUAL_BOUND,
                        "residual_delta_bound": RESIDUAL_BOUND,
                    },
                    "model": model.state_dict(),
                },
                checkpoint_path,
            )
            restored = load_model(checkpoint_path, torch.device("cpu"))
            with torch.no_grad():
                actual = restored(input_image)
        if not all(
            torch.equal(expected_value, actual_value)
            for expected_value, actual_value in zip(expected, actual)
        ):
            raise ValueError(f"existing checkpoint load changed for {model_type}")
        result[model_type] = {
            "parameter_count": parameter_count,
            "state_tensor_count": state_count,
            "checkpoint_roundtrip_exact": True,
        }
    return result


def run_audit(seed: int = 20260810) -> dict[str, object]:
    torch.manual_seed(seed)
    input_image = torch.rand(2, 3, 16, 16) * 0.8 + 0.1
    existing_models = audit_existing_models(input_image)

    historical_trainer = ROOT / "scripts/train/train_patch_cleanup_erasemap_probe.py"
    historical_trainer_sha256 = sha256_file(historical_trainer)
    if historical_trainer_sha256 != HISTORICAL_TRAINER_SHA256:
        raise ValueError("historical cleanup trainer changed")
    training_cli_enabled = MODEL_TYPE in historical_trainer.read_text(
        encoding="utf-8"
    )
    if training_cli_enabled:
        raise ValueError("synthetic-only model is enabled in the historical trainer")

    model = MonotonicResidualEraseCleanupNet(RESIDUAL_BOUND).eval()
    zero_output_projection_init = all(
        bool((tensor.detach() == 0).all())
        for head in (model.edit_support_head, model.bright_magnitude_head)
        for tensor in (head[-1].weight, head[-1].bias)
    )
    if not zero_output_projection_init:
        raise ValueError("output projections are not exactly zero")
    with torch.no_grad():
        components = model.forward_components(input_image)
    exact_identity_init = bool(
        torch.equal(components["candidate"], input_image)
        and torch.equal(components["clean_candidate"], input_image)
    )
    if not exact_identity_init:
        raise ValueError("monotonic model is not exact identity initialized")
    parameter_names = {name for name, _parameter in model.named_parameters()}
    if any("dark" in name or "route" in name for name in parameter_names):
        raise ValueError("monotonic model contains a competing route or dark branch")
    if any("global" in name and "scale" in name for name in parameter_names):
        raise ValueError("monotonic model contains a global scale")
    if set(inspect.signature(model.forward).parameters) != {"x"}:
        raise ValueError("monotonic forward accepts more than input x")

    with torch.no_grad():
        model.edit_support_head[-1].bias.fill_(100.0)
        model.bright_magnitude_head[-1].bias.fill_(100.0)
        forced_candidate = model(input_image)[0]
    forced_delta = forced_candidate - input_image
    forced_case = {
        "delta_min": float(forced_delta.min()),
        "delta_max": float(forced_delta.max()),
        "negative_pixel_count": int((forced_delta < -1e-8).sum()),
        "positive_pixel_count": int((forced_delta > 1e-8).sum()),
    }
    if forced_case["negative_pixel_count"] != 0:
        raise ValueError("forced monotonic output emitted a negative delta")
    if forced_case["delta_max"] > RESIDUAL_BOUND + 1e-7:
        raise ValueError("forced monotonic output exceeded the residual bound")
    if forced_case["positive_pixel_count"] == 0:
        raise ValueError("forced monotonic output produced no movement")

    gradient_cases = [gradient_case(delta) for delta in (0.05, -0.05, 0.0)]
    brighten_gradient, darker_gradient, identity_gradient = gradient_cases
    if (
        brighten_gradient["support_gradient_abs"] <= 0.0
        or brighten_gradient["magnitude_gradient_abs"] <= 0.0
    ):
        raise ValueError("brighten support or magnitude gradient died")
    for preserve_case in (darker_gradient, identity_gradient):
        if preserve_case["support_gradient_abs"] <= 0.0:
            raise ValueError("preserve support gradient died")
        if preserve_case["magnitude_gradient_abs"] != 0.0:
            raise ValueError("preserve target activated brighten magnitude")

    optimization_cases = [
        optimization_case(delta) for delta in (0.05, -0.05, 0.0)
    ]
    brighten_case, darker_case, identity_case = optimization_cases
    if brighten_case["delta_max"] <= 0.0 or brighten_case["exact_noop"]:
        raise ValueError("brighten optimization produced no movement")
    if brighten_case["delta_max"] > RESIDUAL_BOUND + 1e-7:
        raise ValueError("brighten optimization exceeded the residual bound")
    if brighten_case["negative_pixel_count"] != 0:
        raise ValueError("brighten optimization emitted a negative delta")
    for preserve_case in (darker_case, identity_case):
        if not preserve_case["exact_noop"]:
            raise ValueError("preserve optimization moved the candidate")
        if preserve_case["bright_magnitude_abs_max"] != 0.0:
            raise ValueError("preserve optimization activated magnitude")

    serialization_model = MonotonicResidualEraseCleanupNet(
        RESIDUAL_BOUND
    ).eval()
    with torch.no_grad():
        serialization_model.edit_support_head[-1].bias.fill_(1.0)
        serialization_model.bright_magnitude_head[-1].bias.fill_(0.25)
        serialization_expected = serialization_model(input_image)[0]
    serialization_delta_abs_max = float(
        (serialization_expected - input_image).abs().max()
    )
    if serialization_delta_abs_max <= 0.0:
        raise ValueError("serialization fixture produced no movement")
    with TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "monotonic.pt"
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
        restored = load_monotonic_residual_erase_model(
            checkpoint_path,
            torch.device("cpu"),
        )
        with torch.no_grad():
            serialization_actual = restored(input_image)[0]
    serialization_exact = bool(
        torch.equal(serialization_actual, serialization_expected)
    )
    if not serialization_exact:
        raise ValueError("monotonic checkpoint roundtrip changed output")

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

    return {
        "status": "pass",
        "terminal": "PASS",
        "model_type": MODEL_TYPE,
        "residual_delta_bound": RESIDUAL_BOUND,
        "exact_identity_init": exact_identity_init,
        "zero_output_projection_init": zero_output_projection_init,
        "has_competing_route_or_dark_branch": False,
        "has_global_scale": False,
        "training_cli_enabled": training_cli_enabled,
        "training_authorized": False,
        "real_data_access": False,
        "target_decode": False,
        "checkpoint_generated": False,
        "quality_gate_started": False,
        "promotion_enabled": False,
        "reserved_blind_state": "unavailable",
        "historical_trainer_sha256": historical_trainer_sha256,
        "existing_models": existing_models,
        "forced_case": forced_case,
        "gradient_cases": gradient_cases,
        "optimization_cases": optimization_cases,
        "serialization_exact": serialization_exact,
        "serialization_delta_abs_max": serialization_delta_abs_max,
        "forward_parameters": sorted(inspect.signature(model.forward).parameters),
        "inference_parameters": sorted(expected_infer_parameters),
        "state_tensor_count": len(model.state_dict()),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
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
