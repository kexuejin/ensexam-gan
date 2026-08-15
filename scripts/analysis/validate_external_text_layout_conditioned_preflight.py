#!/usr/bin/env python3
"""Validate the external-text-layout conditioned monotonic preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer.monotonic_residual_erase import (  # noqa: E402
    MODEL_TYPE,
    MonotonicResidualEraseCleanupNet,
)
from scripts.infer.run_monotonic_residual_erase_candidate import (  # noqa: E402
    apply_monotonic_candidate_gate,
)


PLAN_PATH = Path("docs/external-text-layout-conditioned-monotonic-preflight-v1.json")
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
OUTPUT_PATH = Path(
    "outputs/external-text-layout-conditioned-preflight-20260815/preflight.json"
)
ACTIVE_ITERATION_ID = "monotonic-residual-erase-support"
SUPPORT_PREREQUISITE_ID = "external_text_layout_support_train_only_diagnostic"

EXPECTED_AUTHORIZATION = {
    "candidate_inference": False,
    "checkpoint_generation": False,
    "holdout40": False,
    "model_training": False,
    "promotion": False,
    "reserved_blind": False,
    "scut115": False,
    "state": "preflight_only",
    "visual_review": False,
}
EXPECTED_CHANNEL_ORDER = [
    "second_stage_r",
    "second_stage_g",
    "second_stage_b",
    "external_text_occupancy",
    "external_text_confidence",
]
EXPECTED_MODEL = {
    "allowed_delta_direction": "nonnegative_rgb_only",
    "input_channels": 5,
    "model_type": MODEL_TYPE,
    "output_channels": 3,
    "residual_delta_bound": 0.08,
}
EXPECTED_TRAINER = {
    "batch_size": 1,
    "device": "cpu",
    "input_channels": 5,
    "log_every": 10,
    "lr": 0.0001,
    "max_steps": 80,
    "output_dir": "artifacts/trials/external-text-layout-conditioned-monotonic-v1",
    "patch_index_file": (
        "hardcase_lists/external-text-layout-conditioned-monotonic-train-patches-v1.csv"
    ),
    "residual_delta_bound": 0.08,
    "save_every": 0,
    "seed": 42,
    "tile_size": 256,
    "validation_enabled": False,
}
EXPECTED_APPLICATION = {
    "baseline_input": "recovered_second_stage_rgb_plus_external_text_layout_channels",
    "device": "cpu",
    "direction_enforcement": "candidate_must_not_darken_any_rgb_channel",
    "edit_probability_threshold": 0.5,
    "minimum_delta_threshold": 12.0,
    "stride": 160,
    "tile_size": 256,
}
EXPECTED_SUPPORT_REQUIREMENTS = {
    "full_auc_ablation_margin_min": 0.03,
    "full_mean_fold_auc_min": 0.65,
    "minimum_passed_folds": 5,
    "positive_mean_above_preserve_min_folds": 4,
    "train_page_count": 275,
}
EXPECTED_PLANNED_OUTPUTS_MUST_BE_ABSENT = {
    "checkpoint_audit": (
        "outputs/external-text-layout-conditioned-monotonic-checkpoint-audit-20260815"
    ),
    "first_gate_candidate": (
        "outputs/external-text-layout-conditioned-monotonic-inner-val15-candidate"
    ),
    "first_gate_score": (
        "outputs/external-text-layout-conditioned-monotonic-inner-val15-score"
    ),
    "patch_index": (
        "hardcase_lists/external-text-layout-conditioned-monotonic-train-patches-v1.csv"
    ),
    "patch_index_summary_dir": (
        "outputs/external-text-layout-conditioned-monotonic-train-patches-v1"
    ),
    "training_output_dir": (
        "artifacts/trials/external-text-layout-conditioned-monotonic-v1"
    ),
}
EXPECTED_TERMINAL_SUCCESSORS = {
    "KILL": (
        "close_layout_conditioned_monotonic_without_threshold_or_layout_transform_rescue"
    ),
    "PASS": "authorize_exact_layout_conditioned_trainer_and_application_implementation_only",
    "PREREQUISITE_NEEDED": "repair_registered_preflight_evidence_or_metadata_only",
}


class PreflightError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PreflightError(f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreflightError(f"JSON must be an object: {path}")
    return value


def repo_path(repo_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PreflightError(f"{label} path must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise PreflightError(f"{label} path must stay inside repository")
    return repo_root / relative


def validate_artifact(repo_root: Path, artifact: Any, label: str) -> Path:
    if not isinstance(artifact, dict):
        raise PreflightError(f"{label} must be an object")
    path = repo_path(repo_root, artifact.get("path"), label)
    expected_hash = artifact.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise PreflightError(f"{label} SHA-256 is invalid")
    if not path.is_file():
        raise PreflightError(f"{label} is missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise PreflightError(
            f"{label} artifact hash mismatch: "
            f"expected={expected_hash} actual={actual_hash}"
        )
    return path


def assert_exact_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != 1:
        raise PreflightError("conditioned plan schema changed")
    if plan.get("state") != "preregistered_external_text_layout_conditioned_preflight":
        raise PreflightError("conditioned plan state changed")
    if plan.get("iteration_id") != ACTIVE_ITERATION_ID:
        raise PreflightError("conditioned plan iteration changed")
    if plan.get("family") != "external_text_layout_conditioned_monotonic_v1":
        raise PreflightError("conditioned family changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise PreflightError("conditioned authorization changed")
    if plan.get("model") != EXPECTED_MODEL:
        raise PreflightError("conditioned model contract changed")
    if plan.get("trainer") != EXPECTED_TRAINER:
        raise PreflightError("conditioned trainer contract changed")
    if plan.get("candidate_application") != EXPECTED_APPLICATION:
        raise PreflightError("conditioned application contract changed")
    if plan.get("support_diagnostic_requirements") != EXPECTED_SUPPORT_REQUIREMENTS:
        raise PreflightError("conditioned support requirements changed")
    if (
        plan.get("planned_outputs_must_be_absent")
        != EXPECTED_PLANNED_OUTPUTS_MUST_BE_ABSENT
    ):
        raise PreflightError("conditioned planned outputs changed")
    if plan.get("terminal_successors") != EXPECTED_TERMINAL_SUCCESSORS:
        raise PreflightError("conditioned terminal successors changed")
    conditioned = plan.get("conditioned_input", {})
    if (
        conditioned.get("channel_order") != EXPECTED_CHANNEL_ORDER
        or conditioned.get("target_access") is not False
        or conditioned.get("shape_rule")
        != "layout_channels_must_match_rgb_height_width"
    ):
        raise PreflightError("conditioned input contract changed")
    first_gate = plan.get("first_quality_gate", {})
    if (
        first_gate.get("role") != "inner_val15"
        or first_gate.get("scut115") is not False
        or first_gate.get("holdout40") is not False
        or first_gate.get("minimum_residual_gain") != 0.0005
    ):
        raise PreflightError("conditioned first gate changed")
    synthetic = plan.get("synthetic_preflight", {})
    if (
        synthetic.get("device") != "cpu"
        or synthetic.get("target_delta") != 0.08
        or synthetic.get("identity_target_must_remain_exact") is not True
        or synthetic.get("layout_channel_gradient_required") is not True
    ):
        raise PreflightError("conditioned synthetic preflight changed")


def validate_ledger_authority(ledger: dict[str, Any]) -> dict[str, Any]:
    active = ledger.get("active_iteration", {})
    if active.get("id") != ACTIVE_ITERATION_ID:
        raise PreflightError("active iteration changed")
    if active.get("terminal") != "PREREQUISITE_NEEDED":
        raise PreflightError("active iteration terminal changed")
    prerequisites = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if prerequisites.get(SUPPORT_PREREQUISITE_ID) != "passed":
        raise PreflightError("external text-layout support diagnostic is not passed")
    for forbidden in ("scut115", "holdout40", "reserved_blind"):
        if forbidden not in active.get("prohibited_before_first_gate", []):
            raise PreflightError(f"{forbidden} is not prohibited before first gate")
    return {
        "active_iteration": active.get("id"),
        "support_diagnostic_status": prerequisites.get(SUPPORT_PREREQUISITE_ID),
    }


def validate_support_audit(
    repo_root: Path, plan: dict[str, Any]
) -> dict[str, Any]:
    evidence = plan.get("evidence")
    expected_names = {
        "effective_plan",
        "layout_materialization_manifest",
        "support_diagnostic",
        "support_diagnostic_decision",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected_names:
        raise PreflightError("conditioned evidence set changed")
    paths = {
        name: validate_artifact(repo_root, artifact, f"evidence.{name}")
        for name, artifact in evidence.items()
    }
    support = read_json(paths["support_diagnostic"])
    requirements = plan["support_diagnostic_requirements"]
    aggregates = support.get("aggregates", {})
    acceptance = support.get("acceptance", {})
    if (
        support.get("terminal") != "PASS"
        or support.get("train_page_count") != requirements["train_page_count"]
        or acceptance.get("passed") is not True
        or support.get("training_started") is not False
        or support.get("candidate_inference_started") is not False
        or support.get("quality_gate_started") is not False
        or support.get("reserved_blind_authorized") is not False
    ):
        raise PreflightError("support diagnostic no longer authorizes preflight")
    if (
        aggregates.get("full_mean_fold_auc")
        < requirements["full_mean_fold_auc_min"]
        or aggregates.get("full_auc_ablation_margin")
        < requirements["full_auc_ablation_margin_min"]
        or aggregates.get("positive_mean_above_preserve_folds")
        < requirements["positive_mean_above_preserve_min_folds"]
    ):
        raise PreflightError("support diagnostic aggregate gate changed")
    materialization = support.get("materialization", {})
    if (
        materialization.get("page_count") != requirements["train_page_count"]
        or materialization.get("recognition") is not False
        or materialization.get("target_access") is not False
    ):
        raise PreflightError("support materialization boundary changed")
    return {
        "aggregates": aggregates,
        "evidence_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "support_audit": str(paths["support_diagnostic"]),
    }


def stack_conditioned_input(
    rgb: torch.Tensor,
    occupancy: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must be NCHW with 3 channels")
    for name, value in (("occupancy", occupancy), ("confidence", confidence)):
        if value.ndim != 4 or value.shape[1] != 1:
            raise ValueError(f"{name} must be NCHW with 1 channel")
        if value.shape[0] != rgb.shape[0] or value.shape[2:] != rgb.shape[2:]:
            raise ValueError(f"{name} shape must match rgb height and width")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")
    if not bool(torch.isfinite(rgb).all()):
        raise ValueError("rgb contains non-finite values")
    if bool(((occupancy != 0.0) & (occupancy != 1.0)).any()):
        raise ValueError("occupancy must be binary")
    if bool((confidence < 0.0).any()) or bool((confidence > 1.0).any()):
        raise ValueError("confidence must stay in [0, 1]")
    return torch.cat([rgb, occupancy, confidence], dim=1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.expand_as(values)
    denominator = weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
    per_sample = (values * weight).sum(dim=(1, 2, 3)) / denominator
    return per_sample.mean()


def conditioned_loss_terms(
    model: MonotonicResidualEraseCleanupNet,
    features: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    luminance_margin_gray: float,
) -> dict[str, torch.Tensor]:
    if model.input_channels != features.shape[1]:
        raise ValueError("model input channel count does not match features")
    baseline_rgb = features[:, :3]
    if target_rgb.shape != baseline_rgb.shape:
        raise ValueError("target_rgb must match baseline RGB")
    components = model.forward_components(features)
    input_luma = baseline_rgb.mean(dim=1, keepdim=True)
    target_luma = target_rgb.mean(dim=1, keepdim=True)
    target_delta = target_luma - input_luma
    positive_mask = target_delta > (luminance_margin_gray / 255.0)
    preserve_mask = ~positive_mask
    positive_bce = masked_mean(
        F.binary_cross_entropy_with_logits(
            components["edit_logits"],
            torch.ones_like(components["edit_logits"]),
            reduction="none",
        ),
        positive_mask.float(),
    )
    preserve_bce = masked_mean(
        F.binary_cross_entropy_with_logits(
            components["edit_logits"],
            torch.zeros_like(components["edit_logits"]),
            reduction="none",
        ),
        preserve_mask.float(),
    )
    desired_magnitude = target_delta.clamp(0.0, model.residual_delta_bound)
    magnitude_l1 = masked_mean(
        torch.abs(components["bright_magnitude"] - desired_magnitude),
        positive_mask.float(),
    )
    preserve_delta_l1 = masked_mean(
        components["signed_delta"].abs().mean(dim=1, keepdim=True),
        preserve_mask.float(),
    )
    return {
        "loss": positive_bce + preserve_bce + magnitude_l1 + preserve_delta_l1,
        "positive_bce": positive_bce,
        "preserve_bce": preserve_bce,
        "magnitude_l1": magnitude_l1,
        "preserve_delta_l1": preserve_delta_l1,
    }


def run_synthetic_preflight(plan: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(20260815)
    model = MonotonicResidualEraseCleanupNet(
        residual_delta_bound=plan["model"]["residual_delta_bound"],
        input_channels=plan["model"]["input_channels"],
    )
    rgb = torch.full((1, 3, 16, 16), 0.5)
    occupancy = torch.zeros((1, 1, 16, 16))
    occupancy[:, :, 4:12, 4:12] = 1.0
    confidence = occupancy * 0.75
    features = stack_conditioned_input(rgb, occupancy, confidence)
    target_rgb = rgb + plan["synthetic_preflight"]["target_delta"]
    with torch.no_grad():
        candidate, edit_alpha, clean_candidate = model(features)
    if not torch.equal(candidate, rgb) or not torch.equal(clean_candidate, rgb):
        raise PreflightError("conditioned model is not exact identity at init")
    if candidate.shape[1] != plan["model"]["output_channels"]:
        raise PreflightError("conditioned model output channel count changed")
    if not torch.equal(edit_alpha, torch.full_like(edit_alpha, 0.5)):
        raise PreflightError("conditioned edit probability initialization changed")

    optimizer = torch.optim.Adam(model.parameters(), lr=plan["trainer"]["lr"])
    for _step in range(2):
        optimizer.zero_grad(set_to_none=True)
        terms = conditioned_loss_terms(
            model,
            features,
            target_rgb,
            luminance_margin_gray=2.0,
        )
        terms["loss"].backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    terms = conditioned_loss_terms(
        model,
        features,
        target_rgb,
        luminance_margin_gray=2.0,
    )
    terms["loss"].backward()
    layout_gradient = float(model.enc1.block[0].weight.grad[:, 3:, :, :].abs().sum())
    with torch.no_grad():
        moved = model(features)[0]
        delta = moved - rgb
    if layout_gradient <= 0.0:
        raise PreflightError("layout channel gradient did not reach encoder")
    if float(delta.min()) < -1e-7:
        raise PreflightError("conditioned model darkened RGB")
    return {
        "delta_max": float(delta.max()),
        "delta_min": float(delta.min()),
        "identity_exact": True,
        "layout_encoder_gradient_abs": layout_gradient,
        "output_channels": int(moved.shape[1]),
    }


def validate_application(plan: dict[str, Any]) -> dict[str, Any]:
    config = plan["candidate_application"]
    baseline = np.full((4, 4, 3), 128, dtype=np.uint8)
    identity, identity_gate, _ = apply_monotonic_candidate_gate(
        baseline,
        baseline.copy(),
        np.full((4, 4), 1.0, dtype=np.float32),
        edit_probability_threshold=config["edit_probability_threshold"],
        minimum_delta_threshold=config["minimum_delta_threshold"],
    )
    if not np.array_equal(identity, baseline) or bool(identity_gate.any()):
        raise PreflightError("conditioned identity application is not a no-op")
    candidate = np.full((4, 4, 3), 141, dtype=np.uint8)
    applied, gate, _ = apply_monotonic_candidate_gate(
        baseline,
        candidate,
        np.full((4, 4), 0.6, dtype=np.float32),
        edit_probability_threshold=config["edit_probability_threshold"],
        minimum_delta_threshold=config["minimum_delta_threshold"],
    )
    if not bool(gate.all()) or not np.array_equal(applied, candidate):
        raise PreflightError("reachable conditioned candidate was rejected")
    try:
        apply_monotonic_candidate_gate(
            baseline,
            np.full((4, 4, 3), 127, dtype=np.uint8),
            np.full((4, 4), 1.0, dtype=np.float32),
            edit_probability_threshold=config["edit_probability_threshold"],
            minimum_delta_threshold=config["minimum_delta_threshold"],
        )
    except ValueError as error:
        if "darkened" not in str(error):
            raise
    else:
        raise PreflightError("conditioned application accepted a darker candidate")
    return {
        "baseline_input": config["baseline_input"],
        "darker_candidate_rejected": True,
        "identity_noop": True,
        "reachable_brighten_applied": True,
        "target_or_route_options_absent": True,
    }


def validate_outputs_absent(repo_root: Path, plan: dict[str, Any]) -> list[str]:
    absent: list[str] = []
    for label, value in sorted(plan["planned_outputs_must_be_absent"].items()):
        path = repo_path(repo_root, value, f"planned output {label}")
        if path.exists():
            raise PreflightError(f"planned output must be absent: {path}")
        absent.append(str(path))
    return absent


def run_preflight(
    *,
    repo_root: Path = ROOT,
    plan_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_plan = plan_path or (repo_root / PLAN_PATH)
    resolved_ledger = ledger_path or (repo_root / LEDGER_PATH)
    try:
        plan = read_json(resolved_plan)
        assert_exact_plan(plan)
        ledger = read_json(resolved_ledger)
        authority = validate_ledger_authority(ledger)
        support = validate_support_audit(repo_root, plan)
        synthetic = run_synthetic_preflight(plan)
        application = validate_application(plan)
        absent = validate_outputs_absent(repo_root, plan)
    except (KeyError, OSError, PreflightError, TypeError, ValueError) as error:
        return {
            "reason": str(error),
            "runnable": False,
            "terminal": "PREREQUISITE_NEEDED",
        }
    return {
        "application": application,
        "authority": authority,
        "candidate_inference_started": False,
        "checkpoint_generated": False,
        "planned_outputs_absent": absent,
        "plan": str(resolved_plan),
        "plan_sha256": sha256_file(resolved_plan),
        "promotion_enabled": False,
        "quality_gate_started": False,
        "real_image_decode": False,
        "reserved_blind_state": "unavailable",
        "runnable": True,
        "schema_version": 1,
        "support": support,
        "synthetic_preflight": synthetic,
        "target_decode": False,
        "terminal": "PASS",
        "training_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    result = run_preflight(
        repo_root=args.repo_root,
        plan_path=args.plan,
        ledger_path=args.ledger,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        path = repo_path(args.repo_root.resolve(), str(args.output_json), "output")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["terminal"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
