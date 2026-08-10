#!/usr/bin/env python3
"""Audit the single registered sign-separated v2 checkpoint before gating."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.validate_sign_separated_candidate_application_preflight import (  # noqa: E402
    APPLICATION_PREREQUISITE_ID,
    PLAN_PATH,
    PLAN_SHA256,
)
from scripts.analysis.validate_sign_separated_training_preflight import (  # noqa: E402
    LEDGER_PATH,
    read_json,
    sha256_file,
    validate_artifact,
)
from scripts.infer.patch_cleanup_erasemap import (  # noqa: E402
    SignSeparatedResidualDeltaCleanupNet,
    resolve_device,
)
from scripts.train.train_sign_separated_residual_probe import (  # noqa: E402
    LOSS_TERM_NAMES,
    MODEL_TYPE,
    TargetDifferencePatchDataset,
)


APPLICATION_RECORD_ID = (
    "sign-separated-residual-candidate-application-preflight"
)
APPLICATION_OUTCOME = (
    "v1_unreachable_scale_rejected_and_v2_bounded_third_stage_application_passed"
)
CHECKPOINT_PREREQUISITE_ID = "sign_separated_residual_checkpoint_audit"


class AuditError(RuntimeError):
    pass


def repo_path(repo_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise AuditError(f"{label} must be a repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise AuditError(f"{label} must stay inside repository")
    return repo_root / path


def validate_authority(
    repo_root: Path,
    ledger: dict[str, Any],
) -> dict[str, str]:
    active = ledger.get("active_iteration", {})
    if active.get("id") != "sign-separated-residual-repair":
        raise AuditError("active iteration changed")
    prerequisites = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if prerequisites.get(APPLICATION_PREREQUISITE_ID) != "passed":
        raise AuditError("candidate application prerequisite is not passed")
    checkpoint_status = prerequisites.get(CHECKPOINT_PREREQUISITE_ID)
    if checkpoint_status not in {"pending", "passed"}:
        raise AuditError("checkpoint audit prerequisite is not registered")
    records = [
        record
        for record in ledger.get("records", [])
        if isinstance(record, dict) and record.get("id") == APPLICATION_RECORD_ID
    ]
    if len(records) != 1:
        raise AuditError("exactly one application preflight PASS record is required")
    record = records[0]
    if (
        record.get("terminal") != "PASS"
        or record.get("outcome") != APPLICATION_OUTCOME
    ):
        raise AuditError("application preflight record has wrong authority")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise AuditError("application preflight record lacks evidence")
    for item in evidence:
        validate_artifact(repo_root, item, "application preflight evidence")
    return {
        "candidate_application": "passed",
        "checkpoint_audit": checkpoint_status,
    }


def validate_history(path: Path, expected_steps: int) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing training history: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    expected_fields = ["step", *LOSS_TERM_NAMES]
    if fieldnames != expected_fields:
        raise AuditError("training history columns changed")
    if len(rows) != expected_steps:
        raise AuditError("training history step count changed")
    if [int(row["step"]) for row in rows] != list(range(1, expected_steps + 1)):
        raise AuditError("training history steps are not contiguous")
    for row in rows:
        for name in LOSS_TERM_NAMES:
            value = float(row[name])
            if not math.isfinite(value) or value < 0.0:
                raise AuditError(f"training history has invalid {name}")
    return {
        "row_count": len(rows),
        "columns": expected_fields,
        "sha256": sha256_file(path),
        "first": {name: float(rows[0][name]) for name in LOSS_TERM_NAMES},
        "last": {name: float(rows[-1][name]) for name in LOSS_TERM_NAMES},
    }


def normalized_checkpoint_args(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(item) if isinstance(item, Path) else item
        for key, item in value.items()
    }


def expected_checkpoint_args(
    repo_root: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    trainer = plan["trainer"]
    base_plan = read_json(
        repo_path(
            repo_root,
            plan["evidence"]["base_training_plan"]["path"],
            "base training plan",
        )
    )
    return {
        "data_root": base_plan["data"]["data_root"],
        "split": trainer["split"],
        "input_dir": trainer["input_dir"],
        "patch_index_file": trainer["patch_index_file"],
        "output_dir": trainer["output_dir"],
        "residual_delta_bound": trainer["residual_delta_bound"],
        "device": trainer["device"],
        "tile_size": trainer["tile_size"],
        "max_steps": trainer["max_steps"],
        "batch_size": trainer["batch_size"],
        "lr": trainer["lr"],
        "seed": trainer["seed"],
        "log_every": trainer["log_every"],
        "save_every": trainer["save_every"],
        "sign_direction_margin": trainer["sign_direction_margin"],
        "route_loss_weight": trainer["route_loss_weight"],
        "bright_magnitude_weight": trainer["bright_magnitude_weight"],
        "dark_magnitude_weight": trainer["dark_magnitude_weight"],
        "identity_delta_weight": trainer["identity_delta_weight"],
        "model_type": MODEL_TYPE,
        "mask_source": "target_delta",
        "validation_enabled": False,
    }


def load_checkpoint(
    checkpoint_path: Path,
    plan: dict[str, Any],
    repo_root: Path,
) -> tuple[SignSeparatedResidualDeltaCleanupNet, dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise AuditError(f"missing checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or set(state) != {"args", "model", "step"}:
        raise AuditError("checkpoint envelope changed")
    if state["step"] != plan["trainer"]["max_steps"]:
        raise AuditError("checkpoint step changed")
    args = state["args"]
    if not isinstance(args, dict):
        raise AuditError("checkpoint args are missing")
    if normalized_checkpoint_args(args) != expected_checkpoint_args(
        repo_root, plan
    ):
        raise AuditError("checkpoint args changed")
    model_state = state["model"]
    if not isinstance(model_state, dict) or len(model_state) != 36:
        raise AuditError("checkpoint model state changed")
    for name, tensor in model_state.items():
        if not isinstance(tensor, torch.Tensor) or not bool(torch.isfinite(tensor).all()):
            raise AuditError(f"checkpoint tensor is invalid: {name}")
    model = SignSeparatedResidualDeltaCleanupNet(
        residual_delta_bound=plan["model"]["residual_delta_bound"]
    )
    model.load_state_dict(model_state)
    model.eval()
    return model, state


def parameter_movement(
    model: SignSeparatedResidualDeltaCleanupNet,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    initial = SignSeparatedResidualDeltaCleanupNet(
        residual_delta_bound=model.residual_delta_bound
    )
    initial_state = initial.state_dict()
    movement = {
        name: float((tensor.detach().cpu() - initial_state[name]).abs().sum())
        for name, tensor in model.state_dict().items()
    }
    total = sum(movement.values())
    output_heads = sum(
        value
        for name, value in movement.items()
        if name.startswith(
            ("route_head", "bright_magnitude_head", "dark_magnitude_head")
        )
    )
    if total <= 0.0 or output_heads <= 0.0:
        raise AuditError("checkpoint is an identity-initialized no-op state")
    return {
        "total_l1": total,
        "output_head_l1": output_heads,
    }


def audit_real_patch_behavior(
    *,
    model: SignSeparatedResidualDeltaCleanupNet,
    plan: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    trainer = plan["trainer"]
    base_plan = read_json(
        repo_path(
            repo_root,
            plan["evidence"]["base_training_plan"]["path"],
            "base training plan",
        )
    )
    dataset = TargetDifferencePatchDataset(
        data_root=repo_path(repo_root, base_plan["data"]["data_root"], "data root"),
        split=trainer["split"],
        input_dir=repo_path(repo_root, trainer["input_dir"], "input dir"),
        patch_index_file=repo_path(
            repo_root, trainer["patch_index_file"], "patch index"
        ),
        tile_size=trainer["tile_size"],
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    device = resolve_device(trainer["device"])
    model = model.to(device)
    totals = {
        "pixel_count": 0,
        "applied_pixel_count": 0,
        "applied_bright_pixel_count": 0,
        "applied_dark_pixel_count": 0,
        "target_directional_applied_pixel_count": 0,
        "target_direction_correct_pixel_count": 0,
        "target_direction_opposed_pixel_count": 0,
        "target_identity_applied_pixel_count": 0,
    }
    route_counts = [0, 0, 0]
    maximum_delta = 0.0
    app = plan["candidate_application"]
    with torch.no_grad():
        for inp, target in loader:
            inp = inp.to(device)
            target = target.to(device)
            components = model.forward_components(inp)
            candidate_delta = (
                components["candidate"] - inp
            ).mean(dim=1, keepdim=True) * 255.0
            target_delta = (target - inp).mean(dim=1, keepdim=True) * 255.0
            applied = (
                components["edit_alpha"] >= app["edit_probability_threshold"]
            ) & (candidate_delta.abs() >= app["minimum_delta_threshold"])
            target_bright = target_delta > trainer["sign_direction_margin"]
            target_dark = target_delta < -trainer["sign_direction_margin"]
            target_directional = target_bright | target_dark
            correct = applied & (
                (target_bright & (candidate_delta > 0))
                | (target_dark & (candidate_delta < 0))
            )
            opposed = applied & (
                (target_bright & (candidate_delta < 0))
                | (target_dark & (candidate_delta > 0))
            )
            totals["pixel_count"] += applied.numel()
            totals["applied_pixel_count"] += int(applied.sum())
            totals["applied_bright_pixel_count"] += int(
                (applied & (candidate_delta > 0)).sum()
            )
            totals["applied_dark_pixel_count"] += int(
                (applied & (candidate_delta < 0)).sum()
            )
            totals["target_directional_applied_pixel_count"] += int(
                (applied & target_directional).sum()
            )
            totals["target_direction_correct_pixel_count"] += int(correct.sum())
            totals["target_direction_opposed_pixel_count"] += int(opposed.sum())
            totals["target_identity_applied_pixel_count"] += int(
                (applied & ~target_directional).sum()
            )
            routes = components["route_logits"].argmax(dim=1)
            for route in range(3):
                route_counts[route] += int((routes == route).sum())
            maximum_delta = max(
                maximum_delta, float(candidate_delta.abs().max())
            )
    if maximum_delta > plan["candidate_application"]["model_output_bound_gray"] + 1e-4:
        raise AuditError("real patch output exceeded the registered bound")
    structural_failures: list[str] = []
    if totals["applied_pixel_count"] == 0:
        structural_failures.append("no_application_eligible_pixels")
    if totals["applied_bright_pixel_count"] == 0:
        structural_failures.append("no_application_eligible_brighten_pixels")
    if totals["applied_dark_pixel_count"] == 0:
        structural_failures.append("no_application_eligible_darken_pixels")
    return {
        "patch_count": len(dataset),
        **totals,
        "applied_ratio": totals["applied_pixel_count"] / totals["pixel_count"],
        "route_argmax_pixel_counts": {
            "identity": route_counts[0],
            "brighten": route_counts[1],
            "darken": route_counts[2],
        },
        "candidate_delta_abs_max": maximum_delta,
        "structural_failures": structural_failures,
    }


def run_audit(
    *,
    repo_root: Path = ROOT,
    plan_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_plan = plan_path or (repo_root / PLAN_PATH)
    resolved_ledger = ledger_path or (repo_root / LEDGER_PATH)
    try:
        if sha256_file(resolved_plan) != PLAN_SHA256:
            raise AuditError("candidate v2 plan hash changed")
        plan = read_json(resolved_plan)
        ledger = read_json(resolved_ledger)
        authority = validate_authority(repo_root, ledger)
        output_dir = repo_path(
            repo_root, plan["trainer"]["output_dir"], "training output"
        )
        if not output_dir.is_dir():
            raise AuditError("registered training output is missing")
        expected_names = {
            "sign_separated_loss_history.csv",
            "sign_separated_probe.pt",
        }
        actual_names = {path.name for path in output_dir.iterdir()}
        if actual_names != expected_names:
            raise AuditError("training output file set changed")
        history = validate_history(
            output_dir / "sign_separated_loss_history.csv",
            plan["trainer"]["max_steps"],
        )
        checkpoint_path = output_dir / "sign_separated_probe.pt"
        model, _state = load_checkpoint(checkpoint_path, plan, repo_root)
        movement = parameter_movement(model, plan["trainer"]["seed"])
        patch_behavior = audit_real_patch_behavior(
            model=model,
            plan=plan,
            repo_root=repo_root,
        )
        gate_outputs = plan["planned_outputs_must_be_absent"]
        forbidden = [
            label
            for label in (
                "first_gate_baseline_pipeline",
                "first_gate_baseline_primary",
                "first_gate_candidate",
                "first_gate_score",
            )
            if repo_path(repo_root, gate_outputs[label], label).exists()
        ]
        if forbidden:
            raise AuditError(f"quality gate started before checkpoint audit: {forbidden}")
    except (AuditError, KeyError, OSError, TypeError, ValueError) as exc:
        return {
            "terminal": "KILL",
            "runnable": False,
            "reason": str(exc),
        }
    result = {
        "status": "pass",
        "terminal": "PASS" if not patch_behavior["structural_failures"] else "KILL",
        "runnable": not patch_behavior["structural_failures"],
        "authority": authority,
        "plan": str(resolved_plan),
        "plan_sha256": sha256_file(resolved_plan),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history": history,
        "parameter_movement": movement,
        "real_patch_behavior": patch_behavior,
        "first_quality_gate_started": False,
        "later_gates_enabled": False,
        "promotion_enabled": False,
        "reserved_blind_state": "unavailable",
        "current_primary_replaced": False,
    }
    if patch_behavior["structural_failures"]:
        result["status"] = "kill"
        result["reason"] = "checkpoint failed structural application audit"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = run_audit(
        repo_root=args.repo_root,
        plan_path=args.plan,
        ledger_path=args.ledger,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["terminal"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
