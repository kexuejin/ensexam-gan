#!/usr/bin/env python3
"""Validate v2 training reachability and candidate application before training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.validate_sign_separated_training_preflight import (  # noqa: E402
    LEDGER_PATH,
    read_json,
    sha256_file,
    validate_artifact,
)
from scripts.infer.run_sign_separated_residual_candidate import (  # noqa: E402
    apply_candidate_gate,
    build_parser as build_inference_parser,
)
from scripts.infer.patch_cleanup_erasemap import (  # noqa: E402
    SignSeparatedResidualDeltaCleanupNet,
)
from scripts.train.train_sign_separated_residual_probe import (  # noqa: E402
    build_parser as build_training_parser,
    compute_sign_separated_loss_terms,
)


PLAN_PATH = Path("docs/sign-separated-residual-candidate-plan-v2.json")
PLAN_SHA256 = "7f2c35a86efe05fb50c91008bba055c1ea5dd6d3578eee1256813721d707e205"
MATERIALIZATION_RECORD_ID = (
    "sign-separated-residual-train275-materialization-audit"
)
MATERIALIZATION_OUTCOME = (
    "exact_train275_frozen_pipeline_and_direction_balanced_patch_materialization_passed"
)
APPLICATION_PREREQUISITE_ID = (
    "sign_separated_residual_candidate_application_preflight"
)


class PreflightError(RuntimeError):
    pass


def repo_path(repo_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PreflightError(f"{label} must be a repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise PreflightError(f"{label} must stay inside repository")
    return repo_root / path


def validate_plan(repo_root: Path, plan_path: Path) -> dict[str, Any]:
    if sha256_file(plan_path) != PLAN_SHA256:
        raise PreflightError("candidate v2 plan hash changed")
    plan = read_json(plan_path)
    if plan.get("schema_version") != 2:
        raise PreflightError("candidate v2 schema changed")
    if plan.get("iteration_id") != "sign-separated-residual-repair-v2":
        raise PreflightError("candidate v2 iteration changed")
    trainer = plan.get("trainer", {})
    application = plan.get("candidate_application", {})
    reachability = plan.get("synthetic_reachability", {})
    exact = {
        "learning_rate": trainer.get("lr"),
        "steps": trainer.get("max_steps"),
        "bound": trainer.get("residual_delta_bound"),
        "minimum_delta": application.get("minimum_delta_threshold"),
        "edit_probability": application.get("edit_probability_threshold"),
        "tile_size": application.get("tile_size"),
        "stride": application.get("stride"),
        "legacy_learning_rate": reachability.get("legacy_learning_rate"),
        "target_delta": reachability.get("target_delta"),
    }
    expected = {
        "learning_rate": 0.0001,
        "steps": 80,
        "bound": 0.08,
        "minimum_delta": 12.0,
        "edit_probability": 0.5,
        "tile_size": 256,
        "stride": 160,
        "legacy_learning_rate": 0.00002,
        "target_delta": 0.08,
    }
    if exact != expected:
        raise PreflightError("candidate v2 causal parameters changed")
    if application.get("baseline_input") != (
        "frozen_current_primary_plus_current_second_stage_prediction"
    ):
        raise PreflightError("candidate baseline input changed")
    if application.get("minimum_delta_threshold") > (
        trainer["residual_delta_bound"] * 255.0
    ):
        raise PreflightError("candidate delta gate is unreachable")
    if plan.get("authorization") != {
        "current_primary_replacement": "disabled",
        "quality_gates": "disabled_until_checkpoint_audit_pass",
        "reserved_blind": "unavailable",
        "state": "candidate_application_preflight_pending",
        "training": "prohibited_until_candidate_application_preflight_pass",
    }:
        raise PreflightError("candidate v2 authorization changed")
    return plan


def validate_evidence(
    repo_root: Path,
    plan: dict[str, Any],
) -> dict[str, str]:
    evidence = plan.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "base_training_plan",
        "candidate_inference",
        "cleanup_model",
        "materialization_audit",
        "materialization_decision",
        "trainer",
    }:
        raise PreflightError("candidate v2 evidence set changed")
    paths = {
        name: validate_artifact(repo_root, artifact, f"evidence.{name}")
        for name, artifact in evidence.items()
    }
    materialization = read_json(paths["materialization_audit"])
    if materialization.get("terminal") != "PASS":
        raise PreflightError("materialization audit is not PASS")
    patch_summary = materialization.get("patch_summary", {})
    if (
        materialization.get("train_count") != 275
        or patch_summary.get("patch_count") != 512
        or patch_summary.get("selected_counts")
        != {"brighten": 256, "darken": 256}
    ):
        raise PreflightError("materialized training inputs changed")
    return {name: sha256_file(path) for name, path in paths.items()}


def validate_ledger_authority(
    repo_root: Path,
    ledger: dict[str, Any],
) -> dict[str, str]:
    active = ledger.get("active_iteration", {})
    if active.get("id") != "sign-separated-residual-repair":
        raise PreflightError("active iteration changed")
    prerequisites = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if prerequisites.get(
        "sign_separated_residual_train_materialization_audit"
    ) != "passed":
        raise PreflightError("materialization prerequisite is not passed")
    status = prerequisites.get(APPLICATION_PREREQUISITE_ID)
    if status not in {"pending", "passed"}:
        raise PreflightError("candidate application prerequisite is not registered")
    records = [
        record
        for record in ledger.get("records", [])
        if isinstance(record, dict)
        and record.get("id") == MATERIALIZATION_RECORD_ID
    ]
    if len(records) != 1:
        raise PreflightError("exactly one materialization PASS record is required")
    record = records[0]
    if (
        record.get("terminal") != "PASS"
        or record.get("outcome") != MATERIALIZATION_OUTCOME
    ):
        raise PreflightError("materialization record has wrong authority")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise PreflightError("materialization record lacks evidence")
    for item in evidence:
        validate_artifact(repo_root, item, "materialization record evidence")
    return {
        "materialization": "passed",
        "candidate_application": status,
    }


def training_args(
    repo_root: Path,
    plan: dict[str, Any],
) -> argparse.Namespace:
    trainer = plan["trainer"]
    base_plan_path = repo_path(
        repo_root,
        plan["evidence"]["base_training_plan"]["path"],
        "base training plan",
    )
    data_root = read_json(base_plan_path)["data"]["data_root"]
    return build_training_parser().parse_args(
        [
            "--data-root",
            data_root,
            "--split",
            trainer["split"],
            "--input-dir",
            trainer["input_dir"],
            "--patch-index-file",
            trainer["patch_index_file"],
            "--output-dir",
            trainer["output_dir"],
            "--residual-delta-bound",
            str(trainer["residual_delta_bound"]),
            "--device",
            trainer["device"],
            "--tile-size",
            str(trainer["tile_size"]),
            "--max-steps",
            str(trainer["max_steps"]),
            "--batch-size",
            str(trainer["batch_size"]),
            "--lr",
            str(trainer["lr"]),
            "--seed",
            str(trainer["seed"]),
            "--log-every",
            str(trainer["log_every"]),
            "--save-every",
            str(trainer["save_every"]),
            "--sign-direction-margin",
            str(trainer["sign_direction_margin"]),
            "--route-loss-weight",
            str(trainer["route_loss_weight"]),
            "--bright-magnitude-weight",
            str(trainer["bright_magnitude_weight"]),
            "--dark-magnitude-weight",
            str(trainer["dark_magnitude_weight"]),
            "--identity-delta-weight",
            str(trainer["identity_delta_weight"]),
        ]
    )


def run_synthetic_case(
    *,
    learning_rate: float,
    steps: int,
    direction: int,
    target_delta: float,
    loss_args: argparse.Namespace,
) -> dict[str, Any]:
    torch.manual_seed(42)
    model = SignSeparatedResidualDeltaCleanupNet(0.08)
    inp = torch.full((1, 3, 8, 8), 0.5)
    target = inp + direction * target_delta
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        terms = compute_sign_separated_loss_terms(
            model, inp, target, loss_args
        )
        if not bool(torch.isfinite(terms["loss"])):
            raise PreflightError("synthetic reachability loss is non-finite")
        terms["loss"].backward()
        optimizer.step()
    with torch.no_grad():
        components = model.forward_components(inp)
        delta = (components["candidate"] - inp) * 255.0
    opposed = int((delta * direction < -1e-7).sum())
    return {
        "direction": direction,
        "delta_abs_max": float(delta.abs().max()),
        "delta_mean": float(delta.mean()),
        "opposed_pixel_count": opposed,
        "edit_probability_mean": float(components["edit_alpha"].mean()),
        "loss": float(terms["loss"].detach()),
    }


def synthetic_reachability(plan: dict[str, Any]) -> dict[str, Any]:
    trainer = plan["trainer"]
    config = plan["synthetic_reachability"]
    loss_args = argparse.Namespace(
        sign_direction_margin=trainer["sign_direction_margin"],
        route_loss_weight=trainer["route_loss_weight"],
        bright_magnitude_weight=trainer["bright_magnitude_weight"],
        dark_magnitude_weight=trainer["dark_magnitude_weight"],
        identity_delta_weight=trainer["identity_delta_weight"],
    )
    legacy_cases = [
        run_synthetic_case(
            learning_rate=config["legacy_learning_rate"],
            steps=config["steps"],
            direction=direction,
            target_delta=config["target_delta"],
            loss_args=loss_args,
        )
        for direction in (1, -1)
    ]
    registered_cases = [
        run_synthetic_case(
            learning_rate=config["registered_learning_rate"],
            steps=config["steps"],
            direction=direction,
            target_delta=config["target_delta"],
            loss_args=loss_args,
        )
        for direction in (1, -1)
    ]
    threshold = config["minimum_delta_threshold"]
    if any(case["delta_abs_max"] >= threshold for case in legacy_cases):
        raise PreflightError("legacy learning rate unexpectedly reached the gate")
    for case in registered_cases:
        if case["delta_abs_max"] < threshold:
            raise PreflightError("registered learning rate cannot reach the gate")
        if case["delta_abs_max"] > 20.4 + 1e-5:
            raise PreflightError("registered synthetic output exceeded the bound")
        if case["opposed_pixel_count"] != 0:
            raise PreflightError("registered synthetic output changed wrong direction")
        if case["edit_probability_mean"] < 0.5:
            raise PreflightError("registered route probability cannot reach the gate")

    torch.manual_seed(42)
    identity_model = SignSeparatedResidualDeltaCleanupNet(0.08)
    identity_input = torch.full((1, 3, 8, 8), 0.5)
    optimizer = torch.optim.Adam(
        identity_model.parameters(), lr=config["registered_learning_rate"]
    )
    for _step in range(config["steps"]):
        optimizer.zero_grad(set_to_none=True)
        terms = compute_sign_separated_loss_terms(
            identity_model, identity_input, identity_input, loss_args
        )
        terms["loss"].backward()
        optimizer.step()
    with torch.no_grad():
        identity_output = identity_model(identity_input)[0]
    if not torch.equal(identity_output, identity_input):
        raise PreflightError("identity target no longer stays exact")
    return {
        "legacy_cases": legacy_cases,
        "registered_cases": registered_cases,
        "identity_target_exact": True,
    }


def validate_application(plan: dict[str, Any]) -> dict[str, Any]:
    config = plan["candidate_application"]
    parser = build_inference_parser()
    args = parser.parse_args(
        [
            "--samples-file",
            plan["first_quality_gate"]["manifest"],
            "--baseline-pred-dir",
            plan["planned_outputs_must_be_absent"][
                "first_gate_baseline_pipeline"
            ],
            "--checkpoint",
            str(
                Path(plan["trainer"]["output_dir"])
                / "sign_separated_probe.pt"
            ),
            "--output-dir",
            plan["planned_outputs_must_be_absent"]["first_gate_candidate"],
            "--device",
            config["device"],
            "--tile-size",
            str(config["tile_size"]),
            "--stride",
            str(config["stride"]),
            "--edit-probability-threshold",
            str(config["edit_probability_threshold"]),
            "--minimum-delta-threshold",
            str(config["minimum_delta_threshold"]),
        ]
    )
    forbidden = {"--base-edit-threshold", "--second-delta-threshold"}
    if set(parser._option_string_actions) & forbidden:
        raise PreflightError("candidate inference exposes a legacy gate")
    baseline = np.full((4, 4, 3), 128, dtype=np.uint8)
    identity, identity_gate, _ = apply_candidate_gate(
        baseline,
        baseline.copy(),
        np.full((4, 4), 1.0, dtype=np.float32),
        edit_probability_threshold=args.edit_probability_threshold,
        minimum_delta_threshold=args.minimum_delta_threshold,
    )
    if not np.array_equal(identity, baseline) or bool(identity_gate.any()):
        raise PreflightError("identity application is not a no-op")
    candidate = np.full((4, 4, 3), 141, dtype=np.uint8)
    applied, gate, _ = apply_candidate_gate(
        baseline,
        candidate,
        np.full((4, 4), 0.6, dtype=np.float32),
        edit_probability_threshold=args.edit_probability_threshold,
        minimum_delta_threshold=args.minimum_delta_threshold,
    )
    if not bool(gate.all()) or not np.array_equal(applied, candidate):
        raise PreflightError("reachable candidate application was rejected")
    return {
        "baseline_input": config["baseline_input"],
        "edit_probability_threshold": args.edit_probability_threshold,
        "minimum_delta_threshold": args.minimum_delta_threshold,
        "identity_noop": True,
        "reachable_delta_applied": True,
        "legacy_gate_options_absent": True,
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
        plan = validate_plan(repo_root, resolved_plan)
        ledger = read_json(resolved_ledger)
        authority = validate_ledger_authority(repo_root, ledger)
        evidence = validate_evidence(repo_root, plan)
        args = training_args(repo_root, plan)
        if (
            args.lr != plan["trainer"]["lr"]
            or args.max_steps != plan["trainer"]["max_steps"]
            or args.output_dir != Path(plan["trainer"]["output_dir"])
        ):
            raise PreflightError("v2 training CLI changed")
        reachability = synthetic_reachability(plan)
        application = validate_application(plan)
        if not torch.backends.mps.is_available():
            raise PreflightError("registered MPS device is unavailable")
        absent = validate_outputs_absent(repo_root, plan)
    except (KeyError, OSError, PreflightError, TypeError, ValueError) as exc:
        return {
            "terminal": "PREREQUISITE_NEEDED",
            "runnable": False,
            "reason": str(exc),
        }
    return {
        "status": "pass",
        "terminal": "PASS",
        "runnable": True,
        "plan": str(resolved_plan),
        "plan_sha256": sha256_file(resolved_plan),
        "authority": authority,
        "evidence_hashes": evidence,
        "training": {
            "learning_rate": args.lr,
            "steps": args.max_steps,
            "device": args.device,
            "real_training_started": False,
        },
        "synthetic_reachability": reachability,
        "candidate_application": application,
        "mps_available": True,
        "planned_outputs_absent": absent,
        "real_image_decode": False,
        "target_decode": False,
        "checkpoint_generated": False,
        "quality_gate_started": False,
        "later_gates_enabled": False,
        "promotion_enabled": False,
        "reserved_blind_state": "unavailable",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = run_preflight(
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
