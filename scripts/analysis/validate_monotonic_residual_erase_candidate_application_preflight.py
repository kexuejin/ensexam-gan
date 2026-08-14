#!/usr/bin/env python3
"""Validate monotonic training reachability and candidate application."""

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

from scripts.analysis.validate_monotonic_residual_erase_training_preflight import (  # noqa: E402
    LEDGER_PATH,
    PreflightError,
    read_json,
    repo_path,
    sha256_file,
    validate_artifact,
)
from scripts.infer.monotonic_residual_erase import (  # noqa: E402
    MonotonicResidualEraseCleanupNet,
)
from scripts.infer.run_monotonic_residual_erase_candidate import (  # noqa: E402
    apply_monotonic_candidate_gate,
    build_parser as build_inference_parser,
)
from scripts.train.train_monotonic_residual_erase import (  # noqa: E402
    build_parser as build_training_parser,
    compute_monotonic_loss_terms,
)


PLAN_PATH = Path("docs/monotonic-residual-erase-candidate-plan-v2.json")
PLAN_SHA256 = "a5089e8a2a0877ef9e34966f7868d82efeeb24d5057d0e5d594c8eda0b1dfc56"
ACTIVE_ITERATION_ID = "monotonic-residual-erase-support"
MATERIALIZATION_PREREQUISITE_ID = (
    "monotonic_residual_erase_train_materialization_audit"
)
MATERIALIZATION_RECORD_ID = (
    "monotonic-residual-erase-train275-materialization-audit"
)
MATERIALIZATION_OUTCOME = (
    "exact_train275_frozen_pipeline_and_brighten_only_patch_materialization_passed"
)
APPLICATION_PREREQUISITE_ID = (
    "monotonic_residual_erase_candidate_application_preflight"
)
APPLICATION_RECORD_ID = (
    "monotonic-residual-erase-candidate-application-preflight"
)
APPLICATION_OUTCOME = (
    "v1_unreachable_learning_and_legacy_gate_rejected_v2_monotonic_application_passed"
)
CHECKPOINT_RECORD_ID = "monotonic-residual-erase-v2-checkpoint"
CHECKPOINT_KILL_OUTCOME = "real_train_patch_gate_collapsed_to_subthreshold_noop"


EXPECTED_AUTHORIZATION = {
    "current_primary_replacement": "disabled",
    "quality_gates": "disabled_until_checkpoint_audit_pass",
    "reserved_blind": "unavailable",
    "state": "candidate_application_preflight_pending",
    "training": "prohibited_until_candidate_application_preflight_pass",
}
EXPECTED_MATERIALIZED_INPUTS = {
    "frozen_pipeline_prediction_content_sha256": (
        "2ffa40fc0c9b2a7e721d560f6f12edfe2ccdc1c1988582fa7a8104665cdc088a"
    ),
    "input_dir": (
        "outputs/monotonic-residual-erase-train275-frozen-pipeline-v1/pred"
    ),
    "label_content_sha256": (
        "dfd459f552bd0828221c90258f33f4eacc54220494c7e02b21a179894853e99e"
    ),
    "patch_count": 256,
    "patch_index": "hardcase_lists/monotonic-residual-erase-train-patches-v1.csv",
    "patch_index_sha256": (
        "2503616f2d94fd5bfd65be4ad61c7c53af8726dc4f2307745ef1da6f74033943"
    ),
    "train_count": 275,
    "train_filename_sha256": (
        "e9ac4d6f700f41ef3a9b7c3f04ce0593f593324a881a0f9fc387901a497f9039"
    ),
}


def validate_plan(plan_path: Path) -> dict[str, Any]:
    if sha256_file(plan_path) != PLAN_SHA256:
        raise PreflightError("monotonic candidate v2 plan hash changed")
    plan = read_json(plan_path)
    if plan.get("schema_version") != 1:
        raise PreflightError("monotonic candidate v2 schema changed")
    if plan.get("iteration_id") != "monotonic-residual-erase-support-v2":
        raise PreflightError("monotonic candidate v2 iteration changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise PreflightError("monotonic candidate v2 authorization changed")
    if plan.get("materialized_training_inputs") != EXPECTED_MATERIALIZED_INPUTS:
        raise PreflightError("monotonic materialized training inputs changed")
    if plan.get("model") != {
        "direction": "preserve_or_brighten_only",
        "model_type": "monotonic_residual_erase",
        "residual_delta_bound": 0.08,
    }:
        raise PreflightError("monotonic candidate model contract changed")

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
        raise PreflightError("monotonic candidate v2 causal parameters changed")
    if application.get("baseline_input") != (
        "frozen_current_primary_plus_current_second_stage_prediction"
    ):
        raise PreflightError("monotonic candidate baseline input changed")
    if application.get("direction_enforcement") != (
        "candidate_must_not_darken_any_channel"
    ):
        raise PreflightError("monotonic application direction guard changed")
    if application.get("minimum_delta_threshold") > (
        trainer["residual_delta_bound"] * 255.0
    ):
        raise PreflightError("monotonic candidate delta gate is unreachable")
    if trainer.get("output_dir") != (
        "artifacts/trials/monotonic-residual-erase-v2"
    ):
        raise PreflightError("monotonic v2 training output changed")
    if trainer.get("validation_enabled") is not False:
        raise PreflightError("monotonic trainer opened validation")
    return plan


def validate_evidence(
    repo_root: Path,
    plan: dict[str, Any],
) -> dict[str, str]:
    evidence = plan.get("evidence")
    expected_names = {
        "base_training_plan",
        "candidate_inference",
        "materialization_audit",
        "materialization_decision",
        "model",
        "trainer",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected_names:
        raise PreflightError("monotonic candidate v2 evidence set changed")
    paths = {
        name: validate_artifact(repo_root, artifact, f"evidence.{name}")
        for name, artifact in evidence.items()
    }
    materialization = read_json(paths["materialization_audit"])
    patch_summary = materialization.get("patch_summary", {})
    if (
        materialization.get("terminal") != "PASS"
        or materialization.get("train_count") != 275
        or patch_summary.get("patch_count") != 256
        or patch_summary.get("selection") != "top_target_lighter_support_only"
        or materialization.get("training_started") is not False
        or materialization.get("first_quality_gate_started") is not False
    ):
        raise PreflightError("monotonic materialization evidence changed")
    return {name: sha256_file(path) for name, path in paths.items()}


def validate_ledger_authority(
    repo_root: Path,
    ledger: dict[str, Any],
) -> dict[str, str]:
    active = ledger.get("active_iteration", {})
    if active.get("id") != ACTIVE_ITERATION_ID:
        raise PreflightError("active iteration is not monotonic residual erase")
    prerequisites = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if prerequisites.get(MATERIALIZATION_PREREQUISITE_ID) != "passed":
        raise PreflightError("monotonic materialization prerequisite is not passed")
    application_status = prerequisites.get(APPLICATION_PREREQUISITE_ID)
    if application_status not in {"pending", "passed"}:
        raise PreflightError(
            "monotonic candidate application prerequisite is not registered"
        )

    records = [
        record
        for record in ledger.get("records", [])
        if isinstance(record, dict)
        and record.get("id") == MATERIALIZATION_RECORD_ID
    ]
    if len(records) != 1:
        raise PreflightError("exactly one monotonic materialization record is required")
    record = records[0]
    if (
        record.get("terminal") != "PASS"
        or record.get("outcome") != MATERIALIZATION_OUTCOME
    ):
        raise PreflightError("monotonic materialization record has wrong authority")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise PreflightError("monotonic materialization record lacks evidence")
    for item in evidence:
        validate_artifact(repo_root, item, "monotonic materialization evidence")

    if application_status == "passed":
        application_records = [
            item
            for item in ledger.get("records", [])
            if isinstance(item, dict)
            and item.get("id") == APPLICATION_RECORD_ID
        ]
        if len(application_records) != 1:
            raise PreflightError(
                "exactly one monotonic application PASS record is required"
            )
        application_record = application_records[0]
        if (
            application_record.get("terminal") != "PASS"
            or application_record.get("outcome") != APPLICATION_OUTCOME
        ):
            raise PreflightError(
                "monotonic candidate application record has wrong authority"
            )
        application_evidence = application_record.get("evidence")
        if not isinstance(application_evidence, list) or not application_evidence:
            raise PreflightError(
                "monotonic candidate application PASS record lacks evidence"
            )
        for item in application_evidence:
            validate_artifact(
                repo_root, item, "monotonic candidate application evidence"
            )
    return {
        "materialization": "passed",
        "candidate_application": application_status,
    }


def validate_checkpoint_kill_authority(
    repo_root: Path,
    ledger: dict[str, Any],
) -> bool:
    records = [
        item
        for item in ledger.get("records", [])
        if isinstance(item, dict) and item.get("id") == CHECKPOINT_RECORD_ID
    ]
    if not records:
        return False
    if len(records) != 1:
        raise PreflightError("monotonic checkpoint KILL record count changed")
    record = records[0]
    if (
        record.get("terminal") != "KILL"
        or record.get("outcome") != CHECKPOINT_KILL_OUTCOME
    ):
        raise PreflightError("monotonic checkpoint record has wrong authority")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise PreflightError("monotonic checkpoint KILL record lacks evidence")
    for item in evidence:
        validate_artifact(repo_root, item, "monotonic checkpoint KILL evidence")
    return True


def training_args(repo_root: Path, plan: dict[str, Any]) -> argparse.Namespace:
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
            "--luminance-margin-gray",
            str(trainer["luminance_margin_gray"]),
            "--support-positive-weight",
            str(trainer["support_positive_weight"]),
            "--support-preserve-weight",
            str(trainer["support_preserve_weight"]),
            "--magnitude-weight",
            str(trainer["magnitude_weight"]),
            "--preserve-delta-weight",
            str(trainer["preserve_delta_weight"]),
        ]
    )


def loss_args(plan: dict[str, Any]) -> argparse.Namespace:
    trainer = plan["trainer"]
    return argparse.Namespace(
        luminance_margin_gray=trainer["luminance_margin_gray"],
        support_positive_weight=trainer["support_positive_weight"],
        support_preserve_weight=trainer["support_preserve_weight"],
        magnitude_weight=trainer["magnitude_weight"],
        preserve_delta_weight=trainer["preserve_delta_weight"],
    )


def run_synthetic_case(
    *,
    learning_rate: float,
    steps: int,
    target_delta: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    torch.manual_seed(42)
    model = MonotonicResidualEraseCleanupNet(0.08)
    inp = torch.full((1, 3, 8, 8), 0.5)
    target = inp + target_delta
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        terms = compute_monotonic_loss_terms(model, inp, target, args)
        if not bool(torch.isfinite(terms["loss"])):
            raise PreflightError("monotonic synthetic loss is non-finite")
        terms["loss"].backward()
        optimizer.step()
    with torch.no_grad():
        components = model.forward_components(inp)
        delta = components["signed_delta"] * 255.0
    return {
        "delta_max_gray": float(delta.max()),
        "delta_mean_gray": float(delta.mean()),
        "edit_probability_mean": float(components["edit_alpha"].mean()),
        "negative_delta_pixel_count": int((delta < -1e-7).sum()),
        "loss": float(terms["loss"].detach()),
    }


def run_preserve_case(
    *,
    steps: int,
    target_delta: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    torch.manual_seed(42)
    model = MonotonicResidualEraseCleanupNet(0.08)
    inp = torch.full((1, 3, 8, 8), 0.5)
    target = inp + target_delta
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        terms = compute_monotonic_loss_terms(model, inp, target, args)
        terms["loss"].backward()
        optimizer.step()
    with torch.no_grad():
        output = model(inp)[0]
    return {
        "target_delta": target_delta,
        "exact_identity_output": bool(torch.equal(output, inp)),
        "delta_max_gray": float(((output - inp) * 255.0).abs().max()),
    }


def synthetic_reachability(plan: dict[str, Any]) -> dict[str, Any]:
    config = plan["synthetic_reachability"]
    args = loss_args(plan)
    legacy = run_synthetic_case(
        learning_rate=config["legacy_learning_rate"],
        steps=config["steps"],
        target_delta=config["target_delta"],
        args=args,
    )
    registered = run_synthetic_case(
        learning_rate=config["registered_learning_rate"],
        steps=config["steps"],
        target_delta=config["target_delta"],
        args=args,
    )
    threshold = config["minimum_delta_threshold"]
    if legacy["delta_max_gray"] >= threshold:
        raise PreflightError("legacy monotonic learning rate unexpectedly reached gate")
    if registered["delta_max_gray"] < threshold:
        raise PreflightError("registered monotonic learning rate cannot reach gate")
    if registered["delta_max_gray"] > 20.4 + 1e-5:
        raise PreflightError("registered monotonic output exceeded bound")
    if registered["negative_delta_pixel_count"] != 0:
        raise PreflightError("registered monotonic output darkened pixels")
    if registered["edit_probability_mean"] < 0.5:
        raise PreflightError("registered monotonic support cannot reach gate")

    preserve_cases = [
        run_preserve_case(steps=config["steps"], target_delta=value, args=args)
        for value in (0.0, -0.08)
    ]
    if not all(case["exact_identity_output"] for case in preserve_cases):
        raise PreflightError("monotonic preserve target no longer stays exact")
    return {
        "legacy_case": legacy,
        "registered_case": registered,
        "preserve_cases": preserve_cases,
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
                / "monotonic_residual_erase_probe.pt"
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
    forbidden = {
        "--base-edit-threshold",
        "--second-delta-threshold",
        "--label-dir",
        "--target-dir",
        "--route-override",
    }
    if set(parser._option_string_actions) & forbidden:
        raise PreflightError("monotonic candidate inference exposes forbidden input")

    baseline = np.full((4, 4, 3), 128, dtype=np.uint8)
    identity, identity_gate, _ = apply_monotonic_candidate_gate(
        baseline,
        baseline.copy(),
        np.full((4, 4), 1.0, dtype=np.float32),
        edit_probability_threshold=args.edit_probability_threshold,
        minimum_delta_threshold=args.minimum_delta_threshold,
    )
    if not np.array_equal(identity, baseline) or bool(identity_gate.any()):
        raise PreflightError("monotonic identity application is not a no-op")

    candidate = np.full((4, 4, 3), 141, dtype=np.uint8)
    applied, gate, _ = apply_monotonic_candidate_gate(
        baseline,
        candidate,
        np.full((4, 4), 0.6, dtype=np.float32),
        edit_probability_threshold=args.edit_probability_threshold,
        minimum_delta_threshold=args.minimum_delta_threshold,
    )
    if not bool(gate.all()) or not np.array_equal(applied, candidate):
        raise PreflightError("reachable monotonic candidate application was rejected")
    try:
        apply_monotonic_candidate_gate(
            baseline,
            np.full((4, 4, 3), 127, dtype=np.uint8),
            np.full((4, 4), 1.0, dtype=np.float32),
            edit_probability_threshold=args.edit_probability_threshold,
            minimum_delta_threshold=args.minimum_delta_threshold,
        )
    except ValueError as exc:
        if "darkened" not in str(exc):
            raise
    else:
        raise PreflightError("monotonic application accepted a darker candidate")
    return {
        "baseline_input": config["baseline_input"],
        "edit_probability_threshold": args.edit_probability_threshold,
        "minimum_delta_threshold": args.minimum_delta_threshold,
        "identity_noop": True,
        "reachable_brighten_applied": True,
        "darker_candidate_rejected": True,
        "target_or_route_options_absent": True,
    }


def validate_outputs_absent(
    repo_root: Path,
    plan: dict[str, Any],
    *,
    allow_checkpoint_kill_outputs: bool = False,
) -> list[str]:
    absent: list[str] = []
    for label, value in sorted(plan["planned_outputs_must_be_absent"].items()):
        if allow_checkpoint_kill_outputs and label in {
            "checkpoint_audit",
            "training_output_dir",
        }:
            continue
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
        plan = validate_plan(resolved_plan)
        ledger = read_json(resolved_ledger)
        authority = validate_ledger_authority(repo_root, ledger)
        checkpoint_killed = validate_checkpoint_kill_authority(repo_root, ledger)
        evidence = validate_evidence(repo_root, plan)
        args = training_args(repo_root, plan)
        if (
            args.lr != plan["trainer"]["lr"]
            or args.max_steps != plan["trainer"]["max_steps"]
            or args.output_dir != Path(plan["trainer"]["output_dir"])
        ):
            raise PreflightError("monotonic v2 training CLI changed")
        reachability = synthetic_reachability(plan)
        application = validate_application(plan)
        if not torch.backends.mps.is_available():
            raise PreflightError("registered MPS device is unavailable")
        absent = validate_outputs_absent(
            repo_root,
            plan,
            allow_checkpoint_kill_outputs=checkpoint_killed,
        )
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
        "checkpoint_killed": checkpoint_killed,
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
