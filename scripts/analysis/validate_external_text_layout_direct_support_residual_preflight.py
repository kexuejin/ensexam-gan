#!/usr/bin/env python3
"""Validate the direct support-score residual proposal preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.validate_external_text_layout_conditioned_preflight import (  # noqa: E402
    ACTIVE_ITERATION_ID,
    LEDGER_PATH,
    PreflightError,
    read_json,
    repo_path,
    validate_artifact,
)


PLAN_PATH = Path("docs/external-text-layout-direct-support-residual-proposal-v1.json")
OUTPUT_PATH = Path(
    "outputs/external-text-layout-direct-support-residual-preflight-20260815/preflight.json"
)
SUPPORT_PREREQUISITE_ID = "external_text_layout_support_train_only_diagnostic"
CONDITIONED_CHECKPOINT_AUDIT_ID = (
    "external_text_layout_conditioned_monotonic_checkpoint_audit"
)
DIRECT_SUPPORT_PREFLIGHT_ID = (
    "external_text_layout_direct_support_residual_preflight"
)
CHECKPOINT_KILL_RECORD_ID = "external-text-layout-conditioned-monotonic-checkpoint-audit"
CHECKPOINT_KILL_OUTCOME = "conditioned_checkpoint_subthreshold_noop_before_candidate_gate"

EXPECTED_AUTHORIZATION = {
    "candidate_inference": False,
    "checkpoint_generation": False,
    "holdout40": False,
    "inner_val15": False,
    "model_training": False,
    "promotion": False,
    "reserved_blind": False,
    "scut115": False,
    "state": "preflight_only",
    "visual_review": False,
}
EXPECTED_INPUT_CHANNELS = [
    "second_stage_r",
    "second_stage_g",
    "second_stage_b",
    "external_text_occupancy",
    "external_text_confidence",
]
EXPECTED_PROJECTION = {
    "calibration_source": "external_text_layout_support_train_only_diagnostic_full_fit",
    "delta_bound_gray": 20.4,
    "direction_enforcement": "nonnegative_rgb_only",
    "input_channels": EXPECTED_INPUT_CHANNELS,
    "normalization": "fold_fit_preserve_to_positive_score_span",
    "output_channels": 3,
    "score_below_preserve_center_delta_gray": 0.0,
    "score_at_or_above_positive_center_delta_gray": 20.4,
    "target_access_at_application": False,
    "training": False,
}
EXPECTED_DIAGNOSTIC = {
    "allowed_roles": ["train"],
    "candidate_inference": False,
    "fit_source": "support_diagnostic_fold_fits_only",
    "gate_threshold_gray": 12.0,
    "maximum_preserve_gate_ratio": 0.005,
    "minimum_positive_gate_ratio": 0.05,
    "minimum_positive_over_preserve_gate_margin": 0.04,
    "minimum_reachable_patch_ratio": 0.1,
    "output_dir": (
        "outputs/external-text-layout-direct-support-residual-proposal-diagnostic-20260815"
    ),
    "required_patch_count": 256,
    "target_access": "train_labels_only_for_reachability_measurement",
    "validation_roles_forbidden": [
        "inner_val15",
        "scut115",
        "holdout40",
        "reserved_blind",
    ],
}
EXPECTED_PLANNED_OUTPUTS_MUST_BE_ABSENT = {
    "direct_support_diagnostic": (
        "outputs/external-text-layout-direct-support-residual-proposal-diagnostic-20260815"
    ),
    "first_gate_candidate": (
        "outputs/external-text-layout-direct-support-residual-inner-val15-candidate"
    ),
    "first_gate_score": (
        "outputs/external-text-layout-direct-support-residual-inner-val15-score"
    ),
    "holdout40_candidate": (
        "outputs/external-text-layout-direct-support-residual-holdout40-candidate"
    ),
    "scut115_candidate": (
        "outputs/external-text-layout-direct-support-residual-scut115-candidate"
    ),
}
EXPECTED_TERMINAL_SUCCESSORS = {
    "KILL": "close_direct_support_residual_without_score_delta_threshold_or_layout_rescue",
    "PASS": "authorize_train_only_direct_support_reachability_diagnostic_only",
    "PREREQUISITE_NEEDED": "repair_registered_preflight_evidence_or_metadata_only",
}
EXPECTED_FORBIDDEN_RESCUES = [
    "detector_threshold_tuning",
    "layout_transform_tuning",
    "repeat_conditioned_monotonic_training",
    "learning_rate_or_step_sweep",
    "loss_weight_sweep",
    "candidate_gate_lowering",
    "scut115_before_inner_val15",
    "holdout40_before_inner_val15",
    "reserved_blind_before_development_gates",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_exact_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != 1:
        raise PreflightError("direct support plan schema changed")
    if (
        plan.get("state")
        != "preregistered_external_text_layout_direct_support_residual_preflight"
    ):
        raise PreflightError("direct support plan state changed")
    if plan.get("iteration_id") != ACTIVE_ITERATION_ID:
        raise PreflightError("direct support iteration changed")
    if plan.get("family") != "external_text_layout_direct_support_residual_v1":
        raise PreflightError("direct support family changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise PreflightError("direct support authorization changed")
    if plan.get("direct_support_projection") != EXPECTED_PROJECTION:
        raise PreflightError("direct support projection changed")
    if plan.get("train_only_reachability_diagnostic") != EXPECTED_DIAGNOSTIC:
        raise PreflightError("direct support diagnostic contract changed")
    if (
        plan.get("planned_outputs_must_be_absent")
        != EXPECTED_PLANNED_OUTPUTS_MUST_BE_ABSENT
    ):
        raise PreflightError("direct support planned outputs changed")
    if plan.get("terminal_successors") != EXPECTED_TERMINAL_SUCCESSORS:
        raise PreflightError("direct support terminal successors changed")
    if plan.get("forbidden_rescues") != EXPECTED_FORBIDDEN_RESCUES:
        raise PreflightError("direct support forbidden rescues changed")
    causal = plan.get("causal_change")
    if not isinstance(causal, str) or "closed-form support-score" not in causal:
        raise PreflightError("direct support causal change changed")


def validate_ledger_authority(
    repo_root: Path,
    ledger: dict[str, Any],
) -> dict[str, str]:
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
    required = {
        SUPPORT_PREREQUISITE_ID: "passed",
        CONDITIONED_CHECKPOINT_AUDIT_ID: "passed",
    }
    for prerequisite, expected in required.items():
        if prerequisites.get(prerequisite) != expected:
            raise PreflightError(f"required prerequisite is not {expected}: {prerequisite}")
    direct_status = prerequisites.get(DIRECT_SUPPORT_PREFLIGHT_ID, "not_started")
    if direct_status not in {"not_started", "passed"}:
        raise PreflightError("direct support preflight status changed")

    records = [
        item
        for item in ledger.get("records", [])
        if isinstance(item, dict) and item.get("id") == CHECKPOINT_KILL_RECORD_ID
    ]
    if len(records) != 1:
        raise PreflightError("conditioned checkpoint KILL record count changed")
    record = records[0]
    if record.get("terminal") != "KILL" or record.get("outcome") != CHECKPOINT_KILL_OUTCOME:
        raise PreflightError("conditioned checkpoint KILL authority changed")
    for item in record.get("evidence", []):
        if not str(item.get("path", "")).startswith("artifacts/trials/"):
            validate_artifact(repo_root, item, "conditioned checkpoint KILL evidence")
    return {
        "conditioned_checkpoint_audit": "passed",
        "conditioned_checkpoint_terminal": "KILL",
        "direct_support_preflight": direct_status,
        "support_diagnostic": "passed",
    }


def validate_evidence(repo_root: Path, plan: dict[str, Any]) -> dict[str, str]:
    evidence = plan.get("evidence")
    expected = {
        "conditioned_checkpoint_audit",
        "conditioned_checkpoint_decision",
        "conditioned_preflight_plan",
        "support_diagnostic",
        "support_diagnostic_decision",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected:
        raise PreflightError("direct support evidence set changed")
    paths = {
        name: validate_artifact(repo_root, artifact, f"evidence.{name}")
        for name, artifact in evidence.items()
    }
    support = read_json(paths["support_diagnostic"])
    checkpoint = read_json(paths["conditioned_checkpoint_audit"])
    if support.get("terminal") != "PASS" or support.get("acceptance", {}).get("passed") is not True:
        raise PreflightError("support diagnostic no longer passes")
    if checkpoint.get("terminal") != "KILL" or checkpoint.get("candidate_inference_started") is not False:
        raise PreflightError("conditioned checkpoint audit no longer closes candidate inference")
    patch_summary = checkpoint.get("patch_summary", {})
    if (
        patch_summary.get("positive_gate_ratio") != 0.0
        or patch_summary.get("preserve_gate_ratio") != 0.0
        or patch_summary.get("patches_with_gate") != 0
    ):
        raise PreflightError("conditioned checkpoint no-op evidence changed")
    return {name: sha256_file(path) for name, path in paths.items()}


def score_to_delta_gray(
    scores: np.ndarray,
    *,
    preserve_center: float,
    positive_center: float,
    delta_bound_gray: float,
) -> np.ndarray:
    if positive_center <= preserve_center:
        raise ValueError("positive score center must exceed preserve center")
    if delta_bound_gray <= 0.0:
        raise ValueError("delta bound must be positive")
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("scores contain non-finite values")
    normalized = (values - preserve_center) / (positive_center - preserve_center)
    return np.clip(normalized, 0.0, 1.0) * delta_bound_gray


def run_synthetic_projection(plan: dict[str, Any]) -> dict[str, Any]:
    projection = plan["direct_support_projection"]
    scores = np.asarray([-0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    delta = score_to_delta_gray(
        scores,
        preserve_center=0.0,
        positive_center=1.0,
        delta_bound_gray=projection["delta_bound_gray"],
    )
    if float(delta.min()) < 0.0:
        raise PreflightError("direct support projection darkened pixels")
    if float(delta.max()) != projection["delta_bound_gray"]:
        raise PreflightError("direct support projection bound changed")
    if float(delta[1]) != 0.0:
        raise PreflightError("preserve-center score should produce no edit")
    if float(delta[3]) != projection["delta_bound_gray"]:
        raise PreflightError("positive-center score should reach the bound")
    gate = delta >= plan["train_only_reachability_diagnostic"]["gate_threshold_gray"]
    return {
        "delta_bound_gray": float(delta.max()),
        "delta_min_gray": float(delta.min()),
        "gate_count": int(gate.sum()),
        "nonnegative": True,
        "score_count": int(len(scores)),
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
        authority = validate_ledger_authority(repo_root, ledger)
        evidence_hashes = validate_evidence(repo_root, plan)
        synthetic = run_synthetic_projection(plan)
        absent = validate_outputs_absent(repo_root, plan)
    except (KeyError, OSError, PreflightError, TypeError, ValueError) as error:
        return {
            "reason": str(error),
            "runnable": False,
            "terminal": "PREREQUISITE_NEEDED",
        }
    return {
        "authority": authority,
        "candidate_inference_started": False,
        "checkpoint_generated": False,
        "evidence_hashes": evidence_hashes,
        "model_training_started": False,
        "planned_outputs_absent": absent,
        "plan": str(resolved_plan),
        "plan_sha256": sha256_file(resolved_plan),
        "promotion_enabled": False,
        "quality_gate_started": False,
        "reserved_blind_state": "unavailable",
        "runnable": True,
        "schema_version": 1,
        "synthetic_projection": synthetic,
        "target_decode": False,
        "terminal": "PASS",
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
        path = args.output_json
        if not path.is_absolute():
            path = args.repo_root.resolve() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["terminal"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
