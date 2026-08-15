#!/usr/bin/env python3
"""Validate the incremental support-score residual proposal preflight."""

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


PLAN_PATH = Path("docs/external-text-layout-incremental-support-residual-proposal-v1.json")
OUTPUT_PATH = Path(
    "outputs/external-text-layout-incremental-support-residual-preflight-20260815/preflight.json"
)
DIRECT_SUPPORT_DIAGNOSTIC_ID = (
    "external_text_layout_direct_support_residual_reachability_diagnostic"
)
DIRECT_SUPPORT_RECORD_ID = "external-text-layout-direct-support-residual-reachability-diagnostic"
DIRECT_SUPPORT_OUTCOME = "direct_support_projection_reachable_but_preserve_gate_unsafe_before_candidate_surface"
INCREMENTAL_SUPPORT_PREFLIGHT_ID = (
    "external_text_layout_incremental_support_residual_preflight"
)
SUPPORT_PREREQUISITE_ID = "external_text_layout_support_train_only_diagnostic"

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
    "ablation_fit": "support_diagnostic_fold_rgb_ablation_fit",
    "calibration_source": "page_folded_train_patch_incremental_score_centers",
    "delta_bound_gray": 20.4,
    "direction_enforcement": "nonnegative_rgb_only",
    "full_fit": "support_diagnostic_fold_full_fit",
    "input_channels": EXPECTED_INPUT_CHANNELS,
    "normalization": "heldout_fold_preserve_to_positive_incremental_score_span",
    "output_channels": 3,
    "score_formula": "support_full_score_minus_rgb_ablation_score",
    "score_below_preserve_center_delta_gray": 0.0,
    "score_at_or_above_positive_center_delta_gray": 20.4,
    "target_access_at_application": False,
    "training": False,
}
EXPECTED_DIAGNOSTIC = {
    "allowed_roles": ["train"],
    "calibration": "page_folded_train_patch_incremental_preserve_to_positive_span",
    "candidate_inference": False,
    "fit_source": "support_diagnostic_full_and_rgb_ablation_fold_fits_only",
    "gate_threshold_gray": 12.0,
    "maximum_preserve_gate_ratio": 0.005,
    "minimum_ordered_center_folds": 5,
    "minimum_positive_gate_ratio": 0.05,
    "minimum_positive_over_preserve_gate_margin": 0.04,
    "minimum_reachable_patch_ratio": 0.1,
    "output_dir": "outputs/external-text-layout-incremental-support-residual-diagnostic-20260815",
    "required_patch_count": 256,
    "target_access": "train_labels_only_for_page_folded_reachability_measurement",
    "validation_roles_forbidden": [
        "inner_val15",
        "scut115",
        "holdout40",
        "reserved_blind",
    ],
}
EXPECTED_PLANNED_OUTPUTS_MUST_BE_ABSENT = {
    "first_gate_candidate": (
        "outputs/external-text-layout-incremental-support-residual-inner-val15-candidate"
    ),
    "first_gate_score": (
        "outputs/external-text-layout-incremental-support-residual-inner-val15-score"
    ),
    "holdout40_candidate": (
        "outputs/external-text-layout-incremental-support-residual-holdout40-candidate"
    ),
    "incremental_support_diagnostic": (
        "outputs/external-text-layout-incremental-support-residual-diagnostic-20260815"
    ),
    "scut115_candidate": (
        "outputs/external-text-layout-incremental-support-residual-scut115-candidate"
    ),
}
EXPECTED_PREFLIGHT_SUCCESSORS = {
    "KILL": "close_incremental_support_residual_without_score_delta_threshold_or_gate_rescue",
    "PASS": "authorize_train_only_incremental_support_preserve_separation_diagnostic_only",
    "PREREQUISITE_NEEDED": "repair_registered_preflight_evidence_or_metadata_only",
}
EXPECTED_DIAGNOSTIC_SUCCESSORS = {
    "KILL": "close_incremental_support_residual_without_score_delta_threshold_or_gate_rescue",
    "PASS": "authorize_separate_target_free_application_preflight_only",
    "PREREQUISITE_NEEDED": "repair_registered_diagnostic_evidence_or_metadata_only",
}
EXPECTED_FORBIDDEN_RESCUES = [
    "detector_threshold_tuning",
    "layout_transform_tuning",
    "repeat_direct_support_residual_v1",
    "score_normalization_rescue",
    "gate_threshold_rescue",
    "candidate_gate_lowering",
    "candidate_inference_before_train_only_diagnostic",
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
        raise PreflightError("incremental support plan schema changed")
    if (
        plan.get("state")
        != "preregistered_external_text_layout_incremental_support_residual_preflight"
    ):
        raise PreflightError("incremental support plan state changed")
    if plan.get("iteration_id") != ACTIVE_ITERATION_ID:
        raise PreflightError("incremental support iteration changed")
    if plan.get("family") != "external_text_layout_incremental_support_residual_v1":
        raise PreflightError("incremental support family changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise PreflightError("incremental support authorization changed")
    if plan.get("incremental_support_projection") != EXPECTED_PROJECTION:
        raise PreflightError("incremental support projection changed")
    if plan.get("train_only_preserve_separation_diagnostic") != EXPECTED_DIAGNOSTIC:
        raise PreflightError("incremental support diagnostic contract changed")
    if (
        plan.get("planned_outputs_must_be_absent")
        != EXPECTED_PLANNED_OUTPUTS_MUST_BE_ABSENT
    ):
        raise PreflightError("incremental support planned outputs changed")
    if plan.get("preflight_successors") != EXPECTED_PREFLIGHT_SUCCESSORS:
        raise PreflightError("incremental support preflight successors changed")
    if plan.get("diagnostic_successors") != EXPECTED_DIAGNOSTIC_SUCCESSORS:
        raise PreflightError("incremental support diagnostic successors changed")
    if plan.get("forbidden_rescues") != EXPECTED_FORBIDDEN_RESCUES:
        raise PreflightError("incremental support forbidden rescues changed")
    causal = plan.get("causal_change")
    if not isinstance(causal, str) or "full_fit score minus RGB-ablation score" not in causal:
        raise PreflightError("incremental support causal change changed")


def validate_ledger_authority(repo_root: Path, ledger: dict[str, Any]) -> dict[str, str]:
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
        DIRECT_SUPPORT_DIAGNOSTIC_ID: "passed",
        SUPPORT_PREREQUISITE_ID: "passed",
    }
    for prerequisite, expected in required.items():
        if prerequisites.get(prerequisite) != expected:
            raise PreflightError(f"required prerequisite is not {expected}: {prerequisite}")
    incremental_status = prerequisites.get(INCREMENTAL_SUPPORT_PREFLIGHT_ID, "not_started")
    if incremental_status not in {"not_started", "passed"}:
        raise PreflightError("incremental support preflight status changed")

    records = [
        item
        for item in ledger.get("records", [])
        if isinstance(item, dict) and item.get("id") == DIRECT_SUPPORT_RECORD_ID
    ]
    if len(records) != 1:
        raise PreflightError("direct support KILL record count changed")
    record = records[0]
    if record.get("terminal") != "KILL" or record.get("outcome") != DIRECT_SUPPORT_OUTCOME:
        raise PreflightError("direct support KILL authority changed")
    for item in record.get("evidence", []):
        validate_artifact(repo_root, item, "direct support KILL evidence")
    return {
        "direct_support_terminal": "KILL",
        "incremental_support_preflight": incremental_status,
        "support_diagnostic": "passed",
    }


def validate_evidence(repo_root: Path, plan: dict[str, Any]) -> dict[str, str]:
    evidence = plan.get("evidence")
    expected = {
        "direct_support_diagnostic",
        "direct_support_kill_decision",
        "direct_support_plan",
        "registered_patch_index",
        "support_diagnostic",
        "support_diagnostic_decision",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected:
        raise PreflightError("incremental support evidence set changed")
    paths = {
        name: validate_artifact(repo_root, artifact, f"evidence.{name}")
        for name, artifact in evidence.items()
    }
    support = read_json(paths["support_diagnostic"])
    direct = read_json(paths["direct_support_diagnostic"])
    if support.get("terminal") != "PASS" or support.get("acceptance", {}).get("passed") is not True:
        raise PreflightError("support diagnostic no longer passes")
    if direct.get("terminal") != "KILL" or direct.get("candidate_inference_started") is not False:
        raise PreflightError("direct support diagnostic no longer closes candidate inference")
    conditions = direct.get("acceptance", {}).get("conditions", {})
    if conditions.get("preserve_gate_ratio") is not False:
        raise PreflightError("direct support KILL no longer proves preserve failure")
    return {name: sha256_file(path) for name, path in paths.items()}


def incremental_score_to_delta_gray(
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
    projection = plan["incremental_support_projection"]
    scores = np.asarray([-0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    delta = incremental_score_to_delta_gray(
        scores,
        preserve_center=0.0,
        positive_center=1.0,
        delta_bound_gray=projection["delta_bound_gray"],
    )
    if float(delta.min()) < 0.0:
        raise PreflightError("incremental support projection darkened pixels")
    if float(delta.max()) != projection["delta_bound_gray"]:
        raise PreflightError("incremental support projection bound changed")
    if float(delta[1]) != 0.0:
        raise PreflightError("preserve-center score should produce no edit")
    if float(delta[3]) != projection["delta_bound_gray"]:
        raise PreflightError("positive-center score should reach the bound")
    gate = delta >= plan["train_only_preserve_separation_diagnostic"]["gate_threshold_gray"]
    return {
        "delta_bound_gray": float(delta.max()),
        "delta_min_gray": float(delta.min()),
        "gate_count": int(gate.sum()),
        "nonnegative": True,
        "score_count": int(len(scores)),
        "score_formula": projection["score_formula"],
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
