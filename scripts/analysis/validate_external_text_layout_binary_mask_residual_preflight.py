#!/usr/bin/env python3
"""Validate the binary external text-layout mask residual proposal preflight."""

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


PLAN_PATH = Path("docs/external-text-layout-binary-mask-residual-proposal-v1.json")
OUTPUT_PATH = Path(
    "outputs/external-text-layout-binary-mask-residual-preflight-20260815/preflight.json"
)
SUPPORT_PREREQUISITE_ID = "external_text_layout_support_train_only_diagnostic"
INCREMENTAL_DIAGNOSTIC_ID = (
    "external_text_layout_incremental_support_residual_reachability_diagnostic"
)
INCREMENTAL_RECORD_ID = "external-text-layout-incremental-support-residual-reachability-diagnostic"
INCREMENTAL_OUTCOME = (
    "incremental_support_projection_reachable_but_not_preserve_separating_before_candidate_surface"
)
BINARY_MASK_PREFLIGHT_ID = "external_text_layout_binary_mask_residual_preflight"

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
EXPECTED_PROJECTION = {
    "calibration_source": "none_binary_mask",
    "delta_bound_gray": 20.4,
    "direction_enforcement": "nonnegative_rgb_only",
    "input_channels": ["external_text_occupancy"],
    "mask_formula": "external_text_occupancy_equals_one",
    "output_channels": 3,
    "target_access_at_application": False,
    "training": False,
}
EXPECTED_DIAGNOSTIC = {
    "allowed_roles": ["train"],
    "candidate_inference": False,
    "gate_threshold_gray": 12.0,
    "maximum_preserve_gate_ratio": 0.005,
    "minimum_positive_gate_ratio": 0.05,
    "minimum_positive_over_preserve_gate_margin": 0.04,
    "minimum_reachable_patch_ratio": 0.1,
    "output_dir": "outputs/external-text-layout-binary-mask-residual-diagnostic-20260815",
    "required_patch_count": 256,
    "target_access": "train_labels_only_for_binary_mask_reachability_measurement",
    "validation_roles_forbidden": [
        "inner_val15",
        "scut115",
        "holdout40",
        "reserved_blind",
    ],
}
EXPECTED_PLANNED_OUTPUTS_MUST_BE_ABSENT = {
    "binary_mask_diagnostic": (
        "outputs/external-text-layout-binary-mask-residual-diagnostic-20260815"
    ),
    "first_gate_candidate": (
        "outputs/external-text-layout-binary-mask-residual-inner-val15-candidate"
    ),
    "first_gate_score": "outputs/external-text-layout-binary-mask-residual-inner-val15-score",
    "holdout40_candidate": "outputs/external-text-layout-binary-mask-residual-holdout40-candidate",
    "scut115_candidate": "outputs/external-text-layout-binary-mask-residual-scut115-candidate",
}
EXPECTED_PREFLIGHT_SUCCESSORS = {
    "KILL": "close_binary_mask_residual_without_threshold_or_confidence_rescue",
    "PASS": "authorize_train_only_binary_mask_preserve_separation_diagnostic_only",
    "PREREQUISITE_NEEDED": "repair_registered_preflight_evidence_or_metadata_only",
}
EXPECTED_FORBIDDEN_RESCUES = [
    "detector_threshold_tuning",
    "layout_transform_tuning",
    "repeat_direct_support_residual_v1",
    "repeat_incremental_support_residual_v1",
    "binary_mask_threshold_rescue",
    "confidence_threshold_sweep",
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
        raise PreflightError("binary mask plan schema changed")
    if plan.get("state") != "preregistered_external_text_layout_binary_mask_residual_preflight":
        raise PreflightError("binary mask plan state changed")
    if plan.get("iteration_id") != ACTIVE_ITERATION_ID:
        raise PreflightError("binary mask iteration changed")
    if plan.get("family") != "external_text_layout_binary_mask_residual_v1":
        raise PreflightError("binary mask family changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise PreflightError("binary mask authorization changed")
    if plan.get("binary_mask_projection") != EXPECTED_PROJECTION:
        raise PreflightError("binary mask projection changed")
    if plan.get("train_only_preserve_separation_diagnostic") != EXPECTED_DIAGNOSTIC:
        raise PreflightError("binary mask diagnostic contract changed")
    if plan.get("planned_outputs_must_be_absent") != EXPECTED_PLANNED_OUTPUTS_MUST_BE_ABSENT:
        raise PreflightError("binary mask planned outputs changed")
    if plan.get("preflight_successors") != EXPECTED_PREFLIGHT_SUCCESSORS:
        raise PreflightError("binary mask preflight successors changed")
    if plan.get("forbidden_rescues") != EXPECTED_FORBIDDEN_RESCUES:
        raise PreflightError("binary mask forbidden rescues changed")
    causal = plan.get("causal_change")
    if not isinstance(causal, str) or "binary external-text occupancy mask" not in causal:
        raise PreflightError("binary mask causal change changed")


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
    for prerequisite in (SUPPORT_PREREQUISITE_ID, INCREMENTAL_DIAGNOSTIC_ID):
        if prerequisites.get(prerequisite) != "passed":
            raise PreflightError(f"required prerequisite is not passed: {prerequisite}")
    binary_status = prerequisites.get(BINARY_MASK_PREFLIGHT_ID, "not_started")
    if binary_status not in {"not_started", "passed"}:
        raise PreflightError("binary mask preflight status changed")

    records = [
        item
        for item in ledger.get("records", [])
        if isinstance(item, dict) and item.get("id") == INCREMENTAL_RECORD_ID
    ]
    if len(records) != 1:
        raise PreflightError("incremental support KILL record count changed")
    record = records[0]
    if record.get("terminal") != "KILL" or record.get("outcome") != INCREMENTAL_OUTCOME:
        raise PreflightError("incremental support KILL authority changed")
    for item in record.get("evidence", []):
        validate_artifact(repo_root, item, "incremental support KILL evidence")
    return {
        "binary_mask_preflight": binary_status,
        "incremental_support_terminal": "KILL",
        "support_diagnostic": "passed",
    }


def validate_evidence(repo_root: Path, plan: dict[str, Any]) -> dict[str, str]:
    evidence = plan.get("evidence")
    expected = {
        "incremental_support_diagnostic",
        "incremental_support_kill_decision",
        "registered_patch_index",
        "support_diagnostic",
        "support_diagnostic_decision",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected:
        raise PreflightError("binary mask evidence set changed")
    paths = {
        name: validate_artifact(repo_root, artifact, f"evidence.{name}")
        for name, artifact in evidence.items()
    }
    support = read_json(paths["support_diagnostic"])
    incremental = read_json(paths["incremental_support_diagnostic"])
    if support.get("terminal") != "PASS" or support.get("acceptance", {}).get("passed") is not True:
        raise PreflightError("support diagnostic no longer passes")
    if incremental.get("terminal") != "KILL" or incremental.get("candidate_inference_started") is not False:
        raise PreflightError("incremental support diagnostic no longer closes candidate inference")
    return {name: sha256_file(path) for name, path in paths.items()}


def binary_mask_to_delta_gray(mask: np.ndarray, *, delta_bound_gray: float) -> np.ndarray:
    if delta_bound_gray <= 0.0:
        raise ValueError("delta bound must be positive")
    values = np.asarray(mask)
    if values.dtype != np.bool_:
        raise ValueError("binary mask must be boolean")
    return values.astype(np.float64) * delta_bound_gray


def run_synthetic_projection(plan: dict[str, Any]) -> dict[str, Any]:
    mask = np.asarray([False, True, False, True], dtype=np.bool_)
    delta = binary_mask_to_delta_gray(
        mask,
        delta_bound_gray=plan["binary_mask_projection"]["delta_bound_gray"],
    )
    if float(delta.min()) < 0.0:
        raise PreflightError("binary mask projection darkened pixels")
    gate = delta >= plan["train_only_preserve_separation_diagnostic"]["gate_threshold_gray"]
    return {
        "delta_bound_gray": float(delta.max()),
        "delta_min_gray": float(delta.min()),
        "gate_count": int(gate.sum()),
        "mask_count": int(mask.sum()),
        "nonnegative": True,
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
