#!/usr/bin/env python3
"""Validate monotonic residual-erase roles without decoding pixels."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.validate_sign_separated_data_roles import (  # noqa: E402
    FORBIDDEN_PIXEL_MODULES,
    ROLE_ORDER,
    PreflightError,
    derive_effective_roles,
    read_json,
    repo_path,
    sha256_file,
    validate_artifact,
    validate_baseline,
    validate_reserved_blind,
    validate_role_sources,
    validate_zero_overlap,
)


PLAN_PATH = Path("docs/monotonic-residual-erase-data-roles.json")
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
MODEL_TYPE = "monotonic_residual_erase"
ACTIVE_ITERATION_ID = "monotonic-residual-erase-support"
SYNTHETIC_RECORD_ID = "monotonic-residual-erase-synthetic-prerequisite"
SYNTHETIC_OUTCOME = (
    "identity_initialized_nonnegative_bounded_erase_contract_passed"
)
DATA_ROLE_PREREQUISITE_ID = "monotonic_residual_erase_data_role_preflight"


EXPECTED_AUTHORIZATION = {
    "checkpoint_generation": "prohibited",
    "current_primary_replacement": "prohibited",
    "image_mask_or_target_pixel_decode": "prohibited",
    "prediction_generation": "prohibited",
    "quality_gates": "prohibited",
    "reserved_blind": "unavailable",
    "training": "prohibited",
}
EXPECTED_FIRST_GATE = {
    "manifest_role": "inner_val15",
    "minimum_residual_gain": 0.0005,
    "requires_measurable_movement": True,
    "requires_no_aggregate_overerase_regression": True,
    "requires_no_aggregate_residual_regression": True,
    "requires_no_page_overerase_regression": True,
    "requires_no_page_residual_regression": True,
}
EXPECTED_SUPERVISION = {
    "edit_positive": "target_luma_minus_input_luma_gt_margin",
    "inference_allowed_inputs": [
        "frozen_current_primary_plus_current_second_stage_prediction"
    ],
    "inference_forbidden_inputs": [
        "target",
        "label",
        "mask",
        "split",
        "domain",
        "route_override",
    ],
    "luminance_margin_gray": 2,
    "model_output": "bounded_nonnegative_luminance_delta",
    "preserve_negative": "target_luma_minus_input_luma_le_margin",
    "preserve_negative_includes": [
        "target_darker",
        "identity",
        "submargin_target_lighter",
    ],
    "residual_delta_bound": 0.08,
    "target_access_roles": ["train"],
    "target_forbidden_roles": [
        "inner_val15",
        "development_train160",
        "development_next120",
        "scut115",
        "holdout40",
        "reserved_blind",
    ],
}
EXPECTED_ABSENT_OUTPUTS = {
    "target_derived_patch_index": (
        "hardcase_lists/monotonic-residual-erase-train-patches-v1.csv"
    ),
    "training_output": "artifacts/trials/monotonic-residual-erase-v1",
    "training_plan": "docs/monotonic-residual-erase-training-plan.json",
    "training_preflight": (
        "outputs/monotonic-residual-erase-training-preflight-20260810"
    ),
}


def validate_plan_header(plan: dict[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "iteration_id": ACTIVE_ITERATION_ID,
        "state": "metadata_only_preregistered",
        "identity_format": "<domain>/<split>/<basename>",
        "role_contract_mode": "reuse_exact_frozen_effective_roles",
        "target_or_image_pixel_access": "prohibited",
        "training_cli_model_type_enabled": False,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise PreflightError(f"monotonic role plan {key} changed")
    if plan.get("authorization") != EXPECTED_AUTHORIZATION:
        raise PreflightError("monotonic role plan authorization changed")
    if plan.get("first_quality_gate") != EXPECTED_FIRST_GATE:
        raise PreflightError("monotonic first quality gate changed")
    if plan.get("supervision_contract") != EXPECTED_SUPERVISION:
        raise PreflightError("monotonic supervision contract changed")
    if plan.get("planned_outputs_must_be_absent") != EXPECTED_ABSENT_OUTPUTS:
        raise PreflightError("monotonic planned outputs changed")


def validate_plan_evidence(
    repo_root: Path,
    plan: dict[str, Any],
) -> dict[str, Path]:
    evidence = plan.get("evidence")
    expected_names = {
        "base_role_contract",
        "model",
        "preregistration",
        "synthetic_audit",
        "synthetic_decision",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected_names:
        raise PreflightError("monotonic evidence set changed")
    return {
        name: validate_artifact(repo_root, artifact, f"evidence.{name}")
        for name, artifact in evidence.items()
    }


def validate_synthetic_audit(audit: dict[str, Any]) -> None:
    required = {
        "terminal": "PASS",
        "model_type": MODEL_TYPE,
        "exact_identity_init": True,
        "zero_output_projection_init": True,
        "has_competing_route_or_dark_branch": False,
        "has_global_scale": False,
        "training_cli_enabled": False,
        "training_authorized": False,
        "real_data_access": False,
        "target_decode": False,
        "checkpoint_generated": False,
        "quality_gate_started": False,
        "promotion_enabled": False,
        "reserved_blind_state": "unavailable",
        "serialization_exact": True,
        "residual_delta_bound": 0.08,
    }
    for key, value in required.items():
        if audit.get(key) != value:
            raise PreflightError(f"synthetic audit field changed: {key}")
    if audit.get("forward_parameters") != ["x"]:
        raise PreflightError("synthetic model forward surface changed")
    gradients = audit.get("gradient_cases")
    if not isinstance(gradients, list) or len(gradients) != 3:
        raise PreflightError("synthetic gradient evidence changed")
    brighten = gradients[0]
    if (
        brighten.get("target_delta") != 0.05
        or float(brighten.get("support_gradient_abs", 0.0)) <= 0.0
        or float(brighten.get("magnitude_gradient_abs", 0.0)) <= 0.0
    ):
        raise PreflightError("synthetic brighten gradients are invalid")
    for preserve in gradients[1:]:
        if (
            float(preserve.get("support_gradient_abs", 0.0)) <= 0.0
            or float(preserve.get("magnitude_gradient_abs", -1.0)) != 0.0
        ):
            raise PreflightError("synthetic preserve gradients are invalid")


def validate_ledger_authority(
    repo_root: Path,
    ledger: dict[str, Any],
) -> dict[str, Any]:
    active = ledger.get("active_iteration", {})
    if active.get("id") != ACTIVE_ITERATION_ID:
        raise PreflightError("active iteration is not monotonic residual erase")
    prerequisites = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    expected_passed = {
        "monotonic_residual_erase_preregistration",
        "monotonic_residual_erase_synthetic_prerequisite",
    }
    if any(prerequisites.get(item) != "passed" for item in expected_passed):
        raise PreflightError("monotonic prerequisite authority changed")
    data_status = prerequisites.get(DATA_ROLE_PREREQUISITE_ID)
    if data_status not in {"pending", "passed"}:
        raise PreflightError("monotonic data-role preflight status is invalid")

    records = [
        record
        for record in ledger.get("records", [])
        if isinstance(record, dict) and record.get("id") == SYNTHETIC_RECORD_ID
    ]
    if len(records) != 1:
        raise PreflightError("ledger requires one monotonic synthetic PASS record")
    record = records[0]
    if (
        record.get("terminal") != "PASS"
        or record.get("outcome") != SYNTHETIC_OUTCOME
    ):
        raise PreflightError("monotonic synthetic record has wrong authority")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise PreflightError("monotonic synthetic PASS record lacks evidence")
    validated = [
        validate_artifact(repo_root, item, "monotonic synthetic record evidence")
        for item in evidence
    ]
    return {
        "record_id": SYNTHETIC_RECORD_ID,
        "evidence_count": len(validated),
        "data_role_ledger_status": data_status,
    }


def pixel_decoder_imports() -> list[str]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    return sorted(imported & FORBIDDEN_PIXEL_MODULES)


def validate_training_cli_closed(repo_root: Path) -> bool:
    enabled = any(
        MODEL_TYPE in path.read_text(encoding="utf-8")
        for path in sorted((repo_root / "scripts/train").glob("*.py"))
    )
    if enabled:
        raise PreflightError("monotonic model is enabled in a training CLI")
    return False


def validate_outputs_absent(repo_root: Path) -> list[str]:
    absent = []
    for label, raw_path in sorted(EXPECTED_ABSENT_OUTPUTS.items()):
        path = repo_path(repo_root, raw_path, f"planned output {label}")
        if path.exists():
            raise PreflightError(f"planned output must be absent: {path}")
        absent.append(str(path))
    return absent


def run_preflight(
    *,
    repo_root: Path = ROOT,
    role_plan_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_plan = role_plan_path or (repo_root / PLAN_PATH)
    resolved_ledger = ledger_path or (repo_root / LEDGER_PATH)
    try:
        plan = read_json(resolved_plan)
        ledger = read_json(resolved_ledger)
        validate_plan_header(plan)
        evidence = validate_plan_evidence(repo_root, plan)
        authority = validate_ledger_authority(repo_root, ledger)
        synthetic_audit = read_json(evidence["synthetic_audit"])
        validate_synthetic_audit(synthetic_audit)
        inherited = read_json(evidence["base_role_contract"])
        baseline = validate_baseline(repo_root, inherited, ledger)
        roles = inherited.get("roles")
        if not isinstance(roles, dict):
            raise PreflightError("inherited role contract has no roles")
        validate_reserved_blind(roles)
        raw_identities, source_summary = validate_role_sources(repo_root, roles)
        effective, role_summary = derive_effective_roles(roles, raw_identities)
        overlap_count = validate_zero_overlap(effective)
        decoder_imports = pixel_decoder_imports()
        if decoder_imports:
            raise PreflightError(
                f"metadata validator imports pixel decoder modules: {decoder_imports}"
            )
        training_cli_enabled = validate_training_cli_closed(repo_root)
        absent_outputs = validate_outputs_absent(repo_root)
    except (KeyError, OSError, PreflightError, TypeError, ValueError) as exc:
        return {
            "reason": str(exc),
            "runnable": False,
            "terminal": "PREREQUISITE_NEEDED",
        }

    role_counts = {
        name: int(role_summary[name]["count"])
        for name in ROLE_ORDER
    }
    role_counts["reserved_blind"] = 0
    return {
        "status": "pass",
        "terminal": "PASS",
        "runnable": True,
        "metadata_only": True,
        "role_plan": str(resolved_plan),
        "role_plan_sha256": sha256_file(resolved_plan),
        "inherited_role_contract": str(evidence["base_role_contract"]),
        "role_counts": dict(sorted(role_counts.items())),
        "role_summary": role_summary,
        "source_summary": source_summary,
        "train_domain_counts": role_summary["train"]["domain_counts"],
        "overlap_count": overlap_count,
        "supervision_contract": plan["supervision_contract"],
        "reserved_blind_state": "unavailable",
        "reserved_blind_authorized": False,
        "training_cli_enabled": training_cli_enabled,
        "pixel_decoder_imports": decoder_imports,
        "planned_outputs_absent": absent_outputs,
        "baseline": baseline,
        "synthetic_authority": authority,
        "first_quality_gate": plan["first_quality_gate"],
        "real_image_decode": False,
        "mask_decode": False,
        "target_decode": False,
        "target_patch_materialized": False,
        "training_started": False,
        "checkpoint_generated": False,
        "prediction_artifacts_generated": False,
        "quality_gate_started": False,
        "promotion_enabled": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--role-plan", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = run_preflight(
        repo_root=args.repo_root,
        role_plan_path=args.role_plan,
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
