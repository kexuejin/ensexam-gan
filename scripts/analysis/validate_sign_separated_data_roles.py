#!/usr/bin/env python3
"""Metadata-only role preflight for sign-separated residual repair."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ROLE_PLAN_PATH = Path("docs/sign-separated-residual-data-roles.json")
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
SYNTHETIC_AUDIT_PATH = Path(
    "outputs/sign-separated-residual-repair-synthetic-preflight-20260809/"
    "audit-final.json"
)
SYNTHETIC_RECORD_ID = "sign-separated-residual-repair-synthetic-prerequisite"
SYNTHETIC_OUTCOME = (
    "identity_signed_routes_and_branch_isolation_contract_passed"
)
MODEL_TYPE = "sign_separated_residual_delta"
ROLE_ORDER = (
    "inner_val15",
    "development_train160",
    "development_next120",
    "scut115",
    "holdout40",
    "train",
)
EXPECTED_PRIMARY_PROTOCOL = {
    "batch_size": 8,
    "change_threshold": 12,
    "copy_input_outside_mask": "mb",
    "copy_mask_dilate": 0,
    "copy_mask_threshold_auto": "mb_cov8_step",
    "eval_threshold": 12,
    "page_overlap": 32,
}
EXPECTED_SECOND_STAGE_PROTOCOL = {
    "base_edit_threshold": 12,
    "cleanup_alpha_threshold": 0.3,
    "cleanup_stride": 160,
    "cleanup_tile_size": 160,
    "dark_threshold": 0,
    "second_delta_threshold": 32,
}
FORBIDDEN_PIXEL_MODULES = {"cv2", "imageio", "numpy", "PIL", "torch"}


class PreflightError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_identities(identities: set[str]) -> str:
    payload = "\n".join(sorted(identities)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PreflightError(f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreflightError(f"JSON must be an object: {path}")
    return value


def repo_path(repo_root: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise PreflightError(f"{label} path must be a non-empty string")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise PreflightError(f"{label} path must stay inside repository")
    return repo_root / relative


def validate_artifact(
    repo_root: Path,
    artifact: Any,
    label: str,
) -> Path:
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


def read_manifest(path: Path) -> list[str]:
    rows = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        raise PreflightError(f"manifest is empty: {path}")
    if len(rows) != len(set(rows)):
        raise PreflightError(f"manifest contains duplicate rows: {path}")
    return rows


def identities_from_role(role: dict[str, Any], manifest_path: Path) -> set[str]:
    rows = read_manifest(manifest_path)
    manifest_type = role.get("manifest_type")
    if manifest_type == "scut_paths":
        domain = role.get("identity_domain")
        split = role.get("identity_split")
        if domain != "scut" or split not in {"train", "test"}:
            raise PreflightError("scut_paths role has invalid domain/split")
        return {f"{domain}/{split}/{Path(row).name}" for row in rows}
    if manifest_type == "mixed_prefixed":
        identities: set[str] = set()
        for row in rows:
            match = re.fullmatch(r"(scut|hw5k)_(.+)", Path(row).name)
            if match is None:
                raise PreflightError(f"mixed manifest row lacks domain prefix: {row}")
            identities.add(f"{match.group(1)}/train/{match.group(2)}")
        if len(identities) != len(rows):
            raise PreflightError("mixed manifest identities are not unique")
        return identities
    raise PreflightError(f"unsupported manifest_type: {manifest_type}")


def validate_role_sources(
    repo_root: Path,
    roles: dict[str, Any],
) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]]]:
    raw_identities: dict[str, set[str]] = {}
    source_summary: dict[str, dict[str, Any]] = {}
    for role_name in ROLE_ORDER:
        role = roles.get(role_name)
        if not isinstance(role, dict):
            raise PreflightError(f"missing role: {role_name}")
        manifest_path = validate_artifact(
            repo_root, role.get("manifest"), f"roles.{role_name}.manifest"
        )
        identities = identities_from_role(role, manifest_path)
        expected_source_count = int(role.get("source_count", -1))
        if len(identities) != expected_source_count:
            raise PreflightError(
                f"{role_name} source count mismatch: "
                f"expected={expected_source_count} actual={len(identities)}"
            )
        raw_identities[role_name] = identities
        source_summary[role_name] = {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "source_count": len(identities),
        }
        if role_name == "train":
            domain_counts = {
                domain: sum(
                    identity.startswith(f"{domain}/") for identity in identities
                )
                for domain in ("hw5k", "scut")
            }
            if domain_counts != role.get("source_domain_counts"):
                raise PreflightError(
                    f"train source domain counts mismatch: {domain_counts}"
                )
            source_summary[role_name]["source_domain_counts"] = domain_counts
    return raw_identities, source_summary


def derive_effective_roles(
    roles: dict[str, Any],
    raw_identities: dict[str, set[str]],
) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]]]:
    effective: dict[str, set[str]] = {}
    summary: dict[str, dict[str, Any]] = {}
    for role_name in ROLE_ORDER:
        role = roles[role_name]
        exclusions = role.get("exclude_roles")
        if not isinstance(exclusions, list):
            raise PreflightError(f"{role_name}.exclude_roles must be a list")
        unknown = [name for name in exclusions if name not in effective]
        if unknown:
            raise PreflightError(
                f"{role_name} exclusions are unknown or out of order: {unknown}"
            )
        excluded = set().union(*(effective[name] for name in exclusions))
        identities = raw_identities[role_name] - excluded
        expected_count = int(role.get("effective_count", -1))
        expected_hash = role.get("effective_identity_sha256")
        actual_hash = sha256_identities(identities)
        if len(identities) != expected_count:
            raise PreflightError(
                f"{role_name} effective count mismatch: "
                f"expected={expected_count} actual={len(identities)}"
            )
        if actual_hash != expected_hash:
            raise PreflightError(
                f"{role_name} effective identity hash mismatch: "
                f"expected={expected_hash} actual={actual_hash}"
            )
        effective[role_name] = identities
        summary[role_name] = {
            "count": len(identities),
            "identity_sha256": actual_hash,
            "excluded_count": len(raw_identities[role_name] & excluded),
        }
        if role_name == "train":
            domain_counts = {
                domain: sum(
                    identity.startswith(f"{domain}/") for identity in identities
                )
                for domain in ("hw5k", "scut")
            }
            if domain_counts != role.get("effective_domain_counts"):
                raise PreflightError(
                    f"train effective domain counts mismatch: {domain_counts}"
                )
            summary[role_name]["domain_counts"] = domain_counts
    return effective, summary


def validate_reserved_blind(roles: dict[str, Any]) -> None:
    reserved = roles.get("reserved_blind")
    if not isinstance(reserved, dict):
        raise PreflightError("reserved blind role is missing")
    if reserved != {
        "authorized": False,
        "effective_count": 0,
        "manifest": None,
        "state": "unavailable",
    }:
        raise PreflightError("reserved blind must remain unavailable and unauthorized")


def validate_zero_overlap(effective: dict[str, set[str]]) -> int:
    overlaps = []
    for left, right in combinations(ROLE_ORDER, 2):
        shared = effective[left] & effective[right]
        if shared:
            overlaps.append((left, right, sorted(shared)[:5]))
    if overlaps:
        raise PreflightError(f"effective roles overlap: {overlaps}")
    return 0


def validate_baseline(
    repo_root: Path,
    plan: dict[str, Any],
    ledger: dict[str, Any],
) -> dict[str, str]:
    baseline = plan.get("baseline")
    if not isinstance(baseline, dict):
        raise PreflightError("role plan baseline is missing")
    primary = baseline.get("current_primary")
    second_stage = baseline.get("current_second_stage")
    if not isinstance(primary, dict) or not isinstance(second_stage, dict):
        raise PreflightError("role plan baseline entries are missing")
    ledger_baseline = ledger.get("baseline", {})
    if primary.get("config") != ledger_baseline.get("config"):
        raise PreflightError("current-primary config does not match ledger baseline")
    if primary.get("checkpoint") != ledger_baseline.get("checkpoint"):
        raise PreflightError("current-primary checkpoint does not match ledger baseline")
    if primary.get("matched_copy_protocol") != EXPECTED_PRIMARY_PROTOCOL:
        raise PreflightError("current-primary matched-copy protocol changed")
    if second_stage.get("protocol") != EXPECTED_SECOND_STAGE_PROTOCOL:
        raise PreflightError("current second-stage protocol changed")
    primary_config = validate_artifact(
        repo_root, primary.get("config"), "baseline.current_primary.config"
    )
    primary_checkpoint = validate_artifact(
        repo_root,
        primary.get("checkpoint"),
        "baseline.current_primary.checkpoint",
    )
    second_checkpoint = validate_artifact(
        repo_root,
        second_stage.get("checkpoint"),
        "baseline.current_second_stage.checkpoint",
    )
    inference_script = validate_artifact(
        repo_root,
        second_stage.get("inference_script"),
        "baseline.current_second_stage.inference_script",
    )
    return {
        "current_primary_config_sha256": sha256_file(primary_config),
        "current_primary_checkpoint_sha256": sha256_file(primary_checkpoint),
        "current_second_stage_checkpoint_sha256": sha256_file(second_checkpoint),
        "current_second_stage_inference_sha256": sha256_file(inference_script),
    }


def validate_evidence_path(
    repo_root: Path,
    evidence: dict[str, Any],
) -> Path:
    return validate_artifact(repo_root, evidence, "synthetic prerequisite evidence")


def validate_synthetic_audit(audit: dict[str, Any]) -> None:
    required = {
        "terminal": "PASS",
        "model_type": MODEL_TYPE,
        "exact_identity_init": True,
        "zero_magnitude_projection_init": True,
        "has_global_scale": False,
        "training_cli_enabled": False,
        "opposed_pixel_count": 0,
        "serialization_exact": True,
        "residual_delta_bound": 0.08,
    }
    mismatches = {
        key: audit.get(key)
        for key, expected in required.items()
        if audit.get(key) != expected
    }
    if mismatches:
        raise PreflightError(f"synthetic audit contract mismatch: {mismatches}")
    cases = audit.get("gradient_cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise PreflightError("synthetic audit gradient cases are missing")
    by_direction = {case.get("direction"): case for case in cases}
    if set(by_direction) != {-1, 1}:
        raise PreflightError("synthetic audit directions must be -1 and 1")
    bright = by_direction[1]
    dark = by_direction[-1]
    if (
        float(bright.get("bright_gradient_abs", 0.0)) <= 0.0
        or float(bright.get("dark_gradient_abs", -1.0)) != 0.0
        or float(dark.get("dark_gradient_abs", 0.0)) <= 0.0
        or float(dark.get("bright_gradient_abs", -1.0)) != 0.0
        or float(bright.get("route_gradient_abs", 0.0)) <= 0.0
        or float(dark.get("route_gradient_abs", 0.0)) <= 0.0
    ):
        raise PreflightError("synthetic audit branch isolation is invalid")


def validate_synthetic_authority(
    repo_root: Path,
    ledger: dict[str, Any],
) -> dict[str, Any]:
    active = ledger.get("active_iteration", {})
    if active.get("id") != "sign-separated-residual-repair":
        raise PreflightError("active iteration is not sign-separated residual repair")
    prerequisites = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if prerequisites.get("sign_separated_residual_synthetic_preflight") != "passed":
        raise PreflightError("synthetic prerequisite is not marked passed")
    data_status = prerequisites.get("sign_separated_residual_data_role_preflight")
    if data_status not in {"pending", "passed"}:
        raise PreflightError("data-role preflight ledger status is invalid")
    records = [
        record
        for record in ledger.get("records", [])
        if isinstance(record, dict) and record.get("id") == SYNTHETIC_RECORD_ID
    ]
    if len(records) != 1:
        raise PreflightError("ledger requires one synthetic prerequisite PASS record")
    record = records[0]
    if record.get("terminal") != "PASS" or record.get("outcome") != SYNTHETIC_OUTCOME:
        raise PreflightError("synthetic prerequisite PASS record has wrong authority")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise PreflightError("synthetic prerequisite PASS record lacks evidence")
    validated = [validate_evidence_path(repo_root, item) for item in evidence]
    audit_path = repo_root / SYNTHETIC_AUDIT_PATH
    if audit_path not in validated:
        raise PreflightError("synthetic prerequisite PASS record lacks audit-final.json")
    audit = read_json(audit_path)
    validate_synthetic_audit(audit)
    return {
        "record_id": SYNTHETIC_RECORD_ID,
        "audit_sha256": sha256_file(audit_path),
        "evidence_count": len(validated),
        "data_role_ledger_status": data_status,
    }


def validate_training_cli_closed(repo_root: Path) -> bool:
    training_script = repo_root / "scripts/train/train_patch_cleanup_erasemap_probe.py"
    if not training_script.is_file():
        raise PreflightError("cleanup training script is missing")
    enabled = MODEL_TYPE in training_script.read_text(encoding="utf-8")
    if enabled:
        raise PreflightError("sign-separated model is enabled in the training CLI")
    return enabled


def validate_planned_outputs_absent(
    repo_root: Path,
    plan: dict[str, Any],
) -> list[str]:
    outputs = plan.get("planned_outputs_must_be_absent")
    if not isinstance(outputs, dict) or not outputs:
        raise PreflightError("planned output paths are missing")
    validated = []
    for label, raw_path in sorted(outputs.items()):
        path = repo_path(repo_root, raw_path, f"planned_outputs.{label}")
        if path.exists():
            raise PreflightError(f"planned output must be absent: {path}")
        validated.append(str(path))
    return validated


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


def validate_plan_header(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != 1:
        raise PreflightError("role plan schema_version must be 1")
    if plan.get("state") != "metadata_only_preregistered":
        raise PreflightError("role plan state must be metadata_only_preregistered")
    if plan.get("target_or_image_pixel_access") != "prohibited":
        raise PreflightError("role plan must prohibit target/image pixel access")
    if plan.get("training_cli_model_type_enabled") is not False:
        raise PreflightError("role plan must keep training CLI disabled")
    if plan.get("identity_format") != "<domain>/<split>/<basename>":
        raise PreflightError("role identity format changed")
    gate = plan.get("first_quality_gate")
    expected_gate = {
        "manifest_role": "inner_val15",
        "minimum_residual_gain": 0.0005,
        "requires_measurable_movement": True,
        "requires_no_aggregate_overerase_regression": True,
        "requires_no_aggregate_residual_regression": True,
        "requires_no_page_overerase_regression": True,
        "requires_no_page_residual_regression": True,
    }
    if gate != expected_gate:
        raise PreflightError("first quality gate contract changed")


def run_preflight(
    *,
    repo_root: Path = ROOT,
    role_plan_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_plan = role_plan_path or (repo_root / ROLE_PLAN_PATH)
    resolved_ledger = ledger_path or (repo_root / LEDGER_PATH)
    try:
        plan = read_json(resolved_plan)
        ledger = read_json(resolved_ledger)
        validate_plan_header(plan)
        authority = validate_synthetic_authority(repo_root, ledger)
        baseline = validate_baseline(repo_root, plan, ledger)
        roles = plan.get("roles")
        if not isinstance(roles, dict):
            raise PreflightError("role plan roles must be an object")
        validate_reserved_blind(roles)
        raw_identities, sources = validate_role_sources(repo_root, roles)
        effective, role_summary = derive_effective_roles(roles, raw_identities)
        overlap_count = validate_zero_overlap(effective)
        decoder_imports = pixel_decoder_imports()
        if decoder_imports:
            raise PreflightError(
                f"metadata validator imports pixel decoder modules: {decoder_imports}"
            )
        training_cli_enabled = validate_training_cli_closed(repo_root)
        absent_outputs = validate_planned_outputs_absent(repo_root, plan)
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
        "role_counts": dict(sorted(role_counts.items())),
        "role_summary": role_summary,
        "source_summary": sources,
        "train_domain_counts": role_summary["train"]["domain_counts"],
        "overlap_count": overlap_count,
        "reserved_blind_state": "unavailable",
        "reserved_blind_authorized": False,
        "training_cli_enabled": training_cli_enabled,
        "pixel_decoder_imports": decoder_imports,
        "planned_outputs_absent": absent_outputs,
        "baseline": baseline,
        "synthetic_authority": authority,
        "first_quality_gate": plan["first_quality_gate"],
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
