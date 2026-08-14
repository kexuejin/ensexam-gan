#!/usr/bin/env python3
"""Fail-closed real preflight for the preregistered universal-sidecar D5 run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from networks.generator import Generator  # noqa: E402
from scripts.analysis.audit_universal_sidecar_structure import run_audit  # noqa: E402
from scripts.analysis.validate_universal_sidecar_d4_preflight import (  # noqa: E402
    FORBIDDEN_D3_FIELDS,
    PreflightError,
    assert_frozen_current_primary_matches_ledger,
    flatten,
    read_json,
    read_name_manifest,
    read_yaml,
    resolve_repo_path,
    sha256_file,
)
from train import (  # noqa: E402
    apply_generator_trainable_patterns,
    freeze_batchnorm_running_stats,
    validate_universal_sidecar_config,
)


D4_CONFIG = Path(
    "configs/local/"
    "config.local-universal-sidecar-d4-d1-mixed-scut130-hw5k260-step80-"
    "primary-edit-direction-mps.yaml"
)
D5_CONFIG = Path(
    "configs/local/"
    "config.local-universal-sidecar-d5-d1-mixed-scut130-hw5k260-step80-"
    "primary-edit-direction-folded-mps.yaml"
)
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
SYNTHETIC_AUDIT_PATH = Path(
    "outputs/primary-edit-direction-folded-sidecar-preflight-20260809/"
    "audit-final.json"
)
SYNTHETIC_RECORD_ID = (
    "universal-sidecar-d5-folded-direction-magnitude-synthetic-prerequisite"
)
SYNTHETIC_OUTCOME = (
    "synthetic_folded_direction_both_sign_gradient_contract_passed"
)
EXPECTED_DIFFERENCES = {
    "model.universal_residual_adapter_sidecar.residual_parameterization": (
        "primary_edit_direction_folded"
    ),
    "train.save_dir": (
        "./artifacts/trials/"
        "universal-sidecar-d5-d1-mixed-scut130-hw5k260-step80-"
        "primary-edit-direction-folded-20260809"
    ),
}


def assert_exact_config_delta(d4: dict[str, Any], d5: dict[str, Any]) -> None:
    left = flatten(d4)
    right = flatten(d5)
    changed = {
        key: right.get(key)
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }
    if changed != EXPECTED_DIFFERENCES:
        raise PreflightError(
            "D5/D4 semantic differences do not match preregistration: "
            f"{changed}"
        )


def assert_no_forbidden_d3_fields(d5: dict[str, Any]) -> None:
    flattened = flatten(d5)
    present = sorted(key for key in FORBIDDEN_D3_FIELDS if key in flattened)
    if present:
        raise PreflightError(f"D3 cache/baseline-tail fields must be absent: {present}")


def assert_gate_isolation(d5: dict[str, Any]) -> None:
    evaluation = d5.get("evaluation", {})
    if evaluation.get("skip_validation") is not True:
        raise PreflightError("D5 must keep validation disabled during training")
    if evaluation.get("skip_final_test") is not True:
        raise PreflightError("D5 must keep final test disabled during training")
    if evaluation.get("standalone_test_mode") not in {None, "none"}:
        raise PreflightError("standalone test gate is enabled")
    if evaluation.get("final_test_mode") not in {None, "none"}:
        raise PreflightError("final test gate is enabled")
    flattened = flatten(d5)
    forbidden_tokens = ("reserved_blind", "scut115", "holdout40", "promotion")
    enabled_forbidden = [
        key
        for key, value in flattened.items()
        if any(token in key.lower() for token in forbidden_tokens)
        and value not in {None, False, "", "none", 0}
    ]
    if enabled_forbidden:
        raise PreflightError(f"later gates enabled in D5: {enabled_forbidden}")


def assert_output_dir_missing(repo_root: Path, d5: dict[str, Any]) -> Path:
    output_dir = resolve_repo_path(repo_root, d5["train"]["save_dir"])
    if output_dir.exists():
        raise PreflightError(f"D5 save_dir must not exist yet: {output_dir}")
    return output_dir


def assert_train_manifest_matches_d4(
    repo_root: Path,
    d4: dict[str, Any],
    d5: dict[str, Any],
) -> dict[str, Any]:
    d4_manifest = resolve_repo_path(repo_root, d4["data"]["train_file_list"])
    d5_manifest = resolve_repo_path(repo_root, d5["data"]["train_file_list"])
    d4_names = read_name_manifest(d4_manifest)
    d5_names = read_name_manifest(d5_manifest)
    if d5_names != d4_names:
        raise PreflightError("D5 train manifest must match D4 exactly")
    inner_val = (
        repo_root
        / "hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt"
    )
    inner_names = set(read_name_manifest(inner_val))
    overlap = sorted(set(d5_names) & inner_names)
    if overlap:
        raise PreflightError(f"D5 train manifest overlaps inner-val15: {overlap[:5]}")
    return {
        "path": str(d5_manifest),
        "sha256": sha256_file(d5_manifest),
        "sample_count": len(d5_names),
        "inner_val_overlap": 0,
    }


def _validated_evidence_path(
    repo_root: Path,
    evidence: dict[str, Any],
) -> Path:
    raw_path = evidence.get("path")
    expected_hash = evidence.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise PreflightError("synthetic prerequisite evidence entry is incomplete")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise PreflightError("synthetic prerequisite evidence path must stay in repo")
    path = repo_root / relative
    if not path.is_file():
        raise PreflightError(f"synthetic prerequisite evidence is missing: {relative}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise PreflightError(
            "synthetic prerequisite evidence hash mismatch: "
            f"{relative} expected={expected_hash} actual={actual_hash}"
        )
    return path


def _assert_synthetic_audit_contract(
    audit: dict[str, Any],
    ledger: dict[str, Any],
) -> None:
    required_values = {
        "terminal": "PASS",
        "mode": "primary_edit_direction_folded",
        "strict_current_primary_load": True,
        "exact_zero_init": True,
        "trainable_tensors": 17,
        "positive_negative_fold_equal": True,
        "opposed_channel_count": 0,
        "zero_primary_edit_noop": True,
        "public_interface_unchanged": True,
    }
    mismatches = {
        key: audit.get(key)
        for key, expected in required_values.items()
        if audit.get(key) != expected
    }
    if mismatches:
        raise PreflightError(f"synthetic audit contract mismatch: {mismatches}")

    baseline = ledger.get("baseline", {})
    expected_config_hash = baseline.get("config", {}).get("sha256")
    expected_checkpoint_hash = baseline.get("checkpoint", {}).get("sha256")
    if audit.get("current_primary_config_sha256") != expected_config_hash:
        raise PreflightError("synthetic audit current-primary config hash mismatch")
    if audit.get("current_primary_checkpoint_sha256") != expected_checkpoint_hash:
        raise PreflightError("synthetic audit current-primary checkpoint hash mismatch")

    base_tensors = int(audit.get("base_parameter_tensors", 0))
    frozen_base_tensors = int(audit.get("frozen_base_parameter_tensors", -1))
    if base_tensors <= 0 or frozen_base_tensors != base_tensors:
        raise PreflightError("synthetic audit did not freeze every base tensor")
    if int(audit.get("sidecar_only_missing_key_count", -1)) != 17:
        raise PreflightError("synthetic audit sidecar missing-key count must be 17")
    residual_abs_max = float(audit.get("residual_abs_max", 0.0))
    residual_bound = float(audit.get("residual_bound", 0.0))
    if residual_abs_max < 0.0 or residual_bound <= 0.0:
        raise PreflightError("synthetic audit residual-bound evidence is missing")
    if residual_abs_max > residual_bound + 1e-7:
        raise PreflightError("synthetic audit residual exceeded its bound")

    probes = audit.get("two_step_probes")
    if not isinstance(probes, list) or len(probes) != 2:
        raise PreflightError("synthetic audit must contain exactly two sign probes")
    by_sign = {probe.get("raw_sign"): probe for probe in probes}
    if set(by_sign) != {-1, 1}:
        raise PreflightError("synthetic audit must contain raw signs -1 and 1")
    for raw_sign, probe in by_sign.items():
        if int(probe.get("folded_support_count", 0)) <= 0:
            raise PreflightError(f"raw_sign={raw_sign} has no folded support")
        if float(probe.get("first_projection_gradient_min", 0.0)) <= 0.0:
            raise PreflightError(f"raw_sign={raw_sign} first projection gradient died")
        if float(probe.get("second_projection_gradient_min", 0.0)) <= 0.0:
            raise PreflightError(f"raw_sign={raw_sign} second projection gradient died")
        if float(probe.get("second_scale_gradient_abs", 0.0)) <= 0.0:
            raise PreflightError(f"raw_sign={raw_sign} second scale gradient died")
        initial_scale = probe.get("initial_global_residual_scale")
        final_scale = probe.get("final_global_residual_scale")
        if not isinstance(initial_scale, (int, float)) or not isinstance(
            final_scale, (int, float)
        ):
            raise PreflightError(f"raw_sign={raw_sign} scale movement is missing")
        if float(final_scale) == float(initial_scale):
            raise PreflightError(f"raw_sign={raw_sign} global scale did not move")


def assert_synthetic_prerequisite_authority(
    repo_root: Path,
    ledger: dict[str, Any],
) -> dict[str, Any]:
    active = ledger.get("active_iteration", {})
    if active.get("id") != "universal-sidecar-d5-folded-direction-magnitude":
        raise PreflightError("ledger active iteration is not D5 folded direction")
    prerequisites = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if prerequisites.get("d5_folded_magnitude_synthetic_preflight") != "passed":
        raise PreflightError("D5 synthetic prerequisite is not marked passed")
    real_preflight_status = prerequisites.get("d5_real_preflight")
    if real_preflight_status not in {"pending", "passed"}:
        raise PreflightError("D5 real preflight ledger status is not pending or passed")

    records = [
        record
        for record in ledger.get("records", [])
        if isinstance(record, dict) and record.get("id") == SYNTHETIC_RECORD_ID
    ]
    if len(records) != 1:
        raise PreflightError("ledger requires exactly one synthetic prerequisite PASS record")
    record = records[0]
    if record.get("terminal") != "PASS" or record.get("outcome") != SYNTHETIC_OUTCOME:
        raise PreflightError("synthetic prerequisite PASS record has wrong authority")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise PreflightError("synthetic prerequisite PASS record lacks evidence")
    validated = [_validated_evidence_path(repo_root, item) for item in evidence]
    audit_path = repo_root / SYNTHETIC_AUDIT_PATH
    if audit_path not in validated:
        raise PreflightError("synthetic prerequisite PASS record lacks audit-final.json")
    audit = read_json(audit_path)
    _assert_synthetic_audit_contract(audit, ledger)
    return {
        "record_id": SYNTHETIC_RECORD_ID,
        "audit": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "evidence_count": len(validated),
        "real_preflight_ledger_status": real_preflight_status,
        "terminal": "PASS",
    }


def assert_structure_and_training_scope(d5: dict[str, Any]) -> dict[str, Any]:
    model_cfg = d5.get("model", {})
    train_cfg = d5.get("train", {})
    mode = (
        model_cfg.get("universal_residual_adapter_sidecar", {})
        .get("residual_parameterization")
    )
    if mode != "primary_edit_direction_folded":
        raise PreflightError(
            "D5 must use primary_edit_direction_folded residual_parameterization"
        )
    try:
        validate_universal_sidecar_config(d5)
        structure = run_audit(
            model_cfg=model_cfg,
            trainable_patterns=train_cfg.get("trainable_generator_patterns"),
        )
    except (TypeError, ValueError) as exc:
        raise PreflightError(f"runtime config/structure validation failed: {exc}") from exc

    generator = Generator(cfg=model_cfg)
    trainable = apply_generator_trainable_patterns(
        generator,
        train_cfg.get("trainable_generator_patterns"),
    )
    trainable_names = [
        name
        for name, parameter in generator.named_parameters()
        if parameter.requires_grad
    ]
    if trainable["trainable_tensors"] != 17 or any(
        not name.startswith("universal_residual_adapter_sidecar.")
        for name in trainable_names
    ):
        raise PreflightError(
            "D5 must expose exactly 17 sidecar-only trainable tensors"
        )
    frozen_batchnorm = freeze_batchnorm_running_stats(generator)
    if frozen_batchnorm <= 0:
        raise PreflightError("D5 did not freeze any BatchNorm layers")
    if any(
        module.training
        for module in generator.modules()
        if isinstance(
            module,
            (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.SyncBatchNorm,
            ),
        )
    ):
        raise PreflightError("D5 left a BatchNorm layer in training mode")
    return {
        "structure_audit": structure,
        "trainable_tensors": trainable["trainable_tensors"],
        "frozen_tensors": trainable["frozen_tensors"],
        "trainable_params": trainable["trainable_params"],
        "frozen_batchnorm_layers": frozen_batchnorm,
    }


def run_preflight(
    *,
    repo_root: Path = ROOT,
    d4_config: Path | None = None,
    d5_config: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    d4_path = d4_config or (repo_root / D4_CONFIG)
    d5_path = d5_config or (repo_root / D5_CONFIG)
    resolved_ledger = ledger_path or (repo_root / LEDGER_PATH)
    try:
        d4 = read_yaml(d4_path)
        d5 = read_yaml(d5_path)
        ledger = read_json(resolved_ledger)
        authority = assert_synthetic_prerequisite_authority(repo_root, ledger)
        assert_no_forbidden_d3_fields(d5)
        assert_gate_isolation(d5)
        assert_exact_config_delta(d4, d5)
        output_dir = assert_output_dir_missing(repo_root, d5)
        manifest = assert_train_manifest_matches_d4(repo_root, d4, d5)
        checkpoint = assert_frozen_current_primary_matches_ledger(
            repo_root, ledger, d5
        )
        if checkpoint["sidecar_missing_keys"] != 17:
            raise PreflightError("D5 current-primary load must miss 17 sidecar keys")
        if checkpoint["sidecar_unexpected_keys"] != 0:
            raise PreflightError("D5 current-primary load has unexpected keys")
        structure = assert_structure_and_training_scope(d5)
    except (KeyError, OSError, PreflightError, TypeError, ValueError) as exc:
        return {
            "first_gate": "scut_inner_val15",
            "reason": str(exc),
            "runnable": False,
            "terminal": "PREREQUISITE_NEEDED",
        }
    return {
        "checkpoint_audit": checkpoint,
        "config": str(d5_path),
        "first_gate": "scut_inner_val15",
        "output_dir": str(output_dir),
        "runnable": True,
        "structure_audit": structure,
        "synthetic_prerequisite": authority,
        "terminal": "PASS",
        "train_manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_preflight(repo_root=args.repo_root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if result["terminal"] == "PASS" else 2)


if __name__ == "__main__":
    main()
