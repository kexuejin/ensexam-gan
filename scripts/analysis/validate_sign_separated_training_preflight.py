#!/usr/bin/env python3
"""Fail-closed training preflight for sign-separated residual repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.build_sign_separated_residual_patch_index import (  # noqa: E402
    effective_train_filenames,
    sha256_rows,
)
from scripts.infer.patch_cleanup_erasemap import (  # noqa: E402
    SignSeparatedResidualDeltaCleanupNet,
)
from scripts.train.train_sign_separated_residual_probe import (  # noqa: E402
    MASK_SOURCE,
    MODEL_TYPE,
    build_parser,
    compute_sign_separated_loss_terms,
)


TRAINING_PLAN_PATH = Path("docs/sign-separated-residual-training-plan.json")
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
DATA_ROLE_RECORD_ID = "sign-separated-residual-data-role-preflight"
DATA_ROLE_OUTCOME = "metadata_only_roles_baseline_hashes_and_zero_overlap_passed"
EXPECTED_AUTHORIZATION = {
    "checkpoint_generation": "prohibited_until_preflight_pass",
    "image_or_target_decode": "prohibited_until_preflight_pass",
    "inference": "prohibited_until_preflight_pass",
    "promotion": "disabled",
    "reserved_blind": "unavailable",
    "state": "training_preflight_pending",
    "target_derived_patch_construction": "prohibited_until_preflight_pass",
    "training": "prohibited_until_preflight_pass",
}
EXPECTED_DATA = {
    "data_root": "data-links/samples/SCUT-HW5K-mixed-20260729",
    "effective_train_count": 275,
    "effective_train_domain_counts": {"hw5k": 253, "scut": 22},
    "effective_train_filename_sha256": (
        "e9ac4d6f700f41ef3a9b7c3f04ce0593f593324a881a0f9fc387901a497f9039"
    ),
    "label_dir": (
        "data-links/samples/SCUT-HW5K-mixed-20260729/train/all_labels"
    ),
    "source_dir": (
        "data-links/samples/SCUT-HW5K-mixed-20260729/train/all_images"
    ),
    "split": "train",
}
EXPECTED_MODEL = {
    "global_residual_scale": "absent",
    "model_type": "sign_separated_residual_delta",
    "residual_delta_bound": 0.08,
    "route_classes": ["identity", "brighten", "darken"],
}
EXPECTED_PATCH_BUILDER = {
    "direction_margin": 2.0,
    "input_dir": (
        "outputs/sign-separated-residual-repair-train275-frozen-pipeline-v1/pred"
    ),
    "min_support_ratio": 0.001,
    "output_csv": (
        "hardcase_lists/sign-separated-residual-repair-train-patches-v1.csv"
    ),
    "output_json": (
        "outputs/sign-separated-residual-repair-train-patches-v1/summary.json"
    ),
    "overlap": 96,
    "tile_size": 256,
    "top_k_per_direction": 256,
}
EXPECTED_PRIMARY = {
    "batch_size": 8,
    "copy_input_outside_mask": "mb",
    "copy_mask_dilate": 0,
    "copy_mask_threshold": 70,
    "copy_mask_threshold_auto": "mb_cov8_step",
    "device": "mps",
    "output_dir": "outputs/sign-separated-residual-repair-train275-primary-v1",
    "page_overlap": 32,
    "samples_file": (
        "hardcase_lists/sign-separated-residual-repair-train275-v1.txt"
    ),
    "skip_label_metrics": True,
}
EXPECTED_SECOND_STAGE = {
    "base_edit_threshold": 12,
    "change_threshold": 12,
    "cleanup_alpha_threshold": 0.3,
    "cleanup_stride": 160,
    "cleanup_tile_size": 160,
    "dark_threshold": 0,
    "device": "mps",
    "eval_threshold": 12,
    "output_dir": (
        "outputs/sign-separated-residual-repair-train275-frozen-pipeline-v1"
    ),
    "primary_pred_dir": (
        "outputs/sign-separated-residual-repair-train275-primary-v1/pred"
    ),
    "samples_file": (
        "hardcase_lists/sign-separated-residual-repair-train275-v1.txt"
    ),
    "second_delta_threshold": 32,
}
EXPECTED_TRAINER = {
    "batch_size": 1,
    "bright_magnitude_weight": 1.0,
    "dark_magnitude_weight": 1.0,
    "device": "mps",
    "identity_delta_weight": 1.0,
    "init_checkpoint": "",
    "input_dir": (
        "outputs/sign-separated-residual-repair-train275-frozen-pipeline-v1/pred"
    ),
    "log_every": 10,
    "lr": 0.00002,
    "mask_source": "target_delta",
    "mask_threshold": 12,
    "max_steps": 80,
    "model_type": "sign_separated_residual_delta",
    "output_dir": "artifacts/trials/sign-separated-residual-repair-v1",
    "patch_index_file": (
        "hardcase_lists/sign-separated-residual-repair-train-patches-v1.csv"
    ),
    "residual_delta_bound": 0.08,
    "route_loss_weight": 1.0,
    "save_every": 0,
    "seed": 42,
    "sign_direction_margin": 2.0,
    "tile_size": 256,
    "validation_enabled": False,
}
EXPECTED_GATE = {
    "manifest_role": "inner_val15",
    "minimum_residual_gain": 0.0005,
    "requires_measurable_movement": True,
    "requires_no_aggregate_overerase_regression": True,
    "requires_no_aggregate_residual_regression": True,
    "requires_no_page_overerase_regression": True,
    "requires_no_page_residual_regression": True,
}
EXPECTED_OUTPUTS = {
    "first_gate_output_dir": (
        "outputs/sign-separated-residual-repair-inner-val15-v1"
    ),
    "patch_index": (
        "hardcase_lists/sign-separated-residual-repair-train-patches-v1.csv"
    ),
    "patch_index_summary_dir": (
        "outputs/sign-separated-residual-repair-train-patches-v1"
    ),
    "primary_prediction_dir": (
        "outputs/sign-separated-residual-repair-train275-primary-v1"
    ),
    "sample_manifest": (
        "hardcase_lists/sign-separated-residual-repair-train275-v1.txt"
    ),
    "training_input_dir": (
        "outputs/sign-separated-residual-repair-train275-frozen-pipeline-v1"
    ),
    "training_output_dir": "artifacts/trials/sign-separated-residual-repair-v1",
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


def assert_exact_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != 1:
        raise PreflightError("training plan schema_version must be 1")
    if plan.get("iteration_id") != "sign-separated-residual-repair":
        raise PreflightError("training plan iteration changed")
    expected_sections = {
        "authorization": EXPECTED_AUTHORIZATION,
        "data": EXPECTED_DATA,
        "model": EXPECTED_MODEL,
        "patch_builder": EXPECTED_PATCH_BUILDER,
        "first_quality_gate": EXPECTED_GATE,
        "planned_outputs_must_be_absent": EXPECTED_OUTPUTS,
        "trainer": EXPECTED_TRAINER,
    }
    for key, expected in expected_sections.items():
        if plan.get(key) != expected:
            raise PreflightError(f"training plan {key} changed")
    preparation = plan.get("pipeline_preparation")
    if preparation != {
        "primary": EXPECTED_PRIMARY,
        "second_stage": EXPECTED_SECOND_STAGE,
    }:
        raise PreflightError("training plan pipeline preparation changed")


def validate_ledger_authority(
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
        raise PreflightError("synthetic prerequisite is not passed")
    if prerequisites.get("sign_separated_residual_data_role_preflight") != "passed":
        raise PreflightError("data-role prerequisite is not passed")
    training_status = prerequisites.get("sign_separated_residual_training_preflight")
    if training_status not in {"pending", "passed"}:
        raise PreflightError("training preflight status must be pending or passed")

    records = [
        record
        for record in ledger.get("records", [])
        if isinstance(record, dict) and record.get("id") == DATA_ROLE_RECORD_ID
    ]
    if len(records) != 1:
        raise PreflightError("ledger requires exactly one data-role PASS record")
    record = records[0]
    if record.get("terminal") != "PASS" or record.get("outcome") != DATA_ROLE_OUTCOME:
        raise PreflightError("data-role record has wrong authority")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise PreflightError("data-role PASS record lacks evidence")
    validated_paths = [
        validate_artifact(repo_root, item, "data-role evidence")
        for item in evidence
    ]
    role_plan = repo_root / "docs/sign-separated-residual-data-roles.json"
    if role_plan not in validated_paths:
        raise PreflightError("data-role PASS record lacks the frozen role plan")
    return {
        "record_id": DATA_ROLE_RECORD_ID,
        "evidence_count": len(validated_paths),
        "training_preflight_ledger_status": training_status,
    }


def validate_plan_artifacts(
    repo_root: Path,
    plan: dict[str, Any],
    ledger: dict[str, Any],
) -> dict[str, str]:
    evidence = plan.get("evidence")
    if not isinstance(evidence, dict):
        raise PreflightError("training plan evidence is missing")
    validated = {
        name: validate_artifact(repo_root, artifact, f"evidence.{name}")
        for name, artifact in evidence.items()
    }
    expected_names = {
        "cleanup_model",
        "current_primary_checkpoint",
        "current_primary_config",
        "current_second_stage_checkpoint",
        "data_role_plan",
        "patch_builder",
        "primary_inference",
        "second_stage_inference",
        "trainer",
    }
    if set(validated) != expected_names:
        raise PreflightError("training plan evidence set changed")
    baseline = ledger.get("baseline", {})
    if evidence["current_primary_config"] != baseline.get("config"):
        raise PreflightError("training plan current-primary config changed")
    if evidence["current_primary_checkpoint"] != baseline.get("checkpoint"):
        raise PreflightError("training plan current-primary checkpoint changed")
    return {name: sha256_file(path) for name, path in validated.items()}


def validate_train_files(
    repo_root: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    data = plan["data"]
    role_plan = validate_artifact(
        repo_root, plan["evidence"]["data_role_plan"], "data role plan"
    )
    filenames = effective_train_filenames(repo_root, role_plan)
    if len(filenames) != data["effective_train_count"]:
        raise PreflightError("effective train count changed")
    filename_hash = sha256_rows(filenames)
    if filename_hash != data["effective_train_filename_sha256"]:
        raise PreflightError("effective train filename hash changed")
    domain_counts = {
        domain: sum(name.startswith(f"{domain}_") for name in filenames)
        for domain in ("hw5k", "scut")
    }
    if domain_counts != data["effective_train_domain_counts"]:
        raise PreflightError("effective train domain counts changed")
    source_dir = repo_path(repo_root, data["source_dir"], "source_dir")
    label_dir = repo_path(repo_root, data["label_dir"], "label_dir")
    missing_sources = [name for name in filenames if not (source_dir / name).is_file()]
    missing_labels = [name for name in filenames if not (label_dir / name).is_file()]
    if missing_sources:
        raise PreflightError(f"missing train source files: {missing_sources[:5]}")
    if missing_labels:
        raise PreflightError(f"missing train label files: {missing_labels[:5]}")
    return {
        "effective_train_count": len(filenames),
        "effective_train_filename_sha256": filename_hash,
        "domain_counts": domain_counts,
        "source_file_count": len(filenames),
        "label_file_count": len(filenames),
    }


def validate_training_cli(plan: dict[str, Any]) -> dict[str, Any]:
    trainer = plan["trainer"]
    parser = build_parser()
    args = parser.parse_args(
        [
            "--data-root",
            plan["data"]["data_root"],
            "--split",
            plan["data"]["split"],
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
    if MODEL_TYPE != trainer["model_type"] or MASK_SOURCE != trainer["mask_source"]:
        raise PreflightError("dedicated trainer identity/data boundary changed")
    option_names = set(parser._option_string_actions)
    forbidden = {"--model-type", "--mask-source", "--init-checkpoint"}
    if option_names & forbidden or any(
        option.startswith("--val-") for option in option_names
    ):
        raise PreflightError("dedicated trainer exposed a forbidden alternate path")
    return {
        "model_type": MODEL_TYPE,
        "mask_source": MASK_SOURCE,
        "validation_enabled": False,
        "max_steps": args.max_steps,
        "device": args.device,
    }


def synthetic_training_audit(plan: dict[str, Any]) -> dict[str, Any]:
    trainer = plan["trainer"]
    args = argparse.Namespace(
        sign_direction_margin=trainer["sign_direction_margin"],
        route_loss_weight=trainer["route_loss_weight"],
        bright_magnitude_weight=trainer["bright_magnitude_weight"],
        dark_magnitude_weight=trainer["dark_magnitude_weight"],
        identity_delta_weight=trainer["identity_delta_weight"],
    )
    cases = []
    for direction in (1, -1):
        torch.manual_seed(20260810)
        model = SignSeparatedResidualDeltaCleanupNet(
            residual_delta_bound=trainer["residual_delta_bound"]
        )
        inp = torch.full((1, 3, 8, 8), 0.5)
        with torch.no_grad():
            identity_exact = torch.equal(model(inp)[0], inp)
        target = inp + direction * 0.04
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        terms = compute_sign_separated_loss_terms(model, inp, target, args)
        terms["loss"].backward()
        bright_grad = float(
            model.bright_magnitude_head[-1].bias.grad.abs().sum()
        )
        dark_grad = float(model.dark_magnitude_head[-1].bias.grad.abs().sum())
        route_grad = float(model.route_head[-1].bias.grad.abs().sum())
        if direction > 0:
            matching_grad, opposite_grad = bright_grad, dark_grad
        else:
            matching_grad, opposite_grad = dark_grad, bright_grad
        if not identity_exact or matching_grad <= 0.0 or opposite_grad != 0.0:
            raise PreflightError(
                f"sign-separated training gradient contract failed: {direction}"
            )
        if route_grad <= 0.0:
            raise PreflightError("sign-separated route gradient is zero")
        optimizer.step()
        delta = (model(inp)[0] - inp).detach()
        opposed = int((delta < 0).sum()) if direction > 0 else int((delta > 0).sum())
        if opposed != 0 or float(delta.abs().max()) > 0.08 + 1e-7:
            raise PreflightError("sign-separated one-step direction contract failed")
        if float(delta.abs().max()) <= 0.0:
            raise PreflightError("sign-separated one-step update is a no-op")
        cases.append(
            {
                "direction": direction,
                "matching_magnitude_gradient": matching_grad,
                "opposite_magnitude_gradient": opposite_grad,
                "route_gradient": route_grad,
                "opposed_pixel_count": opposed,
                "delta_abs_max": float(delta.abs().max()),
            }
        )
    model = SignSeparatedResidualDeltaCleanupNet(
        residual_delta_bound=trainer["residual_delta_bound"]
    )
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_tensor_count": len(model.state_dict()),
        "exact_identity_init": True,
        "has_global_scale": any("global" in name for name, _ in model.named_parameters()),
        "gradient_cases": cases,
    }


def validate_outputs_absent(repo_root: Path, plan: dict[str, Any]) -> list[str]:
    outputs = plan["planned_outputs_must_be_absent"]
    absent = []
    for label, value in sorted(outputs.items()):
        path = repo_path(repo_root, value, f"planned_outputs.{label}")
        if path.exists():
            raise PreflightError(f"planned output must be absent: {path}")
        absent.append(str(path))
    return absent


def run_preflight(
    *,
    repo_root: Path = ROOT,
    training_plan_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_plan = training_plan_path or (repo_root / TRAINING_PLAN_PATH)
    resolved_ledger = ledger_path or (repo_root / LEDGER_PATH)
    try:
        plan = read_json(resolved_plan)
        ledger = read_json(resolved_ledger)
        assert_exact_plan(plan)
        authority = validate_ledger_authority(repo_root, ledger)
        artifact_hashes = validate_plan_artifacts(repo_root, plan, ledger)
        train_files = validate_train_files(repo_root, plan)
        cli = validate_training_cli(plan)
        synthetic = synthetic_training_audit(plan)
        if synthetic["parameter_count"] != 389253:
            raise PreflightError("sign-separated parameter count changed")
        if synthetic["state_tensor_count"] != 36:
            raise PreflightError("sign-separated state tensor count changed")
        if synthetic["has_global_scale"]:
            raise PreflightError("sign-separated model regained a global scale")
        if not torch.backends.mps.is_available():
            raise PreflightError("registered MPS training device is unavailable")
        absent_outputs = validate_outputs_absent(repo_root, plan)
    except (KeyError, OSError, PreflightError, TypeError, ValueError) as exc:
        return {
            "reason": str(exc),
            "runnable": False,
            "terminal": "PREREQUISITE_NEEDED",
        }
    return {
        "status": "pass",
        "terminal": "PASS",
        "runnable": True,
        "training_plan": str(resolved_plan),
        "training_plan_sha256": sha256_file(resolved_plan),
        "ledger_authority": authority,
        "artifact_hashes": artifact_hashes,
        "train_files": train_files,
        "training_cli": cli,
        "synthetic_training_audit": synthetic,
        "mps_available": True,
        "real_image_decode": False,
        "target_decode": False,
        "target_patch_materialized": False,
        "training_started": False,
        "prediction_artifacts_generated": False,
        "planned_outputs_absent": absent_outputs,
        "first_quality_gate": plan["first_quality_gate"],
        "later_gates_enabled": False,
        "promotion_enabled": False,
        "reserved_blind_state": "unavailable",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--training-plan", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = run_preflight(
        repo_root=args.repo_root,
        training_plan_path=args.training_plan,
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
