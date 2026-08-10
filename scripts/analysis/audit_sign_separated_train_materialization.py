#!/usr/bin/env python3
"""Audit the registered train275 materialization before training is allowed."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.build_sign_separated_residual_patch_index import (  # noqa: E402
    build_candidates,
    effective_train_filenames,
    select_direction_balanced,
    sha256_rows,
)
from scripts.analysis.materialize_sign_separated_train_inputs import (  # noqa: E402
    CONTROL_DIR,
    STAGES,
    assert_manifest,
    patch_command,
    primary_command,
    repo_path,
    second_stage_command,
    validate_authority,
)
from scripts.analysis.validate_sign_separated_training_preflight import (  # noqa: E402
    LEDGER_PATH,
    TRAINING_PLAN_PATH,
    assert_exact_plan,
    read_json,
    sha256_file,
    validate_plan_artifacts,
)


class AuditError(RuntimeError):
    pass


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise AuditError(f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AuditError(f"empty CSV: {path}")
    return rows


def expected_prediction_names(filenames: list[str]) -> list[str]:
    names = sorted(f"{Path(name).stem}.png" for name in filenames)
    if len(names) != len(set(names)):
        raise AuditError("effective train filenames collide after PNG conversion")
    return names


def validate_prediction_set(
    prediction_dir: Path,
    expected_names: list[str],
) -> dict[str, Any]:
    if not prediction_dir.is_dir():
        raise AuditError(f"missing prediction directory: {prediction_dir}")
    actual_names = sorted(
        path.name for path in prediction_dir.iterdir() if path.is_file()
    )
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))[:5]
        extra = sorted(set(actual_names) - set(expected_names))[:5]
        raise AuditError(
            f"prediction set changed: missing={missing} extra={extra}"
        )
    hashes = [f"{name} {sha256_file(prediction_dir / name)}" for name in actual_names]
    return {
        "count": len(actual_names),
        "filename_sha256": sha256_rows(actual_names),
        "content_sha256": sha256_rows(hashes),
    }


def rows_by_file(
    rows: list[dict[str, str]],
    expected_files: list[str],
    label: str,
) -> dict[str, dict[str, str]]:
    mapped: dict[str, dict[str, str]] = {}
    for row in rows:
        file_name = row.get("file", "")
        if not file_name or file_name in mapped:
            raise AuditError(f"{label} has missing or duplicate file row: {file_name}")
        mapped[file_name] = row
    if set(mapped) != set(expected_files):
        raise AuditError(f"{label} file identities changed")
    return mapped


def expected_auto_threshold(coverage: float) -> int:
    if coverage <= 0.129:
        return 8
    if coverage <= 0.421:
        return 76
    return 160


def validate_primary_metrics(
    *,
    repo_root: Path,
    plan: dict[str, Any],
    filenames: list[str],
    manifest_rows: list[str],
    primary_dir: Path,
) -> dict[str, Any]:
    rows = rows_by_file(
        read_csv_rows(primary_dir / "metrics.csv"), filenames, "primary metrics"
    )
    config = plan["pipeline_preparation"]["primary"]
    config_hash = plan["evidence"]["current_primary_config"]["sha256"]
    checkpoint_hash = plan["evidence"]["current_primary_checkpoint"]["sha256"]
    manifest_by_name = {Path(value).name: value for value in manifest_rows}
    for file_name, row in rows.items():
        expected_prediction = primary_dir / "pred" / f"{Path(file_name).stem}.png"
        source = repo_root / manifest_by_name[file_name]
        exact = {
            "image_path": manifest_by_name[file_name],
            "metrics_skipped": "1",
            "primary_config_sha256": config_hash,
            "primary_weights_sha256": checkpoint_hash,
            "page_overlap": str(config["page_overlap"]),
            "batch_size": str(config["batch_size"]),
            "copy_input_outside_mask": config["copy_input_outside_mask"],
            "copy_mask_threshold_auto": config["copy_mask_threshold_auto"],
            "copy_mask_dilate": str(config["copy_mask_dilate"]),
            "image_sha256": sha256_file(source),
            "pred_path": str(expected_prediction),
            "pred_sha256": sha256_file(expected_prediction),
        }
        for key, expected in exact.items():
            if row.get(key) != expected:
                raise AuditError(f"primary metrics {key} changed for {file_name}")
        coverage = float(row["copy_mask_cov8"])
        if int(row["copy_mask_threshold"]) != expected_auto_threshold(coverage):
            raise AuditError(f"primary auto threshold changed for {file_name}")
    return {
        "row_count": len(rows),
        "metrics_sha256": sha256_file(primary_dir / "metrics.csv"),
    }


def validate_second_stage_metrics(
    *,
    plan: dict[str, Any],
    filenames: list[str],
    second_stage_dir: Path,
) -> dict[str, Any]:
    rows = rows_by_file(
        read_csv_rows(second_stage_dir / "metrics.csv"),
        filenames,
        "second-stage metrics",
    )
    config = plan["pipeline_preparation"]["second_stage"]
    for file_name, row in rows.items():
        expected_prediction = (
            second_stage_dir / "pred" / f"{Path(file_name).stem}.png"
        )
        exact = {
            "pred_path": str(expected_prediction),
            "base_edit_threshold": str(float(config["base_edit_threshold"])),
            "second_delta_threshold": str(
                float(config["second_delta_threshold"])
            ),
            "dark_threshold": str(config["dark_threshold"]),
        }
        for key, expected in exact.items():
            if row.get(key) != expected:
                raise AuditError(
                    f"second-stage metrics {key} changed for {file_name}"
                )
        gate_ratio = float(row["gate_ratio"])
        if not 0.0 <= gate_ratio <= 1.0:
            raise AuditError(f"invalid second-stage gate ratio for {file_name}")
        if sha256_file(expected_prediction) == "":
            raise AuditError(f"unhashable second-stage prediction: {file_name}")
    return {
        "row_count": len(rows),
        "metrics_sha256": sha256_file(second_stage_dir / "metrics.csv"),
    }


def expected_stage_command(
    *,
    repo_root: Path,
    plan: dict[str, Any],
    stage: str,
) -> list[str]:
    outputs = plan["planned_outputs_must_be_absent"]
    if stage == "manifest":
        return []
    if stage == "primary":
        final = repo_path(repo_root, outputs["primary_prediction_dir"])
        temporary = final.with_name(f".{final.name}.materializing")
        return primary_command(repo_root, plan, temporary)
    if stage == "second_stage":
        final = repo_path(repo_root, outputs["training_input_dir"])
        temporary = final.with_name(f".{final.name}.materializing")
        return second_stage_command(repo_root, plan, temporary)
    if stage == "patch_index":
        final_csv = repo_path(repo_root, plan["patch_builder"]["output_csv"])
        final_json = repo_path(repo_root, plan["patch_builder"]["output_json"])
        temporary_csv = final_csv.with_name(f".{final_csv.name}.materializing")
        temporary_summary = final_json.parent.with_name(
            f".{final_json.parent.name}.materializing"
        ) / final_json.name
        return patch_command(
            repo_root, plan, temporary_csv, temporary_summary
        )
    raise AuditError(f"unknown materialization stage: {stage}")


def validate_stage_records(
    repo_root: Path,
    plan: dict[str, Any],
    plan_path: Path,
) -> dict[str, str]:
    control_dir = repo_root / CONTROL_DIR
    hashes: dict[str, str] = {}
    for stage in STAGES:
        record_path = control_dir / f"{stage}.json"
        record = read_json(record_path)
        if record.get("terminal") != "PASS" or record.get("stage") != stage:
            raise AuditError(f"materialization stage record failed: {stage}")
        if record.get("training_plan_sha256") != sha256_file(plan_path):
            raise AuditError(f"materialization stage plan hash changed: {stage}")
        actual_command = record.get("command")
        expected_command = expected_stage_command(
            repo_root=repo_root, plan=plan, stage=stage
        )
        if not isinstance(actual_command, list):
            raise AuditError(f"materialization command missing: {stage}")
        if actual_command[1:] != expected_command[1:]:
            raise AuditError(f"materialization command changed: {stage}")
        evidence = record.get("outputs")
        if not isinstance(evidence, list) or not evidence:
            raise AuditError(f"materialization stage lacks outputs: {stage}")
        for artifact in evidence:
            if not isinstance(artifact, dict):
                raise AuditError(f"malformed stage output: {stage}")
            path = repo_path(repo_root, artifact.get("path", ""))
            expected_hash = artifact.get("sha256")
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise AuditError(f"materialization stage output drift: {path}")
        hashes[stage] = sha256_file(record_path)
    return hashes


PATCH_STRING_FIELDS = {"file", "selected_for"}
PATCH_INTEGER_FIELDS = {"x1", "y1", "x2", "y2"}
PATCH_FLOAT_FIELDS = {
    "rank_score",
    "brighten_ratio",
    "darken_ratio",
    "brighten_mean_delta",
    "darken_mean_delta",
    "brighten_score",
    "darken_score",
}


def compare_selected_rows(
    actual: list[dict[str, str]],
    expected: list[dict[str, Any]],
) -> None:
    if len(actual) != len(expected):
        raise AuditError("patch selection count changed")
    for index, (actual_row, expected_row) in enumerate(zip(actual, expected)):
        for key in PATCH_STRING_FIELDS:
            if actual_row.get(key) != str(expected_row[key]):
                raise AuditError(f"patch row {index} {key} changed")
        for key in PATCH_INTEGER_FIELDS:
            if int(actual_row[key]) != int(expected_row[key]):
                raise AuditError(f"patch row {index} {key} changed")
        for key in PATCH_FLOAT_FIELDS:
            if not math.isclose(
                float(actual_row[key]),
                float(expected_row[key]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise AuditError(f"patch row {index} {key} changed")


def validate_patch_index(
    *,
    repo_root: Path,
    plan: dict[str, Any],
    filenames: list[str],
    patch_csv: Path,
    patch_json: Path,
) -> dict[str, Any]:
    config = plan["patch_builder"]
    input_dir = repo_path(repo_root, config["input_dir"])
    label_dir = repo_path(repo_root, plan["data"]["label_dir"])
    candidates, content_hashes = build_candidates(
        filenames=filenames,
        label_dir=label_dir,
        input_dir=input_dir,
        tile_size=config["tile_size"],
        overlap=config["overlap"],
        direction_margin=config["direction_margin"],
        min_support_ratio=config["min_support_ratio"],
    )
    expected = select_direction_balanced(
        candidates, config["top_k_per_direction"]
    )
    actual = read_csv_rows(patch_csv)
    compare_selected_rows(actual, expected)
    summary = read_json(patch_json)
    selected_counts = {
        direction: sum(direction in row["selected_for"] for row in actual)
        for direction in ("brighten", "darken")
    }
    exact = {
        "status": "pass",
        "terminal": "PASS",
        "train_role_count": len(filenames),
        "train_role_sha256": sha256_rows(filenames),
        "candidate_count": len(candidates),
        "patch_count": len(actual),
        "page_count": len({row["file"] for row in actual}),
        "selected_counts": selected_counts,
        "patch_index": str(patch_csv),
        "patch_index_sha256": sha256_file(patch_csv),
        **content_hashes,
    }
    if summary != exact:
        raise AuditError("patch summary changed")
    top_k = config["top_k_per_direction"]
    if selected_counts != {"brighten": top_k, "darken": top_k}:
        raise AuditError("patch index lacks registered directional support")
    return exact


def run_audit(
    *,
    repo_root: Path = ROOT,
    training_plan_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    plan_path = training_plan_path or (repo_root / TRAINING_PLAN_PATH)
    resolved_ledger = ledger_path or (repo_root / LEDGER_PATH)
    try:
        plan = read_json(plan_path)
        ledger = read_json(resolved_ledger)
        assert_exact_plan(plan)
        authority = validate_authority(repo_root, ledger)
        artifact_hashes = validate_plan_artifacts(repo_root, plan, ledger)
        filenames = effective_train_filenames(
            repo_root,
            repo_path(repo_root, plan["evidence"]["data_role_plan"]["path"]),
        )
        outputs = plan["planned_outputs_must_be_absent"]
        manifest = repo_path(repo_root, outputs["sample_manifest"])
        manifest_rows = assert_manifest(repo_root, plan, manifest)
        expected_names = expected_prediction_names(filenames)

        primary_dir = repo_path(repo_root, outputs["primary_prediction_dir"])
        primary_predictions = validate_prediction_set(
            primary_dir / "pred", expected_names
        )
        primary_metrics = validate_primary_metrics(
            repo_root=repo_root,
            plan=plan,
            filenames=filenames,
            manifest_rows=manifest_rows,
            primary_dir=primary_dir,
        )
        second_stage_dir = repo_path(repo_root, outputs["training_input_dir"])
        second_stage_predictions = validate_prediction_set(
            second_stage_dir / "pred", expected_names
        )
        second_stage_metrics = validate_second_stage_metrics(
            plan=plan,
            filenames=filenames,
            second_stage_dir=second_stage_dir,
        )
        patch_csv_value = plan["patch_builder"]["output_csv"]
        patch_summary = validate_patch_index(
            repo_root=repo_root,
            plan=plan,
            filenames=filenames,
            patch_csv=repo_path(repo_root, patch_csv_value),
            patch_json=repo_path(repo_root, plan["patch_builder"]["output_json"]),
        )
        if patch_csv_value != outputs["patch_index"]:
            raise AuditError("patch output registration changed")
        stage_records = validate_stage_records(repo_root, plan, plan_path)
        forbidden = {
            "training_output_dir": repo_path(
                repo_root, outputs["training_output_dir"]
            ),
            "first_gate_output_dir": repo_path(
                repo_root, outputs["first_gate_output_dir"]
            ),
        }
        existing_forbidden = [
            label for label, path in forbidden.items() if path.exists()
        ]
        if existing_forbidden:
            raise AuditError(
                f"training or quality gate started early: {existing_forbidden}"
            )
    except (AuditError, KeyError, OSError, TypeError, ValueError) as exc:
        return {
            "terminal": "PREREQUISITE_NEEDED",
            "runnable": False,
            "reason": str(exc),
        }
    return {
        "status": "pass",
        "terminal": "PASS",
        "runnable": True,
        "authority": authority,
        "training_plan": str(plan_path),
        "training_plan_sha256": sha256_file(plan_path),
        "artifact_hashes": artifact_hashes,
        "train_count": len(filenames),
        "train_filename_sha256": sha256_rows(filenames),
        "sample_manifest_sha256": sha256_file(manifest),
        "primary_predictions": primary_predictions,
        "primary_metrics": primary_metrics,
        "second_stage_predictions": second_stage_predictions,
        "second_stage_metrics": second_stage_metrics,
        "patch_summary": patch_summary,
        "stage_record_hashes": stage_records,
        "training_started": False,
        "checkpoint_generated": False,
        "first_quality_gate_started": False,
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
    result = run_audit(
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
