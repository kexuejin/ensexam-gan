#!/usr/bin/env python3
"""Materialize the registered train275 inputs one atomic stage at a time."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.build_sign_separated_residual_patch_index import (  # noqa: E402
    effective_train_filenames,
    sha256_rows,
)
from scripts.analysis.validate_sign_separated_training_preflight import (  # noqa: E402
    LEDGER_PATH,
    TRAINING_PLAN_PATH,
    assert_exact_plan,
    read_json,
    sha256_file,
    validate_artifact,
    validate_ledger_authority,
    validate_plan_artifacts,
)


CONTROL_DIR = Path(
    "outputs/sign-separated-residual-repair-train275-materialization-v1"
)
TRAINING_RECORD_ID = "sign-separated-residual-training-preflight"
TRAINING_RECORD_OUTCOME = (
    "dedicated_train275_identity_initialized_training_path_passed_without_pixel_decode"
)
MATERIALIZATION_PREREQUISITE_ID = (
    "sign_separated_residual_train_materialization_audit"
)
STAGES = ("manifest", "primary", "second_stage", "patch_index")


def repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"registered path must stay inside repository: {value}")
    return repo_root / path


def validate_authority(
    repo_root: Path,
    ledger: dict[str, Any],
) -> dict[str, str]:
    validate_ledger_authority(repo_root, ledger)
    active = ledger.get("active_iteration", {})
    prerequisites = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if prerequisites.get("sign_separated_residual_training_preflight") != "passed":
        raise ValueError("training preflight is not passed")
    materialization_status = prerequisites.get(MATERIALIZATION_PREREQUISITE_ID)
    if materialization_status not in {"pending", "passed"}:
        raise ValueError("materialization audit is not registered")
    records = [
        record
        for record in ledger.get("records", [])
        if isinstance(record, dict) and record.get("id") == TRAINING_RECORD_ID
    ]
    if len(records) != 1:
        raise ValueError("exactly one training preflight PASS record is required")
    record = records[0]
    if (
        record.get("terminal") != "PASS"
        or record.get("outcome") != TRAINING_RECORD_OUTCOME
    ):
        raise ValueError("training preflight record has wrong authority")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("training preflight record lacks evidence")
    for item in evidence:
        validate_artifact(repo_root, item, "training preflight evidence")
    return {
        "training_preflight": "passed",
        "materialization_audit": materialization_status,
    }


def expected_manifest_rows(
    repo_root: Path,
    plan: dict[str, Any],
) -> list[str]:
    role_plan = validate_artifact(
        repo_root,
        plan["evidence"]["data_role_plan"],
        "data role plan",
    )
    filenames = effective_train_filenames(repo_root, role_plan)
    if len(filenames) != plan["data"]["effective_train_count"]:
        raise ValueError("effective train count changed")
    if sha256_rows(filenames) != plan["data"]["effective_train_filename_sha256"]:
        raise ValueError("effective train filename hash changed")
    source_dir = Path(plan["data"]["source_dir"])
    rows = [(source_dir / name).as_posix() for name in filenames]
    missing = [row for row in rows if not (repo_root / row).is_file()]
    if missing:
        raise FileNotFoundError(f"missing train sources: {missing[:5]}")
    return rows


def assert_manifest(
    repo_root: Path,
    plan: dict[str, Any],
    manifest_path: Path,
) -> list[str]:
    expected = expected_manifest_rows(repo_root, plan)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    actual = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if actual != expected:
        raise ValueError("materialized train manifest changed")
    return actual


def atomic_write_manifest(path: Path, rows: list[str]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.materializing")
    if temporary.exists():
        raise FileExistsError(f"stale manifest materialization: {temporary}")
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_atomic_directory_command(
    *,
    repo_root: Path,
    final_dir: Path,
    command_builder,
    log_path: Path,
) -> list[str]:
    if final_dir.exists():
        raise FileExistsError(final_dir)
    temporary = final_dir.with_name(f".{final_dir.name}.materializing")
    if temporary.exists():
        raise FileExistsError(f"stale materialization directory: {temporary}")
    command = command_builder(temporary)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise FileExistsError(log_path)
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"materialization command failed ({result.returncode}); see {log_path}"
        )
    if not temporary.is_dir():
        raise RuntimeError(f"command did not create expected directory: {temporary}")
    temporary.replace(final_dir)
    rewrite_metrics_paths(final_dir / "metrics.csv", temporary, final_dir)
    return command


def rewrite_metrics_paths(
    metrics_path: Path,
    temporary_dir: Path,
    final_dir: Path,
) -> None:
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"metrics CSV lacks a header: {metrics_path}")
    old = str(temporary_dir)
    new = str(final_dir)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, str) and old in value:
                row[key] = value.replace(old, new)
    rewritten = metrics_path.with_name(f".{metrics_path.name}.rewriting")
    with rewritten.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    rewritten.replace(metrics_path)


def primary_command(
    repo_root: Path,
    plan: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    config = plan["pipeline_preparation"]["primary"]
    return [
        sys.executable,
        str(repo_root / plan["evidence"]["primary_inference"]["path"]),
        "--samples-file",
        config["samples_file"],
        "--output-dir",
        str(output_dir),
        "--primary-config",
        plan["evidence"]["current_primary_config"]["path"],
        "--primary-weights",
        plan["evidence"]["current_primary_checkpoint"]["path"],
        "--device",
        config["device"],
        "--page-overlap",
        str(config["page_overlap"]),
        "--batch-size",
        str(config["batch_size"]),
        "--copy-input-outside-mask",
        config["copy_input_outside_mask"],
        "--copy-mask-threshold",
        str(config["copy_mask_threshold"]),
        "--copy-mask-threshold-auto",
        config["copy_mask_threshold_auto"],
        "--copy-mask-dilate",
        str(config["copy_mask_dilate"]),
        "--skip-label-metrics",
    ]


def second_stage_command(
    repo_root: Path,
    plan: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    config = plan["pipeline_preparation"]["second_stage"]
    return [
        sys.executable,
        str(repo_root / plan["evidence"]["second_stage_inference"]["path"]),
        "--samples-file",
        config["samples_file"],
        "--output-dir",
        str(output_dir),
        "--primary-pred-dir",
        config["primary_pred_dir"],
        "--cleanup-checkpoint",
        plan["evidence"]["current_second_stage_checkpoint"]["path"],
        "--device",
        config["device"],
        "--cleanup-tile-size",
        str(config["cleanup_tile_size"]),
        "--cleanup-stride",
        str(config["cleanup_stride"]),
        "--cleanup-alpha-threshold",
        str(config["cleanup_alpha_threshold"]),
        "--base-edit-threshold",
        str(config["base_edit_threshold"]),
        "--second-delta-threshold",
        str(config["second_delta_threshold"]),
        "--dark-threshold",
        str(config["dark_threshold"]),
        "--change-threshold",
        str(config["change_threshold"]),
        "--eval-threshold",
        str(config["eval_threshold"]),
    ]


def patch_command(
    repo_root: Path,
    plan: dict[str, Any],
    temporary_csv: Path,
    temporary_json: Path,
) -> list[str]:
    config = plan["patch_builder"]
    return [
        sys.executable,
        str(repo_root / plan["evidence"]["patch_builder"]["path"]),
        "--repo-root",
        str(repo_root),
        "--role-plan",
        plan["evidence"]["data_role_plan"]["path"],
        "--data-root",
        plan["data"]["data_root"],
        "--input-dir",
        config["input_dir"],
        "--output-csv",
        str(temporary_csv),
        "--output-json",
        str(temporary_json),
        "--tile-size",
        str(config["tile_size"]),
        "--overlap",
        str(config["overlap"]),
        "--direction-margin",
        str(config["direction_margin"]),
        "--min-support-ratio",
        str(config["min_support_ratio"]),
        "--top-k-per-direction",
        str(config["top_k_per_direction"]),
    ]


def write_stage_record(
    *,
    repo_root: Path,
    plan_path: Path,
    stage: str,
    command: list[str],
    outputs: list[Path],
) -> None:
    control_dir = repo_root / CONTROL_DIR
    control_dir.mkdir(parents=True, exist_ok=True)
    record_path = control_dir / f"{stage}.json"
    if record_path.exists():
        raise FileExistsError(record_path)
    record = {
        "terminal": "PASS",
        "stage": stage,
        "training_plan": str(plan_path),
        "training_plan_sha256": sha256_file(plan_path),
        "command": command,
        "outputs": [
            {
                "path": str(path.relative_to(repo_root)),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
            for path in outputs
        ],
    }
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_stage(
    *,
    repo_root: Path,
    plan: dict[str, Any],
    plan_path: Path,
    stage: str,
) -> None:
    outputs = plan["planned_outputs_must_be_absent"]
    manifest = repo_path(repo_root, outputs["sample_manifest"])
    control_dir = repo_root / CONTROL_DIR
    if stage == "manifest":
        rows = expected_manifest_rows(repo_root, plan)
        atomic_write_manifest(manifest, rows)
        write_stage_record(
            repo_root=repo_root,
            plan_path=plan_path,
            stage=stage,
            command=[],
            outputs=[manifest],
        )
        return

    assert_manifest(repo_root, plan, manifest)
    if stage == "primary":
        final_dir = repo_path(repo_root, outputs["primary_prediction_dir"])
        command = run_atomic_directory_command(
            repo_root=repo_root,
            final_dir=final_dir,
            command_builder=lambda temporary: primary_command(
                repo_root, plan, temporary
            ),
            log_path=control_dir / "primary.log",
        )
        stage_outputs = [final_dir / "metrics.csv"]
    elif stage == "second_stage":
        primary_dir = repo_path(repo_root, outputs["primary_prediction_dir"])
        if not (primary_dir / "metrics.csv").is_file():
            raise FileNotFoundError("primary materialization is incomplete")
        final_dir = repo_path(repo_root, outputs["training_input_dir"])
        command = run_atomic_directory_command(
            repo_root=repo_root,
            final_dir=final_dir,
            command_builder=lambda temporary: second_stage_command(
                repo_root, plan, temporary
            ),
            log_path=control_dir / "second_stage.log",
        )
        stage_outputs = [final_dir / "metrics.csv"]
    elif stage == "patch_index":
        training_input = repo_path(repo_root, outputs["training_input_dir"])
        if not (training_input / "metrics.csv").is_file():
            raise FileNotFoundError("second-stage materialization is incomplete")
        final_csv = repo_path(repo_root, plan["patch_builder"]["output_csv"])
        final_json = repo_path(repo_root, plan["patch_builder"]["output_json"])
        if final_csv.exists() or final_json.exists():
            raise FileExistsError("registered patch outputs already exist")
        temporary_csv = final_csv.with_name(f".{final_csv.name}.materializing")
        temporary_summary_dir = final_json.parent.with_name(
            f".{final_json.parent.name}.materializing"
        )
        temporary_json = temporary_summary_dir / final_json.name
        if temporary_csv.exists() or temporary_summary_dir.exists():
            raise FileExistsError("stale patch materialization output")
        command = patch_command(
            repo_root, plan, temporary_csv, temporary_json
        )
        log_path = control_dir / "patch_index.log"
        if log_path.exists():
            raise FileExistsError(log_path)
        with log_path.open("x", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=repo_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"patch materialization failed ({result.returncode}); see {log_path}"
            )
        final_csv.parent.mkdir(parents=True, exist_ok=True)
        temporary_csv.replace(final_csv)
        temporary_summary_dir.replace(final_json.parent)
        summary = read_json(final_json)
        summary["patch_index"] = str(final_csv)
        summary["patch_index_sha256"] = sha256_file(final_csv)
        final_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage_outputs = [final_csv, final_json]
    else:
        raise ValueError(f"unsupported materialization stage: {stage}")

    write_stage_record(
        repo_root=repo_root,
        plan_path=plan_path,
        stage=stage,
        command=command,
        outputs=stage_outputs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--training-plan", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--stage", choices=(*STAGES, "all"), required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    plan_path = args.training_plan or (repo_root / TRAINING_PLAN_PATH)
    ledger_path = args.ledger or (repo_root / LEDGER_PATH)
    plan = read_json(plan_path)
    ledger = read_json(ledger_path)
    assert_exact_plan(plan)
    validate_authority(repo_root, ledger)
    validate_plan_artifacts(repo_root, plan, ledger)
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"materializing stage={stage}", flush=True)
        run_stage(
            repo_root=repo_root,
            plan=plan,
            plan_path=plan_path,
            stage=stage,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
