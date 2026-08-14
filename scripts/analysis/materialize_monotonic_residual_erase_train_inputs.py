#!/usr/bin/env python3
"""Materialize registered monotonic train275 inputs in atomic stages."""

from __future__ import annotations

import argparse
import json
import os
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
from scripts.analysis.validate_monotonic_residual_erase_training_preflight import (  # noqa: E402
    LEDGER_PATH,
    TRAINING_PLAN_PATH,
    assert_exact_plan,
    read_json,
    sha256_file,
    validate_artifact,
    validate_plan_artifacts,
)


CONTROL_DIR = Path("outputs/monotonic-residual-erase-train275-materialization-v1")
ARCHIVE_AUDIT = Path(
    "outputs/sign-separated-residual-repair-train275-materialization-audit-20260810/audit.json"
)
ARCHIVE_AUDIT_SHA256 = (
    "d91c3ec25e5ec861aa38a05dbd27448f981befc70c814d4a60ae8f972f5ca741"
)
ARCHIVE_MANIFEST = Path(
    "hardcase_lists/archive/sign-separated-residual-repair-20260810-train275-v1.txt"
)
ARCHIVE_PRIMARY_DIR = Path(
    "outputs/archive/sign-separated-residual-repair-20260810/train275-primary"
)
ARCHIVE_SECOND_STAGE_DIR = Path(
    "outputs/archive/sign-separated-residual-repair-20260810/train275-frozen-pipeline"
)
TRAINING_RECORD_ID = "monotonic-residual-erase-training-preflight"
TRAINING_RECORD_OUTCOME = (
    "class_balanced_monotonic_train275_configuration_passed_without_pixel_decode"
)
MATERIALIZATION_PREREQUISITE_ID = (
    "monotonic_residual_erase_train_materialization_audit"
)
STAGES = ("manifest", "primary", "second_stage", "patch_index")


def repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"registered path must stay inside repository: {value}")
    return repo_root / path


def validate_authority(
    repo_root: Path, ledger: dict[str, Any]
) -> dict[str, str]:
    active = ledger.get("active_iteration", {})
    if active.get("id") != "monotonic-residual-erase-support":
        raise ValueError("active iteration is not monotonic residual erase")
    prerequisites = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if prerequisites.get("monotonic_residual_erase_training_preflight") != "passed":
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
        raise ValueError("exactly one monotonic training PASS record is required")
    record = records[0]
    if (
        record.get("terminal") != "PASS"
        or record.get("outcome") != TRAINING_RECORD_OUTCOME
    ):
        raise ValueError("monotonic training record has wrong authority")
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("monotonic training PASS record lacks evidence")
    for item in evidence:
        validate_artifact(repo_root, item, "monotonic training evidence")
    return {
        "training_preflight": "passed",
        "materialization_audit": materialization_status,
    }


def expected_manifest_rows(
    repo_root: Path, plan: dict[str, Any]
) -> list[str]:
    role_plan = validate_artifact(
        repo_root,
        plan["evidence"]["base_role_contract"],
        "base role contract",
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
    repo_root: Path, plan: dict[str, Any], manifest_path: Path
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
        raise ValueError("materialized monotonic train manifest changed")
    return actual


def validate_reuse_source(
    repo_root: Path, plan: dict[str, Any]
) -> dict[str, Any]:
    audit_path = repo_root / ARCHIVE_AUDIT
    if sha256_file(audit_path) != ARCHIVE_AUDIT_SHA256:
        raise ValueError("archived materialization audit hash changed")
    audit = read_json(audit_path)
    if audit.get("terminal") != "PASS" or audit.get("train_count") != 275:
        raise ValueError("archived materialization is not a train275 PASS")
    if audit.get("train_filename_sha256") != plan["data"][
        "effective_train_filename_sha256"
    ]:
        raise ValueError("archived train role differs from monotonic plan")
    artifact_hashes = audit.get("artifact_hashes", {})
    expected_artifacts = {
        "current_primary_checkpoint": plan["evidence"][
            "current_primary_checkpoint"
        ]["sha256"],
        "current_primary_config": plan["evidence"]["current_primary_config"][
            "sha256"
        ],
        "current_second_stage_checkpoint": plan["evidence"][
            "current_second_stage_checkpoint"
        ]["sha256"],
        "primary_inference": plan["evidence"]["primary_inference"]["sha256"],
        "second_stage_inference": plan["evidence"]["second_stage_inference"][
            "sha256"
        ],
    }
    for name, expected in expected_artifacts.items():
        if artifact_hashes.get(name) != expected:
            raise ValueError(f"archived pipeline artifact changed: {name}")
    if sha256_file(repo_root / ARCHIVE_MANIFEST) != audit.get(
        "sample_manifest_sha256"
    ):
        raise ValueError("archived manifest hash changed")
    primary_metrics = repo_root / ARCHIVE_PRIMARY_DIR / "metrics.csv"
    second_metrics = repo_root / ARCHIVE_SECOND_STAGE_DIR / "metrics.csv"
    if sha256_file(primary_metrics) != audit.get("primary_metrics", {}).get(
        "metrics_sha256"
    ):
        raise ValueError("archived primary metrics changed")
    if sha256_file(second_metrics) != audit.get("second_stage_metrics", {}).get(
        "metrics_sha256"
    ):
        raise ValueError("archived second-stage metrics changed")
    return {
        "audit_path": str(audit_path),
        "audit_sha256": ARCHIVE_AUDIT_SHA256,
        "manifest_sha256": audit["sample_manifest_sha256"],
        "primary_predictions": audit["primary_predictions"],
        "second_stage_predictions": audit["second_stage_predictions"],
        "primary_metrics_sha256": audit["primary_metrics"]["metrics_sha256"],
        "second_stage_metrics_sha256": audit["second_stage_metrics"][
            "metrics_sha256"
        ],
    }


def atomic_write_manifest(path: Path, rows: list[str]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.materializing")
    if temporary.exists():
        raise FileExistsError(f"stale manifest materialization: {temporary}")
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_relative_symlink(final_path: Path, source_path: Path) -> str:
    if final_path.exists() or final_path.is_symlink():
        raise FileExistsError(final_path)
    if not source_path.is_dir():
        raise FileNotFoundError(source_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.name}.materializing")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"stale symlink materialization: {temporary}")
    target = os.path.relpath(source_path, start=final_path.parent)
    temporary.symlink_to(target, target_is_directory=True)
    temporary.replace(final_path)
    return target


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
        "--repo-root", str(repo_root),
        "--role-plan", plan["evidence"]["base_role_contract"]["path"],
        "--data-root", plan["data"]["data_root"],
        "--input-dir", config["input_dir"],
        "--output-csv", str(temporary_csv),
        "--output-json", str(temporary_json),
        "--tile-size", str(config["tile_size"]),
        "--overlap", str(config["overlap"]),
        "--luminance-margin-gray", str(config["luminance_margin_gray"]),
        "--min-positive-ratio", str(config["min_positive_ratio"]),
        "--top-k", str(config["top_k"]),
    ]


def write_stage_record(
    *,
    repo_root: Path,
    plan_path: Path,
    stage: str,
    command: list[str],
    outputs: list[Path],
    reuse_source: Path | None = None,
    link_target: str | None = None,
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
                "sha256": sha256_file(path),
            }
            for path in outputs
        ],
        "reuse_source": (
            str(reuse_source.relative_to(repo_root)) if reuse_source else None
        ),
        "link_target": link_target,
    }
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_stage(
    *, repo_root: Path, plan: dict[str, Any], plan_path: Path, stage: str
) -> None:
    outputs = plan["planned_outputs_must_be_absent"]
    manifest = repo_path(repo_root, outputs["sample_manifest"])
    if stage == "manifest":
        rows = expected_manifest_rows(repo_root, plan)
        archive_rows = [
            line.strip()
            for line in (repo_root / ARCHIVE_MANIFEST)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if archive_rows != rows:
            raise ValueError("archived manifest content differs from exact train275")
        atomic_write_manifest(manifest, rows)
        write_stage_record(
            repo_root=repo_root,
            plan_path=plan_path,
            stage=stage,
            command=["write-exact-manifest"],
            outputs=[manifest],
            reuse_source=repo_root / ARCHIVE_MANIFEST,
        )
        return

    assert_manifest(repo_root, plan, manifest)
    if stage in {"primary", "second_stage"}:
        key = "primary_prediction_dir" if stage == "primary" else "training_input_dir"
        source = (
            repo_root / ARCHIVE_PRIMARY_DIR
            if stage == "primary"
            else repo_root / ARCHIVE_SECOND_STAGE_DIR
        )
        final = repo_path(repo_root, outputs[key])
        link_target = atomic_relative_symlink(final, source)
        command = [
            "reuse-frozen-output",
            str(source.relative_to(repo_root)),
            str(final.relative_to(repo_root)),
        ]
        write_stage_record(
            repo_root=repo_root,
            plan_path=plan_path,
            stage=stage,
            command=command,
            outputs=[final / "metrics.csv"],
            reuse_source=source,
            link_target=link_target,
        )
        return

    if stage != "patch_index":
        raise ValueError(f"unsupported materialization stage: {stage}")
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
    command = patch_command(repo_root, plan, temporary_csv, temporary_json)
    log_path = repo_root / CONTROL_DIR / "patch_index.log"
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
    write_stage_record(
        repo_root=repo_root,
        plan_path=plan_path,
        stage=stage,
        command=command,
        outputs=[final_csv, final_json],
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
    validate_reuse_source(repo_root, plan)
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
