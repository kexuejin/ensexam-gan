#!/usr/bin/env python3
"""Audit monotonic train275 materialization before training is allowed."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.audit_sign_separated_train_materialization import (  # noqa: E402
    expected_auto_threshold,
    expected_prediction_names,
    read_csv_rows,
    rows_by_file,
    validate_prediction_set,
)
from scripts.analysis.build_monotonic_residual_erase_patch_index import (  # noqa: E402
    build_candidates,
    select_top_brighten,
)
from scripts.analysis.build_sign_separated_residual_patch_index import (  # noqa: E402
    effective_train_filenames,
    sha256_rows,
)
from scripts.analysis.materialize_monotonic_residual_erase_train_inputs import (  # noqa: E402
    ARCHIVE_PRIMARY_DIR,
    ARCHIVE_SECOND_STAGE_DIR,
    CONTROL_DIR,
    STAGES,
    assert_manifest,
    patch_command,
    repo_path,
    validate_authority,
    validate_reuse_source,
)
from scripts.analysis.validate_monotonic_residual_erase_training_preflight import (  # noqa: E402
    LEDGER_PATH,
    TRAINING_PLAN_PATH,
    assert_exact_plan,
    read_json,
    sha256_file,
    validate_plan_artifacts,
)


class AuditError(RuntimeError):
    pass


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
    manifest_by_name = {Path(value).name: value for value in manifest_rows}
    for file_name, row in rows.items():
        prediction = primary_dir / "pred" / f"{Path(file_name).stem}.png"
        source = repo_root / manifest_by_name[file_name]
        exact = {
            "image_path": manifest_by_name[file_name],
            "metrics_skipped": "1",
            "primary_config_sha256": plan["evidence"]["current_primary_config"][
                "sha256"
            ],
            "primary_weights_sha256": plan["evidence"][
                "current_primary_checkpoint"
            ]["sha256"],
            "page_overlap": str(config["page_overlap"]),
            "batch_size": str(config["batch_size"]),
            "copy_input_outside_mask": config["copy_input_outside_mask"],
            "copy_mask_threshold_auto": config["copy_mask_threshold_auto"],
            "copy_mask_dilate": str(config["copy_mask_dilate"]),
            "image_sha256": sha256_file(source),
            "pred_sha256": sha256_file(prediction),
        }
        for key, expected in exact.items():
            if row.get(key) != expected:
                raise AuditError(f"primary metrics {key} changed for {file_name}")
        if Path(row.get("pred_path", "")).name != prediction.name:
            raise AuditError(f"primary prediction identity changed for {file_name}")
        coverage = float(row["copy_mask_cov8"])
        if int(row["copy_mask_threshold"]) != expected_auto_threshold(coverage):
            raise AuditError(f"primary auto threshold changed for {file_name}")
    return {
        "row_count": len(rows),
        "metrics_sha256": sha256_file(primary_dir / "metrics.csv"),
    }


def validate_second_stage_metrics(
    *, plan: dict[str, Any], filenames: list[str], second_stage_dir: Path
) -> dict[str, Any]:
    rows = rows_by_file(
        read_csv_rows(second_stage_dir / "metrics.csv"),
        filenames,
        "second-stage metrics",
    )
    config = plan["pipeline_preparation"]["second_stage"]
    for file_name, row in rows.items():
        prediction = second_stage_dir / "pred" / f"{Path(file_name).stem}.png"
        exact = {
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
        if Path(row.get("pred_path", "")).name != prediction.name:
            raise AuditError(f"second-stage prediction identity changed for {file_name}")
        if not 0.0 <= float(row["gate_ratio"]) <= 1.0:
            raise AuditError(f"invalid second-stage gate ratio for {file_name}")
        sha256_file(prediction)
    return {
        "row_count": len(rows),
        "metrics_sha256": sha256_file(second_stage_dir / "metrics.csv"),
    }


PATCH_STRING_FIELDS = {"file"}
PATCH_INTEGER_FIELDS = {"x1", "y1", "x2", "y2"}
PATCH_FLOAT_FIELDS = {
    "edit_positive_score",
    "edit_positive_ratio",
    "edit_positive_mean_delta",
    "preserve_negative_ratio",
}


def compare_selected_rows(
    actual: list[dict[str, str]], expected: list[dict[str, Any]]
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
    candidates, content_hashes = build_candidates(
        filenames=filenames,
        label_dir=repo_path(repo_root, plan["data"]["label_dir"]),
        input_dir=repo_path(repo_root, config["input_dir"]),
        tile_size=config["tile_size"],
        overlap=config["overlap"],
        luminance_margin_gray=config["luminance_margin_gray"],
        min_positive_ratio=config["min_positive_ratio"],
    )
    expected = select_top_brighten(candidates, config["top_k"])
    actual = read_csv_rows(patch_csv)
    compare_selected_rows(actual, expected)
    summary = read_json(patch_json)
    registered_patch_index = Path(str(summary.get("patch_index", "")))
    expected_relative = Path(config["output_csv"])
    if (
        len(registered_patch_index.parts) < len(expected_relative.parts)
        or registered_patch_index.parts[-len(expected_relative.parts) :]
        != expected_relative.parts
    ):
        raise AuditError("monotonic patch summary path changed")
    summary["patch_index"] = str(patch_csv)
    exact = {
        "status": "pass",
        "terminal": "PASS",
        "selection": "top_target_lighter_support_only",
        "train_role_count": len(filenames),
        "train_role_sha256": sha256_rows(filenames),
        "candidate_count": len(candidates),
        "patch_count": len(actual),
        "page_count": len({row["file"] for row in actual}),
        "patch_index": str(patch_csv),
        "patch_index_sha256": sha256_file(patch_csv),
        **content_hashes,
    }
    if summary != exact:
        raise AuditError("monotonic patch summary changed")
    positive_ratios = [float(row["edit_positive_ratio"]) for row in actual]
    preserve_ratios = [float(row["preserve_negative_ratio"]) for row in actual]
    if len(actual) != config["top_k"]:
        raise AuditError("patch index lacks registered target-lighter support")
    if min(positive_ratios) < config["min_positive_ratio"]:
        raise AuditError("patch index contains subfloor target-lighter support")
    if min(preserve_ratios) <= 0.0:
        raise AuditError("patch index lacks preserve-negative coverage")
    return {
        **exact,
        "positive_ratio_min": min(positive_ratios),
        "positive_ratio_max": max(positive_ratios),
        "preserve_ratio_min": min(preserve_ratios),
        "preserve_ratio_max": max(preserve_ratios),
    }


def expected_stage_command(
    *, repo_root: Path, plan: dict[str, Any], stage: str
) -> list[str]:
    outputs = plan["planned_outputs_must_be_absent"]
    if stage == "manifest":
        return ["write-exact-manifest"]
    if stage in {"primary", "second_stage"}:
        source = (
            repo_root / ARCHIVE_PRIMARY_DIR
            if stage == "primary"
            else repo_root / ARCHIVE_SECOND_STAGE_DIR
        )
        key = "primary_prediction_dir" if stage == "primary" else "training_input_dir"
        final = repo_path(repo_root, outputs[key])
        return [
            "reuse-frozen-output",
            str(source.relative_to(repo_root)),
            str(final.relative_to(repo_root)),
        ]
    if stage == "patch_index":
        final_csv = repo_path(repo_root, plan["patch_builder"]["output_csv"])
        final_json = repo_path(repo_root, plan["patch_builder"]["output_json"])
        temporary_csv = final_csv.with_name(f".{final_csv.name}.materializing")
        temporary_json = final_json.parent.with_name(
            f".{final_json.parent.name}.materializing"
        ) / final_json.name
        return patch_command(repo_root, plan, temporary_csv, temporary_json)
    raise AuditError(f"unknown materialization stage: {stage}")


def relocate_recorded_command(command: Any, repo_root: Path) -> Any:
    if not isinstance(command, list) or "--repo-root" not in command:
        return command
    root_index = command.index("--repo-root") + 1
    if root_index >= len(command):
        return command
    registered_root = Path(str(command[root_index]))
    if not registered_root.is_absolute():
        return command
    relocated: list[Any] = []
    for value in command:
        if not isinstance(value, str):
            relocated.append(value)
            continue
        path = Path(value)
        try:
            relative = path.relative_to(registered_root)
        except ValueError:
            relocated.append(value)
        else:
            relocated.append(str(repo_root / relative))
    return relocated


def validate_stage_records(
    repo_root: Path, plan: dict[str, Any], plan_path: Path
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for stage in STAGES:
        record_path = repo_root / CONTROL_DIR / f"{stage}.json"
        record = read_json(record_path)
        if record.get("terminal") != "PASS" or record.get("stage") != stage:
            raise AuditError(f"materialization stage record failed: {stage}")
        if record.get("training_plan_sha256") != sha256_file(plan_path):
            raise AuditError(f"materialization stage plan hash changed: {stage}")
        recorded_command = record.get("command")
        if stage == "patch_index":
            recorded_command = relocate_recorded_command(
                recorded_command, repo_root
            )
        if recorded_command != expected_stage_command(
            repo_root=repo_root, plan=plan, stage=stage
        ):
            raise AuditError(f"materialization command changed: {stage}")
        evidence = record.get("outputs")
        if not isinstance(evidence, list) or not evidence:
            raise AuditError(f"materialization stage lacks outputs: {stage}")
        for artifact in evidence:
            path = repo_path(repo_root, artifact.get("path", ""))
            if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
                raise AuditError(f"materialization stage output drift: {path}")
        if stage in {"primary", "second_stage"}:
            key = "primary_prediction_dir" if stage == "primary" else "training_input_dir"
            link = repo_path(
                repo_root, plan["planned_outputs_must_be_absent"][key]
            )
            if not link.is_symlink() or os.readlink(link) != record.get("link_target"):
                raise AuditError(f"frozen reuse link changed: {stage}")
        hashes[stage] = sha256_file(record_path)
    return hashes


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
        artifact_hashes, _paths = validate_plan_artifacts(repo_root, plan, ledger)
        reuse_source = validate_reuse_source(repo_root, plan)
        filenames = effective_train_filenames(
            repo_root,
            repo_path(repo_root, plan["evidence"]["base_role_contract"]["path"]),
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
        patch_csv = repo_path(repo_root, plan["patch_builder"]["output_csv"])
        patch_summary = validate_patch_index(
            repo_root=repo_root,
            plan=plan,
            filenames=filenames,
            patch_csv=patch_csv,
            patch_json=repo_path(repo_root, plan["patch_builder"]["output_json"]),
        )
        if plan["patch_builder"]["output_csv"] != outputs["patch_index"]:
            raise AuditError("patch output registration changed")
        stage_records = validate_stage_records(repo_root, plan, plan_path)
        forbidden = {
            "training_output_dir": repo_path(repo_root, outputs["training_output_dir"]),
            "first_gate_output_dir": repo_path(
                repo_root, outputs["first_gate_output_dir"]
            ),
        }
        existing = [label for label, path in forbidden.items() if path.exists()]
        if existing:
            raise AuditError(f"training or quality gate started early: {existing}")
        if primary_predictions != reuse_source["primary_predictions"]:
            raise AuditError("reused primary content differs from archived PASS")
        if second_stage_predictions != reuse_source["second_stage_predictions"]:
            raise AuditError("reused second-stage content differs from archived PASS")
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
        "reuse_source": reuse_source,
        "train_count": len(filenames),
        "train_filename_sha256": sha256_rows(filenames),
        "sample_manifest_sha256": sha256_file(manifest),
        "primary_predictions": primary_predictions,
        "primary_metrics": primary_metrics,
        "second_stage_predictions": second_stage_predictions,
        "second_stage_metrics": second_stage_metrics,
        "patch_summary": patch_summary,
        "stage_record_hashes": stage_records,
        "real_train_pixels_audited": True,
        "target_decode_roles": ["train"],
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
