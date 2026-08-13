#!/usr/bin/env python3
"""Materialize paired frozen primary outputs on specialist-unseen train pages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import cv2


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.audit_dual_input_support_separation import (  # noqa: E402
    read_json,
    repo_path,
    sha256_file,
)


PLAN_PATH = Path(
    "docs/independent-hw5k-expert-disagreement-support-prerequisite-v1.json"
)
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
OUTPUT_ROOT = Path("outputs/independent-hw5k-expert-support-materialization-20260813")
FAMILY = "independent_hw5k_expert_disagreement_support_v1"
ACTIVE_ITERATION_ID = "monotonic-residual-erase-support"
PREREGISTRATION_ID = "materially_new_support_successor_preregistration_v3"
DIAGNOSTIC_ID = "independent_hw5k_expert_support_train_only_diagnostic"
EXPECTED_METRIC_FIELDS = (
    "file",
    "image_path",
    "image_sha256",
    "pred_path",
    "pred_sha256",
    "metrics_skipped",
    "primary_config_sha256",
    "primary_weights_sha256",
    "page_overlap",
    "batch_size",
    "copy_input_outside_mask",
    "copy_mask_threshold",
    "copy_mask_threshold_auto",
    "copy_mask_dilate",
    "copy_mask_cov8",
)
FORBIDDEN_METADATA_FIELDS = {
    "caller",
    "domain",
    "expert_selection",
    "route",
    "routing",
    "split",
    "target",
    "target_path",
}


class MaterializationError(RuntimeError):
    pass


def sha256_newline_rows(rows: list[str]) -> str:
    payload = "" if not rows else "\n".join(rows) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.materializing")
    if path.exists() or temporary.exists():
        raise MaterializationError(f"refusing to overwrite manifest: {path}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_authority(ledger: dict[str, Any]) -> None:
    program = ledger.get("program", {})
    if (
        program.get("product_default") != "artifacts/current-primary"
        or program.get("promotion_state") != "disabled"
        or program.get("reserved_blind_state") != "disabled"
    ):
        raise MaterializationError("quality-loop program authority changed")
    active = ledger.get("active_iteration", {})
    if (
        active.get("id") != ACTIVE_ITERATION_ID
        or active.get("terminal") != "PREREQUISITE_NEEDED"
        or active.get("first_gate") != "scut_inner_val15"
    ):
        raise MaterializationError("active iteration authority changed")
    statuses = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if statuses.get(PREREGISTRATION_ID) != "passed":
        raise MaterializationError("independent expert preregistration is not passed")
    if statuses.get(DIAGNOSTIC_ID) != "pending":
        raise MaterializationError("independent expert diagnostic is not pending")
    if statuses.get("second_stage_alpha_support_train_only_diagnostic") != "passed":
        raise MaterializationError("prior raw-alpha KILL is not recorded")


def validate_plan(plan: dict[str, Any]) -> None:
    exact = {
        "schema_version": 1,
        "family": FAMILY,
        "iteration_id": ACTIVE_ITERATION_ID,
        "state": "preregistered_pending_paired_expert_materialization",
        "next_boundary_on_pass": "independent_hw5k_expert_data_and_training_preflight_only",
    }
    for key, expected in exact.items():
        if plan.get(key) != expected:
            raise MaterializationError(f"plan field changed: {key}")
    spec = plan.get("paired_expert_materialization", {})
    if spec.get("output_root") != str(OUTPUT_ROOT) or spec.get("target_access") is not False:
        raise MaterializationError("paired materialization boundary changed")
    if spec.get("inference") != (
        "run_both_frozen_checkpoints_on_every_same_source_page_without_routing"
    ):
        raise MaterializationError("paired inference contract changed")
    if spec.get("inference_protocol") != {
        "batch_size": 8,
        "copy_input_outside_mask": "mb",
        "copy_mask_dilate": 0,
        "copy_mask_threshold": 70,
        "copy_mask_threshold_auto": "mb_cov8_step",
        "device": "auto",
        "page_overlap": 32,
        "skip_label_metrics": True,
    }:
        raise MaterializationError("primary inference protocol changed")
    authorization = plan.get("authorization", {})
    forbidden = {
        "candidate_checkpoint",
        "candidate_inference",
        "development_gate",
        "holdout40",
        "inner_val15",
        "model_training",
        "promotion",
        "reserved_blind",
        "scut115",
        "visual_review",
    }
    if any(authorization.get(name) is not False for name in forbidden):
        raise MaterializationError("a candidate or quality surface was opened")
    if (
        authorization.get("paired_expert_materialization") is not True
        or authorization.get("paired_expert_materialization_reads_targets") is not False
        or authorization.get("separability_diagnostic_target_decode_roles") != ["train"]
    ):
        raise MaterializationError("paired materialization authorization changed")
    if plan.get("planned_implementation") != {
        "audit_output": "outputs/independent-hw5k-expert-support-prerequisite-20260813/audit.json",
        "audit_script": "scripts/analysis/audit_independent_hw5k_expert_disagreement_support.py",
        "materialization_output": str(OUTPUT_ROOT),
        "materializer": "scripts/analysis/materialize_independent_hw5k_expert_outputs_train_only.py",
        "test": "tests/test_independent_hw5k_expert_disagreement_support_prerequisite.py",
    }:
        raise MaterializationError("planned implementation changed")


def validate_internal_artifact(
    repo_root: Path, artifact: dict[str, Any], label: str
) -> Path:
    if set(artifact) != {"path", "sha256"}:
        raise MaterializationError(f"{label} must contain path and sha256")
    path = repo_path(repo_root, str(artifact["path"]))
    if not path.is_file() or sha256_file(path) != artifact["sha256"]:
        raise MaterializationError(f"{label} missing or sha256 changed")
    return path


def validate_external_artifact(artifact: dict[str, Any], label: str) -> Path:
    if set(artifact) != {"external_path", "sha256"}:
        raise MaterializationError(f"{label} must contain external_path and sha256")
    path = Path(str(artifact["external_path"]))
    if not path.is_absolute() or not path.is_file():
        raise MaterializationError(f"missing {label}: {path}")
    if sha256_file(path) != artifact["sha256"]:
        raise MaterializationError(f"{label} sha256 changed")
    return path


def read_nonempty_rows(path: Path) -> list[str]:
    return [
        row.strip()
        for row in path.read_text(encoding="utf-8").splitlines()
        if row.strip() and not row.lstrip().startswith("#")
    ]


def derive_eligible_sources(
    repo_root: Path,
    plan: dict[str, Any],
    manifest_path: Path,
    exclusion_path: Path,
) -> list[tuple[str, Path, str]]:
    data = plan["data"]
    manifest_rows = read_nonempty_rows(manifest_path)
    if len(manifest_rows) != int(data["source_count"]):
        raise MaterializationError("source manifest count changed")
    source_names = [Path(row).name for row in manifest_rows]
    if len(source_names) != len(set(source_names)):
        raise MaterializationError("source manifest basenames are not unique")
    exclusion_names = [Path(row).name for row in read_nonempty_rows(exclusion_path)]
    if len(exclusion_names) != len(set(exclusion_names)):
        raise MaterializationError("specialist training manifest basenames are not unique")
    exclusion = set(exclusion_names)
    overlap = [name for name in source_names if name in exclusion]
    eligible_rows = [row for row in manifest_rows if Path(row).name not in exclusion]
    eligible_names = [Path(row).name for row in eligible_rows]
    overlap_counts = {
        "hw5k": sum(name.startswith("hw5k_") for name in overlap),
        "scut": sum(name.startswith("scut_") for name in overlap),
    }
    eligible_counts = {
        "hw5k": sum(name.startswith("hw5k_") for name in eligible_names),
        "scut": sum(name.startswith("scut_") for name in eligible_names),
    }
    if len(overlap) != int(data["overlap_count"]):
        raise MaterializationError("specialist training overlap count changed")
    if overlap_counts != data["overlap_domain_counts"]:
        raise MaterializationError("specialist training overlap domains changed")
    if len(eligible_rows) != int(data["diagnostic_count"]):
        raise MaterializationError("eligible diagnostic count changed")
    if eligible_counts != data["diagnostic_domain_counts"]:
        raise MaterializationError("eligible diagnostic domains changed")
    if set(eligible_names) & exclusion:
        raise MaterializationError("specialist training overlap remains in diagnostic population")
    if sha256_newline_rows(eligible_names) != data["derived_basename_newline_sha256"]:
        raise MaterializationError("eligible basename identity changed")
    if sha256_newline_rows(eligible_rows) != data["derived_full_path_newline_sha256"]:
        raise MaterializationError("eligible full-path identity changed")
    sources: list[tuple[str, Path, str]] = []
    for row, name in zip(eligible_rows, eligible_names, strict=True):
        source = repo_path(repo_root, row)
        if not source.is_file():
            raise MaterializationError(f"eligible source is missing: {row}")
        lowered = {part.lower() for part in source.resolve().parts}
        if lowered & {"label", "labels", "target", "targets", "all_labels"}:
            raise MaterializationError(f"eligible source resolves to target data: {row}")
        sources.append((name, source, row))
    return sources


def validate_inputs(
    repo_root: Path, plan: dict[str, Any], ledger: dict[str, Any]
) -> tuple[dict[str, Path], list[tuple[str, Path, str]]]:
    validate_plan(plan)
    validate_authority(ledger)
    evidence = plan["evidence"]
    paths = {
        "candidate5_routing_closure": validate_internal_artifact(
            repo_root, evidence["candidate5_routing_closure"], "Candidate 5 closure"
        ),
        "current_primary_checkpoint": validate_internal_artifact(
            repo_root, evidence["current_primary_checkpoint"], "current-primary checkpoint"
        ),
        "current_primary_config": validate_internal_artifact(
            repo_root, evidence["current_primary_config"], "current-primary config"
        ),
        "primary_inference_source": validate_internal_artifact(
            repo_root, evidence["primary_inference_source"], "primary inference source"
        ),
        "role_plan": validate_internal_artifact(
            repo_root, evidence["role_plan"], "role plan"
        ),
        "second_stage_alpha_kill": validate_internal_artifact(
            repo_root, evidence["second_stage_alpha_kill"], "prior alpha KILL"
        ),
        "specialist_checkpoint": validate_external_artifact(
            evidence["specialist_checkpoint"], "specialist checkpoint"
        ),
        "specialist_config": validate_external_artifact(
            evidence["specialist_config"], "specialist config"
        ),
        "exclusion_manifest": validate_external_artifact(
            plan["data"]["exclusion_manifest"], "specialist training manifest"
        ),
    }
    manifest = validate_internal_artifact(
        repo_root, plan["data"]["manifest"], "train275 manifest"
    )
    paths["manifest"] = manifest
    sources = derive_eligible_sources(
        repo_root, plan, manifest, paths["exclusion_manifest"]
    )
    output_root = repo_path(repo_root, plan["paired_expert_materialization"]["output_root"])
    temporary_root = output_root.with_name(f".{output_root.name}.materializing")
    if output_root.exists() or temporary_root.exists():
        raise MaterializationError("paired materialization output already exists")
    return paths, sources


def primary_command(
    *,
    repo_root: Path,
    plan: dict[str, Any],
    inference_source: Path,
    samples_file: Path,
    output_dir: Path,
    config: Path,
    checkpoint: Path,
) -> list[str]:
    protocol = plan["paired_expert_materialization"]["inference_protocol"]
    return [
        sys.executable,
        str(inference_source),
        "--samples-file",
        str(samples_file),
        "--output-dir",
        str(output_dir),
        "--primary-config",
        str(config),
        "--primary-weights",
        str(checkpoint),
        "--device",
        str(protocol["device"]),
        "--page-overlap",
        str(protocol["page_overlap"]),
        "--batch-size",
        str(protocol["batch_size"]),
        "--copy-input-outside-mask",
        str(protocol["copy_input_outside_mask"]),
        "--copy-mask-threshold",
        str(protocol["copy_mask_threshold"]),
        "--copy-mask-threshold-auto",
        str(protocol["copy_mask_threshold_auto"]),
        "--copy-mask-dilate",
        str(protocol["copy_mask_dilate"]),
        "--skip-label-metrics",
    ]


def rewrite_metrics_paths(metrics_path: Path, old_root: Path, final_root: Path) -> None:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if tuple(fieldnames or ()) != EXPECTED_METRIC_FIELDS:
        raise MaterializationError("primary metrics schema changed")
    old = str(old_root)
    new = str(final_root)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, str) and old in value:
                row[key] = value.replace(old, new)
    temporary = metrics_path.with_name(f".{metrics_path.name}.rewriting")
    with temporary.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EXPECTED_METRIC_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(metrics_path)


def validate_stage(
    *,
    repo_root: Path,
    stage_root: Path,
    final_stage_root: Path,
    sources: list[tuple[str, Path, str]],
    config: Path,
    checkpoint: Path,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if {path.name for path in stage_root.iterdir()} != {"metrics.csv", "pred"}:
        raise MaterializationError(f"unexpected primary output entries: {stage_root}")
    metrics_path = stage_root / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_METRIC_FIELDS:
            raise MaterializationError("primary metrics schema changed")
        rows = list(reader)
    names = [name for name, _source, _row in sources]
    if [row.get("file") for row in rows] != names:
        raise MaterializationError("primary metrics sample order changed")
    if any(set(row) & FORBIDDEN_METADATA_FIELDS for row in rows):
        raise MaterializationError("routing or target metadata appeared in primary metrics")
    config_sha = sha256_file(config)
    checkpoint_sha = sha256_file(checkpoint)
    result: list[dict[str, Any]] = []
    for row, (name, source, manifest_row) in zip(rows, sources, strict=True):
        expected = {
            "metrics_skipped": "1",
            "primary_config_sha256": config_sha,
            "primary_weights_sha256": checkpoint_sha,
            "page_overlap": str(protocol["page_overlap"]),
            "batch_size": str(protocol["batch_size"]),
            "copy_input_outside_mask": str(protocol["copy_input_outside_mask"]),
            "copy_mask_threshold_auto": str(protocol["copy_mask_threshold_auto"]),
            "copy_mask_dilate": str(protocol["copy_mask_dilate"]),
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise MaterializationError(f"primary protocol changed for {name}: {key}")
        try:
            coverage = float(row["copy_mask_cov8"])
            applied_threshold = int(row["copy_mask_threshold"])
        except (KeyError, TypeError, ValueError) as error:
            raise MaterializationError(
                f"invalid automatic mask evidence for {name}"
            ) from error
        expected_threshold = 8 if coverage <= 0.129 else 76 if coverage <= 0.421 else 160
        if not 0.0 <= coverage <= 1.0 or applied_threshold != expected_threshold:
            raise MaterializationError(
                f"automatic mask threshold changed for {name}"
            )
        if row.get("image_sha256") != sha256_file(source):
            raise MaterializationError(f"source hash changed during inference: {name}")
        pred_path = stage_root / "pred" / f"{Path(name).stem}.png"
        if row.get("image_path") != manifest_row or row.get("pred_path") != str(pred_path):
            raise MaterializationError(f"primary path provenance changed for {name}")
        if not pred_path.is_file() or row.get("pred_sha256") != sha256_file(pred_path):
            raise MaterializationError(f"prediction hash changed during inference: {name}")
        image = cv2.imread(str(pred_path), cv2.IMREAD_COLOR)
        if image is None:
            raise MaterializationError(f"prediction decode failed: {name}")
        result.append(
            {
                "file": name,
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "source_sha256": row["image_sha256"],
                "prediction_sha256": row["pred_sha256"],
            }
        )
    pred_names = sorted(path.name for path in (stage_root / "pred").glob("*.png"))
    expected_pred_names = sorted(f"{Path(name).stem}.png" for name in names)
    if pred_names != expected_pred_names:
        raise MaterializationError("primary prediction filenames changed")
    rewrite_metrics_paths(metrics_path, stage_root, final_stage_root)
    content_rows = [f"{row['file']} {row['prediction_sha256']}" for row in result]
    return result, {
        "count": len(result),
        "directory": str((final_stage_root / "pred").relative_to(repo_root)),
        "filename_sha256": sha256_newline_rows(pred_names),
        "content_sha256": sha256_newline_rows(sorted(content_rows)),
        "metrics_path": str((final_stage_root / "metrics.csv").relative_to(repo_root)),
        "metrics_sha256": sha256_file(metrics_path),
    }


def materialize(
    *,
    repo_root: Path,
    plan_path: Path = PLAN_PATH,
    ledger_path: Path = LEDGER_PATH,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    plan_file = repo_path(repo_root, str(plan_path))
    ledger_file = repo_path(repo_root, str(ledger_path))
    plan = read_json(plan_file)
    ledger = read_json(ledger_file)
    paths, sources = validate_inputs(repo_root, plan, ledger)
    output_root = repo_path(repo_root, plan["paired_expert_materialization"]["output_root"])
    temporary_root = output_root.with_name(f".{output_root.name}.materializing")
    temporary_root.mkdir(parents=True)
    samples_file = temporary_root / "eligible-samples.txt"
    samples_file.write_text(
        "".join(f"{manifest_row}\n" for _name, _source, manifest_row in sources),
        encoding="utf-8",
    )
    branches = (
        (
            "current-primary",
            paths["current_primary_config"],
            paths["current_primary_checkpoint"],
        ),
        ("hw5k-expert", paths["specialist_config"], paths["specialist_checkpoint"]),
    )
    commands: dict[str, list[str]] = {}
    branch_rows: dict[str, list[dict[str, Any]]] = {}
    branch_summaries: dict[str, dict[str, Any]] = {}
    protocol = plan["paired_expert_materialization"]["inference_protocol"]
    for branch, config, checkpoint in branches:
        temporary_branch = temporary_root / branch
        final_branch = output_root / branch
        command = primary_command(
            repo_root=repo_root,
            plan=plan,
            inference_source=paths["primary_inference_source"],
            samples_file=samples_file,
            output_dir=temporary_branch,
            config=config,
            checkpoint=checkpoint,
        )
        commands[branch] = command
        runner(command, cwd=repo_root, check=True)
        rows, summary = validate_stage(
            repo_root=repo_root,
            stage_root=temporary_branch,
            final_stage_root=final_branch,
            sources=sources,
            config=config,
            checkpoint=checkpoint,
            protocol=protocol,
        )
        branch_rows[branch] = rows
        branch_summaries[branch] = summary
        commands[branch] = [
            value.replace(str(temporary_root), str(output_root)) for value in command
        ]
    paired_rows: list[dict[str, Any]] = []
    for current, expert in zip(
        branch_rows["current-primary"], branch_rows["hw5k-expert"], strict=True
    ):
        if (
            current["file"] != expert["file"]
            or current["source_sha256"] != expert["source_sha256"]
            or current["height"] != expert["height"]
            or current["width"] != expert["width"]
        ):
            raise MaterializationError("paired prediction identity or shape changed")
        paired_rows.append(
            {
                "file": current["file"],
                "height": current["height"],
                "width": current["width"],
                "source_sha256": current["source_sha256"],
                "current_primary_prediction_sha256": current["prediction_sha256"],
                "hw5k_expert_prediction_sha256": expert["prediction_sha256"],
            }
        )
    paired_content_sha256 = sha256_newline_rows(
        [
            f"{row['file']} {row['current_primary_prediction_sha256']} "
            f"{row['hw5k_expert_prediction_sha256']}"
            for row in paired_rows
        ]
    )
    manifest = {
        "schema_version": 1,
        "terminal": "PASS",
        "family": FAMILY,
        "target_access": False,
        "routing_metadata": False,
        "output_root": str(OUTPUT_ROOT),
        "eligible_count": len(sources),
        "eligible_samples_file": str(
            (output_root / "eligible-samples.txt").relative_to(repo_root)
        ),
        "eligible_basename_newline_sha256": sha256_newline_rows(
            [name for name, _source, _row in sources]
        ),
        "eligible_full_path_newline_sha256": sha256_file(samples_file),
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_file)},
        "source_manifest": {
            "path": plan["data"]["manifest"]["path"],
            "sha256": sha256_file(paths["manifest"]),
        },
        "specialist_training_manifest": {
            "external_path": str(paths["exclusion_manifest"]),
            "sha256": sha256_file(paths["exclusion_manifest"]),
        },
        "primary_inference_source": {
            "path": plan["evidence"]["primary_inference_source"]["path"],
            "sha256": sha256_file(paths["primary_inference_source"]),
        },
        "inference_protocol": protocol,
        "commands": commands,
        "current_primary": branch_summaries["current-primary"],
        "hw5k_expert": branch_summaries["hw5k-expert"],
        "paired_content_sha256": paired_content_sha256,
        "pages": paired_rows,
    }
    atomic_write_json(temporary_root / "manifest.json", manifest)
    temporary_root.replace(output_root)
    return {
        "terminal": "PASS",
        "output_root": str(output_root),
        "manifest": str(output_root / "manifest.json"),
        "eligible_count": len(sources),
        "paired_content_sha256": paired_content_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = materialize(
            repo_root=args.repo_root,
            plan_path=args.plan,
            ledger_path=args.ledger,
        )
    except (
        MaterializationError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"terminal=PREREQUISITE_NEEDED reason={error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
