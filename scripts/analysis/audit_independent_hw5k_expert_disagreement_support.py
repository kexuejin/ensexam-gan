#!/usr/bin/env python3
"""Audit independent HW5K-expert RGB support before any model training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.audit_dual_input_support_separation import (  # noqa: E402
    AuditError,
    auc_rank,
    balanced_indices,
    fit_closed_form_ridge,
    fold_for_name,
    read_json,
    read_rgb,
    repo_path,
    sha256_file,
    validate_label_set,
)
from scripts.analysis.materialize_independent_hw5k_expert_outputs_train_only import (  # noqa: E402
    ACTIVE_ITERATION_ID,
    DIAGNOSTIC_ID,
    EXPECTED_METRIC_FIELDS,
    FAMILY,
    OUTPUT_ROOT as MATERIALIZATION_ROOT,
    PLAN_PATH,
    PREREGISTRATION_ID,
    derive_eligible_sources,
    primary_command,
    sha256_newline_rows,
    validate_external_artifact,
    validate_internal_artifact,
)


LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
OUTPUT_PATH = Path(
    "outputs/independent-hw5k-expert-support-prerequisite-20260813/audit.json"
)
CHANNELS = (
    "current_primary_r",
    "current_primary_g",
    "current_primary_b",
    "frozen_hw5k_expert_r",
    "frozen_hw5k_expert_g",
    "frozen_hw5k_expert_b",
)
ABLATION_CHANNELS = CHANNELS[:3]
FORBIDDEN_COMMAND_TOKENS = {
    "--caller",
    "--domain",
    "--expert-selection",
    "--route",
    "--routing",
    "--split",
    "--target",
    "--target-path",
}


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.materializing")
    if path.exists() or temporary.exists():
        raise AuditError(f"refusing to overwrite audit: {path}")
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
        raise AuditError("quality-loop program authority changed")
    active = ledger.get("active_iteration", {})
    if (
        active.get("id") != ACTIVE_ITERATION_ID
        or active.get("terminal") != "PREREQUISITE_NEEDED"
        or active.get("first_gate") != "scut_inner_val15"
    ):
        raise AuditError("active iteration authority changed")
    statuses = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if statuses.get(PREREGISTRATION_ID) != "passed":
        raise AuditError("independent expert preregistration is not passed")
    if statuses.get(DIAGNOSTIC_ID) != "pending":
        raise AuditError("independent expert diagnostic is not pending")
    if statuses.get("second_stage_alpha_support_train_only_diagnostic") != "passed":
        raise AuditError("prior raw-alpha KILL is not recorded")


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
            raise AuditError(f"plan field changed: {key}")
    data = plan.get("data", {})
    required_data = {
        "data_root": "data-links/samples/SCUT-HW5K-mixed-20260729",
        "derived_basename_newline_sha256": (
            "e2921d717086c080606acd69dbec2de0e4a97281edc460aa6c0b74af41097698"
        ),
        "derived_full_path_newline_sha256": (
            "ad7c794706edb1b832cb30af978663853fb10646531bc4cf83011023338d81e2"
        ),
        "diagnostic_count": 123,
        "diagnostic_domain_counts": {"hw5k": 123, "scut": 0},
        "overlap_count": 152,
        "overlap_domain_counts": {"hw5k": 130, "scut": 22},
        "population_derivation": (
            "preserve_train275_order_and_exclude_every_basename_in_specialist_training_manifest"
        ),
        "source_count": 275,
        "split": "train",
        "target_lighter_margin_gray": 2.0,
    }
    for key, expected in required_data.items():
        if data.get(key) != expected:
            raise AuditError(f"data field changed: {key}")
    if plan.get("representation") != {
        "ablation_is_identical_probe_without_specialist_rgb": True,
        "channels": list(CHANNELS),
        "feature_count": 6,
        "forbidden_features": [
            "derived_expert_difference",
            "domain_or_caller_label",
            "split_path_route_or_expert_selection_metadata",
            "page_scalars_masks_source_rgb_or_second_stage_rgb",
            "thresholds_neighborhoods_transforms_or_alternative_checkpoints",
            "nonlinear_probe",
        ],
        "single_causal_change": (
            "add_frozen_independently_trained_hw5k_expert_rgb_to_current_primary_rgb_support_evidence"
        ),
    }:
        raise AuditError("registered independent-expert representation changed")
    diagnostic = plan.get("diagnostic", {})
    required_diagnostic = {
        "ablation_features": list(ABLATION_CHANNELS),
        "auc_tie_policy": "average_rank",
        "class_balance": "equal_per_page_target_lighter_and_preserve",
        "coordinate_rank": (
            "splitmix64(row_major_index XOR first_64_bits(sha256(utf8_basename))) "
            "then unsigned_hash_y_x"
        ),
        "feature_normalization": "rgb_divided_by_255",
        "fold_assignment": "int(sha256(utf8_basename), 16) % 5",
        "fold_count": 5,
        "full_feature_count": 6,
        "lambda": 1.0,
        "luma_definition": "arithmetic_mean_rgb_gray_units",
        "max_samples_per_class_per_page": 1024,
        "numeric_type": "float64",
        "optimizer": "none_closed_form_ridge",
        "ridge_intercept": "unpenalized",
        "sampling_count": (
            "min(1024, target_lighter_count, preserve_count) per class per page"
        ),
        "standardization": "fitting_fold_mean_and_standard_deviation_only",
        "target_definition": (
            "target_luma_minus_current_primary_luma_greater_than_2_gray"
        ),
        "target_encoding": {"preserve": -1, "target_lighter": 1},
        "threshold_derivation": "prohibited",
    }
    for key, expected in required_diagnostic.items():
        if diagnostic.get(key) != expected:
            raise AuditError(f"diagnostic field changed: {key}")
    if plan.get("acceptance") != {
        "full_auc_ablation_margin_min": 0.03,
        "full_fold_auc_min": 0.55,
        "full_mean_fold_auc_min": 0.65,
        "macro_median_page_auc_min": 0.6,
        "positive_mean_above_preserve_min_folds": 4,
        "required_fold_count": 5,
        "required_terminal_on_pass": "PASS",
    }:
        raise AuditError("acceptance contract changed")
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
        raise AuditError("a candidate or quality surface was opened")
    if (
        authorization.get("paired_expert_materialization") is not True
        or authorization.get("paired_expert_materialization_reads_targets") is not False
        or authorization.get("separability_diagnostic_target_decode_roles") != ["train"]
    ):
        raise AuditError("diagnostic authorization changed")
    materialization = plan.get("paired_expert_materialization", {})
    if materialization != {
        "derived_manifest_creation": (
            "reproduce_after_all_source_and_exclusion_hashes_validate_before_image_decode"
        ),
        "inference": (
            "run_both_frozen_checkpoints_on_every_same_source_page_without_routing"
        ),
        "inference_protocol": {
            "batch_size": 8,
            "copy_input_outside_mask": "mb",
            "copy_mask_dilate": 0,
            "copy_mask_threshold": 70,
            "copy_mask_threshold_auto": "mb_cov8_step",
            "device": "auto",
            "page_overlap": 32,
            "skip_label_metrics": True,
        },
        "output_root": str(MATERIALIZATION_ROOT),
        "target_access": False,
    }:
        raise AuditError("paired materialization contract changed")
    if plan.get("planned_implementation") != {
        "audit_output": str(OUTPUT_PATH),
        "audit_script": "scripts/analysis/audit_independent_hw5k_expert_disagreement_support.py",
        "materialization_output": str(MATERIALIZATION_ROOT),
        "materializer": "scripts/analysis/materialize_independent_hw5k_expert_outputs_train_only.py",
        "test": "tests/test_independent_hw5k_expert_disagreement_support_prerequisite.py",
    }:
        raise AuditError("planned implementation changed")


def read_metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_METRIC_FIELDS:
            raise AuditError("materialized primary metrics schema changed")
        return list(reader)


def validate_materialization(
    *,
    repo_root: Path,
    plan: dict[str, Any],
    plan_file: Path,
    paths: dict[str, Path],
    sources: list[tuple[str, Path, str]],
) -> tuple[Path, Path, dict[str, Any]]:
    root = repo_path(repo_root, str(MATERIALIZATION_ROOT))
    if not root.is_dir():
        raise AuditError("paired materialization directory is missing")
    if {path.name for path in root.iterdir()} != {
        "current-primary",
        "eligible-samples.txt",
        "hw5k-expert",
        "manifest.json",
    }:
        raise AuditError("paired materialization entries changed")
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    required = {
        "schema_version": 1,
        "terminal": "PASS",
        "family": FAMILY,
        "target_access": False,
        "routing_metadata": False,
        "output_root": str(MATERIALIZATION_ROOT),
        "eligible_count": len(sources),
        "eligible_samples_file": str(MATERIALIZATION_ROOT / "eligible-samples.txt"),
        "eligible_basename_newline_sha256": plan["data"][
            "derived_basename_newline_sha256"
        ],
        "eligible_full_path_newline_sha256": plan["data"][
            "derived_full_path_newline_sha256"
        ],
        "inference_protocol": plan["paired_expert_materialization"][
            "inference_protocol"
        ],
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise AuditError(f"paired materialization manifest changed: {key}")
    if manifest.get("plan") != {
        "path": str(PLAN_PATH),
        "sha256": sha256_file(plan_file),
    }:
        raise AuditError("paired materialization plan provenance changed")
    if manifest.get("source_manifest") != {
        "path": plan["data"]["manifest"]["path"],
        "sha256": sha256_file(paths["manifest"]),
    }:
        raise AuditError("paired materialization source provenance changed")
    if manifest.get("specialist_training_manifest") != {
        "external_path": str(paths["exclusion_manifest"]),
        "sha256": sha256_file(paths["exclusion_manifest"]),
    }:
        raise AuditError("paired materialization exclusion provenance changed")
    if manifest.get("primary_inference_source") != {
        "path": plan["evidence"]["primary_inference_source"]["path"],
        "sha256": sha256_file(paths["primary_inference_source"]),
    }:
        raise AuditError("paired materialization inference provenance changed")
    samples_file = root / "eligible-samples.txt"
    expected_rows = [row for _name, _source, row in sources]
    if samples_file.read_text(encoding="utf-8") != "".join(
        f"{row}\n" for row in expected_rows
    ):
        raise AuditError("paired materialization sample list changed")

    branch_specs = {
        "current-primary": (
            paths["current_primary_config"],
            paths["current_primary_checkpoint"],
            "current_primary",
        ),
        "hw5k-expert": (
            paths["specialist_config"],
            paths["specialist_checkpoint"],
            "hw5k_expert",
        ),
    }
    page_rows = manifest.get("pages")
    if not isinstance(page_rows, list) or len(page_rows) != len(sources):
        raise AuditError("paired materialization page rows changed")
    page_by_name: dict[str, dict[str, Any]] = {}
    for row in page_rows:
        if not isinstance(row, dict) or set(row) != {
            "current_primary_prediction_sha256",
            "file",
            "height",
            "hw5k_expert_prediction_sha256",
            "source_sha256",
            "width",
        }:
            raise AuditError("paired materialization page schema changed")
        file_name = str(row["file"])
        if file_name in page_by_name:
            raise AuditError("duplicate paired materialization page")
        page_by_name[file_name] = row
    if list(page_by_name) != [name for name, _source, _row in sources]:
        raise AuditError("paired materialization page order changed")

    branch_dirs: dict[str, Path] = {}
    for branch, (config, checkpoint, summary_key) in branch_specs.items():
        branch_root = root / branch
        if {path.name for path in branch_root.iterdir()} != {"metrics.csv", "pred"}:
            raise AuditError(f"paired branch entries changed: {branch}")
        metrics_path = branch_root / "metrics.csv"
        rows = read_metric_rows(metrics_path)
        if [row.get("file") for row in rows] != list(page_by_name):
            raise AuditError(f"paired branch metric order changed: {branch}")
        command = primary_command(
            repo_root=repo_root,
            plan=plan,
            inference_source=paths["primary_inference_source"],
            samples_file=samples_file,
            output_dir=branch_root,
            config=config,
            checkpoint=checkpoint,
        )
        registered_command = manifest.get("commands", {}).get(branch)
        if registered_command != command:
            raise AuditError(f"paired materialization command changed: {branch}")
        if set(command) & FORBIDDEN_COMMAND_TOKENS:
            raise AuditError("routing or target command metadata appeared")
        pred_dir = branch_root / "pred"
        pred_names = sorted(path.name for path in pred_dir.glob("*.png"))
        expected_pred_names = sorted(
            f"{Path(name).stem}.png" for name in page_by_name
        )
        if pred_names != expected_pred_names:
            raise AuditError(f"paired branch prediction names changed: {branch}")
        config_sha = sha256_file(config)
        checkpoint_sha = sha256_file(checkpoint)
        content_rows = []
        for metric_row, (name, source, manifest_row) in zip(
            rows, sources, strict=True
        ):
            prediction = pred_dir / f"{Path(name).stem}.png"
            prediction_sha = sha256_file(prediction)
            paired = page_by_name[name]
            paired_key = (
                "current_primary_prediction_sha256"
                if branch == "current-primary"
                else "hw5k_expert_prediction_sha256"
            )
            if (
                metric_row.get("image_sha256") != sha256_file(source)
                or metric_row.get("pred_sha256") != prediction_sha
                or paired["source_sha256"] != metric_row["image_sha256"]
                or paired[paired_key] != prediction_sha
                or metric_row.get("metrics_skipped") != "1"
                or metric_row.get("primary_config_sha256") != config_sha
                or metric_row.get("primary_weights_sha256") != checkpoint_sha
                or metric_row.get("image_path") != manifest_row
                or metric_row.get("pred_path") != str(prediction)
                or metric_row.get("page_overlap") != "32"
                or metric_row.get("batch_size") != "8"
                or metric_row.get("copy_input_outside_mask") != "mb"
                or metric_row.get("copy_mask_threshold_auto") != "mb_cov8_step"
                or metric_row.get("copy_mask_dilate") != "0"
            ):
                raise AuditError(f"paired branch provenance changed: {branch}/{name}")
            try:
                coverage = float(metric_row["copy_mask_cov8"])
                applied_threshold = int(metric_row["copy_mask_threshold"])
            except (KeyError, TypeError, ValueError) as error:
                raise AuditError(
                    f"invalid automatic mask evidence: {branch}/{name}"
                ) from error
            expected_threshold = (
                8 if coverage <= 0.129 else 76 if coverage <= 0.421 else 160
            )
            if not 0.0 <= coverage <= 1.0 or applied_threshold != expected_threshold:
                raise AuditError(
                    f"automatic mask threshold changed: {branch}/{name}"
                )
            content_rows.append(f"{name} {prediction_sha}")
        summary = manifest.get(summary_key)
        expected_summary = {
            "count": len(rows),
            "directory": str((branch_root / "pred").relative_to(repo_root)),
            "filename_sha256": sha256_newline_rows(pred_names),
            "content_sha256": sha256_newline_rows(sorted(content_rows)),
            "metrics_path": str(metrics_path.relative_to(repo_root)),
            "metrics_sha256": sha256_file(metrics_path),
        }
        if summary != expected_summary:
            raise AuditError(f"paired branch summary changed: {branch}")
        branch_dirs[branch] = pred_dir
    paired_content_sha = sha256_newline_rows(
        [
            f"{row['file']} {row['current_primary_prediction_sha256']} "
            f"{row['hw5k_expert_prediction_sha256']}"
            for row in page_rows
        ]
    )
    if manifest.get("paired_content_sha256") != paired_content_sha:
        raise AuditError("paired prediction content identity changed")
    return branch_dirs["current-primary"], branch_dirs["hw5k-expert"], {
        "manifest_path": str(manifest_path.relative_to(repo_root)),
        "manifest_sha256": sha256_file(manifest_path),
        "target_access": False,
        "routing_metadata": False,
        "paired_content_sha256": paired_content_sha,
    }


def build_page(
    *,
    file_name: str,
    label_dir: Path,
    current_primary_dir: Path,
    hw5k_expert_dir: Path,
    expected_height: int,
    expected_width: int,
    margin_gray: float,
    sample_cap: int,
) -> dict[str, Any]:
    current = read_rgb(current_primary_dir / f"{Path(file_name).stem}.png")
    expert = read_rgb(hw5k_expert_dir / f"{Path(file_name).stem}.png")
    target = read_rgb(label_dir / file_name)
    expected_shape = (expected_height, expected_width, 3)
    if current.shape != expert.shape or current.shape != target.shape:
        raise AuditError(f"paired image shape mismatch for {file_name}")
    if current.shape != expected_shape:
        raise AuditError(f"paired manifest shape changed for {file_name}")
    current_float = current.astype(np.float32)
    expert_float = expert.astype(np.float32)
    target_float = target.astype(np.float32)
    positive_mask = (
        target_float.mean(axis=2) - current_float.mean(axis=2)
    ) > margin_gray
    positive_indices, preserve_indices = balanced_indices(
        positive_mask, file_name, sample_cap
    )
    indices = np.concatenate([positive_indices, preserve_indices])
    labels = np.concatenate(
        [
            np.ones(len(positive_indices), dtype=np.int8),
            -np.ones(len(preserve_indices), dtype=np.int8),
        ]
    )
    current_flat = current_float.reshape(-1, 3)[indices] / 255.0
    expert_flat = expert_float.reshape(-1, 3)[indices] / 255.0
    features = np.column_stack([current_flat, expert_flat]).astype(
        np.float32, copy=False
    )
    if features.shape != (len(indices), len(CHANNELS)):
        raise AuditError(f"independent-expert feature shape changed for {file_name}")
    if not np.isfinite(features).all():
        raise AuditError(f"non-finite independent-expert feature for {file_name}")
    sample_digest = hashlib.sha256()
    sample_digest.update(file_name.encode("utf-8"))
    sample_digest.update(positive_indices.astype("<i8").tobytes())
    sample_digest.update(preserve_indices.astype("<i8").tobytes())
    return {
        "file": file_name,
        "fold": fold_for_name(file_name),
        "features": features,
        "ablation_features": features[:, :3],
        "labels": labels,
        "samples_per_class": len(positive_indices),
        "positive_pixel_count": int(positive_mask.sum()),
        "preserve_pixel_count": int((~positive_mask).sum()),
        "sample_sha256": sample_digest.hexdigest(),
    }


def concatenate_pages(
    pages: list[dict[str, Any]], feature_key: str
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([page[feature_key] for page in pages], axis=0),
        np.concatenate([page["labels"] for page in pages], axis=0),
    )


def evaluate_fold(
    pages: list[dict[str, Any]], fold: int, ridge_lambda: float
) -> dict[str, Any]:
    train_pages = [page for page in pages if page["fold"] != fold]
    test_pages = [page for page in pages if page["fold"] == fold]
    if not train_pages or not test_pages:
        raise AuditError(f"empty fitting or held-out fold: {fold}")
    train_full, train_labels = concatenate_pages(train_pages, "features")
    test_full, test_labels = concatenate_pages(test_pages, "features")
    full_scores, full_fit = fit_closed_form_ridge(
        train_full, train_labels, test_full, ridge_lambda
    )
    train_ablation, _ = concatenate_pages(train_pages, "ablation_features")
    test_ablation, _ = concatenate_pages(test_pages, "ablation_features")
    ablation_scores, ablation_fit = fit_closed_form_ridge(
        train_ablation, train_labels, test_ablation, ridge_lambda
    )
    positive = test_labels > 0
    page_results: list[dict[str, Any]] = []
    offset = 0
    for page in test_pages:
        count = len(page["labels"])
        page_scores = full_scores[offset : offset + count]
        page_results.append(
            {
                "file": page["file"],
                "auc": auc_rank(page["labels"], page_scores),
                "positive_score_mean": float(
                    page_scores[page["labels"] > 0].mean()
                ),
                "preserve_score_mean": float(
                    page_scores[page["labels"] < 0].mean()
                ),
                "samples_per_class": page["samples_per_class"],
            }
        )
        offset += count
    full_auc = auc_rank(test_labels, full_scores)
    ablation_auc = auc_rank(test_labels, ablation_scores)
    return {
        "fold": fold,
        "fit_page_count": len(train_pages),
        "test_page_count": len(test_pages),
        "full_auc": full_auc,
        "ablation_auc": ablation_auc,
        "auc_margin": full_auc - ablation_auc,
        "positive_score_mean": float(full_scores[positive].mean()),
        "preserve_score_mean": float(full_scores[~positive].mean()),
        "page_auc_median": float(
            np.median([row["auc"] for row in page_results])
        ),
        "full_fit": full_fit,
        "ablation_fit": ablation_fit,
        "pages": page_results,
    }


def run_audit(
    *,
    repo_root: Path,
    plan_path: Path = PLAN_PATH,
    ledger_path: Path = LEDGER_PATH,
) -> dict[str, Any]:
    plan_file = repo_path(repo_root, str(plan_path))
    ledger_file = repo_path(repo_root, str(ledger_path))
    plan = read_json(plan_file)
    ledger = read_json(ledger_file)
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
    paths["manifest"] = validate_internal_artifact(
        repo_root, plan["data"]["manifest"], "train275 manifest"
    )
    sources = derive_eligible_sources(
        repo_root, plan, paths["manifest"], paths["exclusion_manifest"]
    )
    current_dir, expert_dir, materialization_summary = validate_materialization(
        repo_root=repo_root,
        plan=plan,
        plan_file=plan_file,
        paths=paths,
        sources=sources,
    )

    all_manifest_rows = [
        row.strip()
        for row in paths["manifest"].read_text(encoding="utf-8").splitlines()
        if row.strip() and not row.lstrip().startswith("#")
    ]
    all_file_names = [Path(row).name for row in all_manifest_rows]
    label_spec = evidence["train_label_set"]
    label_dir = repo_path(repo_root, label_spec["directory"])
    label_summary = validate_label_set(label_dir, all_file_names, label_spec)

    materialization_manifest = read_json(
        repo_path(repo_root, materialization_summary["manifest_path"])
    )
    paired_by_name = {row["file"]: row for row in materialization_manifest["pages"]}
    pages: list[dict[str, Any]] = []
    margin = float(plan["data"]["target_lighter_margin_gray"])
    sample_cap = int(plan["diagnostic"]["max_samples_per_class_per_page"])
    for index, (file_name, _source, _manifest_row) in enumerate(sources, start=1):
        paired = paired_by_name[file_name]
        pages.append(
            build_page(
                file_name=file_name,
                label_dir=label_dir,
                current_primary_dir=current_dir,
                hw5k_expert_dir=expert_dir,
                expected_height=int(paired["height"]),
                expected_width=int(paired["width"]),
                margin_gray=margin,
                sample_cap=sample_cap,
            )
        )
        if index % 25 == 0 or index == len(sources):
            print(
                f"decoded unseen-HW5K train-only pages {index}/{len(sources)}",
                flush=True,
            )
    fold_counts = {
        fold: sum(page["fold"] == fold for page in pages) for fold in range(5)
    }
    if fold_counts != {0: 28, 1: 22, 2: 20, 3: 24, 4: 29}:
        raise AuditError("eligible page fold counts changed")
    folds = [
        evaluate_fold(pages, fold, float(plan["diagnostic"]["lambda"]))
        for fold in range(5)
    ]
    full_aucs = [float(fold["full_auc"]) for fold in folds]
    ablation_aucs = [float(fold["ablation_auc"]) for fold in folds]
    page_aucs = [float(page["auc"]) for fold in folds for page in fold["pages"]]
    positive_above_preserve = sum(
        fold["positive_score_mean"] > fold["preserve_score_mean"]
        for fold in folds
    )
    aggregates = {
        "full_mean_fold_auc": float(np.mean(full_aucs)),
        "full_min_fold_auc": min(full_aucs),
        "ablation_mean_fold_auc": float(np.mean(ablation_aucs)),
        "full_auc_ablation_margin": float(
            np.mean(full_aucs) - np.mean(ablation_aucs)
        ),
        "macro_median_page_auc": float(np.median(page_aucs)),
        "positive_mean_above_preserve_folds": positive_above_preserve,
    }
    contract = plan["acceptance"]
    conditions = {
        "full_mean_fold_auc": (
            aggregates["full_mean_fold_auc"]
            >= contract["full_mean_fold_auc_min"]
        ),
        "every_fold_auc": (
            aggregates["full_min_fold_auc"] >= contract["full_fold_auc_min"]
        ),
        "macro_median_page_auc": (
            aggregates["macro_median_page_auc"]
            >= contract["macro_median_page_auc_min"]
        ),
        "positive_mean_above_preserve": (
            positive_above_preserve
            >= contract["positive_mean_above_preserve_min_folds"]
        ),
        "full_auc_ablation_margin": (
            aggregates["full_auc_ablation_margin"]
            >= contract["full_auc_ablation_margin_min"]
        ),
    }
    terminal = "PASS" if all(conditions.values()) else "KILL"
    return {
        "schema_version": 1,
        "terminal": terminal,
        "family": FAMILY,
        "iteration_id": ACTIVE_ITERATION_ID,
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_file),
        "ledger_path": str(ledger_path),
        "real_data_access": True,
        "target_decode_roles": ["train"],
        "target_decoded_page_count": len(pages),
        "full_train_label_set_validated_before_target_decode": True,
        "training_started": False,
        "checkpoint_generated": False,
        "candidate_inference_started": False,
        "quality_gate_started": False,
        "promotion_enabled": False,
        "reserved_blind_authorized": False,
        "feature_count": len(CHANNELS),
        "feature_names": list(CHANNELS),
        "ablation_feature_names": list(ABLATION_CHANNELS),
        "train_page_count": len(pages),
        "fold_page_counts": fold_counts,
        "materialization": materialization_summary,
        "train_labels": label_summary,
        "sample_count": sum(len(page["labels"]) for page in pages),
        "page_samples": [
            {
                key: page[key]
                for key in (
                    "file",
                    "fold",
                    "samples_per_class",
                    "positive_pixel_count",
                    "preserve_pixel_count",
                    "sample_sha256",
                )
            }
            for page in pages
        ],
        "folds": folds,
        "aggregates": aggregates,
        "acceptance": {
            "contract": contract,
            "conditions": conditions,
            "passed": all(conditions.values()),
        },
        "next_boundary": (
            plan["next_boundary_on_pass"]
            if terminal == "PASS"
            else plan["terminal_successors"]["KILL"]
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path = repo_path(args.repo_root, str(args.output))
    if output_path.parent.exists():
        print(
            "terminal=PREREQUISITE_NEEDED "
            f"reason=output directory already exists: {output_path.parent}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    try:
        result = run_audit(
            repo_root=args.repo_root,
            plan_path=args.plan,
            ledger_path=args.ledger,
        )
    except (AuditError, OSError, ValueError) as error:
        result = {
            "schema_version": 1,
            "terminal": "PREREQUISITE_NEEDED",
            "reason": str(error),
            "training_started": False,
            "checkpoint_generated": False,
            "candidate_inference_started": False,
            "quality_gate_started": False,
            "promotion_enabled": False,
            "reserved_blind_authorized": False,
        }
        return_code = 2
    else:
        return_code = 0 if result["terminal"] == "PASS" else 1
    output_path.parent.mkdir(parents=True)
    atomic_write_json(output_path, result)
    print(f"terminal={result['terminal']} output={output_path}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
