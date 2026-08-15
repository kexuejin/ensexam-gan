#!/usr/bin/env python3
"""Audit incremental external-layout support residual reachability on train patches."""

from __future__ import annotations

import argparse
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
    fold_for_name,
)
from scripts.analysis.audit_external_text_layout_direct_support_residual_reachability import (  # noqa: E402
    PATCH_INDEX_PATH,
    SUPPORT_PLAN_PATH,
    ReachabilityError,
    load_page,
    patch_features,
    parse_patch,
    read_patch_rows,
    support_scores,
    validate_support_inputs,
)
from scripts.analysis.validate_external_text_layout_conditioned_preflight import (  # noqa: E402
    ACTIVE_ITERATION_ID,
    PreflightError,
    validate_artifact,
)
from scripts.analysis.validate_external_text_layout_incremental_support_residual_preflight import (  # noqa: E402
    LEDGER_PATH,
    OUTPUT_PATH as PREFLIGHT_OUTPUT_PATH,
    PLAN_PATH,
    incremental_score_to_delta_gray,
    read_json,
    repo_path,
    sha256_file,
)


OUTPUT_DIR = Path("outputs/external-text-layout-incremental-support-residual-diagnostic-20260815")
OUTPUT_PATH = OUTPUT_DIR / "audit.json"
FAMILY = "external_text_layout_incremental_support_residual_v1"
INCREMENTAL_PREFLIGHT_ID = "external_text_layout_incremental_support_residual_preflight"
DIRECT_SUPPORT_DIAGNOSTIC_ID = (
    "external_text_layout_direct_support_residual_reachability_diagnostic"
)
SUPPORT_PREREQUISITE_ID = "external_text_layout_support_train_only_diagnostic"


class IncrementalSupportError(RuntimeError):
    pass


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.materializing")
    if path.exists() or temporary.exists():
        raise IncrementalSupportError(f"refusing to overwrite diagnostic: {path}")
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def incremental_scores(features: np.ndarray, full_fit: dict[str, Any], ablation_fit: dict[str, Any]) -> np.ndarray:
    full_scores = support_scores(features, full_fit)
    ablation_scores = support_scores(features[:, :3], ablation_fit)
    scores = full_scores - ablation_scores
    if not np.isfinite(scores).all():
        raise IncrementalSupportError("non-finite incremental support scores")
    return scores


def fold_contracts(support_audit: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if support_audit.get("terminal") != "PASS":
        raise IncrementalSupportError("support diagnostic is not PASS")
    folds = support_audit.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise IncrementalSupportError("support diagnostic fold count changed")
    by_fold: dict[int, dict[str, Any]] = {}
    for fold in folds:
        if not isinstance(fold, dict):
            raise IncrementalSupportError("support diagnostic fold schema changed")
        fold_id = int(fold.get("fold"))
        if fold_id in by_fold:
            raise IncrementalSupportError("duplicate support fold")
        full_fit = fold.get("full_fit")
        ablation_fit = fold.get("ablation_fit")
        if not isinstance(full_fit, dict) or not isinstance(ablation_fit, dict):
            raise IncrementalSupportError("support fold missing full/ablation fits")
        if float(fold.get("auc_margin")) < 0.03:
            raise IncrementalSupportError("support fold incremental margin changed")
        by_fold[fold_id] = {
            "ablation_fit": ablation_fit,
            "auc_margin": float(fold["auc_margin"]),
            "full_fit": full_fit,
        }
    if set(by_fold) != set(range(5)):
        raise IncrementalSupportError("support diagnostic fold identities changed")
    return by_fold


def validate_authority(
    *,
    repo_root: Path,
    plan: dict[str, Any],
    ledger: dict[str, Any],
) -> tuple[Path, Path, Path]:
    active = ledger.get("active_iteration", {})
    if active.get("id") != ACTIVE_ITERATION_ID:
        raise IncrementalSupportError("active iteration changed")
    if active.get("terminal") != "PREREQUISITE_NEEDED":
        raise IncrementalSupportError("active iteration terminal changed")
    statuses = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if statuses.get(SUPPORT_PREREQUISITE_ID) != "passed":
        raise IncrementalSupportError("support diagnostic prerequisite is not passed")
    if statuses.get(DIRECT_SUPPORT_DIAGNOSTIC_ID) != "passed":
        raise IncrementalSupportError("direct support diagnostic prerequisite is not passed")
    incremental_status = statuses.get(INCREMENTAL_PREFLIGHT_ID, "not_started")
    if incremental_status not in {"not_started", "passed"}:
        raise IncrementalSupportError("incremental support preflight status changed")

    diagnostic = plan.get("train_only_preserve_separation_diagnostic", {})
    if diagnostic.get("allowed_roles") != ["train"]:
        raise IncrementalSupportError("incremental support diagnostic role changed")
    if diagnostic.get("candidate_inference") is not False:
        raise IncrementalSupportError("candidate inference opened before diagnostic")
    if diagnostic.get("fit_source") != "support_diagnostic_full_and_rgb_ablation_fold_fits_only":
        raise IncrementalSupportError("incremental support fit source changed")
    if diagnostic.get("target_access") != "train_labels_only_for_page_folded_reachability_measurement":
        raise IncrementalSupportError("incremental support target role changed")
    forbidden = set(diagnostic.get("validation_roles_forbidden", []))
    if forbidden != {"inner_val15", "scut115", "holdout40", "reserved_blind"}:
        raise IncrementalSupportError("incremental support validation boundary changed")
    support_path = validate_artifact(
        repo_root,
        plan["evidence"]["support_diagnostic"],
        "support diagnostic",
    )
    validate_artifact(
        repo_root,
        plan["evidence"]["direct_support_diagnostic"],
        "direct support diagnostic",
    )
    patch_path = validate_artifact(
        repo_root,
        plan["evidence"]["registered_patch_index"],
        "registered patch index",
    )
    preflight_path = repo_path(
        repo_root,
        str(PREFLIGHT_OUTPUT_PATH),
        "incremental support preflight output",
    )
    preflight = read_json(preflight_path)
    if preflight.get("terminal") != "PASS" or preflight.get("runnable") is not True:
        raise IncrementalSupportError("incremental support preflight is not PASS")
    plan_file = repo_path(repo_root, str(PLAN_PATH), "incremental support plan")
    if preflight.get("plan_sha256") != sha256_file(plan_file):
        raise IncrementalSupportError("incremental support preflight plan hash changed")
    return support_path, patch_path, preflight_path


def collect_patch_observations(
    *,
    rows: list[dict[str, str]],
    folds: dict[int, dict[str, Any]],
    second_stage_dir: Path,
    label_dir: Path,
    layout_dir: Path,
    second_rows: dict[str, dict[str, str]],
    margin_gray: float,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, float]]]:
    page_cache: dict[str, dict[str, np.ndarray]] = {}
    observations: list[dict[str, Any]] = []
    fold_sums = {
        fold: {
            "positive_count": 0.0,
            "positive_sum": 0.0,
            "preserve_count": 0.0,
            "preserve_sum": 0.0,
        }
        for fold in range(5)
    }
    for index, row in enumerate(rows):
        file_name, x1, y1, x2, y2 = parse_patch(row)
        page = page_cache.get(file_name)
        if page is None:
            page = load_page(
                file_name=file_name,
                second_stage_dir=second_stage_dir,
                label_dir=label_dir,
                layout_dir=layout_dir,
                second_stage_row=second_rows[file_name],
            )
            page_cache[file_name] = page
        fold_id = fold_for_name(file_name)
        features, target_delta = patch_features(page, (x1, y1, x2, y2))
        scores = incremental_scores(
            features,
            folds[fold_id]["full_fit"],
            folds[fold_id]["ablation_fit"],
        )
        positive = target_delta > margin_gray
        preserve = ~positive
        positive_count = int(positive.sum())
        preserve_count = int(preserve.sum())
        if positive_count <= 0 or preserve_count <= 0:
            raise IncrementalSupportError(f"patch lacks both target classes: {file_name}")
        fold_sums[fold_id]["positive_count"] += positive_count
        fold_sums[fold_id]["preserve_count"] += preserve_count
        fold_sums[fold_id]["positive_sum"] += float(scores[positive].sum())
        fold_sums[fold_id]["preserve_sum"] += float(scores[preserve].sum())
        observations.append(
            {
                "file": file_name,
                "fold": fold_id,
                "index": index,
                "positive": positive,
                "scores": scores,
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
            }
        )
    return observations, fold_sums


def build_fold_calibrations(fold_sums: dict[int, dict[str, float]]) -> dict[int, dict[str, Any]]:
    total = {
        "positive_count": sum(item["positive_count"] for item in fold_sums.values()),
        "positive_sum": sum(item["positive_sum"] for item in fold_sums.values()),
        "preserve_count": sum(item["preserve_count"] for item in fold_sums.values()),
        "preserve_sum": sum(item["preserve_sum"] for item in fold_sums.values()),
    }
    calibrations: dict[int, dict[str, Any]] = {}
    for fold, values in fold_sums.items():
        positive_count = total["positive_count"] - values["positive_count"]
        preserve_count = total["preserve_count"] - values["preserve_count"]
        if positive_count <= 0 or preserve_count <= 0:
            raise IncrementalSupportError(f"empty calibration class for fold {fold}")
        positive_center = (total["positive_sum"] - values["positive_sum"]) / positive_count
        preserve_center = (total["preserve_sum"] - values["preserve_sum"]) / preserve_count
        calibrations[fold] = {
            "calibration_positive_pixels": int(positive_count),
            "calibration_preserve_pixels": int(preserve_count),
            "fold": fold,
            "ordered_centers": positive_center > preserve_center,
            "positive_center": float(positive_center),
            "preserve_center": float(preserve_center),
            "score_span": float(positive_center - preserve_center),
        }
    return calibrations


def summarize_reachability(
    *,
    observations: list[dict[str, Any]],
    calibrations: dict[int, dict[str, Any]],
    delta_bound_gray: float,
    gate_threshold_gray: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    counts = {
        "positive": 0,
        "positive_gate": 0,
        "preserve": 0,
        "preserve_gate": 0,
    }
    score_sums = {
        "positive_delta": 0.0,
        "positive_score": 0.0,
        "preserve_delta": 0.0,
        "preserve_score": 0.0,
    }
    max_delta = 0.0
    min_delta = float("inf")
    patches_with_any_gate = 0
    patches_with_positive_gate = 0
    page_summaries: dict[str, dict[str, Any]] = {}
    patch_summaries: list[dict[str, Any]] = []
    for item in observations:
        fold = item["fold"]
        calibration = calibrations[fold]
        positive = item["positive"]
        preserve = ~positive
        scores = item["scores"]
        if calibration["ordered_centers"]:
            delta = incremental_score_to_delta_gray(
                scores,
                preserve_center=calibration["preserve_center"],
                positive_center=calibration["positive_center"],
                delta_bound_gray=delta_bound_gray,
            )
        else:
            delta = np.zeros_like(scores, dtype=np.float64)
        gate = delta >= gate_threshold_gray
        positive_count = int(positive.sum())
        preserve_count = int(preserve.sum())
        positive_gate = int((gate & positive).sum())
        preserve_gate = int((gate & preserve).sum())
        counts["positive"] += positive_count
        counts["positive_gate"] += positive_gate
        counts["preserve"] += preserve_count
        counts["preserve_gate"] += preserve_gate
        score_sums["positive_delta"] += float(delta[positive].sum())
        score_sums["preserve_delta"] += float(delta[preserve].sum())
        score_sums["positive_score"] += float(scores[positive].sum())
        score_sums["preserve_score"] += float(scores[preserve].sum())
        max_delta = max(max_delta, float(delta.max()))
        min_delta = min(min_delta, float(delta.min()))
        has_any_gate = bool(gate.any())
        has_positive_gate = positive_gate > 0
        patches_with_any_gate += int(has_any_gate)
        patches_with_positive_gate += int(has_positive_gate)
        file_name = item["file"]
        patch_summaries.append(
            {
                "file": file_name,
                "fold": fold,
                "index": item["index"],
                "ordered_calibration": calibration["ordered_centers"],
                "positive_gate_ratio": positive_gate / positive_count,
                "positive_pixels": positive_count,
                "preserve_gate_ratio": preserve_gate / preserve_count,
                "preserve_pixels": preserve_count,
                "score_mean": float(scores.mean()),
                "x1": item["x1"],
                "x2": item["x2"],
                "y1": item["y1"],
                "y2": item["y2"],
            }
        )
        page_summary = page_summaries.setdefault(
            file_name,
            {
                "file": file_name,
                "fold": fold,
                "patch_count": 0,
                "patches_with_positive_gate": 0,
                "positive_gate_pixels": 0,
                "positive_pixels": 0,
                "preserve_gate_pixels": 0,
                "preserve_pixels": 0,
            },
        )
        page_summary["patch_count"] += 1
        page_summary["patches_with_positive_gate"] += int(has_positive_gate)
        page_summary["positive_gate_pixels"] += positive_gate
        page_summary["positive_pixels"] += positive_count
        page_summary["preserve_gate_pixels"] += preserve_gate
        page_summary["preserve_pixels"] += preserve_count
    positive_count = counts["positive"]
    preserve_count = counts["preserve"]
    if positive_count <= 0 or preserve_count <= 0:
        raise IncrementalSupportError("diagnostic patch set lacks both target classes")
    summary = {
        "max_delta_gray": max_delta,
        "min_delta_gray": min_delta if min_delta != float("inf") else 0.0,
        "ordered_center_folds": sum(
            int(item["ordered_centers"]) for item in calibrations.values()
        ),
        "patch_count": len(observations),
        "patches_with_any_gate": patches_with_any_gate,
        "patches_with_positive_gate": patches_with_positive_gate,
        "positive_delta_mean_gray": score_sums["positive_delta"] / positive_count,
        "positive_gate_ratio": counts["positive_gate"] / positive_count,
        "positive_pixels": positive_count,
        "positive_score_mean": score_sums["positive_score"] / positive_count,
        "preserve_delta_mean_gray": score_sums["preserve_delta"] / preserve_count,
        "preserve_gate_ratio": counts["preserve_gate"] / preserve_count,
        "preserve_pixels": preserve_count,
        "preserve_score_mean": score_sums["preserve_score"] / preserve_count,
        "reachable_patch_ratio": patches_with_positive_gate / len(observations),
    }
    summary["positive_over_preserve_gate_margin"] = (
        summary["positive_gate_ratio"] - summary["preserve_gate_ratio"]
    )
    return summary, patch_summaries, {
        "page_count": len(page_summaries),
        "pages": sorted(page_summaries.values(), key=lambda row: row["file"]),
    }


def evaluate_acceptance(summary: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, Any]:
    conditions = {
        "ordered_center_folds": (
            summary["ordered_center_folds"] >= diagnostic["minimum_ordered_center_folds"]
        ),
        "positive_gate_ratio": (
            summary["positive_gate_ratio"] >= diagnostic["minimum_positive_gate_ratio"]
        ),
        "positive_over_preserve_gate_margin": (
            summary["positive_over_preserve_gate_margin"]
            >= diagnostic["minimum_positive_over_preserve_gate_margin"]
        ),
        "preserve_gate_ratio": (
            summary["preserve_gate_ratio"] <= diagnostic["maximum_preserve_gate_ratio"]
        ),
        "reachable_patch_ratio": (
            summary["reachable_patch_ratio"] >= diagnostic["minimum_reachable_patch_ratio"]
        ),
        "required_patch_count": summary["patch_count"] == diagnostic["required_patch_count"],
    }
    return {
        "conditions": conditions,
        "contract": {
            key: diagnostic[key]
            for key in (
                "gate_threshold_gray",
                "maximum_preserve_gate_ratio",
                "minimum_ordered_center_folds",
                "minimum_positive_gate_ratio",
                "minimum_positive_over_preserve_gate_margin",
                "minimum_reachable_patch_ratio",
                "required_patch_count",
            )
        },
        "passed": all(conditions.values()),
    }


def run_diagnostic(
    *,
    repo_root: Path = ROOT,
    plan_path: Path = PLAN_PATH,
    support_plan_path: Path = SUPPORT_PLAN_PATH,
    ledger_path: Path = LEDGER_PATH,
    patch_index_path: Path = PATCH_INDEX_PATH,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    plan = read_json(repo_path(repo_root, str(plan_path), "incremental support plan"))
    support_plan = read_json(repo_path(repo_root, str(support_plan_path), "support plan"))
    ledger = read_json(repo_path(repo_root, str(ledger_path), "ledger"))
    support_audit_path, registered_patch_path, preflight_path = validate_authority(
        repo_root=repo_root,
        plan=plan,
        ledger=ledger,
    )
    if repo_path(repo_root, str(patch_index_path), "patch index") != registered_patch_path:
        raise IncrementalSupportError("patch index override is not registered")
    support_audit = read_json(support_audit_path)
    folds = fold_contracts(support_audit)
    diagnostic = plan["train_only_preserve_separation_diagnostic"]
    rows = read_patch_rows(registered_patch_path, int(diagnostic["required_patch_count"]))
    second_stage_dir, label_dir, layout_dir, second_rows, _evidence = validate_support_inputs(
        repo_root=repo_root,
        support_plan=support_plan,
        support_audit=support_audit,
    )
    observations, fold_sums = collect_patch_observations(
        rows=rows,
        folds=folds,
        second_stage_dir=second_stage_dir,
        label_dir=label_dir,
        layout_dir=layout_dir,
        second_rows=second_rows,
        margin_gray=float(support_plan["data"]["target_lighter_margin_gray"]),
    )
    calibrations = build_fold_calibrations(fold_sums)
    projection = plan["incremental_support_projection"]
    summary, patch_summaries, page_summary = summarize_reachability(
        observations=observations,
        calibrations=calibrations,
        delta_bound_gray=float(projection["delta_bound_gray"]),
        gate_threshold_gray=float(diagnostic["gate_threshold_gray"]),
    )
    acceptance = evaluate_acceptance(summary, diagnostic)
    terminal = "PASS" if acceptance["passed"] else "KILL"
    output_dir = repo_path(repo_root, diagnostic["output_dir"], "diagnostic output dir")
    return {
        "acceptance": acceptance,
        "candidate_inference_started": False,
        "checkpoint_generated": False,
        "diagnostic_successor": plan["diagnostic_successors"][terminal],
        "family": FAMILY,
        "fit_source": diagnostic["fit_source"],
        "fold_calibrations": [
            calibrations[fold] for fold in sorted(calibrations)
        ],
        "fold_fit_count": len(folds),
        "iteration_id": plan["iteration_id"],
        "model_training_started": False,
        "output_dir": str(output_dir.relative_to(repo_root)),
        "page_summary": page_summary,
        "patch_index": {
            "path": str(registered_patch_path.relative_to(repo_root)),
            "sha256": sha256_file(registered_patch_path),
        },
        "patches": patch_summaries,
        "plan": {
            "path": str(repo_path(repo_root, str(plan_path), "incremental support plan").relative_to(repo_root)),
            "sha256": sha256_file(repo_path(repo_root, str(plan_path), "incremental support plan")),
        },
        "preflight": {
            "path": str(preflight_path.relative_to(repo_root)),
            "sha256": sha256_file(preflight_path),
        },
        "promotion_enabled": False,
        "quality_gate_started": False,
        "reserved_blind_authorized": False,
        "schema_version": 1,
        "score_formula": projection["score_formula"],
        "support_diagnostic": {
            "path": str(support_audit_path.relative_to(repo_root)),
            "sha256": sha256_file(support_audit_path),
        },
        "summary": summary,
        "target_decode_roles": ["train"],
        "terminal": terminal,
        "training_started": False,
        "validation_roles_forbidden": diagnostic["validation_roles_forbidden"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--support-plan", type=Path, default=SUPPORT_PLAN_PATH)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--patch-index", type=Path, default=PATCH_INDEX_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path = repo_path(args.repo_root.resolve(), str(args.output), "output")
    if output_path.exists() or output_path.parent.exists():
        print(
            f"terminal=PREREQUISITE_NEEDED reason=diagnostic output already exists: {output_path.parent}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    try:
        result = run_diagnostic(
            repo_root=args.repo_root,
            plan_path=args.plan,
            support_plan_path=args.support_plan,
            ledger_path=args.ledger,
            patch_index_path=args.patch_index,
        )
    except (
        AuditError,
        OSError,
        PreflightError,
        ReachabilityError,
        IncrementalSupportError,
        ValueError,
    ) as error:
        result = {
            "candidate_inference_started": False,
            "checkpoint_generated": False,
            "model_training_started": False,
            "promotion_enabled": False,
            "quality_gate_started": False,
            "reason": str(error),
            "reserved_blind_authorized": False,
            "schema_version": 1,
            "terminal": "PREREQUISITE_NEEDED",
            "training_started": False,
        }
        return_code = 2
    else:
        return_code = 0 if result["terminal"] == "PASS" else 1
    atomic_write_json(output_path, result)
    print(f"terminal={result['terminal']} output={output_path}", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
