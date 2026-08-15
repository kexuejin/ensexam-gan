#!/usr/bin/env python3
"""Audit binary external-text-layout mask residual reachability on train patches."""

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

from scripts.analysis.audit_dual_input_support_separation import AuditError, fold_for_name  # noqa: E402
from scripts.analysis.audit_external_text_layout_direct_support_residual_reachability import (  # noqa: E402
    PATCH_INDEX_PATH,
    SUPPORT_PLAN_PATH,
    ReachabilityError,
    load_page,
    patch_features,
    parse_patch,
    read_patch_rows,
    validate_support_inputs,
)
from scripts.analysis.validate_external_text_layout_binary_mask_residual_preflight import (  # noqa: E402
    LEDGER_PATH,
    OUTPUT_PATH as PREFLIGHT_OUTPUT_PATH,
    PLAN_PATH,
    binary_mask_to_delta_gray,
    read_json,
    repo_path,
    sha256_file,
)
from scripts.analysis.validate_external_text_layout_conditioned_preflight import (  # noqa: E402
    ACTIVE_ITERATION_ID,
    PreflightError,
    validate_artifact,
)


OUTPUT_DIR = Path("outputs/external-text-layout-binary-mask-residual-diagnostic-20260815")
OUTPUT_PATH = OUTPUT_DIR / "audit.json"
FAMILY = "external_text_layout_binary_mask_residual_v1"
BINARY_PREFLIGHT_ID = "external_text_layout_binary_mask_residual_preflight"
INCREMENTAL_DIAGNOSTIC_ID = (
    "external_text_layout_incremental_support_residual_reachability_diagnostic"
)
SUPPORT_PREREQUISITE_ID = "external_text_layout_support_train_only_diagnostic"


class BinaryMaskError(RuntimeError):
    pass


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.materializing")
    if path.exists() or temporary.exists():
        raise BinaryMaskError(f"refusing to overwrite diagnostic: {path}")
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_authority(
    *,
    repo_root: Path,
    plan: dict[str, Any],
    ledger: dict[str, Any],
) -> tuple[Path, Path, Path]:
    active = ledger.get("active_iteration", {})
    if active.get("id") != ACTIVE_ITERATION_ID:
        raise BinaryMaskError("active iteration changed")
    if active.get("terminal") != "PREREQUISITE_NEEDED":
        raise BinaryMaskError("active iteration terminal changed")
    statuses = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if statuses.get(SUPPORT_PREREQUISITE_ID) != "passed":
        raise BinaryMaskError("support diagnostic prerequisite is not passed")
    if statuses.get(INCREMENTAL_DIAGNOSTIC_ID) != "passed":
        raise BinaryMaskError("incremental support diagnostic prerequisite is not passed")
    binary_status = statuses.get(BINARY_PREFLIGHT_ID, "not_started")
    if binary_status not in {"not_started", "passed"}:
        raise BinaryMaskError("binary mask preflight status changed")

    diagnostic = plan.get("train_only_preserve_separation_diagnostic", {})
    if diagnostic.get("allowed_roles") != ["train"]:
        raise BinaryMaskError("binary mask diagnostic role changed")
    if diagnostic.get("candidate_inference") is not False:
        raise BinaryMaskError("candidate inference opened before diagnostic")
    if diagnostic.get("target_access") != "train_labels_only_for_binary_mask_reachability_measurement":
        raise BinaryMaskError("binary mask target role changed")
    forbidden = set(diagnostic.get("validation_roles_forbidden", []))
    if forbidden != {"inner_val15", "scut115", "holdout40", "reserved_blind"}:
        raise BinaryMaskError("binary mask validation boundary changed")
    support_path = validate_artifact(
        repo_root,
        plan["evidence"]["support_diagnostic"],
        "support diagnostic",
    )
    validate_artifact(
        repo_root,
        plan["evidence"]["incremental_support_diagnostic"],
        "incremental support diagnostic",
    )
    patch_path = validate_artifact(
        repo_root,
        plan["evidence"]["registered_patch_index"],
        "registered patch index",
    )
    preflight_path = repo_path(repo_root, str(PREFLIGHT_OUTPUT_PATH), "binary mask preflight output")
    preflight = read_json(preflight_path)
    if preflight.get("terminal") != "PASS" or preflight.get("runnable") is not True:
        raise BinaryMaskError("binary mask preflight is not PASS")
    plan_file = repo_path(repo_root, str(PLAN_PATH), "binary mask plan")
    if preflight.get("plan_sha256") != sha256_file(plan_file):
        raise BinaryMaskError("binary mask preflight plan hash changed")
    return support_path, patch_path, preflight_path


def summarize_reachability(
    *,
    rows: list[dict[str, str]],
    second_stage_dir: Path,
    label_dir: Path,
    layout_dir: Path,
    second_rows: dict[str, dict[str, str]],
    margin_gray: float,
    delta_bound_gray: float,
    gate_threshold_gray: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    page_cache: dict[str, dict[str, np.ndarray]] = {}
    page_summaries: dict[str, dict[str, Any]] = {}
    patch_summaries: list[dict[str, Any]] = []
    counts = {
        "positive": 0,
        "positive_gate": 0,
        "preserve": 0,
        "preserve_gate": 0,
    }
    patches_with_any_gate = 0
    patches_with_positive_gate = 0
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
        _features, target_delta = patch_features(page, (x1, y1, x2, y2))
        mask = page["occupancy"][y1:y2, x1:x2].reshape(-1).astype(bool)
        delta = binary_mask_to_delta_gray(mask, delta_bound_gray=delta_bound_gray)
        positive = target_delta > margin_gray
        preserve = ~positive
        gate = delta >= gate_threshold_gray
        positive_count = int(positive.sum())
        preserve_count = int(preserve.sum())
        positive_gate = int((gate & positive).sum())
        preserve_gate = int((gate & preserve).sum())
        counts["positive"] += positive_count
        counts["positive_gate"] += positive_gate
        counts["preserve"] += preserve_count
        counts["preserve_gate"] += preserve_gate
        has_any_gate = bool(gate.any())
        has_positive_gate = positive_gate > 0
        patches_with_any_gate += int(has_any_gate)
        patches_with_positive_gate += int(has_positive_gate)
        fold_id = fold_for_name(file_name)
        patch_summaries.append(
            {
                "file": file_name,
                "fold": fold_id,
                "index": index,
                "positive_gate_ratio": positive_gate / positive_count if positive_count else 0.0,
                "positive_pixels": positive_count,
                "preserve_gate_ratio": preserve_gate / preserve_count if preserve_count else 0.0,
                "preserve_pixels": preserve_count,
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
            }
        )
        page_summary = page_summaries.setdefault(
            file_name,
            {
                "file": file_name,
                "fold": fold_id,
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
        raise BinaryMaskError("diagnostic patch set lacks both target classes")
    summary = {
        "max_delta_gray": delta_bound_gray,
        "min_delta_gray": 0.0,
        "patch_count": len(rows),
        "patches_with_any_gate": patches_with_any_gate,
        "patches_with_positive_gate": patches_with_positive_gate,
        "positive_delta_mean_gray": delta_bound_gray * counts["positive_gate"] / positive_count,
        "positive_gate_ratio": counts["positive_gate"] / positive_count,
        "positive_pixels": positive_count,
        "preserve_delta_mean_gray": delta_bound_gray * counts["preserve_gate"] / preserve_count,
        "preserve_gate_ratio": counts["preserve_gate"] / preserve_count,
        "preserve_pixels": preserve_count,
        "reachable_patch_ratio": patches_with_positive_gate / len(rows),
    }
    summary["positive_over_preserve_gate_margin"] = (
        summary["positive_gate_ratio"] - summary["preserve_gate_ratio"]
    )
    return summary, patch_summaries, {
        "page_count": len(page_summaries),
        "pages": sorted(page_summaries.values(), key=lambda item: item["file"]),
    }


def evaluate_acceptance(summary: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, Any]:
    conditions = {
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
    plan = read_json(repo_path(repo_root, str(plan_path), "binary mask plan"))
    support_plan = read_json(repo_path(repo_root, str(support_plan_path), "support plan"))
    ledger = read_json(repo_path(repo_root, str(ledger_path), "ledger"))
    support_audit_path, registered_patch_path, preflight_path = validate_authority(
        repo_root=repo_root,
        plan=plan,
        ledger=ledger,
    )
    if repo_path(repo_root, str(patch_index_path), "patch index") != registered_patch_path:
        raise BinaryMaskError("patch index override is not registered")
    support_audit = read_json(support_audit_path)
    diagnostic = plan["train_only_preserve_separation_diagnostic"]
    rows = read_patch_rows(registered_patch_path, int(diagnostic["required_patch_count"]))
    second_stage_dir, label_dir, layout_dir, second_rows, _evidence = validate_support_inputs(
        repo_root=repo_root,
        support_plan=support_plan,
        support_audit=support_audit,
    )
    projection = plan["binary_mask_projection"]
    summary, patch_summaries, page_summary = summarize_reachability(
        rows=rows,
        second_stage_dir=second_stage_dir,
        label_dir=label_dir,
        layout_dir=layout_dir,
        second_rows=second_rows,
        margin_gray=float(support_plan["data"]["target_lighter_margin_gray"]),
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
        "family": FAMILY,
        "iteration_id": plan["iteration_id"],
        "mask_formula": projection["mask_formula"],
        "model_training_started": False,
        "output_dir": str(output_dir.relative_to(repo_root)),
        "page_summary": page_summary,
        "patch_index": {
            "path": str(registered_patch_path.relative_to(repo_root)),
            "sha256": sha256_file(registered_patch_path),
        },
        "patches": patch_summaries,
        "plan": {
            "path": str(repo_path(repo_root, str(plan_path), "binary mask plan").relative_to(repo_root)),
            "sha256": sha256_file(repo_path(repo_root, str(plan_path), "binary mask plan")),
        },
        "preflight": {
            "path": str(preflight_path.relative_to(repo_root)),
            "sha256": sha256_file(preflight_path),
        },
        "promotion_enabled": False,
        "quality_gate_started": False,
        "reserved_blind_authorized": False,
        "schema_version": 1,
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
    except (AuditError, OSError, PreflightError, ReachabilityError, BinaryMaskError, ValueError) as error:
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
