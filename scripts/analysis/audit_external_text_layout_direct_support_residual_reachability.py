#!/usr/bin/env python3
"""Audit direct support-score residual reachability on registered train patches."""

from __future__ import annotations

import argparse
import csv
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
    metric_float,
    read_metric_rows,
    validate_label_set,
    validate_prediction_set,
)
from scripts.analysis.audit_external_text_layout_support import (  # noqa: E402
    load_layout_npz,
    read_rgb,
)
from scripts.analysis.validate_external_text_layout_conditioned_preflight import (  # noqa: E402
    validate_artifact,
)
from scripts.analysis.validate_external_text_layout_direct_support_residual_preflight import (  # noqa: E402
    DIRECT_SUPPORT_PREFLIGHT_ID,
    LEDGER_PATH,
    PLAN_PATH,
    PreflightError,
    read_json,
    repo_path,
    score_to_delta_gray,
    sha256_file,
)


SUPPORT_PLAN_PATH = Path("docs/external-text-layout-support-prerequisite-v1.json")
PATCH_INDEX_PATH = Path("hardcase_lists/external-text-layout-conditioned-monotonic-train-patches-v1.csv")
OUTPUT_DIR = Path("outputs/external-text-layout-direct-support-residual-proposal-diagnostic-20260815")
OUTPUT_PATH = OUTPUT_DIR / "audit.json"
FAMILY = "external_text_layout_direct_support_residual_v1"
EXPECTED_PATCH_FIELDS = {
    "edit_positive_mean_delta",
    "edit_positive_ratio",
    "edit_positive_score",
    "file",
    "preserve_negative_ratio",
    "text_confidence_mean",
    "text_confidence_occupied_mean",
    "text_occupancy_ratio",
    "x1",
    "x2",
    "y1",
    "y2",
}


class ReachabilityError(RuntimeError):
    pass


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.materializing")
    if path.exists() or temporary.exists():
        raise ReachabilityError(f"refusing to overwrite diagnostic: {path}")
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_patch_rows(path: Path, expected_count: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_count:
        raise ReachabilityError(
            f"expected {expected_count} registered patches, got {len(rows)}"
        )
    if not rows or set(rows[0]) != EXPECTED_PATCH_FIELDS:
        raise ReachabilityError("registered patch-index schema changed")
    return rows


def parse_patch(row: dict[str, str]) -> tuple[str, int, int, int, int]:
    file_name = row["file"]
    if Path(file_name).name != file_name:
        raise ReachabilityError(f"invalid patch filename: {file_name}")
    try:
        x1, y1, x2, y2 = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ReachabilityError(f"invalid patch coordinates for {file_name}") from error
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ReachabilityError(f"invalid patch bounds for {file_name}")
    return file_name, x1, y1, x2, y2


def support_scores(features: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    mean = np.asarray(fit.get("feature_mean"), dtype=np.float64)
    scale = np.asarray(fit.get("feature_scale"), dtype=np.float64)
    weights = np.asarray(fit.get("weights"), dtype=np.float64)
    if values.ndim != 2 or mean.shape != (values.shape[1],) or scale.shape != mean.shape:
        raise ReachabilityError("support fit feature dimensions changed")
    if weights.shape != (values.shape[1] + 1,):
        raise ReachabilityError("support fit weights changed")
    if not np.isfinite(values).all() or not np.isfinite(mean).all():
        raise ReachabilityError("non-finite support features")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ReachabilityError("invalid support fit scale")
    if not np.isfinite(weights).all():
        raise ReachabilityError("non-finite support fit weights")
    standardized = (values - mean) / scale
    design = np.column_stack([standardized, np.ones(len(standardized), dtype=np.float64)])
    scores = np.sum(design * weights.reshape(1, -1), axis=1)
    if not np.isfinite(scores).all():
        raise ReachabilityError("non-finite support scores")
    return scores


def fold_contracts(support_audit: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if support_audit.get("terminal") != "PASS":
        raise ReachabilityError("support diagnostic is not PASS")
    folds = support_audit.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ReachabilityError("support diagnostic fold count changed")
    by_fold: dict[int, dict[str, Any]] = {}
    for fold in folds:
        if not isinstance(fold, dict):
            raise ReachabilityError("support diagnostic fold schema changed")
        fold_id = int(fold.get("fold"))
        if fold_id in by_fold:
            raise ReachabilityError("duplicate support fold")
        positive_center = float(fold.get("positive_score_mean"))
        preserve_center = float(fold.get("preserve_score_mean"))
        if positive_center <= preserve_center:
            raise ReachabilityError("support fold score centers are not ordered")
        by_fold[fold_id] = {
            "fit": fold["full_fit"],
            "positive_center": positive_center,
            "preserve_center": preserve_center,
        }
    if set(by_fold) != set(range(5)):
        raise ReachabilityError("support diagnostic fold identities changed")
    return by_fold


def validate_authority(
    *,
    repo_root: Path,
    plan: dict[str, Any],
    ledger: dict[str, Any],
) -> tuple[Path, Path]:
    active = ledger.get("active_iteration", {})
    if active.get("id") != "monotonic-residual-erase-support":
        raise ReachabilityError("active iteration changed")
    if active.get("terminal") != "PREREQUISITE_NEEDED":
        raise ReachabilityError("active iteration terminal changed")
    statuses = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if statuses.get(DIRECT_SUPPORT_PREFLIGHT_ID) != "passed":
        raise ReachabilityError("direct support preflight is not passed")
    diagnostic = plan.get("train_only_reachability_diagnostic", {})
    if diagnostic.get("allowed_roles") != ["train"]:
        raise ReachabilityError("direct support diagnostic role changed")
    if diagnostic.get("candidate_inference") is not False:
        raise ReachabilityError("candidate inference opened before diagnostic")
    if diagnostic.get("fit_source") != "support_diagnostic_fold_fits_only":
        raise ReachabilityError("direct support fit source changed")
    if diagnostic.get("target_access") != "train_labels_only_for_reachability_measurement":
        raise ReachabilityError("direct support target role changed")
    forbidden = set(diagnostic.get("validation_roles_forbidden", []))
    if forbidden != {"inner_val15", "scut115", "holdout40", "reserved_blind"}:
        raise ReachabilityError("direct support validation boundary changed")
    support_path = validate_artifact(
        repo_root,
        plan["evidence"]["support_diagnostic"],
        "support diagnostic",
    )
    validate_artifact(
        repo_root,
        plan["evidence"]["conditioned_checkpoint_audit"],
        "conditioned checkpoint audit",
    )
    patch_path = repo_path(repo_root, str(PATCH_INDEX_PATH), "patch index")
    if not patch_path.is_file():
        raise ReachabilityError(f"missing patch index: {patch_path}")
    return support_path, patch_path


def validate_support_inputs(
    *,
    repo_root: Path,
    support_plan: dict[str, Any],
    support_audit: dict[str, Any],
) -> tuple[Path, Path, Path, dict[str, dict[str, Any]], dict[str, str]]:
    evidence = support_plan["evidence"]
    file_names = [row["file"] for row in support_audit["page_samples"]]
    if len(file_names) != 275 or len(set(file_names)) != 275:
        raise ReachabilityError("support audit page population changed")
    second_stage_dir, _summary = validate_prediction_set(
        repo_root,
        evidence["second_stage_prediction_set"],
        file_names,
        "second-stage prediction set",
    )
    second_rows = read_metric_rows(repo_path(repo_root, evidence["second_stage_metrics"]["path"], "second-stage metrics"))
    if set(second_rows) != set(file_names):
        raise ReachabilityError("second-stage metric identities changed")
    label_dir = repo_path(repo_root, evidence["train_label_set"]["directory"], "train labels")
    validate_label_set(label_dir, file_names, evidence["train_label_set"])
    materialization = support_audit.get("materialization", {})
    layout_manifest = repo_path(repo_root, materialization.get("manifest_path"), "layout manifest")
    if sha256_file(layout_manifest) != materialization.get("manifest_sha256"):
        raise ReachabilityError("layout manifest hash changed")
    layout_dir = layout_manifest.parent / "pages"
    if not layout_dir.is_dir():
        raise ReachabilityError("layout page directory missing")
    return second_stage_dir, label_dir, layout_dir, second_rows, evidence


def find_prediction(directory: Path, file_name: str) -> Path:
    path = directory / f"{Path(file_name).stem}.png"
    if path.is_file():
        return path
    raise ReachabilityError(f"missing second-stage prediction: {file_name}")


def load_page(
    *,
    file_name: str,
    second_stage_dir: Path,
    label_dir: Path,
    layout_dir: Path,
    second_stage_row: dict[str, str],
) -> dict[str, np.ndarray]:
    second_stage = read_rgb(find_prediction(second_stage_dir, file_name))
    target = read_rgb(label_dir / file_name)
    if second_stage.shape != target.shape:
        raise ReachabilityError(f"image shape mismatch for {file_name}")
    arrays = load_layout_npz(
        layout_dir / f"{Path(file_name).stem}.npz",
        expected_shape=second_stage.shape[:2],
    )
    if (
        metric_float(second_stage_row, "base_edit_threshold", file_name) != 12.0
        or metric_float(second_stage_row, "second_delta_threshold", file_name) != 32.0
        or metric_float(second_stage_row, "dark_threshold", file_name) != 0.0
    ):
        raise ReachabilityError(f"second-stage protocol changed for {file_name}")
    return {
        "confidence": arrays["text_confidence"],
        "occupancy": arrays["text_occupancy"].astype(np.float32),
        "second_stage": second_stage.astype(np.float32),
        "target": target.astype(np.float32),
    }


def patch_features(page: dict[str, np.ndarray], bounds: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bounds
    second_stage = page["second_stage"]
    height, width = second_stage.shape[:2]
    if x2 > width or y2 > height:
        raise ReachabilityError("patch lies outside page")
    rgb = second_stage[y1:y2, x1:x2].reshape(-1, 3) / 255.0
    occupancy = page["occupancy"][y1:y2, x1:x2].reshape(-1, 1)
    confidence = page["confidence"][y1:y2, x1:x2].reshape(-1, 1)
    features = np.column_stack([rgb, occupancy, confidence]).astype(np.float32, copy=False)
    target_delta = (
        page["target"][y1:y2, x1:x2].mean(axis=2)
        - second_stage[y1:y2, x1:x2].mean(axis=2)
    ).reshape(-1)
    return features, target_delta


def summarize_reachability(
    *,
    rows: list[dict[str, str]],
    folds: dict[int, dict[str, Any]],
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
        fold = folds[fold_id]
        features, target_delta = patch_features(page, (x1, y1, x2, y2))
        scores = support_scores(features, fold["fit"])
        delta = score_to_delta_gray(
            scores,
            preserve_center=fold["preserve_center"],
            positive_center=fold["positive_center"],
            delta_bound_gray=delta_bound_gray,
        )
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
        patch_summary = {
            "file": file_name,
            "fold": fold_id,
            "index": index,
            "positive_gate_ratio": positive_gate / positive_count if positive_count else 0.0,
            "positive_pixels": positive_count,
            "preserve_gate_ratio": preserve_gate / preserve_count if preserve_count else 0.0,
            "preserve_pixels": preserve_count,
            "score_mean": float(scores.mean()),
            "x1": x1,
            "x2": x2,
            "y1": y1,
            "y2": y2,
        }
        patch_summaries.append(patch_summary)
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
        raise ReachabilityError("diagnostic patch set lacks both target classes")
    summary = {
        "max_delta_gray": max_delta,
        "min_delta_gray": min_delta,
        "patch_count": len(rows),
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
    plan = read_json(repo_path(repo_root, str(plan_path), "direct support plan"))
    support_plan = read_json(repo_path(repo_root, str(support_plan_path), "support plan"))
    ledger = read_json(repo_path(repo_root, str(ledger_path), "ledger"))
    support_audit_path, registered_patch_path = validate_authority(
        repo_root=repo_root,
        plan=plan,
        ledger=ledger,
    )
    if repo_path(repo_root, str(patch_index_path), "patch index") != registered_patch_path:
        raise ReachabilityError("patch index override is not registered")
    support_audit = read_json(support_audit_path)
    folds = fold_contracts(support_audit)
    diagnostic = plan["train_only_reachability_diagnostic"]
    rows = read_patch_rows(registered_patch_path, int(diagnostic["required_patch_count"]))
    second_stage_dir, label_dir, layout_dir, second_rows, _evidence = validate_support_inputs(
        repo_root=repo_root,
        support_plan=support_plan,
        support_audit=support_audit,
    )
    projection = plan["direct_support_projection"]
    summary, patch_summaries, page_summary = summarize_reachability(
        rows=rows,
        folds=folds,
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
        "fit_source": diagnostic["fit_source"],
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
            "path": str(repo_path(repo_root, str(plan_path), "direct support plan").relative_to(repo_root)),
            "sha256": sha256_file(repo_path(repo_root, str(plan_path), "direct support plan")),
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
    except (AuditError, OSError, PreflightError, ReachabilityError, ValueError) as error:
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
