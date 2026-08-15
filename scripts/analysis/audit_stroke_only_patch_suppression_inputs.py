#!/usr/bin/env python3
"""Audit stroke-only patch suppression preflight input custody.

This is a fail-closed prerequisite check. It does not decode images, generate
predictions, train models, or open validation/blind surfaces. It only verifies
that the train/train160 source rows and their required pixel artifacts are
present before the registered stroke-only generator is allowed to run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.validate_external_text_layout_conditioned_preflight import (  # noqa: E402
    ACTIVE_ITERATION_ID,
    LEDGER_PATH,
    PreflightError,
    read_json,
    repo_path,
    validate_artifact,
)


PLAN_PATH = Path("docs/stroke-only-patch-suppression-preflight-v1.json")
REVIEW_CSV_PATH = Path("docs/product-quality-review-pages.csv")
SELECTOR_REPLAY_PATH = Path(
    "outputs/selector_replay_exact129_outside_edit_lam16_union_train160_20260706/page_choices.csv"
)
OUTPUT_PATH = Path(
    "outputs/stroke-only-patch-suppression-input-custody-audit-20260815/audit.json"
)
PREFLIGHT_ID = "stroke_only_patch_suppression_preflight"
PREREGISTRATION_RECORD_ID = "stroke-only-patch-suppression-preregistration"
PREREGISTRATION_OUTCOME = (
    "non_layout_source_dark_stroke_only_patch_suppression_frozen_pending_train_only_preflight"
)
DEFAULT_REQUIRED_BUCKET = "selector_false_positive_overerase_risk"
DEFAULT_REQUIRED_CANDIDATE = "exact129_lam16_relaxed_interval_rejected"
REQUIRED_FIELDS = (
    "source_input",
    "baseline_pred",
    "candidate_pred",
    "target",
)
SELECTOR_REPLAY_FIELD_MAP = {
    "baseline_pred": "baseline_pred_path",
    "candidate_pred": "candidate_pred_path",
    "source_input": "image_path",
}
EXPECTED_PLANNED_OUTPUTS_MUST_BE_ABSENT = {
    "holdout40_candidate": "outputs/stroke-only-patch-suppression-holdout40-candidate",
    "inner_val15_candidate": "outputs/stroke-only-patch-suppression-inner-val15-candidate",
    "preflight_output": "outputs/stroke-only-patch-suppression-preflight-20260815",
    "reserved_blind_candidate": (
        "outputs/stroke-only-patch-suppression-reserved-blind-candidate"
    ),
    "scut115_candidate": "outputs/stroke-only-patch-suppression-scut115-candidate",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise PreflightError(f"missing review CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise PreflightError(f"review CSV has no rows: {path}")
    return rows


def assert_exact_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != 1:
        raise PreflightError("stroke-only plan schema changed")
    if plan.get("state") != "preregistered_pending_stroke_only_patch_suppression_preflight":
        raise PreflightError("stroke-only plan state changed")
    if plan.get("iteration_id") != ACTIVE_ITERATION_ID:
        raise PreflightError("stroke-only iteration changed")
    if plan.get("family") != "stroke_only_patch_suppression_v1":
        raise PreflightError("stroke-only family changed")
    authorization = plan.get("authorization", {})
    for key in (
        "model_training",
        "checkpoint_generation",
        "candidate_inference",
        "inner_val15",
        "development_gate",
        "scut115",
        "holdout40",
        "reserved_blind",
        "visual_review",
        "promotion",
        "current_primary_replacement",
    ):
        if authorization.get(key) is not False:
            raise PreflightError(f"stroke-only authorization opened: {key}")
    if plan.get("planned_outputs_must_be_absent") != EXPECTED_PLANNED_OUTPUTS_MUST_BE_ABSENT:
        raise PreflightError("stroke-only planned outputs changed")
    allowed = plan.get("inputs", {}).get("allowed_default_splits")
    if allowed != ["train", "train160"]:
        raise PreflightError("stroke-only allowed splits changed")
    if plan.get("stroke_only_mask", {}).get("paper_background_must_remain_baseline") is not True:
        raise PreflightError("stroke-only preserve contract changed")
    evidence = plan.get("evidence")
    if not isinstance(evidence, dict) or not {"implementation", "test"}.issubset(evidence):
        raise PreflightError("stroke-only evidence set changed")


def validate_ledger_authority(repo_root: Path, ledger: dict[str, Any]) -> dict[str, str]:
    active = ledger.get("active_iteration", {})
    if active.get("id") != ACTIVE_ITERATION_ID:
        raise PreflightError("active iteration changed")
    if active.get("terminal") != "PREREQUISITE_NEEDED":
        raise PreflightError("active iteration terminal changed")
    prerequisites = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if prerequisites.get(PREFLIGHT_ID) != "pending":
        raise PreflightError("stroke-only preflight is not pending")
    records = [
        item
        for item in ledger.get("records", [])
        if isinstance(item, dict) and item.get("id") == PREREGISTRATION_RECORD_ID
    ]
    if len(records) != 1:
        raise PreflightError("stroke-only preregistration record count changed")
    record = records[0]
    if record.get("terminal") != "PREREQUISITE_NEEDED":
        raise PreflightError("stroke-only preregistration terminal changed")
    if record.get("outcome") != PREREGISTRATION_OUTCOME:
        raise PreflightError("stroke-only preregistration outcome changed")
    for item in record.get("evidence", []):
        validate_artifact(repo_root, item, "stroke-only preregistration evidence")
    return {
        "active_iteration": ACTIVE_ITERATION_ID,
        "preregistration": "PREREQUISITE_NEEDED",
        "stroke_only_patch_suppression_preflight": "pending",
    }


def select_authorized_rows(
    rows: list[dict[str, str]],
    *,
    allowed_splits: set[str],
    required_bucket: str,
    required_candidate: str,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    selected = [
        row
        for row in rows
        if row.get("bucket") == required_bucket
        and row.get("candidate") == required_candidate
    ]
    if not selected:
        raise PreflightError("no stroke-only source rows selected")
    disallowed = sorted({row.get("split", "") for row in selected if row.get("split") not in allowed_splits})
    if disallowed:
        raise PreflightError(
            "stroke-only source rows include split outside authority: "
            + ", ".join(disallowed)
        )
    split_counts = Counter(row.get("split", "") for row in selected)
    return selected, dict(sorted(split_counts.items()))


def validate_selector_replay_alignment(
    repo_root: Path,
    rows: list[dict[str, str]],
    *,
    selector_replay_path: Path,
) -> list[dict[str, Any]]:
    selector_path = selector_replay_path
    if not selector_path.is_absolute():
        selector_path = repo_root / selector_path
    replay_rows = read_csv_rows(selector_path)
    replay_by_key = {
        (row.get("split", ""), row.get("file", "")): row
        for row in replay_rows
    }
    alignment: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("split", ""), row.get("file", ""))
        sample_key = f"{key[0]}/{key[1]}"
        replay_row = replay_by_key.get(key)
        if replay_row is None:
            raise PreflightError(f"selector replay missing source row: {sample_key}")
        aligned_paths: dict[str, str] = {}
        for review_field, replay_field in SELECTOR_REPLAY_FIELD_MAP.items():
            review_value = row.get(review_field, "")
            replay_value = replay_row.get(replay_field, "")
            if not replay_value:
                raise PreflightError(
                    f"selector replay missing {sample_key}.{replay_field}"
                )
            if review_value != replay_value:
                raise PreflightError(
                    f"review CSV {sample_key}.{review_field} does not match "
                    f"selector replay {replay_field}: {review_value} != {replay_value}"
                )
            aligned_paths[review_field] = replay_value
        alignment.append(
            {
                "candidate_overerase_ratio": replay_row.get("candidate_overerase_ratio"),
                "candidate_residual_ratio": replay_row.get("candidate_residual_ratio"),
                "copy_mask_cov8": replay_row.get("copy_mask_cov8"),
                "overerase_regret": replay_row.get("overerase_regret"),
                "paths": aligned_paths,
                "primary_edit_px": replay_row.get("primary_edit_px"),
                "residual_gain": replay_row.get("residual_gain"),
                "sample_key": sample_key,
            }
        )
    return alignment


def validate_required_paths(
    repo_root: Path,
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    present: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in rows:
        sample_key = f"{row.get('split', '')}/{row.get('file', '')}"
        for field in REQUIRED_FIELDS:
            raw_value = row.get(field)
            try:
                path = repo_path(repo_root, raw_value, f"{sample_key}.{field}")
            except PreflightError as error:
                missing.append(
                    {
                        "field": field,
                        "path": raw_value or "",
                        "reason": str(error),
                        "sample_key": sample_key,
                    }
                )
                continue
            if path.is_file():
                present.append(
                    {
                        "field": field,
                        "path": str(path.relative_to(repo_root)),
                        "sample_key": sample_key,
                        "sha256": sha256_file(path),
                    }
                )
            else:
                missing.append(
                    {
                        "field": field,
                        "path": str(path.relative_to(repo_root)),
                        "reason": "missing_file",
                        "sample_key": sample_key,
                    }
                )
    return present, missing


def validate_outputs_absent(repo_root: Path, plan: dict[str, Any]) -> list[str]:
    absent: list[str] = []
    for label, value in sorted(plan["planned_outputs_must_be_absent"].items()):
        path = repo_path(repo_root, value, f"planned output {label}")
        if path.exists():
            raise PreflightError(f"planned output must be absent: {path}")
        absent.append(str(path.relative_to(repo_root)))
    return absent


def run_audit(
    *,
    repo_root: Path = ROOT,
    plan_path: Path | None = None,
    ledger_path: Path | None = None,
    review_csv: Path | None = None,
    selector_replay_path: Path | None = None,
    required_bucket: str = DEFAULT_REQUIRED_BUCKET,
    required_candidate: str = DEFAULT_REQUIRED_CANDIDATE,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_plan = plan_path or (repo_root / PLAN_PATH)
    resolved_ledger = ledger_path or (repo_root / LEDGER_PATH)
    resolved_review_csv = review_csv or (repo_root / REVIEW_CSV_PATH)
    resolved_selector_replay = selector_replay_path or SELECTOR_REPLAY_PATH
    if not resolved_selector_replay.is_absolute():
        resolved_selector_replay = repo_root / resolved_selector_replay
    try:
        plan = read_json(resolved_plan)
        assert_exact_plan(plan)
        ledger = read_json(resolved_ledger)
        authority = validate_ledger_authority(repo_root, ledger)
        absent = validate_outputs_absent(repo_root, plan)
        rows = read_csv_rows(resolved_review_csv)
        allowed_splits = set(plan["inputs"]["allowed_default_splits"])
        selected_rows, split_counts = select_authorized_rows(
            rows,
            allowed_splits=allowed_splits,
            required_bucket=required_bucket,
            required_candidate=required_candidate,
        )
        selector_replay_alignment = validate_selector_replay_alignment(
            repo_root,
            selected_rows,
            selector_replay_path=resolved_selector_replay,
        )
        present_paths, missing_paths = validate_required_paths(repo_root, selected_rows)
    except (OSError, KeyError, PreflightError, TypeError, ValueError) as error:
        return {
            "reason": str(error),
            "runnable": False,
            "terminal": "PREREQUISITE_NEEDED",
        }

    terminal = "PASS" if not missing_paths else "PREREQUISITE_NEEDED"
    return {
        "allowed_splits": sorted(allowed_splits),
        "authority": authority,
        "candidate_inference_started": False,
        "checkpoint_generated": False,
        "missing_required_path_count": len(missing_paths),
        "missing_required_paths": missing_paths,
        "model_training_started": False,
        "planned_outputs_absent": absent,
        "present_required_path_count": len(present_paths),
        "present_required_paths": present_paths,
        "promotion_enabled": False,
        "quality_gate_started": False,
        "reason": (
            "all_required_paths_present"
            if not missing_paths
            else "missing_required_train_only_source_artifacts"
        ),
        "required_bucket": required_bucket,
        "required_candidate": required_candidate,
        "reserved_blind_state": "unavailable",
        "review_csv": str(resolved_review_csv.relative_to(repo_root)),
        "runnable": not missing_paths,
        "schema_version": 1,
        "selected_row_count": len(selected_rows),
        "selector_replay_alignment": selector_replay_alignment,
        "selector_replay_csv": str(resolved_selector_replay.relative_to(repo_root)),
        "selected_split_counts": split_counts,
        "target_decode": False,
        "terminal": terminal,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--selector-replay", type=Path)
    parser.add_argument("--required-bucket", default=DEFAULT_REQUIRED_BUCKET)
    parser.add_argument("--required-candidate", default=DEFAULT_REQUIRED_CANDIDATE)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    result = run_audit(
        repo_root=args.repo_root,
        plan_path=args.plan,
        ledger_path=args.ledger,
        review_csv=args.review_csv,
        selector_replay_path=args.selector_replay,
        required_bucket=args.required_bucket,
        required_candidate=args.required_candidate,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        output = args.output_json
        if not output.is_absolute():
            output = args.repo_root.resolve() / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["terminal"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
