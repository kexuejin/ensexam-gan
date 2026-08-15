#!/usr/bin/env python3
"""Audit the external-text-layout conditioned monotonic patch materialization."""

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

from scripts.analysis.build_external_text_layout_conditioned_monotonic_patch_index import (  # noqa: E402
    CHANNEL_ORDER,
    SELECTION,
    build_conditioned_candidates,
    select_top_brighten,
    sha256_file,
    summarize_rows,
)
from scripts.analysis.build_sign_separated_residual_patch_index import (  # noqa: E402
    effective_train_filenames,
)
from scripts.analysis.validate_external_text_layout_conditioned_preflight import (  # noqa: E402
    LEDGER_PATH,
    PATCH_MATERIALIZATION_ID,
    PLAN_PATH,
    SUPPORT_PREREQUISITE_ID,
    PreflightError,
    assert_exact_plan,
    read_json,
    repo_path,
)


BASE_ROLE_PLAN_PATH = Path("docs/sign-separated-residual-data-roles.json")
DATA_ROOT = Path("data-links/samples/SCUT-HW5K-mixed-20260729")
PATCH_SUMMARY_PATH = Path(
    "outputs/external-text-layout-conditioned-monotonic-train-patches-v1/summary.json"
)
OUTPUT_PATH = Path(
    "outputs/external-text-layout-conditioned-monotonic-train-patches-v1/audit.json"
)
TILE_SIZE = 256
OVERLAP = 96
LUMINANCE_MARGIN_GRAY = 2.0
MIN_POSITIVE_RATIO = 0.001
TOP_K = 256

STRING_FIELDS = {"file"}
INTEGER_FIELDS = {"x1", "y1", "x2", "y2"}
FLOAT_FIELDS = {
    "edit_positive_mean_delta",
    "edit_positive_ratio",
    "edit_positive_score",
    "preserve_negative_ratio",
    "text_confidence_mean",
    "text_confidence_occupied_mean",
    "text_occupancy_ratio",
}


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


def validate_ledger_authority(ledger: dict[str, Any]) -> dict[str, str]:
    active = ledger.get("active_iteration", {})
    if active.get("id") != "monotonic-residual-erase-support":
        raise AuditError("active iteration changed")
    statuses = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    required = {
        SUPPORT_PREREQUISITE_ID: "passed",
        "external_text_layout_conditioned_monotonic_preflight": "passed",
        "external_text_layout_conditioned_monotonic_surface_integration": "passed",
    }
    for prerequisite, expected in required.items():
        if statuses.get(prerequisite) != expected:
            raise AuditError(f"required prerequisite is not passed: {prerequisite}")
    materialization_status = statuses.get(PATCH_MATERIALIZATION_ID, "pending")
    if materialization_status not in {"pending", "passed"}:
        raise AuditError("conditioned patch materialization status changed")
    return {
        "patch_materialization_status": materialization_status,
        "support_status": statuses[SUPPORT_PREREQUISITE_ID],
        "surface_status": statuses[
            "external_text_layout_conditioned_monotonic_surface_integration"
        ],
    }


def compare_selected_rows(
    actual: list[dict[str, str]],
    expected: list[dict[str, Any]],
) -> None:
    if len(actual) != len(expected):
        raise AuditError("conditioned patch selection count changed")
    for index, (actual_row, expected_row) in enumerate(zip(actual, expected)):
        for key in STRING_FIELDS:
            if actual_row.get(key) != str(expected_row[key]):
                raise AuditError(f"patch row {index} {key} changed")
        for key in INTEGER_FIELDS:
            if int(actual_row[key]) != int(expected_row[key]):
                raise AuditError(f"patch row {index} {key} changed")
        for key in FLOAT_FIELDS:
            if not math.isclose(
                float(actual_row[key]),
                float(expected_row[key]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise AuditError(f"patch row {index} {key} changed")


def normalize_summary_path(
    summary: dict[str, Any],
    patch_csv: Path,
) -> dict[str, Any]:
    normalized = dict(summary)
    registered_path = Path(str(normalized.get("patch_index", "")))
    expected_parts = patch_csv.parts
    if "hardcase_lists" in expected_parts:
        expected_parts = expected_parts[expected_parts.index("hardcase_lists") :]
    if (
        len(registered_path.parts) < len(expected_parts)
        or registered_path.parts[-len(expected_parts) :] != expected_parts
    ):
        raise AuditError("conditioned patch summary path changed")
    normalized["patch_index"] = str(patch_csv)
    return normalized


def validate_materialized_patch_index(
    *,
    repo_root: Path,
    plan: dict[str, Any],
    filenames: list[str],
    patch_csv: Path,
    patch_summary: Path,
) -> dict[str, Any]:
    conditioned_input = plan["conditioned_input"]
    candidates, content_hashes = build_conditioned_candidates(
        filenames=filenames,
        label_dir=repo_root / DATA_ROOT / "train" / "all_labels",
        input_dir=repo_path(repo_root, conditioned_input["rgb_root"], "rgb_root"),
        layout_dir=repo_path(repo_root, conditioned_input["layout_root"], "layout_root"),
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        luminance_margin_gray=LUMINANCE_MARGIN_GRAY,
        min_positive_ratio=MIN_POSITIVE_RATIO,
    )
    expected = select_top_brighten(candidates, TOP_K)
    actual = read_csv_rows(patch_csv)
    compare_selected_rows(actual, expected)
    expected_summary = summarize_rows(
        filenames=filenames,
        candidates=candidates,
        rows=expected,
        output_csv=patch_csv,
        content_hashes=content_hashes,
    )
    summary = normalize_summary_path(read_json(patch_summary), patch_csv)
    if summary != expected_summary:
        raise AuditError("conditioned patch summary changed")
    if summary["channel_order"] != CHANNEL_ORDER or summary["selection"] != SELECTION:
        raise AuditError("conditioned patch semantic contract changed")
    if summary["patch_count"] != TOP_K:
        raise AuditError("conditioned patch count changed")
    if summary["positive_ratio_min"] < MIN_POSITIVE_RATIO:
        raise AuditError("conditioned patch contains subfloor positive support")
    if summary["preserve_ratio_min"] <= 0.0:
        raise AuditError("conditioned patch lacks preserve-negative coverage")
    return summary


def validate_forbidden_outputs(repo_root: Path, plan: dict[str, Any]) -> list[str]:
    outputs = plan["planned_outputs_must_be_absent"]
    forbidden = {
        "checkpoint_audit",
        "first_gate_candidate",
        "first_gate_score",
        "training_output_dir",
    }
    absent: list[str] = []
    for label in sorted(forbidden):
        path = repo_path(repo_root, outputs[label], f"planned output {label}")
        if path.exists():
            raise AuditError(f"training, candidate, or quality output exists: {path}")
        absent.append(str(path))
    return absent


def run_audit(
    *,
    repo_root: Path = ROOT,
    plan_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_plan = plan_path or (repo_root / PLAN_PATH)
    resolved_ledger = ledger_path or (repo_root / LEDGER_PATH)
    try:
        plan = read_json(resolved_plan)
        assert_exact_plan(plan)
        ledger = read_json(resolved_ledger)
        authority = validate_ledger_authority(ledger)
        filenames = effective_train_filenames(repo_root, repo_root / BASE_ROLE_PLAN_PATH)
        outputs = plan["planned_outputs_must_be_absent"]
        patch_csv = repo_path(repo_root, outputs["patch_index"], "patch_index")
        patch_summary = repo_root / PATCH_SUMMARY_PATH
        summary = validate_materialized_patch_index(
            repo_root=repo_root,
            plan=plan,
            filenames=filenames,
            patch_csv=patch_csv,
            patch_summary=patch_summary,
        )
        absent = validate_forbidden_outputs(repo_root, plan)
    except (AuditError, KeyError, OSError, PreflightError, TypeError, ValueError) as exc:
        return {
            "reason": str(exc),
            "runnable": False,
            "terminal": "PREREQUISITE_NEEDED",
        }
    return {
        "absent_later_outputs": absent,
        "authority": authority,
        "candidate_inference_started": False,
        "checkpoint_generated": False,
        "first_quality_gate_started": False,
        "patch_index_sha256": sha256_file(patch_csv),
        "patch_summary": summary,
        "patch_summary_sha256": sha256_file(patch_summary),
        "promotion_enabled": False,
        "runnable": True,
        "schema_version": 1,
        "target_decode_roles": ["train"],
        "terminal": "PASS",
        "training_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    result = run_audit(
        repo_root=args.repo_root,
        plan_path=args.plan,
        ledger_path=args.ledger,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        path = args.output_json
        if not path.is_absolute():
            path = args.repo_root.resolve() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["terminal"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
