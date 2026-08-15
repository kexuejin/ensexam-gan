#!/usr/bin/env python3
"""Fail closed status report for the current-primary quality-improvement loop.

The legacy selector report depends on disposable fixed-set outputs. This tool
instead validates a versioned ledger against the frozen current-primary files
and decision records that define the active generalization program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RECORD_TERMINALS = {"PASS", "KILL", "PREREQUISITE_NEEDED", "PENDING"}
ACTIVE_TERMINALS = {"PREREQUISITE_NEEDED", "PENDING"}
TRACKED_CODE_PREFIXES = {"networks", "scripts", "tests", "tools"}
NONBLOCKING_GAP_CLASSES = {"tracked_code_historical_drift"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "docs" / "current-primary-quality-loop-ledger.json",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--evidence-audit-json", type=Path)
    parser.add_argument(
        "--evidence-audit-only",
        action="store_true",
        help=(
            "Report referenced evidence presence and hashes without claiming "
            "the full quality-loop status is valid."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def require_sha256(value: Any, label: str) -> str:
    digest = require_string(value, label)
    if not SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def repo_path(repo_root: Path, raw_path: Any, label: str) -> Path:
    relative = Path(require_string(raw_path, f"{label}.path"))
    if relative.is_absolute():
        raise ValueError(f"{label}.path must be repository-relative")
    if ".." in relative.parts:
        raise ValueError(f"{label}.path must not contain parent traversal: {relative}")
    # Registered artifacts are symlinks in a clean worktree. Keep the lexical
    # repository containment check while allowing those explicit symlink targets.
    return repo_root.absolute() / relative


def validate_artifact(repo_root: Path, artifact: Any, label: str) -> dict[str, str]:
    data = require_mapping(artifact, label)
    path = repo_path(repo_root, data.get("path"), label)
    expected_sha256 = require_sha256(data.get("sha256"), f"{label}.sha256")
    if not path.is_file():
        raise ValueError(f"{label}.path is not a file: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label}.sha256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )
    return {"path": str(path.relative_to(repo_root.absolute())), "sha256": actual_sha256}


def iter_declared_artifacts(value: Any, label: str = "ledger") -> list[tuple[str, Any]]:
    artifacts: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            artifacts.append((label, value))
        for key, child in value.items():
            artifacts.extend(iter_declared_artifacts(child, f"{label}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            artifacts.extend(iter_declared_artifacts(child, f"{label}[{index}]"))
    return artifacts


def path_prefix_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            prefix = "<invalid>"
        else:
            prefix = raw_path.split("/", 1)[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items()))


def unique_path_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raw_path = "<invalid>"
        counts[raw_path] = counts.get(raw_path, 0) + 1
    return dict(sorted(counts.items()))


def find_git_history_sha256_match(
    repo_root: Path, relative_path: str, expected_sha256: str
) -> dict[str, str] | None:
    if not (repo_root / ".git").exists():
        return None
    try:
        log = subprocess.run(
            ["git", "-C", str(repo_root), "log", "--all", "--format=%H", "--", relative_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    for commit in log.stdout.splitlines():
        if not commit:
            continue
        try:
            show = subprocess.run(
                ["git", "-C", str(repo_root), "show", f"{commit}:{relative_path}"],
                check=False,
                capture_output=True,
            )
        except OSError:
            continue
        if show.returncode != 0:
            continue
        if hashlib.sha256(show.stdout).hexdigest() == expected_sha256:
            return {"commit": commit, "short_commit": commit[:7]}
    return None


def evidence_gap_class(item: dict[str, Any]) -> str:
    raw_path = item.get("path")
    status = item.get("status")
    prefix = raw_path.split("/", 1)[0] if isinstance(raw_path, str) and raw_path else ""
    if status == "missing":
        if prefix == "artifacts":
            return "missing_ignored_artifact"
        if prefix == "outputs":
            return "missing_ignored_output"
        if prefix in {"configs", "docs", "hardcase_lists", "scripts", "tests", "tools"}:
            return "missing_tracked_reference"
        return "missing_other_reference"
    if status == "sha256_mismatch":
        if prefix in TRACKED_CODE_PREFIXES:
            if item.get("historical_git_match") is True:
                return "tracked_code_historical_drift"
            return "tracked_code_hash_drift"
        if prefix == "docs":
            return "tracked_evidence_hash_drift"
        if prefix in {"artifacts", "outputs"}:
            return "ignored_evidence_hash_drift"
        return "other_hash_drift"
    return "invalid_reference"


def gap_class_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts: dict[str, int] = {}
    class_unique_paths: dict[str, dict[str, int]] = {}
    for item in items:
        gap_class = evidence_gap_class(item)
        class_counts[gap_class] = class_counts.get(gap_class, 0) + 1
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raw_path = "<invalid>"
        paths = class_unique_paths.setdefault(gap_class, {})
        paths[raw_path] = paths.get(raw_path, 0) + 1
    return {
        "gap_class_counts": dict(sorted(class_counts.items())),
        "gap_class_unique_path_counts": {
            gap_class: len(paths) for gap_class, paths in sorted(class_unique_paths.items())
        },
        "gap_class_unique_paths": {
            gap_class: dict(sorted(paths.items()))
            for gap_class, paths in sorted(class_unique_paths.items())
        },
    }


def successor_readiness(gap_summary: dict[str, Any]) -> dict[str, Any]:
    class_counts = require_mapping(gap_summary.get("gap_class_counts"), "gap_class_counts")
    class_paths = require_mapping(
        gap_summary.get("gap_class_unique_paths"), "gap_class_unique_paths"
    )
    blocking_class_counts = {
        name: int(count)
        for name, count in sorted(class_counts.items())
        if name not in NONBLOCKING_GAP_CLASSES
    }
    nonblocking_class_counts = {
        name: int(count)
        for name, count in sorted(class_counts.items())
        if name in NONBLOCKING_GAP_CLASSES
    }
    blocking_unique_paths = {
        name: paths
        for name, paths in sorted(class_paths.items())
        if name in blocking_class_counts
    }
    nonblocking_unique_paths = {
        name: paths
        for name, paths in sorted(class_paths.items())
        if name in nonblocking_class_counts
    }
    blocked = bool(blocking_class_counts)
    return {
        "status": (
            "blocked_by_unresolved_evidence"
            if blocked
            else "not_blocked_by_evidence_audit"
        ),
        "blocking_gap_class_counts": blocking_class_counts,
        "blocking_gap_unique_path_counts": {
            name: len(paths) for name, paths in blocking_unique_paths.items()
        },
        "blocking_gap_unique_paths": blocking_unique_paths,
        "nonblocking_gap_class_counts": nonblocking_class_counts,
        "nonblocking_gap_unique_path_counts": {
            name: len(paths) for name, paths in nonblocking_unique_paths.items()
        },
        "nonblocking_gap_unique_paths": nonblocking_unique_paths,
        "next_action": (
            "repair_or_record_decision_for_blocking_evidence_gaps_before_successor_selection"
            if blocked
            else "successor_selection_may_continue_subject_to_ledger_gate_order"
        ),
    }


def audit_declared_evidence(ledger: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    mismatched: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    for label, artifact in iter_declared_artifacts(ledger):
        entry: dict[str, Any] = {"label": label}
        try:
            path = repo_path(repo_root, artifact.get("path"), label)
            expected_sha256 = require_sha256(artifact.get("sha256"), f"{label}.sha256")
            entry.update(
                {
                    "path": str(path.relative_to(repo_root.absolute())),
                    "expected_sha256": expected_sha256,
                }
            )
            if not path.is_file():
                entry["status"] = "missing"
                missing.append(entry)
            else:
                actual_sha256 = sha256_file(path)
                entry["actual_sha256"] = actual_sha256
                if actual_sha256 != expected_sha256:
                    entry["status"] = "sha256_mismatch"
                    raw_path = entry["path"]
                    prefix = raw_path.split("/", 1)[0]
                    if prefix in TRACKED_CODE_PREFIXES:
                        match = find_git_history_sha256_match(
                            repo_root, raw_path, expected_sha256
                        )
                        entry["historical_git_match"] = match is not None
                        if match is not None:
                            entry["historical_git_commit"] = match["commit"]
                            entry["historical_git_short_commit"] = match["short_commit"]
                    mismatched.append(entry)
                else:
                    entry["status"] = "ok"
                    checked.append(entry)
        except Exception as exc:
            entry.update(
                {
                    "path": artifact.get("path") if isinstance(artifact, dict) else None,
                    "status": "invalid_reference",
                    "reason": str(exc),
                }
            )
            invalid.append(entry)

    total = len(checked) + len(missing) + len(mismatched) + len(invalid)
    status = "evidence_complete" if total and not missing and not mismatched and not invalid else "evidence_incomplete"
    missing_unique_paths = unique_path_counts(missing)
    mismatched_unique_paths = unique_path_counts(mismatched)
    gap_summary = gap_class_summary([*missing, *mismatched, *invalid])
    return {
        "status": status,
        "artifact_reference_count": total,
        "ok_count": len(checked),
        "missing_count": len(missing),
        "missing_unique_path_count": len(missing_unique_paths),
        "missing_prefix_counts": path_prefix_counts(missing),
        "missing_unique_paths": missing_unique_paths,
        "mismatch_count": len(mismatched),
        "mismatch_unique_path_count": len(mismatched_unique_paths),
        "mismatch_prefix_counts": path_prefix_counts(mismatched),
        "mismatch_unique_paths": mismatched_unique_paths,
        "invalid_count": len(invalid),
        **gap_summary,
        "successor_readiness": successor_readiness(gap_summary),
        "missing": missing,
        "mismatched": mismatched,
        "invalid": invalid,
    }


def validate_protocol(protocol: Any) -> dict[str, Any]:
    data = require_mapping(protocol, "baseline.matched_copy_protocol")
    expected = {
        "copy_input_outside_mask": "mb",
        "copy_mask_threshold_auto": "mb_cov8_step",
        "copy_mask_dilate": 0,
        "page_overlap": 32,
        "batch_size": 8,
        "change_threshold": 12,
        "eval_threshold": 12,
    }
    if data != expected:
        raise ValueError(
            "baseline.matched_copy_protocol must exactly match the frozen inner-val15 protocol"
        )
    return expected


def validate_evidence_list(repo_root: Path, value: Any, label: str) -> list[dict[str, str]]:
    artifacts = require_list(value, label)
    if not artifacts:
        raise ValueError(f"{label} must contain at least one artifact")
    return [
        validate_artifact(repo_root, artifact, f"{label}[{index}]")
        for index, artifact in enumerate(artifacts)
    ]


def validate_baseline(repo_root: Path, value: Any) -> dict[str, Any]:
    data = require_mapping(value, "baseline")
    product_default = require_string(data.get("product_default"), "baseline.product_default")
    if product_default != "artifacts/current-primary":
        raise ValueError("baseline.product_default must be artifacts/current-primary")
    default_dir = repo_path(repo_root, product_default, "baseline.product_default")
    if not default_dir.is_dir():
        raise ValueError(f"baseline.product_default is not a directory: {default_dir}")
    return {
        "product_default": product_default,
        "config": validate_artifact(repo_root, data.get("config"), "baseline.config"),
        "checkpoint": validate_artifact(repo_root, data.get("checkpoint"), "baseline.checkpoint"),
        "inner_val15_manifest": validate_artifact(
            repo_root, data.get("inner_val15_manifest"), "baseline.inner_val15_manifest"
        ),
        "matched_copy_protocol": validate_protocol(data.get("matched_copy_protocol")),
    }


def validate_calibration(repo_root: Path, value: Any) -> dict[str, Any]:
    data = require_mapping(value, "calibration")
    if data.get("terminal") != "PASS":
        raise ValueError("calibration.terminal must be PASS")
    if data.get("scope") != "scut_inner_val15_current_primary_matched_copy":
        raise ValueError("calibration.scope must name the frozen inner-val15 matched-copy gate")
    if int(data.get("run_count", 0)) < 3:
        raise ValueError("calibration.run_count must be at least 3")
    if int(data.get("pages_per_run", 0)) != 15:
        raise ValueError("calibration.pages_per_run must be 15")
    if data.get("prediction_hashes_identical") is not True:
        raise ValueError("calibration.prediction_hashes_identical must be true")
    gain = float(data.get("minimum_residual_gain", 0.0))
    if gain < 0.0005:
        raise ValueError("calibration.minimum_residual_gain must be at least 0.0005")
    return {
        "terminal": "PASS",
        "scope": data["scope"],
        "run_count": int(data["run_count"]),
        "pages_per_run": int(data["pages_per_run"]),
        "minimum_residual_gain": gain,
        "decision": validate_artifact(repo_root, data.get("decision"), "calibration.decision"),
    }


def validate_records(repo_root: Path, value: Any) -> list[dict[str, Any]]:
    records = require_list(value, "records")
    if not records:
        raise ValueError("records must not be empty")
    seen_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        data = require_mapping(raw, f"records[{index}]")
        record_id = require_string(data.get("id"), f"records[{index}].id")
        if record_id in seen_ids:
            raise ValueError(f"records contains duplicate id: {record_id}")
        seen_ids.add(record_id)
        terminal = require_string(data.get("terminal"), f"records[{index}].terminal")
        if terminal not in RECORD_TERMINALS:
            raise ValueError(f"records[{index}].terminal is unsupported: {terminal}")
        outcome = require_string(data.get("outcome"), f"records[{index}].outcome")
        repeat_policy = require_string(data.get("repeat_policy"), f"records[{index}].repeat_policy")
        if terminal == "KILL" or outcome == "safe_no_lift":
            if repeat_policy != "do_not_repeat":
                raise ValueError(
                    f"records[{index}] must use repeat_policy=do_not_repeat for {terminal}/{outcome}"
                )
        normalized.append(
            {
                "id": record_id,
                "terminal": terminal,
                "outcome": outcome,
                "repeat_policy": repeat_policy,
                "evidence": validate_evidence_list(repo_root, data.get("evidence"), f"records[{index}].evidence"),
            }
        )
    return normalized


def validate_active_iteration(repo_root: Path, value: Any) -> dict[str, Any]:
    data = require_mapping(value, "active_iteration")
    terminal = require_string(data.get("terminal"), "active_iteration.terminal")
    if terminal not in ACTIVE_TERMINALS:
        raise ValueError("active_iteration.terminal must be PREREQUISITE_NEEDED or PENDING")
    first_gate = require_string(data.get("first_gate"), "active_iteration.first_gate")
    if first_gate != "scut_inner_val15":
        raise ValueError("active_iteration.first_gate must be scut_inner_val15")
    prerequisites = require_list(data.get("prerequisites"), "active_iteration.prerequisites")
    if not prerequisites:
        raise ValueError("active_iteration.prerequisites must not be empty")
    pending_prerequisites: list[str] = []
    for index, raw in enumerate(prerequisites):
        prerequisite = require_mapping(raw, f"active_iteration.prerequisites[{index}]")
        prerequisite_id = require_string(
            prerequisite.get("id"), f"active_iteration.prerequisites[{index}].id"
        )
        status = require_string(
            prerequisite.get("status"), f"active_iteration.prerequisites[{index}].status"
        )
        if status not in {"pending", "passed"}:
            raise ValueError(
                f"active_iteration.prerequisites[{index}].status must be pending or passed"
            )
        require_string(prerequisite.get("detail"), f"active_iteration.prerequisites[{index}].detail")
        if status == "pending":
            pending_prerequisites.append(prerequisite_id)
    if terminal == "PREREQUISITE_NEEDED" and not pending_prerequisites:
        raise ValueError("active_iteration with PREREQUISITE_NEEDED requires a pending prerequisite")
    prohibited = [
        require_string(item, "active_iteration.prohibited_before_first_gate[]")
        for item in require_list(
            data.get("prohibited_before_first_gate"), "active_iteration.prohibited_before_first_gate"
        )
    ]
    required_prohibitions = {"scut115", "holdout40", "reserved_blind"}
    if not required_prohibitions.issubset(set(prohibited)):
        raise ValueError(
            "active_iteration.prohibited_before_first_gate must include scut115, holdout40, and reserved_blind"
        )
    return {
        "id": require_string(data.get("id"), "active_iteration.id"),
        "terminal": terminal,
        "failure_bucket": require_string(data.get("failure_bucket"), "active_iteration.failure_bucket"),
        "causal_change": require_string(data.get("causal_change"), "active_iteration.causal_change"),
        "first_gate": first_gate,
        "prerequisites": prerequisites,
        "pending_prerequisites": pending_prerequisites,
        "next_action": require_string(data.get("next_action"), "active_iteration.next_action"),
        "prohibited_before_first_gate": prohibited,
        "evidence": validate_evidence_list(repo_root, data.get("evidence"), "active_iteration.evidence"),
    }


def load_ledger(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"ledger is not a file: {path}")
    try:
        return require_mapping(json.loads(path.read_text(encoding="utf-8")), "ledger")
    except json.JSONDecodeError as exc:
        raise ValueError(f"ledger is not valid JSON: {path}") from exc


def validate_ledger(ledger: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    if ledger.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    program = require_mapping(ledger.get("program"), "program")
    if require_string(program.get("product_default"), "program.product_default") != "artifacts/current-primary":
        raise ValueError("program.product_default must be artifacts/current-primary")
    if program.get("promotion_state") != "disabled":
        raise ValueError("program.promotion_state must be disabled until a candidate passes all gates")
    if program.get("reserved_blind_state") != "disabled":
        raise ValueError("program.reserved_blind_state must be disabled until promotion gates pass")
    return {
        "program": {
            "name": require_string(program.get("name"), "program.name"),
            "product_default": program["product_default"],
            "promotion_state": program["promotion_state"],
            "reserved_blind_state": program["reserved_blind_state"],
        },
        "baseline": validate_baseline(repo_root, ledger.get("baseline")),
        "calibration": validate_calibration(repo_root, ledger.get("calibration")),
        "records": validate_records(repo_root, ledger.get("records")),
        "active_iteration": validate_active_iteration(repo_root, ledger.get("active_iteration")),
    }


def build_report(validated: dict[str, Any]) -> dict[str, Any]:
    active = validated["active_iteration"]
    blockers = [
        f"active iteration {active['id']} is {active['terminal']}",
        *[f"pending prerequisite: {item}" for item in active["pending_prerequisites"]],
        "promotion remains disabled",
        "reserved blind remains disabled",
    ]
    return {
        "status": "active_not_promotion_eligible",
        "baseline_verified": True,
        "candidate_admission_ready": not active["pending_prerequisites"] and active["terminal"] == "PENDING",
        "promotion_eligible": False,
        "reserved_blind_authorized": False,
        "blockers": blockers,
        **validated,
    }


def markdown_report(report: dict[str, Any]) -> str:
    calibration = report["calibration"]
    active = report["active_iteration"]
    lines = [
        "# Current-Primary Quality Loop Status",
        "",
        f"Status: **{report['status']}**",
        f"Baseline verified: **{report['baseline_verified']}**",
        f"Promotion eligible: **{report['promotion_eligible']}**",
        f"Reserved blind authorized: **{report['reserved_blind_authorized']}**",
        "",
        "## Calibration",
        f"- terminal={calibration['terminal']} scope={calibration['scope']}",
        f"- runs={calibration['run_count']} pages_per_run={calibration['pages_per_run']}",
        f"- minimum_residual_gain={calibration['minimum_residual_gain']:.6f}",
        "",
        "## Recorded Directions",
    ]
    for record in report["records"]:
        lines.append(
            f"- `{record['id']}`: terminal={record['terminal']} outcome={record['outcome']} "
            f"repeat_policy={record['repeat_policy']}"
        )
    lines.extend(
        [
            "",
            "## Active Iteration",
            f"- id={active['id']} terminal={active['terminal']}",
            f"- failure_bucket={active['failure_bucket']}",
            f"- first_gate={active['first_gate']}",
            f"- next_action={active['next_action']}",
            "",
            "## Blockers",
        ]
    )
    lines.extend(f"- {blocker}" for blocker in report["blockers"])
    return "\n".join(lines) + "\n"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    ledger = load_ledger(args.ledger)
    evidence_audit: dict[str, Any] | None = None
    if args.evidence_audit_json or args.evidence_audit_only:
        evidence_audit = audit_declared_evidence(ledger, repo_root)
        if args.evidence_audit_json:
            write_json(args.evidence_audit_json, evidence_audit)
        if args.evidence_audit_only:
            print(
                "Evidence audit: "
                f"status={evidence_audit['status']} "
                f"ok={evidence_audit['ok_count']} "
                f"missing={evidence_audit['missing_count']} "
                f"mismatched={evidence_audit['mismatch_count']} "
                f"invalid={evidence_audit['invalid_count']}"
            )
            return
    report = build_report(validate_ledger(ledger, repo_root))
    if evidence_audit is not None:
        report["evidence_audit"] = evidence_audit
    if args.output_json:
        write_json(args.output_json, report)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), end="")


if __name__ == "__main__":
    main()
