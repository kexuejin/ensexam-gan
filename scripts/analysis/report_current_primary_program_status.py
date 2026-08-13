#!/usr/bin/env python3
"""Fail-closed status reporter for the current-primary generalization program.

Parses the single ``yaml ledger`` block in
``docs/current-primary-failure-ledger.md`` and verifies that program status is
reproducible from committed repository state alone: anchor files exist with the
recorded hashes, every bucket cites tracked evidence, and exactly one bucket is
active while the program is active.

The reporter never synthesizes status from stale prose or disposable
``outputs/`` files. Any missing input, hash mismatch, schema violation, or
evidence problem is collected into an explicit failure list, written into the
report, and reflected in a non-zero exit code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

LEDGER_RELPATH = Path("docs/current-primary-failure-ledger.md")
LEDGER_BLOCK_RE = re.compile(r"^```yaml ledger\n(.*?)^```", re.MULTILINE | re.DOTALL)
SUPPORTED_SCHEMA_VERSIONS = {1}
ALLOWED_BUCKET_STATUS = {"active", "exhausted", "killed", "blocked", "out_of_scope"}
ALLOWED_PROGRAM_STATES = {"active", "all_exhausted"}
REQUIRED_TOP_KEYS = (
    "schema_version",
    "updated",
    "program",
    "product_default",
    "program_state",
    "anchors",
    "calibration",
    "buckets",
)
REQUIRED_BUCKET_KEYS = ("name", "status", "summary", "evidence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        default=str(LEDGER_RELPATH),
        help="Ledger markdown path relative to the repository root.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Fresh or empty directory for status.json and status.md.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def git_tracked(root: Path, relpath: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relpath],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_ledger_block(text: str, failures: list[str]) -> dict[str, Any] | None:
    blocks = LEDGER_BLOCK_RE.findall(text)
    if len(blocks) != 1:
        failures.append(
            f"ledger must contain exactly one 'yaml ledger' block, found {len(blocks)}"
        )
        return None
    try:
        data = yaml.safe_load(blocks[0])
    except yaml.YAMLError as exc:
        failures.append(f"ledger YAML block failed to parse: {exc}")
        return None
    if not isinstance(data, dict):
        failures.append("ledger YAML block did not parse to a mapping")
        return None
    return data


def check_schema(data: dict[str, Any], failures: list[str]) -> None:
    for key in REQUIRED_TOP_KEYS:
        if key not in data:
            failures.append(f"ledger missing required top-level key: {key}")
    version = data.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        failures.append(f"unsupported schema_version: {version!r}")
    state = data.get("program_state")
    if state not in ALLOWED_PROGRAM_STATES:
        failures.append(f"program_state must be one of {sorted(ALLOWED_PROGRAM_STATES)}, got {state!r}")


def check_anchors(root: Path, data: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    anchors = data.get("anchors")
    if not isinstance(anchors, dict):
        failures.append("anchors must be a mapping")
        return verified
    for name, anchor in anchors.items():
        if not isinstance(anchor, dict) or "path" not in anchor:
            # Value-only anchors (e.g. readiness_smoke_expected) carry no file.
            verified[name] = {"kind": "value", "value": anchor}
            continue
        relpath = str(anchor["path"])
        path = root / relpath
        entry: dict[str, Any] = {"kind": "file", "path": relpath}
        if not path.is_file():
            failures.append(f"anchor {name}: missing file {relpath}")
            entry["exists"] = False
            verified[name] = entry
            continue
        entry["exists"] = True
        expected = anchor.get("sha256")
        if expected:
            actual = sha256_file(path)
            entry["sha256_expected"] = expected
            entry["sha256_actual"] = actual
            if actual != expected:
                failures.append(
                    f"anchor {name}: sha256 mismatch for {relpath} "
                    f"(expected {expected}, got {actual})"
                )
        verified[name] = entry
    return verified


def check_calibration(data: dict[str, Any], failures: list[str]) -> None:
    calibration = data.get("calibration")
    if not isinstance(calibration, dict):
        failures.append("calibration must be a mapping")
        return
    floor = calibration.get("residual_lift_floor")
    if not isinstance(floor, (int, float)) or floor <= 0:
        failures.append(f"calibration.residual_lift_floor must be > 0, got {floor!r}")
    if not isinstance(calibration.get("baseline_metrics"), dict):
        failures.append("calibration.baseline_metrics must be a mapping")


def check_buckets(root: Path, data: dict[str, Any], failures: list[str]) -> list[dict[str, Any]]:
    buckets = data.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        failures.append("buckets must be a nonempty list")
        return []
    seen_names: set[str] = set()
    summaries: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        label = f"bucket[{index}]"
        if not isinstance(bucket, dict):
            failures.append(f"{label} must be a mapping")
            continue
        name = bucket.get("name", f"<unnamed:{index}>")
        label = f"bucket {name}"
        for key in REQUIRED_BUCKET_KEYS:
            if not bucket.get(key):
                failures.append(f"{label}: missing required key {key}")
        if name in seen_names:
            failures.append(f"{label}: duplicate bucket name")
        seen_names.add(name)
        status = bucket.get("status")
        if status not in ALLOWED_BUCKET_STATUS:
            failures.append(
                f"{label}: status must be one of {sorted(ALLOWED_BUCKET_STATUS)}, got {status!r}"
            )
        if status == "active" and not bucket.get("next_allowed"):
            failures.append(f"{label}: active bucket must define next_allowed")
        evidence = bucket.get("evidence")
        evidence_states: list[dict[str, Any]] = []
        if isinstance(evidence, list) and evidence:
            for relpath in evidence:
                relpath = str(relpath)
                state = {"path": relpath, "exists": False, "tracked": False}
                if relpath.startswith("outputs/"):
                    failures.append(
                        f"{label}: evidence must be durable, not disposable outputs/: {relpath}"
                    )
                if (root / relpath).is_file():
                    state["exists"] = True
                else:
                    failures.append(f"{label}: evidence file missing: {relpath}")
                if git_tracked(root, relpath):
                    state["tracked"] = True
                elif state["exists"]:
                    failures.append(f"{label}: evidence file not tracked by git: {relpath}")
                evidence_states.append(state)
        elif "evidence" in REQUIRED_BUCKET_KEYS:
            # Missing/empty evidence already reported by the required-key loop.
            pass
        summaries.append(
            {
                "name": name,
                "status": status,
                "evidence": evidence_states,
            }
        )
    active = [b["name"] for b in summaries if b["status"] == "active"]
    program_state = data.get("program_state")
    if program_state == "active" and len(active) != 1:
        failures.append(
            f"program_state is active, so exactly one active bucket is required, found {len(active)}: {active}"
        )
    if program_state == "all_exhausted" and active:
        failures.append(
            f"program_state is all_exhausted but active buckets exist: {active}"
        )
    return summaries


def prepare_output_dir(raw: str, failures: list[str]) -> Path | None:
    output_dir = Path(raw)
    if output_dir.exists():
        if not output_dir.is_dir():
            failures.append(f"--output-dir exists and is not a directory: {output_dir}")
            return None
        if any(output_dir.iterdir()):
            failures.append(
                f"--output-dir must be fresh or empty so an earlier report cannot "
                f"stand in for this run: {output_dir}"
            )
            return None
        return output_dir
    output_dir.mkdir(parents=True)
    return output_dir


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Current-Primary Program Status", ""]
    lines.append(f"- ok: `{report['ok']}`")
    lines.append(f"- ledger: `{report['ledger']}`")
    lines.append(f"- ledger_tracked: `{report['ledger_tracked']}`")
    lines.append(f"- updated: `{report.get('updated')}`")
    lines.append(f"- program_state: `{report.get('program_state')}`")
    lines.append(f"- active_bucket: `{report.get('active_bucket')}`")
    lines.append(f"- residual_lift_floor: `{report.get('residual_lift_floor')}`")
    lines.append("")
    lines.append("| bucket | status | evidence ok |")
    lines.append("| --- | --- | --- |")
    for bucket in report.get("buckets", []):
        evidence_ok = all(e["exists"] and e["tracked"] for e in bucket["evidence"]) if bucket["evidence"] else False
        lines.append(f"| {bucket['name']} | {bucket['status']} | {evidence_ok} |")
    lines.append("")
    if report["failures"]:
        lines.append("## Failures")
        lines.append("")
        for failure in report["failures"]:
            lines.append(f"- {failure}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    root = repo_root()

    ledger_path = root / args.ledger
    ledger_tracked = git_tracked(root, args.ledger)
    if not ledger_tracked:
        failures.append(f"ledger is not tracked by git: {args.ledger}")

    data: dict[str, Any] = {}
    if not ledger_path.is_file():
        failures.append(f"ledger file missing: {args.ledger}")
    else:
        parsed = extract_ledger_block(ledger_path.read_text(encoding="utf-8"), failures)
        if parsed is not None:
            data = parsed

    anchors: dict[str, Any] = {}
    buckets: list[dict[str, Any]] = []
    if data:
        check_schema(data, failures)
        anchors = check_anchors(root, data, failures)
        check_calibration(data, failures)
        buckets = check_buckets(root, data, failures)

    active_bucket = next((b["name"] for b in buckets if b["status"] == "active"), None)
    calibration = data.get("calibration") if isinstance(data.get("calibration"), dict) else {}
    report = {
        "ok": not failures,
        "ledger": args.ledger,
        "ledger_tracked": ledger_tracked,
        "updated": data.get("updated"),
        "program": data.get("program"),
        "program_state": data.get("program_state"),
        "product_default": data.get("product_default"),
        "active_bucket": active_bucket,
        "residual_lift_floor": calibration.get("residual_lift_floor"),
        "anchors": anchors,
        "buckets": buckets,
        "failures": failures,
    }

    output_dir = prepare_output_dir(args.output_dir, failures)
    report["ok"] = not failures
    report["failures"] = failures
    if output_dir is not None:
        (output_dir / "status.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "status.md").write_text(render_markdown(report), encoding="utf-8")

    print(
        f"program_status ok={report['ok']} state={report.get('program_state')} "
        f"active_bucket={active_bucket} buckets={len(buckets)} failures={len(failures)}"
    )
    for failure in failures[:20]:
        print(f"FAIL: {failure}")
    if len(failures) > 20:
        print(f"... {len(failures) - 20} more failures in status.json")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
