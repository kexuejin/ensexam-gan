#!/usr/bin/env python3
"""Validate the host-capacity RSS contract for recovered materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT_PATH = Path(
    "docs/external-text-layout-recovered-materializer-host-capacity-rss-launch-v9.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "b776caf4dc57a136f68686f62ee46aca23bab853361ffd7c30709c6ba0ebe34b"
)
RECOVERED_MAX_PROCESS_TREE_RSS_BYTES = 13 * 1024**3
SHARED_MAX_PROCESS_TREE_RSS_BYTES = 10 * 1024**3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read v9 contract evidence: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("v9 contract is not an object")
    return value


def artifact_path(repo_root: Path, artifact: Any, label: str) -> Path:
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise ValueError(f"v9 {label} artifact changed")
    relative = Path(str(artifact["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"v9 {label} path escaped repository")
    path = repo_root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing v9 {label}: {path}")
    return path


def validate_contract(repo_root: Path) -> dict[str, Any]:
    contract_path = repo_root / CONTRACT_PATH
    if contract_path.is_symlink() or not contract_path.is_file():
        raise ValueError("missing launcher v9 contract")
    if sha256_file(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise ValueError("launcher v9 contract sha256 changed")
    contract = read_json(contract_path)
    implementation = contract.get("implementation", {})
    runtime = contract.get("runtime", {})
    host = contract.get("host_capacity_policy", {})
    observed = contract.get("observed_v8_execution", {})
    resume = contract.get("resume_state", {})
    if (
        contract.get("schema_version") != 9
        or contract.get("terminal") != "PREREQUISITE_NEEDED"
        or contract.get("authority", {}).get("result_authority")
        != "recovered_materializer_host_capacity_rss_v9_preregistration_only"
        or implementation.get("allowed_files")
        != [
            "scripts/analysis/external_text_layout_recovered_batch_runtime.py",
            "scripts/analysis/external_text_layout_recovered_v9_contract.py",
            "scripts/analysis/run_external_text_layout_recovered_materialization.py",
            "tests/test_external_text_layout_recovered_batch_runtime.py",
            "tests/test_run_external_text_layout_recovered_materialization.py",
        ]
        or implementation.get("launcher_result_schema_version") != 9
        or implementation.get("new_dependency") is not False
        or implementation.get("shared_materializer_mutation") is not False
        or implementation.get("shared_runtime_mutation") is not False
        or host
        != {
            "physical_memory_bytes": 24 * 1024**3,
            "recovered_rss_cap_bytes": RECOVERED_MAX_PROCESS_TREE_RSS_BYTES,
            "recovered_rss_cap_fraction_of_physical_memory": 13 / 24,
            "runtime_free_memory_floor_percent": 35.0,
            "runtime_swap_growth_cap_bytes": 512 * 1024**2,
            "shared_rss_cap_bytes_unchanged": SHARED_MAX_PROCESS_TREE_RSS_BYTES,
        }
        or runtime
        != {
            "batch_size": 8,
            "batch_timeout_seconds": 900.0,
            "maximum_concurrent_model_processes": 1,
            "maximum_runtime_process_tree_rss_bytes": RECOVERED_MAX_PROCESS_TREE_RSS_BYTES,
            "maximum_swap_growth_bytes": 512 * 1024**2,
            "minimum_launch_memory_free_percent": 70.0,
            "minimum_runtime_memory_free_percent": 35.0,
            "monitor_interval_seconds": 0.25,
            "stability_window_seconds": 60.0,
            "worker_count": 1,
        }
        or observed.get("second_attempt", {}).get("trigger_process_tree_rss_bytes")
        != 11992580096
        or observed.get("second_attempt", {}).get("supervisor_terminal")
        != "NO_PROGRESS"
        or resume.get("completed_count") != 18
        or resume.get("next_file") != "hw5k_1376.jpg"
        or resume.get("next_manifest_index") != 19
        or contract.get("supersedes", {}).get("status")
        != "eighteen_pages_retained_then_zero_progress_rss_rejection"
    ):
        raise ValueError("launcher v9 contract changed")

    evidence = contract.get("evidence", {})
    historical = {
        "launcher_v8_source": (
            "scripts/analysis/run_external_text_layout_recovered_materialization.py",
            "650225d60f5f4168f91da9f08651ed6aa863a46fb95e0a70181f19277c47605f",
        ),
        "launcher_v8_test": (
            "tests/test_run_external_text_layout_recovered_materialization.py",
            "cdd871640a5279efb1c7745c4b54aa03ae0fc00a006f4e0ba6cb2a7e47ee3c57",
        ),
        "recovered_batch_runtime_v8": (
            "scripts/analysis/external_text_layout_recovered_batch_runtime.py",
            "e8647bdb6d63b31f316b7edefe90b7e0efefaf3f1950ab73288b9cecdfb5b645",
        ),
        "recovered_batch_runtime_v8_test": (
            "tests/test_external_text_layout_recovered_batch_runtime.py",
            "72d073d6ae535d3ed9b2893f37c02046dfadd379d7049f0c75151f80c02722b3",
        ),
    }
    for label, (path, sha256) in historical.items():
        if evidence.get(label) != {"path": path, "sha256": sha256}:
            raise ValueError(f"launcher historical {label} evidence changed")

    exact = {
        "launcher_v8_integration": "684ad1fb3f03ba5efaff84767ebc9618c9f6e1a463e3f48e3bbeea561c1321ed",
        "shared_materializer": "2a87d2a21b9141c9ca16e5f11f7ab1f523d59ffacf6f55759517fd0db26aafcf",
        "shared_runtime": "47d3bda97e0c6f100ed556d7260b1467fc0a236e6391e7414cb6aa932dd9d0d4",
    }
    for label, expected_sha256 in exact.items():
        path = artifact_path(repo_root, evidence.get(label), label)
        actual = sha256_file(path)
        if actual != expected_sha256 or actual != evidence[label]["sha256"]:
            raise ValueError(f"launcher v9 {label} evidence changed")
    return contract
