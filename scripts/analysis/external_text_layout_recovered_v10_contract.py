#!/usr/bin/env python3
"""Validate the page-memory-relief contract for recovered materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT_PATH = Path(
    "docs/external-text-layout-recovered-materializer-page-memory-relief-v10.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "947c88aa38f11df43c4aa83d1bd49a9d82fb312ab48daddfa16569cd3480c85d"
)
RECOVERED_MAX_PROCESS_TREE_RSS_BYTES = 13 * 1024**3


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
        raise ValueError(f"cannot read v10 contract evidence: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("v10 contract is not an object")
    return value


def validate_current_artifact(
    repo_root: Path, artifact: Any, label: str, expected_sha256: str
) -> None:
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise ValueError(f"v10 {label} artifact changed")
    relative = Path(str(artifact["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"v10 {label} path escaped repository")
    path = repo_root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing v10 {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256 or actual != artifact["sha256"]:
        raise ValueError(f"launcher v10 {label} evidence changed")


def validate_contract(repo_root: Path) -> dict[str, Any]:
    contract_path = repo_root / CONTRACT_PATH
    if contract_path.is_symlink() or not contract_path.is_file():
        raise ValueError("missing launcher v10 contract")
    if sha256_file(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise ValueError("launcher v10 contract sha256 changed")
    contract = read_json(contract_path)
    implementation = contract.get("implementation", {})
    relief = contract.get("memory_relief", {})
    observed = contract.get("observed_v9_stop", {})
    resume = contract.get("resume_state", {})
    runtime = contract.get("runtime", {})
    if (
        contract.get("schema_version") != 10
        or contract.get("terminal") != "PREREQUISITE_NEEDED"
        or contract.get("authority", {}).get("result_authority")
        != "recovered_materializer_page_memory_relief_v10_preregistration_only"
        or implementation.get("allowed_files")
        != [
            "scripts/analysis/external_text_layout_recovered_batch_runtime.py",
            "scripts/analysis/external_text_layout_recovered_v10_contract.py",
            "scripts/analysis/run_external_text_layout_recovered_materialization.py",
            "tests/test_external_text_layout_recovered_batch_runtime.py",
            "tests/test_run_external_text_layout_recovered_materialization.py",
        ]
        or implementation.get("launcher_result_schema_version") != 10
        or implementation.get("new_dependency") is not False
        or implementation.get("shared_materializer_mutation") is not False
        or implementation.get("shared_runtime_mutation") is not False
        or relief.get("after")
        != "atomic_npz_then_atomic_record_commit_for_each_page"
        or relief.get("before") != "next_page_decode_and_prediction"
        or relief.get("garbage_collection") != "gc.collect"
        or relief.get("malloc_relief")
        != "libSystem.B.dylib_malloc_zone_pressure_relief_null_zone_goal_zero"
        or relief.get("output_semantics_changed") is not False
        or relief.get("platform_probe", {}).get("symbol_available") is not True
        or observed.get("trigger_process_tree_rss_bytes") != 14049755136
        or observed.get("next_attempt_cancelled_before_detector_creation") is not True
        or resume.get("completed_count") != 23
        or resume.get("next_file") != "hw5k_1447.jpg"
        or resume.get("next_manifest_index") != 24
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
        or contract.get("supersedes", {}).get("status")
        != "twenty_three_pages_retained_then_final_rss_rejection"
    ):
        raise ValueError("launcher v10 contract changed")

    evidence = contract.get("evidence", {})
    historical = {
        "launcher_v9_source": (
            "scripts/analysis/run_external_text_layout_recovered_materialization.py",
            "2b430892cc8cede2e76687c102ec8c6f19e862ac5715e619d9b56500f6bee736",
        ),
        "launcher_v9_test": (
            "tests/test_run_external_text_layout_recovered_materialization.py",
            "8338ff5adcecbbb2ad05c4fdff6a347e7b236431f087fb6cff993307ed71ca12",
        ),
        "recovered_batch_runtime_v9": (
            "scripts/analysis/external_text_layout_recovered_batch_runtime.py",
            "8ec933f85b07a9e0aab6f59916b5fa12dbef008b6210d8f2c465b7a898af602e",
        ),
        "recovered_batch_runtime_v9_test": (
            "tests/test_external_text_layout_recovered_batch_runtime.py",
            "8e8a1c89733e4c96dd16a852c75bf30c08d9188f9daff15096a63cf76df8161c",
        ),
    }
    for label, (path, sha256) in historical.items():
        if evidence.get(label) != {"path": path, "sha256": sha256}:
            raise ValueError(f"launcher historical {label} evidence changed")
    exact = {
        "launcher_v9_integration": "8c73bdc4c5fea0e413f051f406b70ce007ba0d468e6bd6fc2ab99084e6507633",
        "shared_materializer": "2a87d2a21b9141c9ca16e5f11f7ab1f523d59ffacf6f55759517fd0db26aafcf",
        "shared_runtime": "47d3bda97e0c6f100ed556d7260b1467fc0a236e6391e7414cb6aa932dd9d0d4",
    }
    for label, sha256 in exact.items():
        validate_current_artifact(repo_root, evidence.get(label), label, sha256)
    return contract
