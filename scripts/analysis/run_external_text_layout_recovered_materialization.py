#!/usr/bin/env python3
"""Launch frozen external-layout materialization with recovered cache evidence."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import materialize_external_text_layout_support_train_only as materializer  # noqa: E402
from scripts.analysis import probe_external_text_layout_runtime_safety as safety_probe  # noqa: E402
from scripts.analysis import external_text_layout_recovered_batch_runtime as batch_runtime  # noqa: E402
from scripts.analysis import external_text_layout_recovered_v9_contract as v9_contract  # noqa: E402
from scripts.analysis import external_text_layout_recovered_v10_contract as v10_contract  # noqa: E402


CONTRACT_PATH = Path("docs/external-text-layout-recovered-materializer-launch-v2.json")
EXPECTED_CONTRACT_SHA256 = (
    "26eec02ad8c7dcbbb7ebfea2e03b2b431c35cacff0e78f5c2029f459b00d6978"
)
V3_CONTRACT_PATH = Path(
    "docs/external-text-layout-recovered-materializer-baseline-relative-launch-v3.json"
)
EXPECTED_V3_CONTRACT_SHA256 = (
    "8602c2ff972ed45ee6e62514f75d3bcade9e8c4056398d0c47976124153e023f"
)
V4_CONTRACT_PATH = Path(
    "docs/external-text-layout-recovered-materializer-formal-rss-launch-v4.json"
)
EXPECTED_V4_CONTRACT_SHA256 = (
    "e05eb01bac1f92bc7e8b1a5c1064c73d1d19eeb5db7a73a2b872e594a0fc682b"
)
V5_CONTRACT_PATH = Path(
    "docs/external-text-layout-recovered-materializer-formal-memory-launch-v5.json"
)
EXPECTED_V5_CONTRACT_SHA256 = (
    "f15bcd18157235fc5957a1674d205fd338c8bec2b558df2a3eb7fc2324405498"
)
V6_CONTRACT_PATH = Path(
    "docs/external-text-layout-recovered-materializer-batched-launch-v6.json"
)
EXPECTED_V6_CONTRACT_SHA256 = (
    "6a47d124168b6a49b37c71a30f5ca670cdf532cb19562ebc32ebd24ab98aecf2"
)
V8_CONTRACT_PATH = Path(
    "docs/external-text-layout-recovered-materializer-bounded-rss-launch-v8.json"
)
EXPECTED_V8_CONTRACT_SHA256 = (
    "b477fded33c5f861dee006256516a270d495fbf5133d3d176342ccd3f184b27c"
)
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
DERIVED_PLAN_PATH = Path(
    "outputs/external-text-layout-recovered-materializer-input-20260815/"
    "effective-plan.json"
)
RESULT_PATH = Path(
    "outputs/external-text-layout-recovered-materializer-input-20260815/result.json"
)
EXPECTED_DERIVED_PLAN_SHA256 = (
    "39d5d801c0507dc965c970927b2d2ea6a2e7d9a2f3f04b27564956093fbca5d4"
)
EXPECTED_SHARED_MATERIALIZER_SHA256 = (
    "2a87d2a21b9141c9ca16e5f11f7ab1f523d59ffacf6f55759517fd0db26aafcf"
)
EXPECTED_SHARED_TEST_SHA256 = (
    "79e64906656871976d29b3ced66fc396f93c1271f25149e587065600ec5f47f2"
)
EXPECTED_SHARED_RUNTIME_SHA256 = (
    "47d3bda97e0c6f100ed556d7260b1467fc0a236e6391e7414cb6aa932dd9d0d4"
)
EXPECTED_SAFETY_PROBE_SHA256 = (
    "55dcf747f40cc789f5c05c4840b783c6cafb28ce240a38f0c80bf0c2250bdb53"
)
FORMAL_MAX_PROCESS_TREE_RSS_BYTES = materializer.MAX_DETECTOR_RSS_BYTES
V8_MAX_PROCESS_TREE_RSS_BYTES = 11 * 1024**3
RECOVERED_MAX_PROCESS_TREE_RSS_BYTES = (
    batch_runtime.MAX_RECOVERED_PROCESS_TREE_RSS_BYTES
)
FORMAL_MIN_RUNTIME_MEMORY_FREE_PERCENT = materializer.MIN_MEMORY_FREE_PERCENT
FORMAL_MONITOR_INTERVAL_SECONDS = 0.25


class RecoveredMaterializationError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecoveredMaterializationError(f"cannot read launcher evidence: {path}") from error
    if not isinstance(value, dict):
        raise RecoveredMaterializationError(f"launcher evidence is not an object: {path}")
    return value


def repo_path(repo_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RecoveredMaterializationError(f"launcher path escaped repository: {value}")
    return repo_root / relative


def validate_artifact(repo_root: Path, artifact: Any, label: str) -> Path:
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise RecoveredMaterializationError(f"{label} artifact contract changed")
    path = repo_path(repo_root, str(artifact["path"]))
    if path.is_symlink() or not path.is_file():
        raise RecoveredMaterializationError(f"missing {label}: {path}")
    actual = materializer.sha256_file(path)
    if actual != artifact["sha256"]:
        raise RecoveredMaterializationError(
            f"{label} sha256 changed: expected {artifact['sha256']}, got {actual}"
        )
    return path


def launcher_safety_limits() -> dict[str, float | int]:
    return {
        "batch_size": batch_runtime.BATCH_SIZE,
        "batch_timeout_seconds": batch_runtime.BATCH_TIMEOUT_SECONDS,
        "launch_memory_free_percent_min": safety_probe.PROBE_MIN_LAUNCH_MEMORY_FREE_PERCENT,
        "launch_process_tree_rss_bytes_max": safety_probe.PROBE_MAX_PROCESS_TREE_RSS_BYTES,
        "launch_stability_sample_interval_seconds": safety_probe.PROBE_LAUNCH_SAMPLE_INTERVAL_SECONDS,
        "launch_stability_window_seconds": safety_probe.PROBE_LAUNCH_STABILITY_SECONDS,
        "runtime_memory_free_percent_min": FORMAL_MIN_RUNTIME_MEMORY_FREE_PERCENT,
        "runtime_monitor_interval_seconds": FORMAL_MONITOR_INTERVAL_SECONDS,
        "runtime_process_tree_rss_bytes_max": RECOVERED_MAX_PROCESS_TREE_RSS_BYTES,
        "runtime_swap_growth_bytes_max": safety_probe.PROBE_MAX_SWAP_GROWTH_BYTES,
    }


def validate_v8_contract(repo_root: Path) -> dict[str, Any]:
    contract_path = repo_root / V8_CONTRACT_PATH
    if materializer.sha256_file(contract_path) != EXPECTED_V8_CONTRACT_SHA256:
        raise RecoveredMaterializationError("launcher v8 contract sha256 changed")
    contract = read_json(contract_path)
    implementation = contract.get("implementation", {})
    runtime_contract = contract.get("runtime", {})
    observed = contract.get("observed_v7_stop", {})
    correction = contract.get("governance_correction", {})
    if (
        contract.get("schema_version") != 8
        or contract.get("terminal") != "PREREQUISITE_NEEDED"
        or contract.get("authority", {}).get("result_authority")
        != "recovered_materializer_bounded_rss_v8_preregistration_only"
        or implementation.get("allowed_files")
        != [
            "scripts/analysis/external_text_layout_recovered_batch_runtime.py",
            "scripts/analysis/run_external_text_layout_recovered_materialization.py",
            "tests/test_external_text_layout_recovered_batch_runtime.py",
            "tests/test_run_external_text_layout_recovered_materialization.py",
        ]
        or implementation.get("launcher_result_schema_version") != 8
        or implementation.get("new_dependency") is not False
        or implementation.get("shared_materializer_mutation") is not False
        or implementation.get("shared_runtime_mutation") is not False
        or runtime_contract
        != {
            "batch_size": batch_runtime.BATCH_SIZE,
            "batch_timeout_seconds": batch_runtime.BATCH_TIMEOUT_SECONDS,
            "maximum_concurrent_model_processes": 1,
            "maximum_runtime_process_tree_rss_bytes": V8_MAX_PROCESS_TREE_RSS_BYTES,
            "maximum_swap_growth_bytes": safety_probe.PROBE_MAX_SWAP_GROWTH_BYTES,
            "minimum_launch_memory_free_percent": safety_probe.PROBE_MIN_LAUNCH_MEMORY_FREE_PERCENT,
            "minimum_runtime_memory_free_percent": FORMAL_MIN_RUNTIME_MEMORY_FREE_PERCENT,
            "monitor_interval_seconds": FORMAL_MONITOR_INTERVAL_SECONDS,
            "shared_default_process_tree_rss_bytes_unchanged": FORMAL_MAX_PROCESS_TREE_RSS_BYTES,
            "stability_window_seconds": safety_probe.PROBE_LAUNCH_STABILITY_SECONDS,
            "worker_count": 1,
        }
        or observed.get("reason")
        != "detector RSS safety limit exceeded: 11147149312 > 10737418240"
        or observed.get("trigger_process_tree_rss_bytes") != 11147149312
        or correction.get("actual_next_file") != "hw5k_121.jpg"
        or correction.get("incorrect_prior_file_is_in_manifest") is not False
        or contract.get("supersedes", {}).get("status")
        != "eight_pages_retained_then_rss_rejected_before_page_nine_completion"
    ):
        raise RecoveredMaterializationError("launcher v8 contract changed")
    evidence = contract.get("evidence", {})
    historical = {
        "launcher_v7_source": (
            "scripts/analysis/run_external_text_layout_recovered_materialization.py",
            "b058013e3ecbcc5c26d7fb88d7913cb7721ddb61c7dd8fd6368f5267f59d1d59",
        ),
        "launcher_v7_test": (
            "tests/test_run_external_text_layout_recovered_materialization.py",
            "fe2536ae3b382738a12dbe3a0c834b834d34b5aa777764f3df7c241c08e3875b",
        ),
        "recovered_batch_runtime_v7": (
            "scripts/analysis/external_text_layout_recovered_batch_runtime.py",
            "057e0fec7eb0b6adfb43135dee66e3277c142c34fc7d98c2ba4b3a5a22669e2f",
        ),
        "recovered_batch_runtime_v7_test": (
            "tests/test_external_text_layout_recovered_batch_runtime.py",
            "fbec10f38613643ae093b26e410a66132cb49f8a96ef9740700d36f56de1b29a",
        ),
    }
    for label, (path, sha256) in historical.items():
        if evidence.get(label) != {"path": path, "sha256": sha256}:
            raise RecoveredMaterializationError(
                f"launcher historical {label} evidence changed"
            )
    exact = {
        "launcher_v7_integration": "1b5676a031a0af080ddad941db047e28dca0952560cf738a62cf5fbab13bdb1d",
        "shared_materializer": EXPECTED_SHARED_MATERIALIZER_SHA256,
        "shared_runtime": EXPECTED_SHARED_RUNTIME_SHA256,
    }
    for label, expected_sha256 in exact.items():
        path = validate_artifact(repo_root, evidence.get(label), f"launcher v8 {label}")
        if materializer.sha256_file(path) != expected_sha256:
            raise RecoveredMaterializationError(f"launcher v8 {label} source changed")
    manifest = contract.get("dataset_shape_audit", {}).get("manifest")
    manifest_path = validate_artifact(repo_root, manifest, "launcher v8 manifest")
    if materializer.sha256_file(manifest_path) != (
        "ba31900496161322f839f366fa40765d71182d99a59ddad2537786310aae432f"
    ):
        raise RecoveredMaterializationError("launcher v8 manifest changed")
    return contract


def validate_v9_contract(repo_root: Path) -> dict[str, Any]:
    try:
        contract = v9_contract.validate_contract(repo_root)
    except (OSError, ValueError) as error:
        raise RecoveredMaterializationError(str(error)) from error
    if (
        v9_contract.RECOVERED_MAX_PROCESS_TREE_RSS_BYTES
        != RECOVERED_MAX_PROCESS_TREE_RSS_BYTES
        or v9_contract.SHARED_MAX_PROCESS_TREE_RSS_BYTES
        != FORMAL_MAX_PROCESS_TREE_RSS_BYTES
    ):
        raise RecoveredMaterializationError("launcher v9 runtime constants changed")
    return contract


def validate_v10_contract(repo_root: Path) -> dict[str, Any]:
    try:
        contract = v10_contract.validate_contract(repo_root)
    except (OSError, ValueError) as error:
        raise RecoveredMaterializationError(str(error)) from error
    if (
        v10_contract.RECOVERED_MAX_PROCESS_TREE_RSS_BYTES
        != RECOVERED_MAX_PROCESS_TREE_RSS_BYTES
    ):
        raise RecoveredMaterializationError("launcher v10 runtime constants changed")
    return contract


def validate_v6_contract(repo_root: Path) -> dict[str, Any]:
    contract_path = repo_root / V6_CONTRACT_PATH
    if materializer.sha256_file(contract_path) != EXPECTED_V6_CONTRACT_SHA256:
        raise RecoveredMaterializationError("launcher v6 contract sha256 changed")
    contract = read_json(contract_path)
    implementation = contract.get("implementation", {})
    runtime_contract = contract.get("runtime", {})
    batch_execution = contract.get("batch_execution", {})
    if (
        contract.get("schema_version") != 6
        or contract.get("terminal") != "PREREQUISITE_NEEDED"
        or contract.get("authority", {}).get("result_authority")
        != "recovered_materializer_launcher_v6_integration_only"
        or implementation.get("allowed_files")
        != [
            "scripts/analysis/external_text_layout_recovered_batch_runtime.py",
            "scripts/analysis/run_external_text_layout_recovered_materialization.py",
            "tests/test_external_text_layout_recovered_batch_runtime.py",
            "tests/test_run_external_text_layout_recovered_materialization.py",
        ]
        or implementation.get("launcher_result_schema_version") != 6
        or implementation.get("new_dependency") is not False
        or implementation.get("persistent_runtime_patch") is not False
        or implementation.get("shared_materializer_mutation") is not False
        or implementation.get("shared_runtime_mutation") is not False
        or batch_execution
        != {
            "batch_size": batch_runtime.BATCH_SIZE,
            "completed_page_retention": "atomic_npz_then_atomic_record_per_page",
            "detector_factory_calls_per_child": 1,
            "detector_lifetime": "one_child_process_for_at_most_eight_serial_source_pages",
            "failure_resume": "retain_every_validated_record_and_recover_orphan_npz_with_existing_prepare_resume_state",
            "final_manifest_builder": "unchanged_shared_materializer_build_manifest_payload",
            "maximum_concurrent_model_processes": 1,
            "source_order": "exact_registered_train275_manifest_order_skipping_only_validated_completed_records",
        }
        or runtime_contract
        != {
            "batch_timeout_seconds": batch_runtime.BATCH_TIMEOUT_SECONDS,
            "launch_maximum_process_tree_rss_bytes": safety_probe.PROBE_MAX_PROCESS_TREE_RSS_BYTES,
            "maximum_runtime_process_tree_rss_bytes": FORMAL_MAX_PROCESS_TREE_RSS_BYTES,
            "maximum_swap_growth_bytes": safety_probe.PROBE_MAX_SWAP_GROWTH_BYTES,
            "minimum_launch_memory_free_percent": safety_probe.PROBE_MIN_LAUNCH_MEMORY_FREE_PERCENT,
            "minimum_runtime_memory_free_percent": FORMAL_MIN_RUNTIME_MEMORY_FREE_PERCENT,
            "monitor_interval_seconds": FORMAL_MONITOR_INTERVAL_SECONDS,
            "stability_window_seconds": safety_probe.PROBE_LAUNCH_STABILITY_SECONDS,
            "worker_count": 1,
        }
        or contract.get("supersedes", {}).get("status")
        != "eight_pages_retained_then_swap_growth_rejected_during_page_nine_detector_loading"
    ):
        raise RecoveredMaterializationError("launcher v6 contract changed")
    evidence = contract.get("evidence", {})
    historical = {
        "launcher_v5_implementation": (
            "scripts/analysis/run_external_text_layout_recovered_materialization.py",
            "ea223d28590b0bcce07dab7c546be6af1d0c91ba95f2cba61c9ae67d8e548361",
        ),
        "launcher_v5_test": (
            "tests/test_run_external_text_layout_recovered_materialization.py",
            "65748b00acbe2cebb6342aab345dd99b3eb862bca66a0b7c5b33e9117c1ab717",
        ),
    }
    for label, (path, sha256) in historical.items():
        if evidence.get(label) != {"path": path, "sha256": sha256}:
            raise RecoveredMaterializationError(
                f"launcher historical {label} evidence changed"
            )
    exact = {
        "launcher_v5_contract": EXPECTED_V5_CONTRACT_SHA256,
        "launcher_v5_integration": "0fcfc659fdb57ce63bc3e98b3c4915a3d6854e83778679fd54572b9c23d5d0b9",
        "shared_materializer": EXPECTED_SHARED_MATERIALIZER_SHA256,
        "shared_materializer_test": EXPECTED_SHARED_TEST_SHA256,
        "shared_runtime": EXPECTED_SHARED_RUNTIME_SHA256,
    }
    for label, expected_sha256 in exact.items():
        path = validate_artifact(repo_root, evidence.get(label), f"launcher v6 {label}")
        if materializer.sha256_file(path) != expected_sha256:
            raise RecoveredMaterializationError(f"launcher v6 {label} source changed")
    return contract


def validate_v5_contract(repo_root: Path) -> dict[str, Any]:
    contract_path = repo_root / V5_CONTRACT_PATH
    if materializer.sha256_file(contract_path) != EXPECTED_V5_CONTRACT_SHA256:
        raise RecoveredMaterializationError("launcher v5 contract sha256 changed")
    contract = read_json(contract_path)
    implementation = contract.get("implementation", {})
    runtime_contract = contract.get("runtime", {})
    if (
        contract.get("schema_version") != 5
        or contract.get("terminal") != "PREREQUISITE_NEEDED"
        or contract.get("authority", {}).get("result_authority")
        != "recovered_materializer_launcher_v5_integration_only"
        or implementation.get("allowed_files")
        != [
            "scripts/analysis/run_external_text_layout_recovered_materialization.py",
            "tests/test_run_external_text_layout_recovered_materialization.py",
        ]
        or implementation.get("launcher_result_schema_version") != 5
        or implementation.get("new_dependency") is not False
        or implementation.get("persistent_runtime_patch") is not False
        or implementation.get("shared_materializer_mutation") is not False
        or implementation.get("shared_runtime_mutation") is not False
        or runtime_contract
        != {
            "launch_maximum_process_tree_rss_bytes": safety_probe.PROBE_MAX_PROCESS_TREE_RSS_BYTES,
            "maximum_runtime_process_tree_rss_bytes": FORMAL_MAX_PROCESS_TREE_RSS_BYTES,
            "maximum_swap_growth_bytes": safety_probe.PROBE_MAX_SWAP_GROWTH_BYTES,
            "minimum_launch_memory_free_percent": safety_probe.PROBE_MIN_LAUNCH_MEMORY_FREE_PERCENT,
            "minimum_runtime_memory_free_percent": FORMAL_MIN_RUNTIME_MEMORY_FREE_PERCENT,
            "monitor_interval_seconds": FORMAL_MONITOR_INTERVAL_SECONDS,
            "page_timeout_seconds": materializer.runtime.PAGE_TIMEOUT_SECONDS,
            "stability_window_seconds": safety_probe.PROBE_LAUNCH_STABILITY_SECONDS,
            "worker_count": 1,
        }
        or contract.get("supersedes", {}).get("status")
        != "eight_pages_completed_then_resource_rejected_with_exact_resume_state"
    ):
        raise RecoveredMaterializationError("launcher v5 contract changed")
    evidence = contract.get("evidence", {})
    historical = {
        "launcher_v4_implementation": (
            "scripts/analysis/run_external_text_layout_recovered_materialization.py",
            "e85b2bf59e612860ae4d65d402aca5f61d8463709266de212d7e09217f57d5bd",
        ),
        "launcher_v4_test": (
            "tests/test_run_external_text_layout_recovered_materialization.py",
            "5f98a4ef18f52dd95dd3a7bdd5f3b1598f12408883ed9c9daaf0df883d1f3d5a",
        ),
    }
    for label, (path, sha256) in historical.items():
        if evidence.get(label) != {"path": path, "sha256": sha256}:
            raise RecoveredMaterializationError(
                f"launcher historical {label} evidence changed"
            )
    exact = {
        "launcher_v4_contract": EXPECTED_V4_CONTRACT_SHA256,
        "launcher_v4_integration": "5817b56f031db1dfcfb334e66c4578d8836396cd878573b23e6f2c3566239867",
        "shared_materializer": EXPECTED_SHARED_MATERIALIZER_SHA256,
        "shared_materializer_test": EXPECTED_SHARED_TEST_SHA256,
        "shared_runtime": EXPECTED_SHARED_RUNTIME_SHA256,
    }
    for label, expected_sha256 in exact.items():
        path = validate_artifact(repo_root, evidence.get(label), f"launcher v5 {label}")
        if materializer.sha256_file(path) != expected_sha256:
            raise RecoveredMaterializationError(f"launcher v5 {label} source changed")
    return contract


def validate_v4_contract(repo_root: Path) -> dict[str, Any]:
    contract_path = repo_root / V4_CONTRACT_PATH
    if materializer.sha256_file(contract_path) != EXPECTED_V4_CONTRACT_SHA256:
        raise RecoveredMaterializationError("launcher v4 contract sha256 changed")
    contract = read_json(contract_path)
    implementation = contract.get("implementation", {})
    runtime_contract = contract.get("runtime", {})
    if (
        contract.get("schema_version") != 4
        or contract.get("terminal") != "PREREQUISITE_NEEDED"
        or contract.get("authority", {}).get("result_authority")
        != "recovered_materializer_launcher_v4_integration_only"
        or implementation.get("allowed_files")
        != [
            "scripts/analysis/run_external_text_layout_recovered_materialization.py",
            "tests/test_run_external_text_layout_recovered_materialization.py",
        ]
        or implementation.get("launcher_result_schema_version") != 4
        or implementation.get("new_dependency") is not False
        or implementation.get("persistent_runtime_patch") is not False
        or implementation.get("shared_materializer_mutation") is not False
        or implementation.get("shared_runtime_mutation") is not False
        or implementation.get("site_packages_write") is not False
        or runtime_contract
        != {
            "child_rechecks_booted_ios_simulator_before_detector_import": True,
            "launch_maximum_process_tree_rss_bytes": safety_probe.PROBE_MAX_PROCESS_TREE_RSS_BYTES,
            "maximum_runtime_process_tree_rss_bytes": FORMAL_MAX_PROCESS_TREE_RSS_BYTES,
            "maximum_swap_growth_bytes": safety_probe.PROBE_MAX_SWAP_GROWTH_BYTES,
            "minimum_launch_memory_free_percent": safety_probe.PROBE_MIN_LAUNCH_MEMORY_FREE_PERCENT,
            "minimum_runtime_memory_free_percent": safety_probe.PROBE_MIN_RUNTIME_MEMORY_FREE_PERCENT,
            "monitor_interval_seconds": FORMAL_MONITOR_INTERVAL_SECONDS,
            "page_timeout_seconds": materializer.runtime.PAGE_TIMEOUT_SECONDS,
            "process_group_termination_on_limit": True,
            "stability_sample_interval_seconds": safety_probe.PROBE_LAUNCH_SAMPLE_INTERVAL_SECONDS,
            "stability_window_seconds": safety_probe.PROBE_LAUNCH_STABILITY_SECONDS,
            "swap_baseline_absolute_maximum_bytes": None,
            "worker_count": 1,
        }
        or contract.get("supersedes", {}).get("status")
        != "one_page_completed_then_resource_rejected_with_exact_resume_state"
    ):
        raise RecoveredMaterializationError("launcher v4 contract changed")
    evidence = contract.get("evidence", {})
    historical = {
        "launcher_v3_implementation": (
            "scripts/analysis/run_external_text_layout_recovered_materialization.py",
            "3c58365db9d33fbc6de81cce06a45938f341920c80cdd637b763ba11cdc57ddf",
        ),
        "launcher_v3_test": (
            "tests/test_run_external_text_layout_recovered_materialization.py",
            "476ee00a7cd4c84b6d44142e2a18c9fd3b3ef5f347ac0e5568576ba539125b9d",
        ),
    }
    for label, (path, sha256) in historical.items():
        if evidence.get(label) != {"path": path, "sha256": sha256}:
            raise RecoveredMaterializationError(
                f"launcher historical {label} evidence changed"
            )
    exact = {
        "launcher_v3_contract": EXPECTED_V3_CONTRACT_SHA256,
        "launcher_v3_integration": "b7fc960325333e7a20788b487ac17d603f7c7e5bc4fdacecd4625c0b53da1ead",
        "shared_materializer": EXPECTED_SHARED_MATERIALIZER_SHA256,
        "shared_materializer_test": EXPECTED_SHARED_TEST_SHA256,
        "shared_runtime": EXPECTED_SHARED_RUNTIME_SHA256,
        "tiled_probe": EXPECTED_SAFETY_PROBE_SHA256,
    }
    for label, expected_sha256 in exact.items():
        path = validate_artifact(repo_root, evidence.get(label), f"launcher v4 {label}")
        if materializer.sha256_file(path) != expected_sha256:
            raise RecoveredMaterializationError(f"launcher v4 {label} source changed")
    return contract


def validate_v3_contract(repo_root: Path) -> dict[str, Any]:
    contract_path = repo_root / V3_CONTRACT_PATH
    if materializer.sha256_file(contract_path) != EXPECTED_V3_CONTRACT_SHA256:
        raise RecoveredMaterializationError("launcher v3 contract sha256 changed")
    contract = read_json(contract_path)
    implementation = contract.get("implementation", {})
    monitor = contract.get("monitor", {})
    if (
        contract.get("schema_version") != 3
        or contract.get("terminal") != "PREREQUISITE_NEEDED"
        or contract.get("authority", {}).get("result_authority")
        != "recovered_materializer_launcher_v3_integration_only"
        or implementation.get("allowed_files")
        != [
            "scripts/analysis/run_external_text_layout_recovered_materialization.py",
            "tests/test_run_external_text_layout_recovered_materialization.py",
        ]
        or implementation.get("launcher_result_schema_version") != 3
        or implementation.get("new_dependency") is not False
        or implementation.get("persistent_runtime_patch") is not False
        or implementation.get("shared_materializer_mutation") is not False
        or implementation.get("shared_runtime_mutation") is not False
        or implementation.get("site_packages_write") is not False
        or monitor
        != {
            "child_rechecks_booted_ios_simulator_before_detector_import": True,
            "maximum_process_tree_rss_bytes": safety_probe.PROBE_MAX_PROCESS_TREE_RSS_BYTES,
            "maximum_swap_growth_bytes": safety_probe.PROBE_MAX_SWAP_GROWTH_BYTES,
            "minimum_launch_memory_free_percent": safety_probe.PROBE_MIN_LAUNCH_MEMORY_FREE_PERCENT,
            "minimum_runtime_memory_free_percent": safety_probe.PROBE_MIN_RUNTIME_MEMORY_FREE_PERCENT,
            "page_timeout_seconds": materializer.runtime.PAGE_TIMEOUT_SECONDS,
            "process_group_termination_on_limit": True,
            "stability_sample_interval_seconds": safety_probe.PROBE_LAUNCH_SAMPLE_INTERVAL_SECONDS,
            "stability_window_seconds": safety_probe.PROBE_LAUNCH_STABILITY_SECONDS,
            "swap_baseline_absolute_maximum_bytes": None,
            "swap_growth_formula": "max(0,current_absolute_swap-launch_swap_baseline_bytes)",
            "worker_count": 1,
        }
        or contract.get("supersedes", {}).get("status")
        != "integration_pass_execution_blocked_before_detector"
    ):
        raise RecoveredMaterializationError("launcher v3 contract changed")
    evidence = contract.get("evidence", {})
    historical = {
        "launcher_v2_implementation": (
            "scripts/analysis/run_external_text_layout_recovered_materialization.py",
            "1e3e37c9b8c5af78c62735fe29b81caa68c0ebcace2b56460ee04e262e19a738",
        ),
        "launcher_v2_test": (
            "tests/test_run_external_text_layout_recovered_materialization.py",
            "c5eac3b537609e2ab53528f6ab635208e2263f2a4ec474358fc3eb48ac2b2e37",
        ),
    }
    for label, (path, sha256) in historical.items():
        if evidence.get(label) != {"path": path, "sha256": sha256}:
            raise RecoveredMaterializationError(
                f"launcher historical {label} evidence changed"
            )
    exact = {
        "launcher_v2_contract": EXPECTED_CONTRACT_SHA256,
        "shared_materializer": EXPECTED_SHARED_MATERIALIZER_SHA256,
        "shared_materializer_test": EXPECTED_SHARED_TEST_SHA256,
        "shared_runtime": EXPECTED_SHARED_RUNTIME_SHA256,
        "tiled_probe": EXPECTED_SAFETY_PROBE_SHA256,
    }
    for label, expected_sha256 in exact.items():
        path = validate_artifact(repo_root, evidence.get(label), f"launcher v3 {label}")
        if materializer.sha256_file(path) != expected_sha256:
            raise RecoveredMaterializationError(
                f"launcher v3 {label} source changed"
            )
    probe_contract = validate_artifact(
        repo_root,
        evidence.get("tiled_probe_contract"),
        "launcher v3 tiled probe contract",
    )
    if materializer.sha256_file(probe_contract) != safety_probe.EXPECTED_PROBE_CONTRACT_SHA256:
        raise RecoveredMaterializationError("launcher v3 probe contract changed")
    return contract


def validate_repository_contract(repo_root: Path) -> dict[str, Any]:
    validate_v10_contract(repo_root)
    validate_v9_contract(repo_root)
    validate_v8_contract(repo_root)
    validate_v6_contract(repo_root)
    validate_v5_contract(repo_root)
    validate_v4_contract(repo_root)
    validate_v3_contract(repo_root)
    contract_path = repo_root / CONTRACT_PATH
    if materializer.sha256_file(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise RecoveredMaterializationError("launcher v2 contract sha256 changed")
    contract = read_json(contract_path)
    derived = contract.get("derived_plan", {})
    implementation = contract.get("implementation", {})
    if (
        contract.get("schema_version") != 2
        or contract.get("terminal") != "PREREQUISITE_NEEDED"
        or derived.get("path") != str(DERIVED_PLAN_PATH)
        or derived.get("expected_sha256") != EXPECTED_DERIVED_PLAN_SHA256
        or implementation.get("allowed_files")
        != [
            "scripts/analysis/run_external_text_layout_recovered_materialization.py",
            "tests/test_run_external_text_layout_recovered_materialization.py",
        ]
        or implementation.get("launcher_result") != str(RESULT_PATH)
        or implementation.get("new_dependency") is not False
        or implementation.get("shared_materializer_mutation") is not False
        or implementation.get("site_packages_write") is not False
        or contract.get("supersedes", {}).get("status")
        != "rejected_before_integration"
    ):
        raise RecoveredMaterializationError("launcher v2 contract changed")
    evidence = contract.get("evidence", {})
    artifacts = {
        label: validate_artifact(repo_root, artifact, f"launcher {label}")
        for label, artifact in evidence.items()
    }
    if (
        materializer.sha256_file(artifacts["shared_materializer"])
        != EXPECTED_SHARED_MATERIALIZER_SHA256
        or materializer.sha256_file(artifacts["shared_materializer_test"])
        != EXPECTED_SHARED_TEST_SHA256
    ):
        raise RecoveredMaterializationError("probe-bound shared source changed")
    archive_result = read_json(artifacts["archive_publication_result"])
    archive_verification = read_json(artifacts["archive_publication_verification"])
    if (
        archive_result.get("terminal") != "PASS"
        or archive_result.get("publication", {}).get("status") != "published"
        or archive_verification.get("terminal") != "PASS"
    ):
        raise RecoveredMaterializationError("archive publication evidence is not PASS")
    return contract


def validate_execution_authority(repo_root: Path) -> None:
    ledger = read_json(repo_root / LEDGER_PATH)
    active = ledger.get("active_iteration", {})
    statuses = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    required = {
        "external_text_layout_recovered_archive_publication": "passed",
        "external_text_layout_recovered_materializer_input_preregistration": "passed",
        "external_text_layout_recovered_materializer_launch_v2_preregistration": "passed",
        "external_text_layout_recovered_materializer_launch_v2_integration": "passed",
        "external_text_layout_recovered_materializer_baseline_relative_launch_v3_preregistration": "passed",
        "external_text_layout_recovered_materializer_baseline_relative_launch_v3_integration": "passed",
        "external_text_layout_recovered_materializer_formal_rss_launch_v4_preregistration": "passed",
        "external_text_layout_recovered_materializer_formal_rss_launch_v4_integration": "passed",
        "external_text_layout_recovered_materializer_formal_memory_launch_v5_preregistration": "passed",
        "external_text_layout_recovered_materializer_formal_memory_launch_v5_integration": "passed",
        "external_text_layout_recovered_materializer_batched_launch_v6_preregistration": "passed",
        "external_text_layout_recovered_materializer_batched_launch_v6_integration": "passed",
        "external_text_layout_recovered_materializer_bounded_rss_launch_v8_preregistration": "passed",
        "external_text_layout_recovered_materializer_bounded_rss_launch_v8_integration": "passed",
        "external_text_layout_recovered_materializer_host_capacity_rss_launch_v9_preregistration": "passed",
        "external_text_layout_recovered_materializer_host_capacity_rss_launch_v9_integration": "passed",
        "external_text_layout_recovered_materializer_page_memory_relief_v10_preregistration": "passed",
        "external_text_layout_recovered_materializer_page_memory_relief_v10_integration": "passed",
        "external_text_layout_support_train_only_diagnostic": "pending",
    }
    if active.get("terminal") != "PREREQUISITE_NEEDED" or any(
        statuses.get(name) != status for name, status in required.items()
    ):
        raise RecoveredMaterializationError("recovered launcher execution authority is closed")


def _validate_relative_link(repo_root: Path, registered: Any, label: str) -> Path:
    if not isinstance(registered, dict) or set(registered) != {"path", "target"}:
        raise RecoveredMaterializationError(f"{label} link contract changed")
    link = repo_path(repo_root, str(registered["path"]))
    if not link.is_symlink():
        raise RecoveredMaterializationError(f"missing {label} archive link")
    target = os.readlink(link)
    if Path(target).is_absolute() or target != registered["target"]:
        raise RecoveredMaterializationError(f"{label} archive target changed")
    try:
        resolved = link.resolve()
    except (OSError, RuntimeError) as error:
        raise RecoveredMaterializationError(f"{label} archive link is broken") from error
    if not resolved.is_dir() or resolved.is_symlink():
        raise RecoveredMaterializationError(f"{label} archive source changed")
    return link


def _prediction_identity(prediction_dir: Path) -> dict[str, Any]:
    if prediction_dir.is_symlink() or not prediction_dir.is_dir():
        raise RecoveredMaterializationError("archive prediction directory changed")
    predictions = list(prediction_dir.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in predictions):
        raise RecoveredMaterializationError("archive prediction surface changed")
    names = sorted(path.name for path in predictions)
    rows = [f"{name} {materializer.sha256_file(prediction_dir / name)}" for name in names]
    return {
        "content_sha256": materializer.sha256_rows(rows),
        "count": len(names),
        "filename_sha256": materializer.sha256_rows(names),
    }


def validate_archive_inputs(
    repo_root: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    evidence = contract["evidence"]
    overlay = read_json(repo_path(repo_root, evidence["recovered_input_v1"]["path"]))
    archive_result = read_json(
        repo_path(repo_root, evidence["archive_publication_result"]["path"])
    )
    required_links = overlay.get("input_overlay", {}).get(
        "required_archive_links", {}
    )
    actual: dict[str, Any] = {}
    expected = archive_result.get("publication", {}).get("linked_caches", {})
    for stage in ("primary", "second_stage"):
        link = _validate_relative_link(
            repo_root, required_links.get(stage), stage.replace("_", "-")
        )
        metrics = link / "metrics.csv"
        if metrics.is_symlink() or not metrics.is_file():
            raise RecoveredMaterializationError(f"{stage} archive metrics changed")
        actual[stage] = {
            "metrics_sha256": materializer.sha256_file(metrics),
            "prediction_set": _prediction_identity(link / "pred"),
        }
        if actual[stage] != expected.get(stage):
            raise RecoveredMaterializationError(
                f"{stage} archive cache identity changed"
            )
    return actual


def build_derived_plan(
    contract: dict[str, Any], original_plan: dict[str, Any]
) -> tuple[dict[str, Any], bytes]:
    semantic = contract["derived_plan"]["semantic_changes_from_original"]
    metrics_change = semantic["evidence.second_stage_metrics.sha256"]
    original_metrics = original_plan.get("evidence", {}).get("second_stage_metrics")
    if not isinstance(original_metrics, dict) or original_metrics.get("sha256") != metrics_change["before"]:
        raise RecoveredMaterializationError("original second-stage metrics identity changed")
    derived_plan = json.loads(json.dumps(original_plan))
    derived_plan["evidence"]["second_stage_metrics"]["sha256"] = metrics_change[
        "after"
    ]
    derived_plan["recovered_input_overlay"] = semantic["recovered_input_overlay"]
    restored = json.loads(json.dumps(derived_plan))
    del restored["recovered_input_overlay"]
    restored["evidence"]["second_stage_metrics"]["sha256"] = metrics_change[
        "before"
    ]
    if restored != original_plan:
        raise RecoveredMaterializationError("derived plan changed outside v2 contract")
    try:
        materializer.validate_plan(derived_plan)
    except RuntimeError as error:
        raise RecoveredMaterializationError(str(error)) from error
    payload = (
        json.dumps(derived_plan, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != contract["derived_plan"]["expected_sha256"]:
        raise RecoveredMaterializationError(
            f"derived plan sha256 changed: {actual_sha256}"
        )
    return derived_plan, payload


def write_or_validate_derived_plan(
    path: Path, payload: bytes, expected_sha256: str
) -> str:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RecoveredMaterializationError("existing derived plan changed")
        if materializer.sha256_file(path) != expected_sha256:
            raise RecoveredMaterializationError("existing derived plan sha256 changed")
        return "existing"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    if temporary.exists() or temporary.is_symlink():
        raise RecoveredMaterializationError("stale derived plan candidate exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    if materializer.sha256_file(path) != expected_sha256:
        raise RecoveredMaterializationError("derived plan write validation failed")
    return "written"


def run_baseline_relative_materializer(
    *,
    repo_root: Path,
    derived_plan_path: Path,
    health_reader: Callable[[int], dict[str, float | int]] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    materialize_runner: Callable[..., dict[str, Any]] | None = None,
    lock_factory: Callable[[Path], Any] | None = None,
    conflict_checker: Callable[[], None] | None = None,
    simulator_checker: Callable[[], int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    absolute_health_reader = health_reader or materializer.runtime.runtime_health
    runner = materialize_runner or batch_runtime.materialize
    acquire_lock = lock_factory or materializer.runtime.exclusive_run_lock
    check_conflicts = conflict_checker or materializer.assert_no_conflicting_model_processes
    check_simulators = simulator_checker or materializer.runtime.assert_no_booted_ios_simulators

    @contextmanager
    def held_lock(_path: Path) -> Iterator[None]:
        yield

    with acquire_lock(materializer.runtime.HOST_USER_RUN_LOCK_PATH):
        check_conflicts()
        check_simulators()
        baseline, launch_health, relative_reader = safety_probe.stable_launch_health(
            health_reader=absolute_health_reader,
            sleeper=sleeper,
            pid=os.getpid(),
        )
        check_conflicts()
        check_simulators()
        result = runner(
            repo_root=repo_root,
            plan_path=derived_plan_path.relative_to(repo_root),
            ledger_path=LEDGER_PATH,
            worker_count=1,
            batch_size=batch_runtime.BATCH_SIZE,
            health_reader=relative_reader,
            maximum_process_tree_rss_bytes=RECOVERED_MAX_PROCESS_TREE_RSS_BYTES,
            minimum_memory_free_percent=FORMAL_MIN_RUNTIME_MEMORY_FREE_PERCENT,
            maximum_swap_used_bytes=safety_probe.PROBE_MAX_SWAP_GROWTH_BYTES,
            batch_timeout_seconds=batch_runtime.BATCH_TIMEOUT_SECONDS,
            monitor_interval_seconds=FORMAL_MONITOR_INTERVAL_SECONDS,
            reject_booted_ios_simulators=True,
            lock_factory=held_lock,
            conflict_checker=check_conflicts,
        )

        if result.get("terminal") != "PASS":
            raise RecoveredMaterializationError("batched materializer did not PASS")
        check_conflicts()
        check_simulators()
        post_run_health = relative_reader(os.getpid())
        batch_runtime.enforce_recovered_health_limits(
            post_run_health,
            maximum_process_tree_rss_bytes=RECOVERED_MAX_PROCESS_TREE_RSS_BYTES,
            minimum_memory_free_percent=FORMAL_MIN_RUNTIME_MEMORY_FREE_PERCENT,
            maximum_swap_used_bytes=safety_probe.PROBE_MAX_SWAP_GROWTH_BYTES,
        )
    runtime_safety = {
        "evidence_state": "live_monitored_execution",
        "launch_health": launch_health,
        "launch_swap_baseline_bytes": baseline,
        "limits": launcher_safety_limits(),
        "materialization_health": {
            "minimum_memory_free_percent": result["minimum_memory_free_percent"],
            "peak_process_tree_rss_bytes": result["peak_process_tree_rss_bytes"],
            "peak_swap_growth_bytes": result["peak_swap_used_bytes"],
        },
        "post_run_health": safety_probe.explicit_swap_growth_evidence(
            post_run_health
        ),
        "terminal": "PASS",
    }
    return result, runtime_safety


def validate_materialization_output(
    repo_root: Path, derived_plan_path: Path, expected_plan_sha256: str
) -> dict[str, Any]:
    output_root = repo_root / materializer.OUTPUT_ROOT
    manifest_path = output_root / "manifest.json"
    page_dir = output_root / "pages"
    if (
        output_root.is_symlink()
        or not output_root.is_dir()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or page_dir.is_symlink()
        or not page_dir.is_dir()
    ):
        raise RecoveredMaterializationError("materialization output surface changed")
    manifest = read_json(manifest_path)
    expected_plan = {
        "path": str(derived_plan_path.relative_to(repo_root)),
        "sha256": expected_plan_sha256,
    }
    rows = manifest.get("pages")
    if (
        manifest.get("terminal") != "PASS"
        or manifest.get("plan") != expected_plan
        or not isinstance(rows, list)
        or len(rows) != 275
    ):
        raise RecoveredMaterializationError("materialization manifest changed")
    expected_pages: set[str] = set()
    content_rows: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("file"), str):
            raise RecoveredMaterializationError("materialization page row changed")
        page = page_dir / f"{Path(row['file']).stem}.npz"
        expected_pages.add(page.name)
        if page.is_symlink() or not page.is_file():
            raise RecoveredMaterializationError("materialization page payload changed")
        actual_sha256 = materializer.sha256_file(page)
        if actual_sha256 != row.get("npz_sha256"):
            raise RecoveredMaterializationError("materialization page sha256 changed")
        content_rows.append(f"{row['file']} {actual_sha256}")
    if len(expected_pages) != len(rows) or {
        path.name for path in page_dir.iterdir()
    } != expected_pages:
        raise RecoveredMaterializationError("materialization page population changed")
    return {
        "content_sha256": materializer.sha256_rows(content_rows),
        "manifest": str(manifest_path.relative_to(repo_root)),
        "manifest_sha256": materializer.sha256_file(manifest_path),
        "output_root": str(output_root.relative_to(repo_root)),
        "terminal": "PASS",
        "train_count": len(rows),
    }


def build_launcher_result(
    contract: dict[str, Any],
    archive_inputs: dict[str, Any],
    materialization: dict[str, Any],
    plan_state_before: str,
    runtime_safety: dict[str, Any],
) -> dict[str, Any]:
    metrics_change = contract["derived_plan"]["semantic_changes_from_original"][
        "evidence.second_stage_metrics.sha256"
    ]
    return {
        "archive_cache_identities": archive_inputs,
        "authority": {
            "candidate_inference": False,
            "quality_evaluation": False,
            "result_authority": "recovered_materializer_launcher_v10",
        },
        "derived_plan": {
            "path": contract["derived_plan"]["path"],
            "sha256": contract["derived_plan"]["expected_sha256"],
            "state_before": plan_state_before,
        },
        "historical_second_stage_metrics_sha256": metrics_change["before"],
        "materialization": materialization,
        "recovered_second_stage_metrics_sha256": metrics_change["after"],
        "runtime_safety": runtime_safety,
        "schema_version": 10,
        "terminal": "PASS",
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(path)
    try:
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def recovered_runtime_safety() -> dict[str, Any]:
    return {
        "evidence_state": "terminal_output_recovery",
        "limits": launcher_safety_limits(),
        "live_health_available": False,
        "terminal": "PASS",
    }


def runtime_number(
    record: dict[str, Any], key: str, *, integer: bool = False
) -> float | int:
    value = record.get(key)
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected):
        raise RecoveredMaterializationError(
            f"launcher runtime safety number changed: {key}"
        )
    if not math.isfinite(float(value)):
        raise RecoveredMaterializationError(
            f"launcher runtime safety number is non-finite: {key}"
        )
    return value


def validate_runtime_safety(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("terminal") != "PASS":
        raise RecoveredMaterializationError("launcher runtime safety evidence changed")
    if value.get("limits") != launcher_safety_limits():
        raise RecoveredMaterializationError("launcher runtime safety limits changed")
    state = value.get("evidence_state")
    if state == "terminal_output_recovery":
        if value != recovered_runtime_safety():
            raise RecoveredMaterializationError(
                "launcher recovered runtime safety evidence changed"
            )
        return value
    launch = value.get("launch_health")
    materialization_health = value.get("materialization_health")
    post = value.get("post_run_health")
    baseline = value.get("launch_swap_baseline_bytes")
    if (
        state != "live_monitored_execution"
        or not isinstance(launch, dict)
        or not isinstance(materialization_health, dict)
        or not isinstance(post, dict)
        or set(launch)
        != {
            "minimum_memory_free_percent",
            "peak_process_tree_rss_bytes",
            "peak_swap_growth_bytes",
            "sample_count",
            "stability_sample_interval_seconds",
            "stability_window_seconds",
        }
        or set(materialization_health)
        != {
            "minimum_memory_free_percent",
            "peak_process_tree_rss_bytes",
            "peak_swap_growth_bytes",
        }
        or set(post)
        != {
            "memory_free_percent",
            "process_tree_rss_bytes",
            "swap_growth_bytes",
        }
        or launch.get("sample_count") != 61
        or launch.get("stability_sample_interval_seconds")
        != safety_probe.PROBE_LAUNCH_SAMPLE_INTERVAL_SECONDS
        or launch.get("stability_window_seconds")
        != safety_probe.PROBE_LAUNCH_STABILITY_SECONDS
    ):
        raise RecoveredMaterializationError("launcher runtime safety bounds changed")
    baseline_value = runtime_number(value, "launch_swap_baseline_bytes", integer=True)
    launch_free = runtime_number(launch, "minimum_memory_free_percent")
    launch_rss = runtime_number(launch, "peak_process_tree_rss_bytes", integer=True)
    launch_swap = runtime_number(launch, "peak_swap_growth_bytes", integer=True)
    materialization_free = runtime_number(
        materialization_health, "minimum_memory_free_percent"
    )
    materialization_rss = runtime_number(
        materialization_health, "peak_process_tree_rss_bytes", integer=True
    )
    materialization_swap = runtime_number(
        materialization_health, "peak_swap_growth_bytes", integer=True
    )
    post_free = runtime_number(post, "memory_free_percent")
    post_rss = runtime_number(post, "process_tree_rss_bytes", integer=True)
    post_swap = runtime_number(post, "swap_growth_bytes", integer=True)
    if (
        baseline_value < 0
        or launch_free < safety_probe.PROBE_MIN_LAUNCH_MEMORY_FREE_PERCENT
        or not 0 <= launch_rss <= safety_probe.PROBE_MAX_PROCESS_TREE_RSS_BYTES
        or launch_swap != 0
        or materialization_free
        < FORMAL_MIN_RUNTIME_MEMORY_FREE_PERCENT
        or not (
            0
            <= materialization_rss
            <= RECOVERED_MAX_PROCESS_TREE_RSS_BYTES
        )
        or not (
            0
            <= materialization_swap
            <= safety_probe.PROBE_MAX_SWAP_GROWTH_BYTES
        )
        or post_free < FORMAL_MIN_RUNTIME_MEMORY_FREE_PERCENT
        or not 0 <= post_rss <= RECOVERED_MAX_PROCESS_TREE_RSS_BYTES
        or not 0 <= post_swap <= safety_probe.PROBE_MAX_SWAP_GROWTH_BYTES
    ):
        raise RecoveredMaterializationError("launcher runtime safety bounds changed")
    return value


def validate_existing_result(
    repo_root: Path,
    contract: dict[str, Any],
    archive_inputs: dict[str, Any],
) -> dict[str, Any]:
    derived_plan_path = repo_path(repo_root, contract["derived_plan"]["path"])
    if (
        derived_plan_path.is_symlink()
        or not derived_plan_path.is_file()
        or materializer.sha256_file(derived_plan_path)
        != contract["derived_plan"]["expected_sha256"]
    ):
        raise RecoveredMaterializationError("terminal derived plan changed")
    materialization = validate_materialization_output(
        repo_root,
        derived_plan_path,
        contract["derived_plan"]["expected_sha256"],
    )
    result_path = repo_root / RESULT_PATH
    if result_path.is_symlink() or not result_path.is_file():
        raise RecoveredMaterializationError("launcher terminal result surface changed")
    result = read_json(result_path)
    if (
        result.get("terminal") != "PASS"
        or result.get("schema_version") != 10
        or result.get("archive_cache_identities") != archive_inputs
        or result.get("materialization") != materialization
        or result.get("authority", {}).get("result_authority")
        != "recovered_materializer_launcher_v10"
        or result.get("derived_plan", {}).get("path")
        != contract["derived_plan"]["path"]
        or result.get("derived_plan", {}).get("sha256")
        != contract["derived_plan"]["expected_sha256"]
        or result.get("derived_plan", {}).get("state_before")
        not in {"absent", "existing"}
    ):
        raise RecoveredMaterializationError("launcher terminal result changed")
    validate_runtime_safety(result.get("runtime_safety"))
    return result


def run_launcher(repo_root: Path) -> dict[str, Any]:
    contract = validate_repository_contract(repo_root)
    original_plan_path = validate_artifact(
        repo_root, contract["evidence"]["original_plan"], "launcher original plan"
    )
    original_plan = read_json(original_plan_path)
    archive_inputs = validate_archive_inputs(repo_root, contract)
    _derived_plan, payload = build_derived_plan(contract, original_plan)
    result_path = repo_root / RESULT_PATH
    if result_path.exists() or result_path.is_symlink():
        return validate_existing_result(repo_root, contract, archive_inputs)
    validate_execution_authority(repo_root)
    derived_plan_path = repo_path(repo_root, contract["derived_plan"]["path"])
    plan_state_before = (
        "existing" if derived_plan_path.exists() or derived_plan_path.is_symlink() else "absent"
    )
    write_or_validate_derived_plan(
        derived_plan_path, payload, contract["derived_plan"]["expected_sha256"]
    )
    output_root = repo_root / materializer.OUTPUT_ROOT
    temporary_root = output_root.with_name(f".{output_root.name}.materializing")
    marker_path, _cleanup_root = materializer.published_transaction_paths(output_root)
    if output_root.exists() and not temporary_root.exists() and not marker_path.exists():
        materialization = validate_materialization_output(
            repo_root,
            derived_plan_path,
            contract["derived_plan"]["expected_sha256"],
        )
        runtime_safety = recovered_runtime_safety()
    else:
        _materializer_result, runtime_safety = run_baseline_relative_materializer(
            repo_root=repo_root,
            derived_plan_path=derived_plan_path,
        )
        materialization = validate_materialization_output(
            repo_root,
            derived_plan_path,
            contract["derived_plan"]["expected_sha256"],
        )
    result = build_launcher_result(
        contract,
        archive_inputs,
        materialization,
        plan_state_before,
        runtime_safety,
    )
    try:
        write_result(result_path, result)
        if read_json(result_path) != result:
            raise RecoveredMaterializationError(
                "launcher result failed read-back validation"
            )
    except BaseException:
        result_path.unlink(missing_ok=True)
        result_path.with_name(f".{result_path.name}.writing").unlink(
            missing_ok=True
        )
        raise
    return result


def main() -> int:
    try:
        result = run_launcher(ROOT)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "reason": str(error),
                    "schema_version": 10,
                    "terminal": "PREREQUISITE_NEEDED",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
