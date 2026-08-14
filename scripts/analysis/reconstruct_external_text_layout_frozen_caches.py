#!/usr/bin/env python3
"""Reconstruct and verify the frozen train275 caches needed by layout support."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import external_text_layout_materialization_runtime as runtime  # noqa: E402
from scripts.analysis import materialize_external_text_layout_support_train_only as materializer  # noqa: E402


CONTRACT_PATH = Path(
    "docs/external-text-layout-tiled-probe-cache-reconstruction-v2.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "a21575643bfedc4daa762658d037c511d6eb3290e935b1d66eb48bc0db317ae1"
)
EXPECTED_PROBE_SOURCE_SHA256 = (
    "9255f285aa37890421519659a4994587c2467e021c7b1488d1665a7cd0edfa2d"
)
MONITOR_CONTRACT_PATH = Path(
    "docs/external-text-layout-cache-reconstruction-runtime-monitor-v1.json"
)
EXPECTED_MONITOR_CONTRACT_SHA256 = (
    "ebed5d80277cc458d505a93588d01c5908f330f5a8e457a4af13027c6e987556"
)
BASELINE_RELATIVE_CONTRACT_PATH = Path(
    "docs/external-text-layout-cache-reconstruction-baseline-relative-swap-v1.json"
)
EXPECTED_BASELINE_RELATIVE_CONTRACT_SHA256 = (
    "a7cafb5358370585da926a99ad7d844a2cb9b5ec676dbead4803f53e35a09b12"
)
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
CONTROL_DIR = Path("outputs/external-text-layout-cache-reconstruction-20260814")
RECONSTRUCTION_ID = "external_text_layout_tiled_probe_cache_reconstruction_v2"
HISTORICAL_HELPER_MODULE = (
    "scripts.analysis.materialize_sign_separated_train_inputs"
)
STAGES = ("preflight", "primary", "second_stage", "publish", "verify", "all")
EXPECTED_AUTHORITY = {
    "candidate_inference": False,
    "formal_external_layout_materialization": False,
    "model_execution": False,
    "product_default": "artifacts/current-primary",
    "promotion_state": "disabled",
    "quality_evaluation": False,
    "reserved_blind_state": "disabled",
    "result_authority": "frozen_cache_reconstruction_only",
    "training": False,
}
EXPECTED_BASELINE_RELATIVE_AUTHORITY = {
    "cache_reconstruction_execution": False,
    "candidate_inference": False,
    "detector_model_execution": False,
    "formal_external_layout_materialization": False,
    "product_default": "artifacts/current-primary",
    "promotion_state": "disabled",
    "quality_evaluation": False,
    "reserved_blind_state": "disabled",
    "result_authority": "cache_reconstruction_baseline_relative_integration_only",
    "training": False,
}
EXPECTED_BUILD = {
    "archive_manifest": (
        "hardcase_lists/archive/"
        "sign-separated-residual-repair-20260810-train275-v1.txt"
    ),
    "archive_primary": (
        "outputs/archive/sign-separated-residual-repair-20260810/"
        "train275-primary"
    ),
    "archive_second_stage": (
        "outputs/archive/sign-separated-residual-repair-20260810/"
        "train275-frozen-pipeline"
    ),
    "primary": "outputs/sign-separated-residual-repair-train275-primary-v1",
    "publication": (
        "relative_symlinks_after_both_caches_pass_exact_hash_validation"
    ),
    "second_stage": (
        "outputs/sign-separated-residual-repair-train275-frozen-pipeline-v1"
    ),
    "stages": ["primary", "second_stage", "publish", "verify"],
}
EXPECTED_FORBIDDEN_ACCESS = [
    "target_or_label_pixels",
    "quality_splits_or_metrics",
    "ocr_recognition_or_recognized_text",
    "optimizer_checkpoint_or_candidate_surfaces",
    "formal_external_layout_outputs_before_cache_verification",
]
EXPECTED_HISTORICAL_RUNTIME = {
    "numpy": "2.2.6",
    "opencv": "5.0.0",
    "opencv_distribution": "5.0.0.93",
    "python": "3.10.11",
    "torch": "2.5.1",
}


class CacheReconstructionError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_rows(rows: list[str]) -> str:
    payload = "\n".join(sorted(rows)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CacheReconstructionError(f"expected JSON object: {path}")
    return value


def repo_path(repo_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise CacheReconstructionError(
            f"registered path must stay repository-relative: {value}"
        )
    return repo_root / relative


def validate_artifact(repo_root: Path, artifact: dict[str, Any], label: str) -> Path:
    if set(artifact) != {"path", "sha256"}:
        raise CacheReconstructionError(f"{label} artifact contract changed")
    path = repo_path(repo_root, str(artifact["path"]))
    if not path.is_file():
        raise CacheReconstructionError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != artifact["sha256"]:
        raise CacheReconstructionError(
            f"{label} sha256 changed: expected {artifact['sha256']}, got {actual}"
        )
    return path


def validate_authority(repo_root: Path, contract: dict[str, Any]) -> None:
    ledger = read_json(repo_root / LEDGER_PATH)
    program = ledger.get("program", {})
    authority = contract.get("authority", {})
    if (
        program.get("product_default") != authority.get("product_default")
        or program.get("promotion_state") != authority.get("promotion_state")
        or program.get("reserved_blind_state")
        != authority.get("reserved_blind_state")
    ):
        raise CacheReconstructionError("quality-loop authority changed")
    active = ledger.get("active_iteration", {})
    prerequisites = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if (
        active.get("terminal") != "PREREQUISITE_NEEDED"
        or prerequisites.get("materially_new_support_successor_preregistration_v4")
        != "passed"
        or prerequisites.get(
            "external_text_layout_tiled_probe_cache_reconstruction_v2_preregistration"
        )
        != "passed"
        or prerequisites.get(
            "external_text_layout_cache_reconstruction_runtime_monitor_preregistration"
        )
        != "passed"
        or prerequisites.get(
            "external_text_layout_cache_reconstruction_baseline_relative_swap_preregistration"
        )
        != "passed"
        or prerequisites.get(
            "external_text_layout_tiled_one_page_runtime_safety_probe"
        )
        != "passed"
        or prerequisites.get("external_text_layout_support_train_only_diagnostic")
        != "pending"
    ):
        raise CacheReconstructionError("external layout authority changed")


def validate_probe_gate_contract(contract: dict[str, Any]) -> None:
    expected = {
        "contract": {
            "path": (
                "docs/external-text-layout-tiled-9x9-one-page-safety-probe-v1.json"
            ),
            "sha256": (
                "1fd02d49250150f85ce190601b21b36d60a308ef92b07e564c8a21575124aee4"
            ),
        },
        "integration_verification": {
            "path": (
                "docs/external-text-layout-tiled-9x9-one-page-integration-"
                "verification-20260814.json"
            ),
            "sha256": (
                "1572b890b76837aa5448d463abb13b9cd152d10c76f22655023ebae004322731"
            ),
        },
        "maximum_process_tree_rss_bytes": 8 * 1024**3,
        "maximum_swap_used_bytes": 512 * 1024**2,
        "minimum_launch_memory_free_percent": 70.0,
        "minimum_runtime_memory_free_percent": 45.0,
        "probe_page": "hw5k_1011.jpg",
        "probe_result": (
            "outputs/external-text-layout-runtime-safety-probe-tiled-9x9-20260814/"
            "result.json"
        ),
        "required_attempt_count": 1,
        "required_page_completed": True,
        "required_probe": "external_text_layout_tiled_9x9_single_page_runtime_safety",
        "required_probe_reason_code": "runtime_safety_probe_passed",
        "required_probe_terminal": "PASS",
        "required_result_authority": "runtime_prerequisite_only",
        "required_residual_model_process_count": 0,
        "required_safety_limits": {
            "detector_process_tree_rss_bytes_max": 8 * 1024**3,
            "launch_memory_free_percent_min": 70.0,
            "runtime_memory_free_percent_min": 45.0,
            "page_timeout_seconds": runtime.PAGE_TIMEOUT_SECONDS,
            "swap_used_bytes_max": 512 * 1024**2,
        },
        "required_thread_caps": {
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
        "result_must_be_absent_before_attempt": True,
    }
    if contract.get("probe_gate") != expected:
        raise CacheReconstructionError("tiled probe safety gate changed")


def validate_reconstruction_gate_contract(contract: dict[str, Any]) -> None:
    expected = {
        "maximum_process_tree_rss_bytes": runtime.MAX_DETECTOR_RSS_BYTES,
        "maximum_swap_used_bytes": runtime.MAX_SWAP_USED_BYTES,
        "minimum_system_free_memory_percent": runtime.MIN_MEMORY_FREE_PERCENT,
        "probe_pass_required_before_helper_import": True,
        "stage_execution": (
            "one_model_process_at_a_time_under_the_external_layout_host_lock"
        ),
    }
    if contract.get("reconstruction_gate") != expected:
        raise CacheReconstructionError("reconstruction safety gate changed")
    if (
        materializer.MAX_DETECTOR_RSS_BYTES != runtime.MAX_DETECTOR_RSS_BYTES
        or materializer.MAX_SWAP_USED_BYTES != runtime.MAX_SWAP_USED_BYTES
        or materializer.MIN_MEMORY_FREE_PERCENT != runtime.MIN_MEMORY_FREE_PERCENT
    ):
        raise CacheReconstructionError("materializer runtime limits disagree")


def validate_reconstruction_boundaries(contract: dict[str, Any]) -> None:
    if contract.get("authority") != EXPECTED_AUTHORITY:
        raise CacheReconstructionError("cache reconstruction authority changed")
    if contract.get("build") != EXPECTED_BUILD:
        raise CacheReconstructionError("cache reconstruction paths changed")
    if contract.get("forbidden_access") != EXPECTED_FORBIDDEN_ACCESS:
        raise CacheReconstructionError("cache reconstruction access boundary changed")
    if contract.get("historical_runtime") != EXPECTED_HISTORICAL_RUNTIME:
        raise CacheReconstructionError("historical reconstruction runtime changed")
    validate_probe_gate_contract(contract)
    validate_reconstruction_gate_contract(contract)


def expected_legacy_runtime_monitor() -> dict[str, Any]:
    return {
        "child_launch": "subprocess.Popen_start_new_session_true",
        "health_reader": (
            "external_text_layout_materialization_runtime.runtime_health_child_pid"
        ),
        "maximum_process_tree_rss_bytes": runtime.MAX_DETECTOR_RSS_BYTES,
        "maximum_swap_used_bytes": runtime.MAX_SWAP_USED_BYTES,
        "minimum_system_free_memory_percent": runtime.MIN_MEMORY_FREE_PERCENT,
        "monitor_interval_seconds": runtime.MONITOR_INTERVAL_SECONDS,
        "process_scope": "entire_primary_or_second_stage_child_process_tree",
        "stages": ["primary", "second_stage"],
        "termination": {
            "grace_seconds": 5.0,
            "initial_signal": "SIGTERM",
            "scope": "child_process_group",
            "terminal_signal": "SIGKILL",
        },
    }


def expected_runtime_monitor() -> dict[str, Any]:
    return {
        "child_launch": "subprocess.Popen_start_new_session_true",
        "health_reader": (
            "baseline_relative_wrapper_around_external_text_layout_"
            "runtime.runtime_health"
        ),
        "maximum_process_tree_rss_bytes": runtime.MAX_DETECTOR_RSS_BYTES,
        "maximum_swap_growth_bytes": runtime.MAX_SWAP_USED_BYTES,
        "minimum_system_free_memory_percent": runtime.MIN_MEMORY_FREE_PERCENT,
        "monitor_interval_seconds": runtime.MONITOR_INTERVAL_SECONDS,
        "process_scope": "entire_primary_or_second_stage_child_process_tree",
        "stages": ["primary", "second_stage"],
        "swap_baseline": (
            "absolute_sample_after_locked_launch_health_passes_before_helper_import"
        ),
        "termination": {
            "grace_seconds": 5.0,
            "initial_signal": "SIGTERM",
            "scope": "child_process_group",
            "terminal_signal": "SIGKILL",
        },
    }


def validate_runtime_monitor_settings(
    monitor_contract: dict[str, Any],
) -> dict[str, Any]:
    monitor = monitor_contract.get("monitor")
    if monitor != expected_runtime_monitor():
        raise CacheReconstructionError("cache reconstruction runtime monitor changed")
    return monitor


def validate_legacy_runtime_monitor_settings(
    monitor_contract: dict[str, Any],
) -> dict[str, Any]:
    monitor = monitor_contract.get("monitor")
    if monitor != expected_legacy_runtime_monitor():
        raise CacheReconstructionError(
            "legacy cache reconstruction runtime monitor changed"
        )
    return monitor


def validate_runtime_monitor_contract(repo_root: Path) -> dict[str, Any]:
    contract_path = repo_path(repo_root, str(MONITOR_CONTRACT_PATH))
    if sha256_file(contract_path) != EXPECTED_MONITOR_CONTRACT_SHA256:
        raise CacheReconstructionError(
            "cache reconstruction runtime monitor contract sha256 changed"
        )
    contract = read_json(contract_path)
    if (
        contract.get("schema_version") != 1
        or contract.get("state")
        != "preregistered_runtime_monitor_integration_only"
        or contract.get("terminal") != "PREREQUISITE_NEEDED"
        or contract.get("implementation", {}).get("allowed_files")
        != [
            "scripts/analysis/reconstruct_external_text_layout_frozen_caches.py",
            "tests/test_external_text_layout_frozen_cache_reconstruction.py",
        ]
        or contract.get("implementation", {}).get("historical_helper_write")
        is not False
        or contract.get("implementation", {}).get("new_dependency") is not False
        or contract.get("implementation", {}).get("site_packages_write") is not False
    ):
        raise CacheReconstructionError(
            "cache reconstruction runtime monitor contract changed"
        )
    validate_legacy_runtime_monitor_settings(contract)
    frozen_inputs = contract.get("frozen_inputs", {})
    for name, label in (
        ("historical_helper", "runtime monitor historical helper"),
        ("materialization_runtime", "runtime monitor shared runtime"),
        ("reconstruction_v2", "runtime monitor reconstruction v2 contract"),
    ):
        validate_artifact(repo_root, frozen_inputs[name], label)
    return contract


def expected_probe_gate_v2() -> dict[str, Any]:
    return {
        "contract": {
            "path": (
                "docs/external-text-layout-tiled-9x9-one-page-safety-probe-v2.json"
            ),
            "sha256": (
                "2fb92aa625e0409fd7ed9db301d854333ca0852d714a8ed5fa8dcfc20e3527f6"
            ),
        },
        "integration_verification": {
            "path": (
                "docs/external-text-layout-baseline-relative-swap-gate-"
                "integration-verification-20260814.json"
            ),
            "sha256": (
                "6177e4e6e8642cc0c4abdf86f9fd93adff0b6b4eea8b35937fa3135b435b02a8"
            ),
        },
        "maximum_process_tree_rss_bytes": 8 * 1024**3,
        "maximum_swap_growth_bytes": 512 * 1024**2,
        "minimum_launch_memory_free_percent": 70.0,
        "minimum_runtime_memory_free_percent": 45.0,
        "probe_page": "hw5k_1011.jpg",
        "probe_result": {
            "path": (
                "outputs/external-text-layout-runtime-safety-probe-tiled-9x9-"
                "20260814/result.json"
            ),
            "sha256": (
                "1909d66f29d18ca5805fb29b8b89ac054e240bc9231b2a7fa96121466e0ad550"
            ),
        },
        "required_attempt_count": 1,
        "required_page_completed": True,
        "required_probe": "external_text_layout_tiled_9x9_single_page_runtime_safety",
        "required_probe_reason_code": "runtime_safety_probe_passed",
        "required_probe_terminal": "PASS",
        "required_result_authority": "runtime_prerequisite_only",
        "required_residual_model_process_count": 0,
        "required_safety_limits": {
            "detector_process_tree_rss_bytes_max": 8 * 1024**3,
            "launch_memory_free_percent_min": 70.0,
            "launch_stability_sample_interval_seconds": 1.0,
            "launch_stability_window_seconds": 60.0,
            "page_timeout_seconds": runtime.PAGE_TIMEOUT_SECONDS,
            "runtime_memory_free_percent_min": 45.0,
            "runtime_swap_growth_bytes_max": 512 * 1024**2,
        },
        "required_schema_version": 2,
        "required_thread_caps": {
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
    }


def expected_baseline_reconstruction_gate() -> dict[str, Any]:
    return {
        "maximum_process_tree_rss_bytes": runtime.MAX_DETECTOR_RSS_BYTES,
        "maximum_swap_growth_bytes": runtime.MAX_SWAP_USED_BYTES,
        "minimum_system_free_memory_percent": runtime.MIN_MEMORY_FREE_PERCENT,
        "probe_pass_required_before_helper_import": True,
        "stage_execution": (
            "one_model_process_at_a_time_under_the_external_layout_host_lock"
        ),
        "swap_baseline_absolute_maximum_bytes": None,
    }


def validate_baseline_relative_contract(repo_root: Path) -> dict[str, Any]:
    contract_path = repo_path(repo_root, str(BASELINE_RELATIVE_CONTRACT_PATH))
    if sha256_file(contract_path) != EXPECTED_BASELINE_RELATIVE_CONTRACT_SHA256:
        raise CacheReconstructionError(
            "cache reconstruction baseline-relative contract sha256 changed"
        )
    contract = read_json(contract_path)
    implementation = contract.get("implementation", {})
    if (
        contract.get("schema_version") != 1
        or contract.get("state")
        != "preregistered_cache_reconstruction_baseline_relative_integration_only"
        or contract.get("terminal") != "PREREQUISITE_NEEDED"
        or contract.get("authority") != EXPECTED_BASELINE_RELATIVE_AUTHORITY
        or implementation.get("allowed_files")
        != [
            "scripts/analysis/reconstruct_external_text_layout_frozen_caches.py",
            "tests/test_external_text_layout_frozen_cache_reconstruction.py",
        ]
        or implementation.get("new_dependency") is not False
        or implementation.get("site_packages_write") is not False
        or contract.get("probe_gate") != expected_probe_gate_v2()
        or contract.get("reconstruction_gate")
        != expected_baseline_reconstruction_gate()
        or contract.get("result_evidence")
        != {
            "launch": ["launch_swap_baseline_bytes", "initial_health"],
            "runtime": "peak_swap_growth_bytes",
            "post_run": "swap_growth_bytes",
        }
    ):
        raise CacheReconstructionError(
            "cache reconstruction baseline-relative contract changed"
        )
    validate_runtime_monitor_settings(contract)
    frozen = contract.get("frozen_inputs", {})
    expected_names = {
        "historical_helper",
        "monitor_v1",
        "probe_contract_v2",
        "probe_integration_v2",
        "probe_result_v2",
        "reconstruction_v2",
        "shared_runtime",
    }
    if set(frozen) != expected_names:
        raise CacheReconstructionError(
            "cache reconstruction baseline-relative frozen inputs changed"
        )
    paths = {
        name: validate_artifact(
            repo_root, frozen[name], f"baseline-relative {name}"
        )
        for name in sorted(expected_names)
    }
    integration = read_json(paths["probe_integration_v2"])
    validate_artifact(
        repo_root,
        integration.get("implementation", {}).get("probe", {}),
        "baseline-relative current probe implementation",
    )
    return contract


def current_reconstruction_runtime() -> dict[str, str]:
    try:
        opencv_distribution = metadata.version("opencv-python")
        torch_version = metadata.version("torch")
    except metadata.PackageNotFoundError as error:
        raise CacheReconstructionError(
            "historical reconstruction runtime lacks a frozen package"
        ) from error
    return {
        "numpy": materializer.np.__version__,
        "opencv": materializer.cv2.__version__,
        "opencv_distribution": opencv_distribution,
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": torch_version,
    }


def validate_reconstruction_runtime(
    contract: dict[str, Any],
    identity_reader: Callable[[], dict[str, str]] = current_reconstruction_runtime,
) -> dict[str, str]:
    actual = identity_reader()
    expected = contract["historical_runtime"]
    if actual != expected:
        raise CacheReconstructionError(
            f"historical reconstruction runtime changed: expected {expected}, "
            f"got {actual}"
        )
    return actual


def _expected_prediction_set(audit: dict[str, Any], name: str) -> dict[str, Any]:
    value = audit.get(name)
    if not isinstance(value, dict):
        raise CacheReconstructionError(f"historical audit lacks {name}")
    return value


def validate_contract(
    repo_root: Path, contract_path: Path = CONTRACT_PATH
) -> dict[str, Any]:
    contract_file = repo_path(repo_root, str(contract_path))
    if sha256_file(contract_file) != EXPECTED_CONTRACT_SHA256:
        raise CacheReconstructionError("cache reconstruction contract sha256 changed")
    contract = read_json(contract_file)
    if (
        contract.get("schema_version") != 2
        or contract.get("state")
        != "preregistered_tiled_probe_cache_reconstruction_integration"
        or contract.get("terminal") != "PREREQUISITE_NEEDED"
    ):
        raise CacheReconstructionError("cache reconstruction contract changed")
    validate_runtime_monitor_contract(repo_root)
    validate_reconstruction_boundaries(contract)
    validate_authority(repo_root, contract)
    validate_artifact(
        repo_root, contract["external_layout_plan"], "external layout plan"
    )
    validate_artifact(
        repo_root,
        contract["probe_gate"]["contract"],
        "tiled probe contract",
    )
    validate_artifact(
        repo_root,
        contract["probe_gate"]["integration_verification"],
        "tiled probe integration verification",
    )
    historical = contract.get("historical_source", {})
    plan_path = validate_artifact(
        repo_root, historical["training_plan"], "historical training plan"
    )
    audit_path = validate_artifact(
        repo_root, historical["materialization_audit"], "historical audit"
    )
    manifest_path = validate_artifact(
        repo_root, historical["sample_manifest"], "historical sample manifest"
    )
    validate_artifact(
        repo_root, historical["materializer"], "historical materializer"
    )
    legacy_probe_source = {
        "path": "scripts/analysis/probe_external_text_layout_runtime_safety.py",
        "sha256": (
            "8b6c563a7e9f5d879cc962b35fcc6ede15ce341a5f790fc49233cda9d235b43e"
        ),
    }
    runtime_repair_evidence = contract.get("runtime_repair_evidence", [])
    if legacy_probe_source not in runtime_repair_evidence:
        raise CacheReconstructionError(
            "legacy probe source binding changed before adapter application"
        )
    for index, artifact in enumerate(runtime_repair_evidence):
        if artifact == legacy_probe_source:
            continue
        validate_artifact(repo_root, artifact, f"runtime repair evidence {index}")

    baseline_relative_contract = validate_baseline_relative_contract(repo_root)

    plan = read_json(plan_path)
    audit = read_json(audit_path)
    if (
        audit.get("terminal") != "PASS"
        or audit.get("train_count") != 275
        or audit.get("training_plan_sha256") != historical["training_plan"]["sha256"]
        or audit.get("sample_manifest_sha256")
        != historical["sample_manifest"]["sha256"]
    ):
        raise CacheReconstructionError("historical train275 audit changed")
    expected = contract.get("expected_outputs", {})
    for stage, audit_prefix in (
        ("primary", "primary"),
        ("second_stage", "second_stage"),
    ):
        stage_expected = expected.get(stage, {})
        if stage_expected.get("metrics_sha256") != audit.get(
            f"{audit_prefix}_metrics", {}
        ).get("metrics_sha256"):
            raise CacheReconstructionError(
                f"{stage} metrics expectation disagrees with historical audit"
            )
        if stage_expected.get("prediction_set") != _expected_prediction_set(
            audit, f"{audit_prefix}_predictions"
        ):
            raise CacheReconstructionError(
                f"{stage} prediction expectation disagrees with historical audit"
            )
    lines = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 275 or len(set(lines)) != 275:
        raise CacheReconstructionError("historical sample manifest population changed")
    missing_sources = [
        value for value in lines if not repo_path(repo_root, value).is_file()
    ]
    if missing_sources:
        raise CacheReconstructionError(
            f"historical sample sources are missing: {missing_sources[:5]}"
        )
    evidence = plan.get("evidence", {})
    for name in (
        "cleanup_model",
        "current_primary_checkpoint",
        "current_primary_config",
        "current_second_stage_checkpoint",
        "data_role_plan",
        "primary_inference",
        "second_stage_inference",
    ):
        validate_artifact(repo_root, evidence[name], f"historical plan {name}")
    return {
        "audit": audit,
        "contract": contract,
        "manifest_lines": lines,
        "manifest_path": manifest_path,
        "monitor_contract": baseline_relative_contract,
        "plan": plan,
        "runtime_contract": baseline_relative_contract,
    }


def expected_prediction_names(manifest_lines: list[str]) -> list[str]:
    names = sorted(f"{Path(value).stem}.png" for value in manifest_lines)
    if len(names) != len(set(names)):
        raise CacheReconstructionError("manifest names collide after PNG conversion")
    return names


def validate_prediction_set(
    prediction_dir: Path,
    expected_names: list[str],
    expected: dict[str, Any],
) -> dict[str, Any]:
    if not prediction_dir.is_dir():
        raise CacheReconstructionError(
            f"prediction directory is missing: {prediction_dir}"
        )
    if prediction_dir.is_symlink():
        raise CacheReconstructionError("prediction directory surface changed")
    predictions = list(prediction_dir.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in predictions):
        raise CacheReconstructionError("prediction directory surface changed")
    actual_names = sorted(path.name for path in predictions)
    if actual_names != expected_names:
        raise CacheReconstructionError("prediction filename population changed")
    rows = [
        f"{name} {sha256_file(prediction_dir / name)}" for name in actual_names
    ]
    actual = {
        "content_sha256": sha256_rows(rows),
        "count": len(actual_names),
        "filename_sha256": sha256_rows(actual_names),
    }
    if actual != expected:
        raise CacheReconstructionError(
            f"prediction content changed: expected {expected}, got {actual}"
        )
    return actual


def validate_cache(
    cache_dir: Path,
    *,
    expected_names: list[str],
    expected: dict[str, Any],
) -> dict[str, Any]:
    if not cache_dir.is_dir():
        raise CacheReconstructionError(f"cache directory is missing: {cache_dir}")
    if {path.name for path in cache_dir.iterdir()} != {"metrics.csv", "pred"}:
        raise CacheReconstructionError(f"cache surface changed: {cache_dir}")
    metrics_path = cache_dir / "metrics.csv"
    if not metrics_path.is_file() or metrics_path.is_symlink():
        raise CacheReconstructionError(f"cache metrics surface changed: {metrics_path}")
    actual_metrics_sha256 = sha256_file(metrics_path)
    if actual_metrics_sha256 != expected["metrics_sha256"]:
        raise CacheReconstructionError(
            f"cache metrics changed: expected {expected['metrics_sha256']}, "
            f"got {actual_metrics_sha256}"
        )
    predictions = validate_prediction_set(
        cache_dir / "pred", expected_names, expected["prediction_set"]
    )
    return {
        "metrics_sha256": actual_metrics_sha256,
        "prediction_set": predictions,
    }


def validate_reconstructed_cache(
    cache_dir: Path,
    *,
    expected_names: list[str],
    expected: dict[str, Any],
) -> dict[str, Any]:
    if cache_dir.is_symlink():
        raise CacheReconstructionError(
            f"reconstructed cache must be a real directory: {cache_dir}"
        )
    return validate_cache(
        cache_dir,
        expected_names=expected_names,
        expected=expected,
    )


def _required_health_number(
    health: Any,
    field: str,
    *,
    label: str,
    integer: bool,
) -> float | int:
    if not isinstance(health, dict) or field not in health:
        raise CacheReconstructionError(f"{label} lacks {field}")
    value = health[field]
    if isinstance(value, bool):
        raise CacheReconstructionError(f"{label} has invalid {field}")
    if integer:
        if not isinstance(value, int) or value < 0:
            raise CacheReconstructionError(f"{label} has invalid {field}")
        return value
    if not isinstance(value, (int, float)):
        raise CacheReconstructionError(f"{label} has invalid {field}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > 100.0:
        raise CacheReconstructionError(f"{label} has invalid {field}")
    return numeric


def validate_health_snapshot(
    health: Any,
    *,
    gate: dict[str, Any],
    label: str,
) -> dict[str, float | int]:
    normalized = {
        "memory_free_percent": _required_health_number(
            health, "memory_free_percent", label=label, integer=False
        ),
        "process_tree_rss_bytes": _required_health_number(
            health, "process_tree_rss_bytes", label=label, integer=True
        ),
        "swap_used_bytes": _required_health_number(
            health, "swap_used_bytes", label=label, integer=True
        ),
    }
    if (
        normalized["memory_free_percent"]
        < float(gate["minimum_system_free_memory_percent"])
        or normalized["process_tree_rss_bytes"]
        > int(gate["maximum_process_tree_rss_bytes"])
        or normalized["swap_used_bytes"]
        > int(gate["maximum_swap_growth_bytes"])
    ):
        raise CacheReconstructionError(f"{label} crossed a runtime limit")
    return normalized


def validate_health_summary(
    health: Any,
    *,
    gate: dict[str, Any],
    label: str,
) -> dict[str, float | int]:
    normalized = {
        "minimum_memory_free_percent": _required_health_number(
            health, "minimum_memory_free_percent", label=label, integer=False
        ),
        "peak_process_tree_rss_bytes": _required_health_number(
            health, "peak_process_tree_rss_bytes", label=label, integer=True
        ),
        "peak_swap_used_bytes": _required_health_number(
            health, "peak_swap_used_bytes", label=label, integer=True
        ),
    }
    if (
        normalized["minimum_memory_free_percent"]
        < float(gate["minimum_system_free_memory_percent"])
        or normalized["peak_process_tree_rss_bytes"]
        > int(gate["maximum_process_tree_rss_bytes"])
        or normalized["peak_swap_used_bytes"]
        > int(gate["maximum_swap_growth_bytes"])
    ):
        raise CacheReconstructionError(f"{label} crossed a runtime limit")
    return normalized


def _probe_health_gate(gate: dict[str, Any], *, launch: bool) -> dict[str, Any]:
    return {
        "maximum_process_tree_rss_bytes": gate["maximum_process_tree_rss_bytes"],
        "maximum_swap_growth_bytes": gate["maximum_swap_growth_bytes"],
        "minimum_system_free_memory_percent": gate[
            "minimum_launch_memory_free_percent"
            if launch
            else "minimum_runtime_memory_free_percent"
        ],
    }


def validate_explicit_growth_snapshot(
    health: Any,
    *,
    gate: dict[str, Any],
    label: str,
) -> dict[str, float | int]:
    normalized = {
        "memory_free_percent": _required_health_number(
            health, "memory_free_percent", label=label, integer=False
        ),
        "process_tree_rss_bytes": _required_health_number(
            health, "process_tree_rss_bytes", label=label, integer=True
        ),
        "swap_growth_bytes": _required_health_number(
            health, "swap_growth_bytes", label=label, integer=True
        ),
    }
    if (
        normalized["memory_free_percent"]
        < float(gate["minimum_system_free_memory_percent"])
        or normalized["process_tree_rss_bytes"]
        > int(gate["maximum_process_tree_rss_bytes"])
        or normalized["swap_growth_bytes"]
        > int(gate["maximum_swap_growth_bytes"])
    ):
        raise CacheReconstructionError(f"{label} crossed a runtime limit")
    return normalized


def validate_explicit_growth_summary(
    health: Any,
    *,
    gate: dict[str, Any],
    label: str,
) -> dict[str, float | int]:
    normalized = {
        "minimum_memory_free_percent": _required_health_number(
            health, "minimum_memory_free_percent", label=label, integer=False
        ),
        "peak_process_tree_rss_bytes": _required_health_number(
            health, "peak_process_tree_rss_bytes", label=label, integer=True
        ),
        "peak_swap_growth_bytes": _required_health_number(
            health, "peak_swap_growth_bytes", label=label, integer=True
        ),
    }
    if (
        normalized["minimum_memory_free_percent"]
        < float(gate["minimum_system_free_memory_percent"])
        or normalized["peak_process_tree_rss_bytes"]
        > int(gate["maximum_process_tree_rss_bytes"])
        or normalized["peak_swap_growth_bytes"]
        > int(gate["maximum_swap_growth_bytes"])
    ):
        raise CacheReconstructionError(f"{label} crossed a runtime limit")
    return normalized


def validate_probe_pass(repo_root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("probe_gate") != expected_probe_gate_v2():
        raise CacheReconstructionError("tiled v2 probe safety gate changed")
    gate = contract["probe_gate"]
    result_path = repo_path(repo_root, gate["probe_result"]["path"])
    if not result_path.is_file():
        raise CacheReconstructionError(
            f"tiled one-page runtime probe is missing: {result_path}"
        )
    result = read_json(result_path)
    attempt_count = result.get("attempt_count")
    residual_model_process_count = result.get("residual_model_process_count")
    booted_ios_simulator_count = result.get("booted_ios_simulator_count")
    if (
        result.get("contract") != gate["contract"]
        or result.get("terminal") != gate["required_probe_terminal"]
        or result.get("reason_code") != gate["required_probe_reason_code"]
        or result.get("formal_evidence") is not False
        or result.get("formal_outputs_written") is not False
        or result.get("target_access") is not False
        or result.get("label_access") is not False
        or result.get("recognition") is not False
        or result.get("routing_metadata_access") is not False
        or result.get("temporary_page_outputs_retained") is not False
        or result.get("result_authority") != gate["required_result_authority"]
        or result.get("probe") != gate["required_probe"]
        or result.get("schema_version") != gate["required_schema_version"]
        or result.get("detector", {}).get("device") != "cpu"
        or result.get("detector", {}).get("engine") != "transformers"
        or result.get("detector", {}).get("model_name")
        != "PP-OCRv6_medium_det"
        or result.get("page", {}).get("file") != gate["probe_page"]
        or result.get("page", {}).get("source_sha256")
        != EXPECTED_PROBE_SOURCE_SHA256
        or type(attempt_count) is not int
        or attempt_count != gate["required_attempt_count"]
        or result.get("page_completed") is not gate["required_page_completed"]
        or type(residual_model_process_count) is not int
        or residual_model_process_count
        != gate["required_residual_model_process_count"]
        or type(booted_ios_simulator_count) is not int
        or booted_ios_simulator_count != 0
    ):
        raise CacheReconstructionError("tiled one-page runtime probe did not pass")
    if result.get("safety_limits") != gate["required_safety_limits"]:
        raise CacheReconstructionError("tiled one-page probe limits changed")
    if result.get("thread_caps") != gate["required_thread_caps"]:
        raise CacheReconstructionError("tiled one-page probe thread caps changed")
    runtime_gate = _probe_health_gate(gate, launch=False)
    launch_gate = _probe_health_gate(gate, launch=True)
    _required_health_number(
        result, "launch_swap_baseline_bytes", label="probe result", integer=True
    )
    launch_health = validate_explicit_growth_summary(
        result.get("launch_health"),
        gate=launch_gate,
        label="probe launch health",
    )
    if (
        launch_health["peak_swap_growth_bytes"] != 0
        or result.get("launch_health", {}).get("sample_count") != 61
        or result.get("launch_health", {}).get(
            "stability_sample_interval_seconds"
        )
        != 1.0
        or result.get("launch_health", {}).get("stability_window_seconds")
        != 60.0
    ):
        raise CacheReconstructionError("probe launch stability evidence changed")
    page_health = validate_explicit_growth_summary(
        result.get("page_health"),
        gate=runtime_gate,
        label="probe page health",
    )
    post_health = validate_explicit_growth_snapshot(
        result.get("post_run_health"),
        gate=runtime_gate,
        label="probe post_run_health",
    )
    peak_growth = _required_health_number(
        result, "peak_swap_growth_bytes", label="probe result", integer=True
    )
    if peak_growth != max(
        int(launch_health["peak_swap_growth_bytes"]),
        int(page_health["peak_swap_growth_bytes"]),
        int(post_health["swap_growth_bytes"]),
    ):
        raise CacheReconstructionError("probe peak swap growth evidence changed")
    if sha256_file(result_path) != gate["probe_result"]["sha256"]:
        raise CacheReconstructionError("tiled one-page runtime probe sha256 changed")
    return result


def reconstruction_plan(state: dict[str, Any]) -> dict[str, Any]:
    plan = copy.deepcopy(state["plan"])
    manifest = state["contract"]["build"]["archive_manifest"]
    plan["pipeline_preparation"]["primary"]["samples_file"] = manifest
    plan["pipeline_preparation"]["second_stage"]["samples_file"] = manifest
    return plan


def stage_paths(repo_root: Path, contract: dict[str, Any]) -> dict[str, Path]:
    build = contract["build"]
    return {
        name: repo_path(repo_root, build[name])
        for name in ("primary", "second_stage", "archive_primary", "archive_second_stage")
    }


def _load_historical_helper() -> Any:
    try:
        return importlib.import_module(HISTORICAL_HELPER_MODULE)
    except ImportError as error:
        raise CacheReconstructionError(
            "historical cache materialization helper is unavailable"
        ) from error


def build_stage_command(
    *,
    repo_root: Path,
    state: dict[str, Any],
    stage: str,
    output_dir: Path,
    helper: Any,
) -> list[str]:
    plan = reconstruction_plan(state)
    if stage == "primary":
        return helper.primary_command(repo_root, plan, output_dir)
    if stage == "second_stage":
        return helper.second_stage_command(repo_root, plan, output_dir)
    raise CacheReconstructionError(f"stage has no model command: {stage}")


def validate_current_launch_health(
    contract: dict[str, Any],
    *,
    health_reader: Callable[[int], dict[str, float | int]] = runtime.runtime_health,
) -> tuple[
    int,
    dict[str, float | int],
    Callable[[int], dict[str, float | int]],
]:
    materializer.assert_no_conflicting_model_processes()
    health = dict(health_reader(os.getpid()))
    launch_swap_baseline_bytes = int(
        _required_health_number(
            health,
            "swap_used_bytes",
            label="current launch health",
            integer=True,
        )
    )

    def relative_reader(pid: int) -> dict[str, float | int]:
        current = dict(health_reader(pid))
        absolute_swap = int(
            _required_health_number(
                current,
                "swap_used_bytes",
                label="cache stage runtime health",
                integer=True,
            )
        )
        current["swap_used_bytes"] = max(
            0, absolute_swap - launch_swap_baseline_bytes
        )
        return current

    relative_launch = dict(health)
    relative_launch["swap_used_bytes"] = 0
    normalized = validate_health_snapshot(
        relative_launch,
        gate=contract["reconstruction_gate"],
        label="current launch health",
    )
    runtime.enforce_health_limits(
        relative_launch,
        maximum_process_tree_rss_bytes=int(
            contract["reconstruction_gate"]["maximum_process_tree_rss_bytes"]
        ),
        minimum_memory_free_percent=float(
            contract["reconstruction_gate"]["minimum_system_free_memory_percent"]
        ),
        maximum_swap_used_bytes=int(
            contract["reconstruction_gate"]["maximum_swap_growth_bytes"]
        ),
    )
    initial_health = {
        "memory_free_percent": normalized["memory_free_percent"],
        "process_tree_rss_bytes": normalized["process_tree_rss_bytes"],
    }
    return launch_swap_baseline_bytes, initial_health, relative_reader


def validate_post_stage_health(
    contract: dict[str, Any],
    health_reader: Callable[[int], dict[str, float | int]],
) -> dict[str, float | int]:
    materializer.assert_no_conflicting_model_processes()
    health = health_reader(os.getpid())
    normalized = validate_health_snapshot(
        health,
        gate=contract["reconstruction_gate"],
        label="current post-run health",
    )
    runtime.enforce_health_limits(
        health,
        maximum_process_tree_rss_bytes=int(
            contract["reconstruction_gate"]["maximum_process_tree_rss_bytes"]
        ),
        minimum_memory_free_percent=float(
            contract["reconstruction_gate"]["minimum_system_free_memory_percent"]
        ),
        maximum_swap_used_bytes=int(
            contract["reconstruction_gate"]["maximum_swap_growth_bytes"]
        ),
    )
    return {
        "memory_free_percent": normalized["memory_free_percent"],
        "process_tree_rss_bytes": normalized["process_tree_rss_bytes"],
        "swap_growth_bytes": normalized["swap_used_bytes"],
    }


def terminate_monitored_process_group(
    process: Any,
    *,
    grace_seconds: float,
) -> None:
    pid = getattr(process, "pid", None)
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise CacheReconstructionError("cache stage child process has no valid pid")
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired as error:
        raise CacheReconstructionError(
            "cache stage process group survived SIGKILL"
        ) from error


def wait_for_monitored_command(
    process: Any,
    *,
    monitor_contract: dict[str, Any],
    health_reader: Callable[[int], dict[str, float | int]],
) -> dict[str, float | int]:
    monitor = validate_runtime_monitor_settings(monitor_contract)
    pid = getattr(process, "pid", None)
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise CacheReconstructionError("cache stage child process did not start")
    observed = runtime.health_summary(
        {
            "memory_free_percent": 100.0,
            "process_tree_rss_bytes": 0,
            "swap_used_bytes": 0,
        }
    )
    try:
        while process.poll() is None:
            health = validate_health_snapshot(
                health_reader(pid),
                gate=monitor,
                label="cache stage runtime health",
            )
            observed = runtime.health_summary(health, observed)
            runtime.enforce_health_limits(
                health,
                observed_health=observed,
                maximum_process_tree_rss_bytes=int(
                    monitor["maximum_process_tree_rss_bytes"]
                ),
                minimum_memory_free_percent=float(
                    monitor["minimum_system_free_memory_percent"]
                ),
                maximum_swap_used_bytes=int(monitor["maximum_swap_growth_bytes"]),
            )
            try:
                process.wait(timeout=float(monitor["monitor_interval_seconds"]))
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        terminate_monitored_process_group(
            process,
            grace_seconds=float(monitor["termination"]["grace_seconds"]),
        )
        raise
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            f"materialization command failed ({returncode}); child process exited"
        )
    return {
        "minimum_memory_free_percent": observed["minimum_memory_free_percent"],
        "peak_process_tree_rss_bytes": observed["peak_process_tree_rss_bytes"],
        "peak_swap_growth_bytes": observed["peak_swap_used_bytes"],
    }


def run_monitored_atomic_directory_command(
    *,
    repo_root: Path,
    final_dir: Path,
    command_builder: Callable[[Path], list[str]],
    log_path: Path,
    monitor_contract: dict[str, Any],
    helper: Any,
    health_reader: Callable[[int], dict[str, float | int]],
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> tuple[list[str], dict[str, float | int]]:
    validate_runtime_monitor_settings(monitor_contract)
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError(final_dir)
    temporary = final_dir.with_name(f".{final_dir.name}.materializing")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"stale materialization directory: {temporary}")
    command = command_builder(temporary)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() or log_path.is_symlink():
        raise FileExistsError(log_path)
    with log_path.open("x", encoding="utf-8") as log:
        process = popen_factory(
            command,
            cwd=repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        observed = wait_for_monitored_command(
            process,
            monitor_contract=monitor_contract,
            health_reader=health_reader,
        )
    if not temporary.is_dir():
        raise RuntimeError(f"command did not create expected directory: {temporary}")
    temporary.replace(final_dir)
    helper.rewrite_metrics_paths(final_dir / "metrics.csv", temporary, final_dir)
    return command, observed


def reconstruct_stage(
    *,
    repo_root: Path,
    state: dict[str, Any],
    stage: str,
    helper_loader: Callable[[], Any] = _load_historical_helper,
) -> dict[str, Any]:
    if stage not in {"primary", "second_stage"}:
        raise CacheReconstructionError(f"unsupported reconstruction stage: {stage}")
    contract = state["contract"]
    runtime_contract = state["runtime_contract"]
    validate_probe_pass(repo_root, runtime_contract)
    validate_reconstruction_runtime(contract)
    paths = stage_paths(repo_root, contract)
    names = expected_prediction_names(state["manifest_lines"])
    if stage == "second_stage":
        validate_reconstructed_cache(
            paths["primary"],
            expected_names=names,
            expected=contract["expected_outputs"]["primary"],
        )
    cache_dir = paths[stage]
    expected = contract["expected_outputs"][stage]
    if cache_dir.exists() or cache_dir.is_symlink():
        return {
            "cache": validate_reconstructed_cache(
                cache_dir, expected_names=names, expected=expected
            ),
            "stage": stage,
            "status": "already_reconstructed",
        }
    log_path = repo_root / CONTROL_DIR / f"{stage}.log"
    with runtime.exclusive_run_lock(runtime.HOST_USER_RUN_LOCK_PATH):
        (
            launch_swap_baseline_bytes,
            initial_health,
            relative_health_reader,
        ) = validate_current_launch_health(runtime_contract)
        helper = helper_loader()
        command, runtime_health = run_monitored_atomic_directory_command(
            repo_root=repo_root,
            final_dir=cache_dir,
            command_builder=lambda temporary: build_stage_command(
                repo_root=repo_root,
                state=state,
                stage=stage,
                output_dir=temporary,
                helper=helper,
            ),
            log_path=log_path,
            monitor_contract=state["monitor_contract"],
            helper=helper,
            health_reader=relative_health_reader,
        )
        post_health = validate_post_stage_health(
            runtime_contract, relative_health_reader
        )
    return {
        "cache": validate_reconstructed_cache(
            cache_dir, expected_names=names, expected=expected
        ),
        "command": command,
        "initial_health": initial_health,
        "launch_swap_baseline_bytes": launch_swap_baseline_bytes,
        "post_health": post_health,
        "runtime_health": runtime_health,
        "stage": stage,
        "status": "reconstructed",
    }


def validate_publication_destination(
    final_path: Path, source_path: Path
) -> str | None:
    if final_path.exists() or final_path.is_symlink():
        if not final_path.is_symlink():
            raise CacheReconstructionError(
                f"archive cache path already has different content: {final_path}"
            )
        target = os.readlink(final_path)
        if Path(target).is_absolute() or final_path.resolve() != source_path.resolve():
            raise CacheReconstructionError(
                f"archive cache path is not the registered relative symlink: "
                f"{final_path}"
            )
        return target
    return None


def atomic_relative_symlink(final_path: Path, source_path: Path) -> str:
    if not source_path.is_dir():
        raise CacheReconstructionError(f"cache publication source is missing: {source_path}")
    existing = validate_publication_destination(final_path, source_path)
    if existing is not None:
        return existing
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.name}.publishing")
    if temporary.exists() or temporary.is_symlink():
        raise CacheReconstructionError(f"stale cache publication: {temporary}")
    target = os.path.relpath(source_path, start=final_path.parent)
    temporary.symlink_to(target, target_is_directory=True)
    temporary.replace(final_path)
    return target


def publish_caches(repo_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    contract = state["contract"]
    paths = stage_paths(repo_root, contract)
    names = expected_prediction_names(state["manifest_lines"])
    for stage in ("primary", "second_stage"):
        validate_reconstructed_cache(
            paths[stage],
            expected_names=names,
            expected=contract["expected_outputs"][stage],
        )
    for stage, archive_name in (
        ("primary", "archive_primary"),
        ("second_stage", "archive_second_stage"),
    ):
        validate_publication_destination(paths[archive_name], paths[stage])
    links = {
        "primary": atomic_relative_symlink(
            paths["archive_primary"], paths["primary"]
        ),
        "second_stage": atomic_relative_symlink(
            paths["archive_second_stage"], paths["second_stage"]
        ),
    }
    return {"links": links, "status": "published"}


def verify_published_caches(
    repo_root: Path, state: dict[str, Any]
) -> dict[str, Any]:
    contract = state["contract"]
    paths = stage_paths(repo_root, contract)
    names = expected_prediction_names(state["manifest_lines"])
    result: dict[str, Any] = {}
    for stage, archive_name in (
        ("primary", "archive_primary"),
        ("second_stage", "archive_second_stage"),
    ):
        archive = paths[archive_name]
        if (
            not archive.is_symlink()
            or Path(os.readlink(archive)).is_absolute()
            or archive.resolve() != paths[stage].resolve()
        ):
            raise CacheReconstructionError(
                f"published {stage} cache is not the registered relative symlink"
            )
        result[stage] = validate_cache(
            archive,
            expected_names=names,
            expected=contract["expected_outputs"][stage],
        )
    return {"caches": result, "status": "verified"}


def preflight_report(repo_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    contract = state["contract"]
    runtime_contract = state["runtime_contract"]
    paths = stage_paths(repo_root, contract)
    probe_gate = runtime_contract["probe_gate"]
    probe_path = repo_path(repo_root, probe_gate["probe_result"]["path"])
    validate_probe_pass(repo_root, runtime_contract)
    try:
        actual_runtime = current_reconstruction_runtime()
        runtime_ready = actual_runtime == contract["historical_runtime"]
    except CacheReconstructionError as error:
        actual_runtime = {"error": str(error)}
        runtime_ready = False
    return {
        "archive_paths_absent": all(
            not paths[name].exists() and not paths[name].is_symlink()
            for name in ("archive_primary", "archive_second_stage")
        ),
        "build_paths_absent": all(
            not paths[name].exists() and not paths[name].is_symlink()
            for name in ("primary", "second_stage")
        ),
        "execution_authorized": False,
        "historical_manifest_count": len(state["manifest_lines"]),
        "historical_runtime_ready": runtime_ready,
        "observed_runtime": actual_runtime,
        "probe_result_present": probe_path.is_file(),
        "probe_result_sha256": sha256_file(probe_path),
        "reconstruction_id": RECONSTRUCTION_ID,
        "result_authority": "frozen_cache_reconstruction_preflight_only",
        "static_terminal": "PASS",
        "terminal": "PREREQUISITE_NEEDED",
    }


def run_stage(repo_root: Path, contract_path: Path, stage: str) -> dict[str, Any]:
    state = validate_contract(repo_root, contract_path)
    if stage == "preflight":
        return preflight_report(repo_root, state)
    if stage == "primary":
        return reconstruct_stage(repo_root=repo_root, state=state, stage=stage)
    if stage == "second_stage":
        return reconstruct_stage(repo_root=repo_root, state=state, stage=stage)
    if stage == "publish":
        return publish_caches(repo_root, state)
    if stage == "verify":
        return verify_published_caches(repo_root, state)
    if stage == "all":
        return {
            "primary": reconstruct_stage(
                repo_root=repo_root, state=state, stage="primary"
            ),
            "second_stage": reconstruct_stage(
                repo_root=repo_root, state=state, stage="second_stage"
            ),
            "publish": publish_caches(repo_root, state),
            "verify": verify_published_caches(repo_root, state),
        }
    raise CacheReconstructionError(f"unsupported stage: {stage}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--stage", choices=STAGES, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_stage(args.repo_root.resolve(), args.contract, args.stage)
    except (KeyError, RuntimeError, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "reason": str(error),
                    "reconstruction_id": RECONSTRUCTION_ID,
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
