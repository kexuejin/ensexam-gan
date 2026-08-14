#!/usr/bin/env python3
"""Run one target-free detector page as a nonformal runtime-safety probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import external_text_layout_materialization_runtime as runtime  # noqa: E402
from scripts.analysis import materialize_external_text_layout_support_train_only as materializer  # noqa: E402


PLAN_PATH = Path("docs/external-text-layout-support-prerequisite-v1.json")
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
PROBE_CONTRACT_PATH = Path(
    "docs/external-text-layout-tiled-9x9-one-page-safety-probe-v1.json"
)
EXPECTED_PROBE_CONTRACT_SHA256 = (
    "1fd02d49250150f85ce190601b21b36d60a308ef92b07e564c8a21575124aee4"
)
DEFAULT_RESULT_PATH = Path(
    "outputs/external-text-layout-runtime-safety-probe-tiled-9x9-20260814/result.json"
)
PROBE_MAX_PROCESS_TREE_RSS_BYTES = 8 * 1024**3
PROBE_MIN_LAUNCH_MEMORY_FREE_PERCENT = 70.0
PROBE_MIN_RUNTIME_MEMORY_FREE_PERCENT = 45.0
PROBE_MAX_SWAP_USED_BYTES = 512 * 1024**2
THREAD_CAP_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
DETECTOR_ARTIFACT_NAMES = (
    "gitattributes",
    "config_json",
    "inference_yml",
    "model_safetensors",
    "preprocessor_config_json",
    "readme_md",
)


ProbeRunner = Callable[..., dict[str, float | int]]
HealthReader = Callable[[int], dict[str, float | int]]
SimulatorCounter = Callable[[], int]


def select_probe_source(repo_root: Path, plan: dict[str, Any]) -> Path:
    """Select the first frozen raw source without consulting metrics or labels."""
    manifest = plan.get("data", {}).get("manifest", {})
    if set(manifest) != {"path", "sha256"}:
        raise materializer.MaterializationError("probe manifest contract changed")
    manifest_path = materializer.repo_path(repo_root, str(manifest["path"]))
    if not manifest_path.is_file():
        raise materializer.MaterializationError("probe manifest is missing")
    if materializer.sha256_file(manifest_path) != manifest["sha256"]:
        raise materializer.MaterializationError("probe manifest sha256 changed")
    rows = [row.strip() for row in manifest_path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row]
    if not rows:
        raise materializer.MaterializationError("probe manifest is empty")
    source_path = materializer.repo_path(repo_root, rows[0])
    materializer.assert_source_path(source_path)
    if not source_path.is_file():
        raise materializer.MaterializationError("probe source is missing")
    return source_path


def validate_probe_detector(plan: dict[str, Any]) -> dict[str, str]:
    external = plan.get("evidence", {}).get("official_text_detector", {})
    if (
        external.get("license") != "Apache-2.0"
        or external.get("model_name") != "PP-OCRv6_medium_det"
    ):
        raise materializer.MaterializationError(
            "probe detector identity or license changed"
        )
    if set(external) != {"license", "model_name", *DETECTOR_ARTIFACT_NAMES}:
        raise materializer.MaterializationError(
            "probe detector artifact contract changed"
        )
    paths = {
        key: materializer.validate_external_artifact(
            external[key], f"probe detector {key}"
        )
        for key in DETECTOR_ARTIFACT_NAMES
    }
    model_dir = Path(plan["external_text_layout_materialization"]["model_dir"])
    if not model_dir.is_dir() or any(path.parent != model_dir for path in paths.values()):
        raise materializer.MaterializationError("probe detector directory changed")
    model_files = {path.name for path in model_dir.iterdir() if path.is_file()}
    model_directories = {path.name for path in model_dir.iterdir() if path.is_dir()}
    if model_files != {
        ".gitattributes",
        "README.md",
        "config.json",
        "inference.yml",
        "model.safetensors",
        "preprocessor_config.json",
    } or model_directories != {".cache"}:
        raise materializer.MaterializationError(
            "probe detector directory surface changed"
        )
    return {key: materializer.sha256_file(path) for key, path in paths.items()}


def safety_limits() -> dict[str, float | int]:
    return {
        "detector_process_tree_rss_bytes_max": PROBE_MAX_PROCESS_TREE_RSS_BYTES,
        "launch_memory_free_percent_min": PROBE_MIN_LAUNCH_MEMORY_FREE_PERCENT,
        "runtime_memory_free_percent_min": PROBE_MIN_RUNTIME_MEMORY_FREE_PERCENT,
        "page_timeout_seconds": runtime.PAGE_TIMEOUT_SECONDS,
        "swap_used_bytes_max": PROBE_MAX_SWAP_USED_BYTES,
    }


def validate_thread_caps(values: dict[str, str | None]) -> None:
    if values != {name: "1" for name in THREAD_CAP_NAMES}:
        raise materializer.MaterializationError(
            "detector thread caps must all be set to 1 before process start"
        )


def base_result(source_path: Path, contract_path: Path) -> dict[str, Any]:
    return {
        "attempt_count": 0,
        "contract": {
            "path": str(contract_path),
            "sha256": EXPECTED_PROBE_CONTRACT_SHA256,
        },
        "detector": {
            "device": "cpu",
            "engine": "transformers",
            "model_name": "PP-OCRv6_medium_det",
        },
        "formal_evidence": False,
        "formal_outputs_written": False,
        "label_access": False,
        "page_completed": False,
        "page": {
            "file": source_path.name,
            "source_sha256": materializer.sha256_file(source_path),
        },
        "probe": "external_text_layout_tiled_9x9_single_page_runtime_safety",
        "recognition": False,
        "result_authority": "runtime_prerequisite_only",
        "routing_metadata_access": False,
        "safety_limits": safety_limits(),
        "schema_version": 1,
        "target_access": False,
        "temporary_page_outputs_retained": False,
        "thread_caps": {name: os.environ.get(name) for name in THREAD_CAP_NAMES},
    }


def validate_probe_contract(
    repo_root: Path,
    contract_path: Path,
    plan_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    contract_file = materializer.repo_path(repo_root, str(contract_path))
    if not contract_file.is_file():
        raise materializer.MaterializationError("tiled probe contract is missing")
    if materializer.sha256_file(contract_file) != EXPECTED_PROBE_CONTRACT_SHA256:
        raise materializer.MaterializationError("tiled probe contract sha256 changed")
    contract = materializer.read_json(contract_file)
    if (
        contract.get("state")
        != "preregistered_integration_allowed_execution_disabled_until_all_host_gates_pass"
        or contract.get("terminal") != "PREREQUISITE_NEEDED"
        or contract.get("execution", {}).get("exact_attempt_count") != 1
        or contract.get("probe", {}).get("tiled_output_rows") != 4
    ):
        raise materializer.MaterializationError("tiled probe authority changed")
    if Path(contract["execution"]["result_path"]) != result_path:
        raise materializer.MaterializationError("tiled probe result path changed")
    if Path(contract["frozen_inputs"]["support_plan"]["path"]) != plan_path:
        raise materializer.MaterializationError("tiled probe support plan path changed")
    launch = contract.get("host_launch_gates", {})
    acceptance = contract.get("acceptance", {})
    if (
        launch.get("minimum_system_free_memory_percent")
        != PROBE_MIN_LAUNCH_MEMORY_FREE_PERCENT
        or launch.get("maximum_swap_used_bytes") != PROBE_MAX_SWAP_USED_BYTES
        or launch.get("booted_ios_simulator_count") != 0
        or launch.get("no_conflicting_model_processes") is not True
        or launch.get("result_path_absent") is not True
        or acceptance.get("minimum_system_free_memory_percent")
        != PROBE_MIN_RUNTIME_MEMORY_FREE_PERCENT
        or acceptance.get("maximum_process_tree_rss_bytes")
        != PROBE_MAX_PROCESS_TREE_RSS_BYTES
        or acceptance.get("maximum_swap_used_bytes") != PROBE_MAX_SWAP_USED_BYTES
    ):
        raise materializer.MaterializationError("tiled probe safety gates changed")
    for label in ("repair", "repair_test", "repair_verification", "support_plan"):
        materializer.validate_internal_artifact(
            repo_root,
            contract["frozen_inputs"][label],
            f"tiled probe {label}",
        )
    return contract


def validate_probe_source_binding(
    repo_root: Path,
    source_path: Path,
    contract: dict[str, Any],
) -> None:
    expected = contract.get("frozen_inputs", {}).get("page", {})
    if set(expected) != {"path", "sha256"}:
        raise materializer.MaterializationError("tiled probe page contract changed")
    try:
        relative = source_path.relative_to(repo_root)
    except ValueError as error:
        raise materializer.MaterializationError(
            "tiled probe source escaped repository"
        ) from error
    if str(relative) != expected["path"]:
        raise materializer.MaterializationError("tiled probe source path changed")
    if materializer.sha256_file(source_path) != expected["sha256"]:
        raise materializer.MaterializationError("tiled probe source sha256 changed")


def validate_result_path(
    repo_root: Path, result_path: Path, plan: dict[str, Any]
) -> Path:
    destination = materializer.repo_path(repo_root, str(result_path))
    formal_paths = (
        materializer.repo_path(
            repo_root, plan["external_text_layout_materialization"]["output_root"]
        ),
        materializer.repo_path(
            repo_root, plan["planned_implementation"]["audit_output"]
        ).parent,
    )
    resolved = destination.resolve()
    if any(
        resolved == formal.resolve() or resolved.is_relative_to(formal.resolve())
        for formal in formal_paths
    ):
        raise materializer.MaterializationError(
            "runtime probe result must stay outside formal evidence paths"
        )
    return destination


def run_runtime_probe(
    *,
    repo_root: Path = ROOT,
    plan_path: Path = PLAN_PATH,
    ledger_path: Path = LEDGER_PATH,
    contract_path: Path = PROBE_CONTRACT_PATH,
    result_path: Path = DEFAULT_RESULT_PATH,
    page_runner: ProbeRunner | None = None,
    health_reader: HealthReader | None = None,
    simulator_counter: SimulatorCounter | None = None,
    lock_path: Path = runtime.HOST_USER_RUN_LOCK_PATH,
) -> dict[str, Any]:
    contract = validate_probe_contract(
        repo_root,
        contract_path,
        plan_path,
        result_path,
    )
    plan_file = materializer.repo_path(repo_root, str(plan_path))
    ledger_file = materializer.repo_path(repo_root, str(ledger_path))
    plan = materializer.read_json(plan_file)
    ledger = materializer.read_json(ledger_file)
    materializer.validate_plan(plan)
    materializer.validate_authority(ledger)
    registered_runtime = materializer.validate_runtime(plan["evidence"]["runtime"])
    detector_files = validate_probe_detector(plan)
    source_path = select_probe_source(repo_root, plan)
    validate_probe_source_binding(repo_root, source_path, contract)
    spec = plan["external_text_layout_materialization"]
    destination = validate_result_path(repo_root, result_path, plan)
    if destination.exists():
        raise materializer.MaterializationError(
            "tiled probe result already exists; retry is prohibited"
        )
    page_runner = page_runner or materializer.run_isolated_page
    health_reader = health_reader or runtime.runtime_health
    result = base_result(source_path, contract_path)
    result["detector_files"] = detector_files
    result["runtime"] = registered_runtime

    temporary_root = Path(tempfile.mkdtemp(prefix="ensexam-external-layout-probe-"))
    page_dir = temporary_root / "pages"
    record_dir = temporary_root / "records"
    page_dir.mkdir()
    record_dir.mkdir()
    record_path = record_dir / f"{source_path.stem}.json"
    failure: Exception | None = None
    write_result = False
    try:
        with runtime.exclusive_run_lock(lock_path):
            if destination.exists():
                raise materializer.MaterializationError(
                    "tiled probe result already exists; retry is prohibited"
                )
            write_result = True
            validate_thread_caps(result["thread_caps"])
            materializer.assert_no_conflicting_model_processes()
            result["booted_ios_simulator_count"] = (
                runtime.assert_no_booted_ios_simulators(simulator_counter)
            )
            initial_health = health_reader(os.getpid())
            result["initial_health"] = initial_health
            materializer.enforce_health_limits(
                initial_health,
                maximum_process_tree_rss_bytes=PROBE_MAX_PROCESS_TREE_RSS_BYTES,
                minimum_memory_free_percent=PROBE_MIN_LAUNCH_MEMORY_FREE_PERCENT,
                maximum_swap_used_bytes=PROBE_MAX_SWAP_USED_BYTES,
            )
            result.update(
                {
                    "attempt_count": 1,
                    "reason_code": "runtime_safety_probe_running",
                    "terminal": "RUNNING",
                }
            )
            materializer.atomic_write_json(destination, result)
            try:
                result["page_health"] = page_runner(
                    spec=spec,
                    file_name=source_path.name,
                    source_path=source_path,
                    page_dir=page_dir,
                    record_path=record_path,
                    maximum_process_tree_rss_bytes=(
                        PROBE_MAX_PROCESS_TREE_RSS_BYTES
                    ),
                    minimum_memory_free_percent=(
                        PROBE_MIN_RUNTIME_MEMORY_FREE_PERCENT
                    ),
                    maximum_swap_used_bytes=PROBE_MAX_SWAP_USED_BYTES,
                    reject_booted_ios_simulators=True,
                )
            except (materializer.MaterializationError, OSError, ValueError) as error:
                failure = error
            try:
                result["post_run_health"] = health_reader(os.getpid())
                materializer.enforce_health_limits(
                    result["post_run_health"],
                    maximum_process_tree_rss_bytes=(
                        PROBE_MAX_PROCESS_TREE_RSS_BYTES
                    ),
                    minimum_memory_free_percent=(
                        PROBE_MIN_RUNTIME_MEMORY_FREE_PERCENT
                    ),
                    maximum_swap_used_bytes=PROBE_MAX_SWAP_USED_BYTES,
                )
            except (materializer.MaterializationError, OSError, ValueError) as error:
                if failure is None:
                    failure = error
            try:
                materializer.assert_no_conflicting_model_processes()
                result["residual_model_process_count"] = 0
            except (materializer.MaterializationError, OSError, ValueError) as error:
                if failure is None:
                    failure = error
            if failure is None:
                materializer.validate_page_record(
                    materializer.read_json(record_path),
                    file_name=source_path.name,
                    source_path=source_path,
                    npz_path=page_dir / f"{source_path.stem}.npz",
                )
                result["page_completed"] = True
    except (materializer.MaterializationError, OSError, ValueError) as error:
        failure = error
    finally:
        try:
            shutil.rmtree(temporary_root)
        except OSError as error:
            if failure is None:
                failure = materializer.MaterializationError(
                    f"temporary probe output cleanup failed: {error}"
                )
        result["temporary_page_outputs_retained"] = temporary_root.exists()

    if isinstance(failure, runtime.ResourceLimitError):
        result["failure_health"] = failure.evidence()
    if failure is None:
        result.update(
            {
                "reason_code": "runtime_safety_probe_passed",
                "terminal": "PASS",
            }
        )
    else:
        result.update(
            {
                "reason": str(failure),
                "reason_code": "runtime_resource_prerequisite_failed",
                "terminal": "PREREQUISITE_NEEDED",
            }
        )
    if write_result:
        materializer.atomic_write_json(destination, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--contract", type=Path, default=PROBE_CONTRACT_PATH)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_runtime_probe(
            repo_root=args.repo_root,
            plan_path=args.plan,
            ledger_path=args.ledger,
            contract_path=args.contract,
            result_path=args.result,
        )
    except (materializer.MaterializationError, OSError, ValueError) as error:
        print(f"terminal=PREREQUISITE_NEEDED reason={error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["terminal"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
