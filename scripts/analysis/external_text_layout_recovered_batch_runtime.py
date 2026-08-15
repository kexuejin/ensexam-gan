#!/usr/bin/env python3
"""Run recovered external-layout materialization in fixed serial CPU batches."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Callable


from scripts.analysis import materialize_external_text_layout_support_train_only as materializer


BATCH_SIZE = 8
BATCH_TIMEOUT_SECONDS = 15 * 60.0
MONITOR_INTERVAL_SECONDS = 0.25
MAX_RECOVERED_PROCESS_TREE_RSS_BYTES = 11 * 1024**3


def enforce_recovered_health_limits(
    health: dict[str, float | int],
    *,
    observed_health: dict[str, float | int] | None = None,
    maximum_process_tree_rss_bytes: int = MAX_RECOVERED_PROCESS_TREE_RSS_BYTES,
    minimum_memory_free_percent: float = materializer.MIN_MEMORY_FREE_PERCENT,
    maximum_swap_used_bytes: int = materializer.MAX_SWAP_USED_BYTES,
) -> None:
    if (
        maximum_process_tree_rss_bytes > MAX_RECOVERED_PROCESS_TREE_RSS_BYTES
        or minimum_memory_free_percent < materializer.MIN_MEMORY_FREE_PERCENT
        or maximum_swap_used_bytes > materializer.MAX_SWAP_USED_BYTES
    ):
        raise materializer.MaterializationError(
            "recovered health limit override would weaken defaults"
        )
    rss = int(health["process_tree_rss_bytes"])
    free = float(health["memory_free_percent"])
    swap = int(health["swap_used_bytes"])
    observed = observed_health or materializer.runtime.health_summary(health)
    if rss > maximum_process_tree_rss_bytes:
        raise materializer.runtime.ResourceLimitError(
            "detector RSS safety limit exceeded: "
            f"{rss} > {maximum_process_tree_rss_bytes}",
            trigger_health=health,
            observed_health=observed,
        )
    if free < minimum_memory_free_percent:
        raise materializer.runtime.ResourceLimitError(
            f"system memory safety limit crossed: {free:.1f}% free",
            trigger_health=health,
            observed_health=observed,
        )
    if swap > maximum_swap_used_bytes:
        raise materializer.runtime.ResourceLimitError(
            f"swap safety limit exceeded: {swap} > {maximum_swap_used_bytes}",
            trigger_health=health,
            observed_health=observed,
        )


def _validate_batch_items(items: list[tuple[str, str, str]]) -> None:
    if not 1 <= len(items) <= BATCH_SIZE:
        raise materializer.MaterializationError(
            f"external detector batch must contain 1..{BATCH_SIZE} pages"
        )
    file_names = [file_name for file_name, _source, _record in items]
    if len(set(file_names)) != len(file_names):
        raise materializer.MaterializationError("external detector batch repeats a page")
    for file_name, source_path, record_path in items:
        if (
            Path(source_path).name != file_name
            or Path(record_path).stem != Path(file_name).stem
        ):
            raise materializer.MaterializationError(
                f"external detector batch path changed: {file_name}"
            )


def materialize_batch_pages(
    *,
    spec: dict[str, Any],
    items: list[tuple[str, str, str]],
    page_dir: Path,
    detector_factory: Callable[[dict[str, Any]], Any] | None = None,
) -> list[str]:
    """Create one detector and atomically commit each serial page in a batch."""
    _validate_batch_items(items)
    if spec.get("device") != "cpu":
        raise materializer.MaterializationError("recovered detector must remain CPU-only")
    factory = detector_factory or materializer.create_detector
    detector = factory(spec)
    completed: list[str] = []
    try:
        for file_name, source_path_value, record_path_value in items:
            row = materializer.materialize_one(
                detector=detector,
                file_name=file_name,
                source_path=Path(source_path_value),
                spec=spec,
                page_dir=page_dir,
            )
            materializer.atomic_write_json(Path(record_path_value), row)
            completed.append(file_name)
    finally:
        close = getattr(detector, "close", None)
        if callable(close):
            close()
    return completed


def materialize_batch_child(
    spec: dict[str, Any],
    items: list[tuple[str, str, str]],
    page_dir_value: str,
    reject_booted_ios_simulators: bool,
) -> None:
    os.setsid()
    if reject_booted_ios_simulators:
        materializer.runtime.assert_no_booted_ios_simulators()
    materialize_batch_pages(
        spec=spec,
        items=items,
        page_dir=Path(page_dir_value),
    )


def wait_for_batch_process(
    process: Any,
    *,
    health_reader: Callable[[int], dict[str, float | int]] | None = None,
    maximum_process_tree_rss_bytes: int = MAX_RECOVERED_PROCESS_TREE_RSS_BYTES,
    minimum_memory_free_percent: float = materializer.MIN_MEMORY_FREE_PERCENT,
    maximum_swap_used_bytes: int = materializer.MAX_SWAP_USED_BYTES,
    batch_timeout_seconds: float = BATCH_TIMEOUT_SECONDS,
    monitor_interval_seconds: float = MONITOR_INTERVAL_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, float | int]:
    if process.pid is None:
        raise materializer.MaterializationError("detector batch process did not start")
    if (
        isinstance(batch_timeout_seconds, bool)
        or not isinstance(batch_timeout_seconds, (int, float))
        or batch_timeout_seconds <= 0
        or batch_timeout_seconds > BATCH_TIMEOUT_SECONDS
        or isinstance(monitor_interval_seconds, bool)
        or not isinstance(monitor_interval_seconds, (int, float))
        or monitor_interval_seconds <= 0
        or monitor_interval_seconds > MONITOR_INTERVAL_SECONDS
    ):
        raise materializer.MaterializationError("detector batch timing changed")
    read_health = health_reader or materializer.runtime.runtime_health
    started = clock()
    observed = materializer.runtime.health_summary(
        {
            "memory_free_percent": 100.0,
            "process_tree_rss_bytes": 0,
            "swap_used_bytes": 0,
        }
    )
    try:
        while process.is_alive():
            health = read_health(process.pid)
            observed = materializer.runtime.health_summary(health, observed)
            enforce_recovered_health_limits(
                health,
                observed_health=observed,
                maximum_process_tree_rss_bytes=maximum_process_tree_rss_bytes,
                minimum_memory_free_percent=minimum_memory_free_percent,
                maximum_swap_used_bytes=maximum_swap_used_bytes,
            )
            if clock() - started > batch_timeout_seconds:
                raise materializer.MaterializationError(
                    "detector batch timeout exceeded"
                )
            process.join(timeout=float(monitor_interval_seconds))
    except BaseException:
        materializer.runtime.terminate_page_process(process)
        raise
    process.join()
    if process.exitcode != 0:
        raise materializer.MaterializationError(
            f"isolated detector batch failed with exit code {process.exitcode}"
        )
    return observed


def run_isolated_batch(
    *,
    spec: dict[str, Any],
    items: list[tuple[str, str, str]],
    page_dir: Path,
    health_reader: Callable[[int], dict[str, float | int]] | None = None,
    maximum_process_tree_rss_bytes: int = MAX_RECOVERED_PROCESS_TREE_RSS_BYTES,
    minimum_memory_free_percent: float = materializer.MIN_MEMORY_FREE_PERCENT,
    maximum_swap_used_bytes: int = materializer.MAX_SWAP_USED_BYTES,
    batch_timeout_seconds: float = BATCH_TIMEOUT_SECONDS,
    monitor_interval_seconds: float = MONITOR_INTERVAL_SECONDS,
    reject_booted_ios_simulators: bool = True,
) -> dict[str, float | int]:
    _validate_batch_items(items)
    if reject_booted_ios_simulators is not True:
        raise materializer.MaterializationError(
            "detector child Simulator recheck cannot be disabled"
        )
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=materialize_batch_child,
        args=(spec, items, str(page_dir), reject_booted_ios_simulators),
        daemon=False,
    )
    process.start()
    return wait_for_batch_process(
        process,
        health_reader=health_reader,
        maximum_process_tree_rss_bytes=maximum_process_tree_rss_bytes,
        minimum_memory_free_percent=minimum_memory_free_percent,
        maximum_swap_used_bytes=maximum_swap_used_bytes,
        batch_timeout_seconds=batch_timeout_seconds,
        monitor_interval_seconds=monitor_interval_seconds,
    )


def materialize(
    *,
    repo_root: Path,
    plan_path: Path = materializer.PLAN_PATH,
    ledger_path: Path = materializer.LEDGER_PATH,
    worker_count: int = 1,
    batch_size: int = BATCH_SIZE,
    health_reader: Callable[[int], dict[str, float | int]] | None = None,
    maximum_process_tree_rss_bytes: int = MAX_RECOVERED_PROCESS_TREE_RSS_BYTES,
    minimum_memory_free_percent: float = materializer.MIN_MEMORY_FREE_PERCENT,
    maximum_swap_used_bytes: int = materializer.MAX_SWAP_USED_BYTES,
    batch_timeout_seconds: float = BATCH_TIMEOUT_SECONDS,
    monitor_interval_seconds: float = MONITOR_INTERVAL_SECONDS,
    reject_booted_ios_simulators: bool = True,
    lock_factory: Callable[[Path], Any] | None = None,
    conflict_checker: Callable[[], None] | None = None,
    batch_runner: Callable[..., dict[str, float | int]] | None = None,
) -> dict[str, Any]:
    if worker_count != 1 or batch_size != BATCH_SIZE:
        raise materializer.MaterializationError(
            "recovered detector execution is fixed at one eight-page batch child"
        )
    if reject_booted_ios_simulators is not True:
        raise materializer.MaterializationError(
            "detector child Simulator recheck cannot be disabled"
        )
    read_health = health_reader or materializer.runtime.runtime_health
    acquire_lock = lock_factory or materializer.runtime.exclusive_run_lock
    check_conflicts = (
        conflict_checker or materializer.assert_no_conflicting_model_processes
    )
    run_batch = batch_runner or run_isolated_batch

    plan_file = materializer.repo_path(repo_root, str(plan_path))
    ledger_file = materializer.repo_path(repo_root, str(ledger_path))
    plan = materializer.read_json(plan_file)
    ledger = materializer.read_json(ledger_file)
    registered = materializer.validate_registered_inputs(
        repo_root, plan, ledger, require_output_absent=False
    )
    output_root: Path = registered["output_root"]
    temporary_root: Path = registered["temporary_root"]
    marker_path, cleanup_root = materializer.published_transaction_paths(output_root)
    spec = plan["external_text_layout_materialization"]
    if spec.get("device") != "cpu":
        raise materializer.MaterializationError("recovered detector must remain CPU-only")

    with acquire_lock(materializer.runtime.HOST_USER_RUN_LOCK_PATH):
        if output_root.exists():
            if not temporary_root.exists() and not marker_path.exists():
                raise materializer.MaterializationError(
                    "external text-layout output already exists"
                )
            rows = materializer.validate_published_materialization(
                repo_root=repo_root,
                plan_file=plan_file,
                registered=registered,
            )
            if temporary_root.exists():
                _page_dir, _record_dir, completed = materializer.prepare_resume_state(
                    repo_root=repo_root,
                    plan_file=plan_file,
                    registered=registered,
                )
                resumed_rows = [
                    completed.get(file_name) for file_name in registered["file_names"]
                ]
                if resumed_rows != rows:
                    raise materializer.MaterializationError(
                        "published materialization disagrees with resumable page records"
                    )
            materializer.finalize_published_cleanup(
                repo_root=repo_root, registered=registered
            )
            peak_rss = 0
            minimum_free = 100.0
            peak_swap = 0
        else:
            if marker_path.exists() or cleanup_root.exists():
                raise materializer.MaterializationError(
                    "published transaction exists without final output"
                )
            rows = []

        if not rows:
            check_conflicts()
            enforce_recovered_health_limits(
                read_health(os.getpid()),
                maximum_process_tree_rss_bytes=maximum_process_tree_rss_bytes,
                minimum_memory_free_percent=minimum_memory_free_percent,
                maximum_swap_used_bytes=maximum_swap_used_bytes,
            )
            page_dir, record_dir, completed = materializer.prepare_resume_state(
                repo_root=repo_root,
                plan_file=plan_file,
                registered=registered,
            )
            sources = registered["sources"]
            remaining = [
                (index, file_name, source_path)
                for index, (file_name, _relative, source_path) in enumerate(
                    sources, start=1
                )
                if file_name not in completed
            ]
            peak_rss = 0
            minimum_free = 100.0
            peak_swap = 0
            for offset in range(0, len(remaining), BATCH_SIZE):
                batch = remaining[offset : offset + BATCH_SIZE]
                check_conflicts()
                enforce_recovered_health_limits(
                    read_health(os.getpid()),
                    maximum_process_tree_rss_bytes=maximum_process_tree_rss_bytes,
                    minimum_memory_free_percent=minimum_memory_free_percent,
                    maximum_swap_used_bytes=maximum_swap_used_bytes,
                )
                items = [
                    (
                        file_name,
                        str(source_path),
                        str(record_dir / f"{Path(file_name).stem}.json"),
                    )
                    for _index, file_name, source_path in batch
                ]
                health = run_batch(
                    spec=spec,
                    items=items,
                    page_dir=page_dir,
                    health_reader=read_health,
                    maximum_process_tree_rss_bytes=maximum_process_tree_rss_bytes,
                    minimum_memory_free_percent=minimum_memory_free_percent,
                    maximum_swap_used_bytes=maximum_swap_used_bytes,
                    batch_timeout_seconds=batch_timeout_seconds,
                    monitor_interval_seconds=monitor_interval_seconds,
                    reject_booted_ios_simulators=reject_booted_ios_simulators,
                )
                peak_rss = max(
                    peak_rss, int(health["peak_process_tree_rss_bytes"])
                )
                minimum_free = min(
                    minimum_free, float(health["minimum_memory_free_percent"])
                )
                peak_swap = max(peak_swap, int(health["peak_swap_used_bytes"]))
                for index, file_name, source_path in batch:
                    record_path = record_dir / f"{Path(file_name).stem}.json"
                    row = materializer.validate_page_record(
                        materializer.read_json(record_path),
                        file_name=file_name,
                        source_path=source_path,
                        npz_path=page_dir / f"{Path(file_name).stem}.npz",
                    )
                    completed[file_name] = row
                first_index, first_name, _source = batch[0]
                last_index, last_name, _source = batch[-1]
                print(
                    f"batch={offset // BATCH_SIZE + 1} "
                    f"pages={first_index}-{last_index}/{len(sources)} "
                    f"files={first_name}..{last_name} "
                    f"peak_rss_bytes={int(health['peak_process_tree_rss_bytes'])} "
                    f"minimum_memory_free_percent="
                    f"{float(health['minimum_memory_free_percent']):.1f}",
                    flush=True,
                )
            rows = [completed[name] for name, _relative, _source in sources]
            if len(rows) != len(sources):
                raise materializer.MaterializationError(
                    "materialization resume population is incomplete"
                )
            materializer.publish_completed_materialization(
                repo_root=repo_root,
                plan_file=plan_file,
                registered=registered,
                page_dir=page_dir,
                rows=rows,
            )

    manifest_path = output_root / "manifest.json"
    return {
        "content_sha256": materializer.sha256_rows(
            [f"{row['file']} {row['npz_sha256']}" for row in rows]
        ),
        "manifest": str(manifest_path),
        "manifest_sha256": materializer.sha256_file(manifest_path),
        "minimum_memory_free_percent": minimum_free,
        "output_root": str(output_root),
        "peak_process_tree_rss_bytes": peak_rss,
        "peak_swap_used_bytes": peak_swap,
        "terminal": "PASS",
        "train_count": len(rows),
    }
