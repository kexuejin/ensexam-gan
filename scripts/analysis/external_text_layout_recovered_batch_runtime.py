#!/usr/bin/env python3
"""Run recovered external-layout materialization in fixed serial CPU batches."""

from __future__ import annotations

import ctypes
import gc
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable


from scripts.analysis import materialize_external_text_layout_support_train_only as materializer


BATCH_SIZE = 8
BATCH_TIMEOUT_SECONDS = 15 * 60.0
MONITOR_INTERVAL_SECONDS = 0.25
MAX_RECOVERED_PROCESS_TREE_RSS_BYTES = 13 * 1024**3
PAGE_TILE_MAX_PIXELS = 4_250_000
PAGE_TILE_OVERLAP_PIXELS = 128
_MALLOC_ZONE_LIBRARY: Any | None = None
_MALLOC_ZONE_PRESSURE_RELIEF: Any | None = None


def release_page_memory() -> int:
    global _MALLOC_ZONE_LIBRARY, _MALLOC_ZONE_PRESSURE_RELIEF
    gc.collect()
    if _MALLOC_ZONE_PRESSURE_RELIEF is None:
        try:
            library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            relief = library.malloc_zone_pressure_relief
            relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            relief.restype = ctypes.c_size_t
        except (AttributeError, OSError, TypeError) as error:
            raise materializer.MaterializationError(
                "page memory relief is unavailable"
            ) from error
        _MALLOC_ZONE_LIBRARY = library
        _MALLOC_ZONE_PRESSURE_RELIEF = relief
    try:
        return int(_MALLOC_ZONE_PRESSURE_RELIEF(None, 0))
    except Exception as error:
        raise materializer.MaterializationError("page memory relief failed") from error


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


class TiledPageDetector:
    """Limit per-inference image size while preserving page-coordinate outputs."""

    def __init__(
        self,
        spec: dict[str, Any],
        *,
        detector_factory: Callable[[dict[str, Any]], Any] = materializer.create_detector,
        max_tile_pixels: int = PAGE_TILE_MAX_PIXELS,
        overlap_pixels: int = PAGE_TILE_OVERLAP_PIXELS,
    ) -> None:
        if (
            isinstance(max_tile_pixels, bool)
            or max_tile_pixels <= 0
            or isinstance(overlap_pixels, bool)
            or overlap_pixels < 0
        ):
            raise materializer.MaterializationError("tiled detector limits changed")
        self._detector = detector_factory(spec)
        self._max_tile_pixels = int(max_tile_pixels)
        self._overlap_pixels = int(overlap_pixels)

    def close(self) -> None:
        close = getattr(self._detector, "close", None)
        if callable(close):
            close()

    def _core_spans(self, *, height: int, width: int) -> list[tuple[int, int]]:
        if height <= 0 or width <= 0:
            raise materializer.MaterializationError("invalid tiled detector input")
        if height * width <= self._max_tile_pixels:
            return [(0, height)]
        max_tile_height = self._max_tile_pixels // width
        core_height = max_tile_height - (2 * self._overlap_pixels)
        if core_height <= 0:
            raise materializer.MaterializationError(
                "tiled detector input exceeds memory-safe geometry"
            )
        spans: list[tuple[int, int]] = []
        start = 0
        while start < height:
            end = min(height, start + core_height)
            spans.append((start, end))
            start = end
        return spans

    def predict(self, *, input: str, **kwargs: Any) -> list[dict[str, Any]]:
        import cv2
        import numpy as np

        source_path = Path(input)
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise materializer.MaterializationError(
                f"source image decode failed: {source_path}"
            )
        height, width = image.shape[:2]
        spans = self._core_spans(height=height, width=width)
        if len(spans) == 1 and spans[0] == (0, height):
            results = self._detector.predict(input=input, **kwargs)
            return results if isinstance(results, list) else list(results)

        polygons: list[np.ndarray] = []
        scores: list[float] = []
        with tempfile.TemporaryDirectory(prefix="ensexam-layout-tile-") as raw:
            tile_root = Path(raw)
            for index, (core_start, core_end) in enumerate(spans):
                tile_start = max(0, core_start - self._overlap_pixels)
                tile_end = min(height, core_end + self._overlap_pixels)
                tile = image[tile_start:tile_end, :]
                tile_path = tile_root / f"tile-{index:04d}.png"
                if not cv2.imwrite(str(tile_path), tile):
                    raise materializer.MaterializationError(
                        "tiled detector input write failed"
                    )
                tile_results = self._detector.predict(
                    input=str(tile_path), **kwargs
                )
                tile_list = (
                    tile_results
                    if isinstance(tile_results, list)
                    else list(tile_results)
                )
                if len(tile_list) != 1:
                    raise materializer.MaterializationError(
                        "tiled detector did not return exactly one tile result"
                    )
                tile_polygons, tile_scores = materializer.extract_result(tile_list[0])
                polygon_array = np.asarray(tile_polygons)
                score_array = np.asarray(tile_scores, dtype=np.float32)
                if polygon_array.size == 0:
                    continue
                if polygon_array.ndim != 3 or polygon_array.shape[1:] != (4, 2):
                    raise materializer.MaterializationError(
                        "tiled detector polygons must have shape [N,4,2]"
                    )
                if score_array.ndim != 1 or len(score_array) != len(polygon_array):
                    raise materializer.MaterializationError(
                        "tiled detector score count changed"
                    )
                shifted = polygon_array.astype(np.float64, copy=True)
                shifted[:, :, 1] += float(tile_start)
                centers_y = shifted[:, :, 1].mean(axis=1)
                keep = (centers_y >= core_start) & (centers_y < core_end)
                for polygon, score in zip(shifted[keep], score_array[keep], strict=True):
                    polygons.append(polygon)
                    scores.append(float(score))

        if not polygons:
            return [{"dt_polys": np.empty((0, 4, 2)), "dt_scores": np.empty((0,))}]
        return [
            {
                "dt_polys": np.stack(polygons, axis=0),
                "dt_scores": np.asarray(scores, dtype=np.float32),
            }
        ]


def create_recovered_detector(spec: dict[str, Any]) -> TiledPageDetector:
    return TiledPageDetector(spec)


def materialize_batch_pages(
    *,
    spec: dict[str, Any],
    items: list[tuple[str, str, str]],
    page_dir: Path,
    detector_factory: Callable[[dict[str, Any]], Any] | None = None,
    memory_releaser: Callable[[], int] | None = None,
) -> list[str]:
    """Atomically commit each page with a fresh detector lifetime."""
    _validate_batch_items(items)
    if spec.get("device") != "cpu":
        raise materializer.MaterializationError("recovered detector must remain CPU-only")
    factory = detector_factory or create_recovered_detector
    release_memory = memory_releaser or release_page_memory
    completed: list[str] = []
    for file_name, source_path_value, record_path_value in items:
        detector = factory(spec)
        try:
            row = materializer.materialize_one(
                detector=detector,
                file_name=file_name,
                source_path=Path(source_path_value),
                spec=spec,
                page_dir=page_dir,
            )
            materializer.atomic_write_json(Path(record_path_value), row)
        finally:
            close = getattr(detector, "close", None)
            if callable(close):
                close()
        try:
            release_memory()
        except materializer.MaterializationError:
            raise
        except Exception as error:
            raise materializer.MaterializationError(
                "page memory relief failed"
            ) from error
        completed.append(file_name)
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
    observed: dict[str, float | int] = {
        "minimum_memory_free_percent": 100.0,
        "peak_process_tree_rss_bytes": 0,
        "peak_swap_used_bytes": 0,
    }
    for item in items:
        process = context.Process(
            target=materialize_batch_child,
            args=(spec, [item], str(page_dir), reject_booted_ios_simulators),
            daemon=False,
        )
        process.start()
        page_health = wait_for_batch_process(
            process,
            health_reader=health_reader,
            maximum_process_tree_rss_bytes=maximum_process_tree_rss_bytes,
            minimum_memory_free_percent=minimum_memory_free_percent,
            maximum_swap_used_bytes=maximum_swap_used_bytes,
            batch_timeout_seconds=batch_timeout_seconds,
            monitor_interval_seconds=monitor_interval_seconds,
        )
        observed = {
            "minimum_memory_free_percent": min(
                float(observed["minimum_memory_free_percent"]),
                float(page_health["minimum_memory_free_percent"]),
            ),
            "peak_process_tree_rss_bytes": max(
                int(observed["peak_process_tree_rss_bytes"]),
                int(page_health["peak_process_tree_rss_bytes"]),
            ),
            "peak_swap_used_bytes": max(
                int(observed["peak_swap_used_bytes"]),
                int(page_health["peak_swap_used_bytes"]),
            ),
        }
    return observed


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
