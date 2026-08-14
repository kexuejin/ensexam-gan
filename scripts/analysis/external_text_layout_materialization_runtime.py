#!/usr/bin/env python3
"""Crash-resilient runtime support for external text-layout materialization."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable, Iterator

import numpy as np


NPZ_KEYS = {
    "polygons",
    "scores",
    "text_confidence",
    "text_occupancy",
}
PAGE_ROW_KEYS = {
    "confidence_max",
    "confidence_mean",
    "detection_count",
    "file",
    "height",
    "npz_sha256",
    "occupancy_pixels",
    "source_sha256",
    "width",
}
MAX_DETECTOR_RSS_BYTES = 10 * 1024**3
MIN_MEMORY_FREE_PERCENT = 35.0
MAX_SWAP_USED_BYTES = 512 * 1024**2
PAGE_TIMEOUT_SECONDS = 15 * 60.0
MONITOR_INTERVAL_SECONDS = 1.0
SYSTEM_COMMAND_TIMEOUT_SECONDS = 5.0
HOST_USER_RUN_LOCK_PATH = (
    Path(tempfile.gettempdir())
    / f"ensexam-gan-{os.getuid()}-external-layout-materializer.lock"
)
CONFLICTING_MODEL_COMMANDS = (
    "scripts/infer/run_primary_full_page.py",
    "scripts/infer/run_second_stage_residual_repair.py",
    "scripts/run_second_stage_residual_repair.py",
    "scripts/infer/run_hybrid_second_stage_gate.py",
    "scripts/run_hybrid_second_stage_gate.py",
    "scripts/micro_train_region_probe.py",
    "scripts/analysis/train_page_selector_ranker.py",
    "scripts/analysis/train_region_component_ranker.py",
    "scripts/train/",
    "/meta_train.py",
    " meta_train.py",
    "/train.py",
    " train.py",
)


class MaterializationError(RuntimeError):
    pass


class ResourceLimitError(MaterializationError):
    def __init__(
        self,
        message: str,
        *,
        trigger_health: dict[str, float | int],
        observed_health: dict[str, float | int],
    ) -> None:
        super().__init__(message)
        self.trigger_health = dict(trigger_health)
        self.observed_health = dict(observed_health)

    def evidence(self) -> dict[str, dict[str, float | int]]:
        return {
            "observed_health": dict(self.observed_health),
            "trigger_health": dict(self.trigger_health),
        }


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
        raise MaterializationError(f"expected JSON object: {path}")
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_npz(
    path: Path,
    *,
    polygons: np.ndarray,
    scores: np.ndarray,
    confidence: np.ndarray,
    occupancy: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise MaterializationError(f"refusing to overwrite page output: {path.name}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(
                handle,
                polygons=polygons,
                scores=scores,
                text_confidence=confidence,
                text_occupancy=occupancy,
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_run_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MaterializationError(
                "another external detector materializer is already active"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def command_lines() -> list[tuple[int, str]]:
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            check=True,
            capture_output=True,
            text=True,
            timeout=SYSTEM_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise MaterializationError("could not inspect active model processes") from error
    rows: list[tuple[int, str]] = []
    for raw in result.stdout.splitlines():
        value = raw.strip()
        if not value:
            continue
        pid_value, separator, command = value.partition(" ")
        if not separator:
            continue
        try:
            rows.append((int(pid_value), command.strip()))
        except ValueError:
            continue
    return rows


def assert_no_conflicting_model_processes(
    rows: list[tuple[int, str]] | None = None,
) -> None:
    active = command_lines() if rows is None else rows
    conflicts = [
        (pid, command)
        for pid, command in active
        if pid != os.getpid()
        and any(marker in command for marker in CONFLICTING_MODEL_COMMANDS)
    ]
    if conflicts:
        summary = ", ".join(f"pid={pid}" for pid, _command in conflicts[:5])
        raise MaterializationError(
            f"conflicting model process is active; OCR must run alone: {summary}"
        )


def parse_memory_free_percent(output: str) -> float:
    match = re.search(r"System-wide memory free percentage:\s*([0-9.]+)%", output)
    if match is None:
        raise MaterializationError("could not parse memory_pressure output")
    return float(match.group(1))


def parse_swap_used_bytes(output: str) -> int:
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", output)
    if match is None:
        raise MaterializationError("could not parse vm.swapusage output")
    multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return int(float(match.group(1)) * multiplier)


def process_tree_rss_bytes(root_pid: int, ps_output: str | None = None) -> int:
    if ps_output is None:
        try:
            result = subprocess.run(
                ["ps", "-axo", "pid=,ppid=,rss="],
                check=True,
                capture_output=True,
                text=True,
                timeout=SYSTEM_COMMAND_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise MaterializationError("could not inspect detector RSS") from error
        ps_output = result.stdout
    parents: dict[int, int] = {}
    rss_kib: dict[int, int] = {}
    for raw in ps_output.splitlines():
        values = raw.split()
        if len(values) != 3:
            continue
        try:
            pid, parent, rss = (int(value) for value in values)
        except ValueError:
            continue
        parents[pid] = parent
        rss_kib[pid] = rss
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(rss_kib.get(pid, 0) for pid in descendants) * 1024


def runtime_health(root_pid: int) -> dict[str, float | int]:
    try:
        memory = subprocess.run(
            ["memory_pressure"],
            check=True,
            capture_output=True,
            text=True,
            timeout=SYSTEM_COMMAND_TIMEOUT_SECONDS,
        )
        swap = subprocess.run(
            ["sysctl", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
            timeout=SYSTEM_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise MaterializationError("could not inspect system memory safety") from error
    return {
        "memory_free_percent": parse_memory_free_percent(memory.stdout),
        "process_tree_rss_bytes": process_tree_rss_bytes(root_pid),
        "swap_used_bytes": parse_swap_used_bytes(swap.stdout),
    }


def health_summary(
    health: dict[str, float | int],
    previous: dict[str, float | int] | None = None,
) -> dict[str, float | int]:
    prior = previous or {
        "minimum_memory_free_percent": 100.0,
        "peak_process_tree_rss_bytes": 0,
        "peak_swap_used_bytes": 0,
    }
    return {
        "minimum_memory_free_percent": min(
            float(prior["minimum_memory_free_percent"]),
            float(health["memory_free_percent"]),
        ),
        "peak_process_tree_rss_bytes": max(
            int(prior["peak_process_tree_rss_bytes"]),
            int(health["process_tree_rss_bytes"]),
        ),
        "peak_swap_used_bytes": max(
            int(prior["peak_swap_used_bytes"]),
            int(health["swap_used_bytes"]),
        ),
    }


def enforce_health_limits(
    health: dict[str, float | int],
    *,
    observed_health: dict[str, float | int] | None = None,
) -> None:
    rss = int(health["process_tree_rss_bytes"])
    free = float(health["memory_free_percent"])
    swap = int(health["swap_used_bytes"])
    observed = observed_health or health_summary(health)
    if rss > MAX_DETECTOR_RSS_BYTES:
        raise ResourceLimitError(
            f"detector RSS safety limit exceeded: {rss} > {MAX_DETECTOR_RSS_BYTES}",
            trigger_health=health,
            observed_health=observed,
        )
    if free < MIN_MEMORY_FREE_PERCENT:
        raise ResourceLimitError(
            f"system memory safety limit crossed: {free:.1f}% free",
            trigger_health=health,
            observed_health=observed,
        )
    if swap > MAX_SWAP_USED_BYTES:
        raise ResourceLimitError(
            f"swap safety limit exceeded: {swap} > {MAX_SWAP_USED_BYTES}",
            trigger_health=health,
            observed_health=observed,
        )


def load_page_npz(
    path: Path,
    *,
    height: int,
    width: int,
    rasterize: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != NPZ_KEYS:
                raise MaterializationError(f"page NPZ schema changed: {path.name}")
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise MaterializationError(f"page NPZ is unreadable: {path.name}") from error
    polygons = arrays["polygons"]
    scores = arrays["scores"]
    occupancy = arrays["text_occupancy"]
    confidence = arrays["text_confidence"]
    if polygons.dtype != np.int32 or polygons.ndim != 3 or polygons.shape[1:] != (4, 2):
        raise MaterializationError(f"page polygon contract changed: {path.name}")
    if scores.dtype != np.float32 or scores.shape != (len(polygons),):
        raise MaterializationError(f"page score contract changed: {path.name}")
    if occupancy.dtype != np.uint8 or occupancy.shape != (height, width):
        raise MaterializationError(f"page occupancy contract changed: {path.name}")
    if confidence.dtype != np.float32 or confidence.shape != (height, width):
        raise MaterializationError(f"page confidence contract changed: {path.name}")
    if not np.isfinite(scores).all() or not np.isfinite(confidence).all():
        raise MaterializationError(f"page NPZ contains non-finite values: {path.name}")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise MaterializationError(f"page scores escaped unit interval: {path.name}")
    if not np.isin(occupancy, [0, 1]).all():
        raise MaterializationError(f"page occupancy is not binary: {path.name}")
    if len(polygons) and (
        np.any(polygons[:, :, 0] < 0)
        or np.any(polygons[:, :, 0] >= width)
        or np.any(polygons[:, :, 1] < 0)
        or np.any(polygons[:, :, 1] >= height)
    ):
        raise MaterializationError(f"page polygons escaped source bounds: {path.name}")
    expected_occupancy, expected_confidence = rasterize(
        polygons, scores, height=height, width=width
    )
    if not np.array_equal(occupancy, expected_occupancy):
        raise MaterializationError(f"page occupancy raster changed: {path.name}")
    if not np.array_equal(confidence, expected_confidence):
        raise MaterializationError(f"page confidence raster changed: {path.name}")
    return arrays


def validate_page_record(
    row: dict[str, Any],
    *,
    file_name: str,
    source_path: Path,
    npz_path: Path,
    rasterize: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    if set(row) != PAGE_ROW_KEYS or row.get("file") != file_name:
        raise MaterializationError(f"completed page record changed: {file_name}")
    if row.get("source_sha256") != sha256_file(source_path):
        raise MaterializationError(f"completed page source changed: {file_name}")
    height = row.get("height")
    width = row.get("width")
    if (
        not isinstance(height, int)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or isinstance(width, bool)
        or min(height, width) <= 0
    ):
        raise MaterializationError(f"completed page dimensions changed: {file_name}")
    if not npz_path.is_file() or row.get("npz_sha256") != sha256_file(npz_path):
        raise MaterializationError(f"completed page NPZ changed: {file_name}")
    arrays = load_page_npz(
        npz_path, height=height, width=width, rasterize=rasterize
    )
    occupancy = arrays["text_occupancy"]
    confidence = arrays["text_confidence"]
    if row.get("detection_count") != len(arrays["polygons"]):
        raise MaterializationError(f"completed detection count changed: {file_name}")
    if row.get("occupancy_pixels") != int(occupancy.sum()):
        raise MaterializationError(f"completed occupancy count changed: {file_name}")
    if row.get("confidence_max") != float(confidence.max()):
        raise MaterializationError(f"completed confidence max changed: {file_name}")
    if abs(float(row.get("confidence_mean", -1.0)) - float(confidence.mean())) > 1e-7:
        raise MaterializationError(f"completed confidence mean changed: {file_name}")
    return row


def prepare_resume_state(
    *,
    repo_root: Path,
    plan_file: Path,
    registered: dict[str, Any],
    family: str,
    rasterize: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> tuple[Path, Path, dict[str, dict[str, Any]]]:
    temporary_root: Path = registered["temporary_root"]
    page_dir = temporary_root / "pages"
    record_dir = temporary_root / "records"
    progress_path = temporary_root / "progress.json"
    initializing_root = temporary_root.with_name(
        f"{temporary_root.name}.initializing"
    )
    expected_progress = {
        "expected_filename_sha256": sha256_rows(registered["file_names"]),
        "expected_train_count": len(registered["file_names"]),
        "family": family,
        "plan_sha256": sha256_file(plan_file),
        "schema_version": 1,
        "source_manifest_sha256": sha256_file(registered["manifest_path"]),
        "temporary_output": str(temporary_root.relative_to(repo_root)),
    }
    if not temporary_root.exists():
        if initializing_root.exists():
            if not initializing_root.is_dir():
                raise MaterializationError(
                    "materialization initialization root is not a directory"
                )
            shutil.rmtree(initializing_root)
            fsync_directory(initializing_root.parent)
        initializing_root.mkdir(parents=True)
        initializing_page_dir = initializing_root / "pages"
        initializing_record_dir = initializing_root / "records"
        initializing_page_dir.mkdir()
        initializing_record_dir.mkdir()
        atomic_write_json(initializing_root / "progress.json", expected_progress)
        fsync_directory(initializing_page_dir)
        fsync_directory(initializing_record_dir)
        fsync_directory(initializing_root)
        initializing_root.replace(temporary_root)
        fsync_directory(temporary_root.parent)
    else:
        if initializing_root.exists():
            raise MaterializationError(
                "materialization initialization states overlap"
            )
        if not temporary_root.is_dir():
            raise MaterializationError("materialization resume root is not a directory")
        shutil.rmtree(temporary_root / "complete", ignore_errors=True)
        for path in temporary_root.rglob(".*.tmp"):
            if path.is_file():
                path.unlink()
        if {path.name for path in temporary_root.iterdir()} != {
            "pages",
            "progress.json",
            "records",
        }:
            raise MaterializationError("materialization resume surface changed")
        if not page_dir.is_dir() or not record_dir.is_dir():
            raise MaterializationError("materialization resume directories changed")
        if read_json(progress_path) != expected_progress:
            raise MaterializationError("materialization resume provenance changed")
    sources = {
        file_name: source_path
        for file_name, _relative, source_path in registered["sources"]
    }
    stems = {Path(file_name).stem: file_name for file_name in sources}
    if len(stems) != len(sources):
        raise MaterializationError("train source names collide after removing suffixes")
    page_entries = list(page_dir.iterdir())
    record_entries = list(record_dir.iterdir())
    page_files = {
        path.stem: path
        for path in page_entries
        if path.is_file() and path.suffix == ".npz"
    }
    record_files = {
        path.name.removesuffix(".json"): path
        for path in record_entries
        if path.is_file() and path.suffix == ".json"
    }
    if len(page_files) != len(page_entries) or len(record_files) != len(record_entries):
        raise MaterializationError("materialization resume page surface changed")
    if (set(page_files) | set(record_files)) - set(stems):
        raise MaterializationError("materialization resume contains an unknown page")
    completed: dict[str, dict[str, Any]] = {}
    for stem in sorted(set(page_files) | set(record_files)):
        page_path = page_files.get(stem)
        record_path = record_files.get(stem)
        if record_path is None:
            if page_path is not None:
                page_path.unlink()
                fsync_directory(page_dir)
            continue
        if page_path is None:
            raise MaterializationError(f"completed page payload is missing: {stems[stem]}")
        file_name = stems[stem]
        completed[file_name] = validate_page_record(
            read_json(record_path),
            file_name=file_name,
            source_path=sources[file_name],
            npz_path=page_path,
            rasterize=rasterize,
        )
    return page_dir, record_dir, completed


def terminate_page_process(process: Any) -> None:
    if process.pid is None or not process.is_alive():
        return
    try:
        if os.getpgid(process.pid) == process.pid:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    process.join(timeout=5.0)
    if process.is_alive():
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.join(timeout=5.0)


def wait_for_page_process(
    process: Any,
    *,
    health_reader: Callable[[int], dict[str, float | int]] = runtime_health,
) -> dict[str, float | int]:
    if process.pid is None:
        raise MaterializationError("detector process did not start")
    started = time.monotonic()
    observed = health_summary(
        {
            "memory_free_percent": 100.0,
            "process_tree_rss_bytes": 0,
            "swap_used_bytes": 0,
        }
    )
    try:
        while process.is_alive():
            health = health_reader(process.pid)
            observed = health_summary(health, observed)
            enforce_health_limits(health, observed_health=observed)
            if time.monotonic() - started > PAGE_TIMEOUT_SECONDS:
                raise MaterializationError("detector page timeout exceeded")
            process.join(timeout=MONITOR_INTERVAL_SECONDS)
    except BaseException:
        terminate_page_process(process)
        raise
    process.join()
    if process.exitcode != 0:
        raise MaterializationError(
            f"isolated detector process failed with exit code {process.exitcode}"
        )
    return observed
