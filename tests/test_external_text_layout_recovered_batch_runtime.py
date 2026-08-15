from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from typing import Iterator
import unittest
from unittest import mock

import cv2
import numpy as np

from scripts.analysis import external_text_layout_recovered_batch_runtime as batch_runtime


materializer = batch_runtime.materializer


@contextmanager
def noop_lock(_path: Path) -> Iterator[None]:
    yield


def safe_health(_pid: int) -> dict[str, float | int]:
    return {
        "memory_free_percent": 80.0,
        "process_tree_rss_bytes": 1024,
        "swap_used_bytes": 0,
    }


def write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((2, 2, 3), value, dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write fixture image: {path}")


def make_fixture(root: Path, count: int) -> dict[str, object]:
    file_names = [f"page{index:02d}.png" for index in range(count)]
    plan_path = root / "docs" / "plan.json"
    ledger_path = root / "docs" / "ledger.json"
    manifest_path = root / "hardcase_lists" / "train.txt"
    plan_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    spec = {
        "batch_size": 1,
        "box_thresh": 0.45,
        "device": "cpu",
        "engine": "transformers",
        "limit_side_len": 736,
        "limit_type": "min",
        "max_side_limit": 4000,
        "model_dir": "/tmp/frozen-model",
        "model_name": "PP-OCRv6_medium_det",
        "thresh": 0.2,
        "unclip_ratio": 1.4,
    }
    plan_path.write_text(
        json.dumps({"external_text_layout_materialization": spec}),
        encoding="utf-8",
    )
    ledger_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text("\n".join(file_names) + "\n", encoding="utf-8")
    sources: list[tuple[str, str, Path]] = []
    for index, file_name in enumerate(file_names):
        source = root / "sources" / file_name
        write_rgb(source, index)
        sources.append((file_name, f"sources/{file_name}", source))
    output_root = root / "outputs" / "materialized"
    registered = {
        "file_names": file_names,
        "manifest_path": manifest_path,
        "model_paths": {},
        "output_root": output_root,
        "runtime": {},
        "sources": sources,
        "temporary_root": output_root.with_name(".materialized.materializing"),
    }
    return {
        "file_names": file_names,
        "ledger_path": ledger_path.relative_to(root),
        "plan_path": plan_path.relative_to(root),
        "registered": registered,
        "spec": spec,
    }


class EmptyDetector:
    def __init__(self, calls: list[str] | None = None) -> None:
        self.calls = calls
        self.closed = False

    def predict(self, *, input: str, **_kwargs: object) -> list[dict[str, object]]:
        if self.calls is not None:
            self.calls.append(Path(input).name)
        return [{"dt_polys": [], "dt_scores": []}]

    def close(self) -> None:
        self.closed = True


class RecoveredBatchRuntimeTest(unittest.TestCase):
    def test_exact_eight_pages_use_one_cpu_detector_and_commit_in_order(self) -> None:
        items = [
            (f"page{index}.png", f"/source/page{index}.png", f"/records/page{index}.json")
            for index in range(8)
        ]
        detector = EmptyDetector()
        factory = mock.Mock(return_value=detector)
        events: list[str] = []

        def materialize_one(**kwargs: object) -> dict[str, object]:
            file_name = str(kwargs["file_name"])
            self.assertIs(kwargs["detector"], detector)
            events.append(f"npz:{file_name}")
            return {"file": file_name}

        def write_record(path: Path, row: dict[str, object]) -> None:
            events.append(f"record:{path.name}:{row['file']}")

        with (
            mock.patch.object(
                materializer, "materialize_one", side_effect=materialize_one
            ),
            mock.patch.object(
                materializer, "atomic_write_json", side_effect=write_record
            ),
        ):
            completed = batch_runtime.materialize_batch_pages(
                spec={"device": "cpu"},
                items=items,
                page_dir=Path("/pages"),
                detector_factory=factory,
            )

        self.assertEqual(completed, [item[0] for item in items])
        factory.assert_called_once_with({"device": "cpu"})
        self.assertTrue(detector.closed)
        expected: list[str] = []
        for file_name, _source, record_path in items:
            expected.extend(
                [f"npz:{file_name}", f"record:{Path(record_path).name}:{file_name}"]
            )
        self.assertEqual(events, expected)

    def test_batch_is_cpu_only_fixed_size_and_unique(self) -> None:
        one = [("page.png", "/source/page.png", "/records/page.json")]
        factory = mock.Mock(return_value=EmptyDetector())
        with self.assertRaisesRegex(materializer.MaterializationError, "CPU-only"):
            batch_runtime.materialize_batch_pages(
                spec={"device": "mps"},
                items=one,
                page_dir=Path("/pages"),
                detector_factory=factory,
            )
        factory.assert_not_called()
        with self.assertRaisesRegex(materializer.MaterializationError, "1..8"):
            batch_runtime.materialize_batch_pages(
                spec={"device": "cpu"},
                items=one * 9,
                page_dir=Path("/pages"),
                detector_factory=factory,
            )
        with self.assertRaisesRegex(materializer.MaterializationError, "repeats"):
            batch_runtime.materialize_batch_pages(
                spec={"device": "cpu"},
                items=one * 2,
                page_dir=Path("/pages"),
                detector_factory=factory,
            )

    def test_child_checks_simulator_before_detector_creation(self) -> None:
        events: list[str] = []
        items = [("page.png", "/source/page.png", "/records/page.json")]
        with (
            mock.patch.object(
                batch_runtime.os, "setsid", side_effect=lambda: events.append("setsid")
            ),
            mock.patch.object(
                materializer.runtime,
                "assert_no_booted_ios_simulators",
                side_effect=lambda: events.append("simulator"),
            ),
            mock.patch.object(
                batch_runtime,
                "materialize_batch_pages",
                side_effect=lambda **_kwargs: events.append("detector"),
            ),
        ):
            batch_runtime.materialize_batch_child(
                {"device": "cpu"}, items, "/pages", True
            )
        self.assertEqual(events, ["setsid", "simulator", "detector"])

    def test_isolated_batch_starts_exactly_one_spawn_child(self) -> None:
        items = [
            (f"page{index}.png", f"/source/page{index}.png", f"/records/page{index}.json")
            for index in range(8)
        ]

        class Process:
            pid = 123
            exitcode = 0

            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.started = False

            def start(self) -> None:
                self.started = True

            def is_alive(self) -> bool:
                return False

            def join(self, timeout: float | None = None) -> None:
                del timeout

        process = Process()
        context = mock.Mock()
        context.Process.return_value = process
        with mock.patch.object(
            batch_runtime.multiprocessing, "get_context", return_value=context
        ) as get_context:
            batch_runtime.run_isolated_batch(
                spec={"device": "cpu"},
                items=items,
                page_dir=Path("/pages"),
                health_reader=safe_health,
            )
        get_context.assert_called_once_with("spawn")
        context.Process.assert_called_once()
        call = context.Process.call_args.kwargs
        self.assertIs(call["target"], batch_runtime.materialize_batch_child)
        self.assertEqual(call["args"][1], items)
        self.assertFalse(call["daemon"])
        self.assertTrue(process.started)

    def test_rss_free_swap_and_timeout_terminate_the_batch_process(self) -> None:
        class Process:
            pid = 123
            exitcode = 0

            def is_alive(self) -> bool:
                return True

            def join(self, timeout: float | None = None) -> None:
                del timeout

        cases = {
            "RSS": {
                "memory_free_percent": 80.0,
                "process_tree_rss_bytes": (
                    batch_runtime.MAX_RECOVERED_PROCESS_TREE_RSS_BYTES + 1
                ),
                "swap_used_bytes": 0,
            },
            "memory": {
                "memory_free_percent": materializer.MIN_MEMORY_FREE_PERCENT - 0.1,
                "process_tree_rss_bytes": 0,
                "swap_used_bytes": 0,
            },
            "swap": {
                "memory_free_percent": 80.0,
                "process_tree_rss_bytes": 0,
                "swap_used_bytes": materializer.MAX_SWAP_USED_BYTES + 1,
            },
        }
        for label, health in cases.items():
            with self.subTest(label=label):
                process = Process()
                with mock.patch.object(
                    materializer.runtime, "terminate_page_process"
                ) as terminate:
                    with self.assertRaises(materializer.runtime.ResourceLimitError):
                        batch_runtime.wait_for_batch_process(
                            process, health_reader=lambda _pid, value=health: value
                        )
                terminate.assert_called_once_with(process)

        ticks = iter([0.0, batch_runtime.BATCH_TIMEOUT_SECONDS + 1.0])
        process = Process()
        with mock.patch.object(
            materializer.runtime, "terminate_page_process"
        ) as terminate:
            with self.assertRaisesRegex(
                materializer.MaterializationError, "batch timeout"
            ):
                batch_runtime.wait_for_batch_process(
                    process,
                    health_reader=safe_health,
                    clock=lambda: next(ticks),
                )
        terminate.assert_called_once_with(process)

        for timing in (
            {"batch_timeout_seconds": batch_runtime.BATCH_TIMEOUT_SECONDS + 1.0},
            {
                "monitor_interval_seconds": (
                    batch_runtime.MONITOR_INTERVAL_SECONDS + 0.01
                )
            },
        ):
            with self.subTest(timing=timing):
                with self.assertRaisesRegex(
                    materializer.MaterializationError, "timing changed"
                ):
                    batch_runtime.wait_for_batch_process(
                        Process(), health_reader=safe_health, **timing
                    )

    def test_recovered_rss_cap_accepts_observed_page_without_weakening_shared_default(
        self,
    ) -> None:
        observed = {
            "memory_free_percent": 74.0,
            "process_tree_rss_bytes": 11_992_580_096,
            "swap_used_bytes": 0,
        }
        batch_runtime.enforce_recovered_health_limits(observed)
        self.assertEqual(materializer.MAX_DETECTOR_RSS_BYTES, 10 * 1024**3)
        self.assertEqual(
            batch_runtime.MAX_RECOVERED_PROCESS_TREE_RSS_BYTES, 13 * 1024**3
        )
        with self.assertRaisesRegex(
            materializer.runtime.ResourceLimitError, "RSS safety limit"
        ):
            batch_runtime.enforce_recovered_health_limits(
                {
                    **observed,
                    "process_tree_rss_bytes": (
                        batch_runtime.MAX_RECOVERED_PROCESS_TREE_RSS_BYTES + 1
                    ),
                }
            )
        with self.assertRaisesRegex(
            materializer.MaterializationError, "would weaken"
        ):
            batch_runtime.enforce_recovered_health_limits(
                observed,
                maximum_process_tree_rss_bytes=(
                    batch_runtime.MAX_RECOVERED_PROCESS_TREE_RSS_BYTES + 1
                ),
            )

    def test_resume_skips_eight_pages_recovers_orphan_and_publishes_shared_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_fixture(root, 18)
            registered = fixture["registered"]
            spec = fixture["spec"]
            page_dir, record_dir, _completed = materializer.prepare_resume_state(
                repo_root=root,
                plan_file=root / fixture["plan_path"],
                registered=registered,
            )
            seed_detector = EmptyDetector()
            for file_name, _relative, source_path in registered["sources"][:8]:
                row = materializer.materialize_one(
                    detector=seed_detector,
                    file_name=file_name,
                    source_path=source_path,
                    spec=spec,
                    page_dir=page_dir,
                )
                materializer.atomic_write_json(
                    record_dir / f"{Path(file_name).stem}.json", row
                )
            orphan_name = fixture["file_names"][8]
            materializer.atomic_write_npz(
                page_dir / f"{Path(orphan_name).stem}.npz",
                polygons=np.empty((0, 4, 2), dtype=np.int32),
                scores=np.empty((0,), dtype=np.float32),
                confidence=np.zeros((2, 2), dtype=np.float32),
                occupancy=np.zeros((2, 2), dtype=np.uint8),
            )

            calls: list[str] = []
            detectors: list[EmptyDetector] = []
            batch_lengths: list[int] = []

            def run_batch(**kwargs: object) -> dict[str, float | int]:
                items = kwargs["items"]
                batch_lengths.append(len(items))
                detector = EmptyDetector(calls)
                detectors.append(detector)
                batch_runtime.materialize_batch_pages(
                    spec=kwargs["spec"],
                    items=items,
                    page_dir=kwargs["page_dir"],
                    detector_factory=lambda _spec: detector,
                )
                return {
                    "minimum_memory_free_percent": 70.0,
                    "peak_process_tree_rss_bytes": 1024,
                    "peak_swap_used_bytes": 0,
                }

            conflicts = mock.Mock()
            with mock.patch.object(
                materializer, "validate_registered_inputs", return_value=registered
            ):
                result = batch_runtime.materialize(
                    repo_root=root,
                    plan_path=fixture["plan_path"],
                    ledger_path=fixture["ledger_path"],
                    health_reader=safe_health,
                    lock_factory=noop_lock,
                    conflict_checker=conflicts,
                    batch_runner=run_batch,
                )

            self.assertEqual(batch_lengths, [8, 2])
            self.assertEqual(calls, fixture["file_names"][8:])
            self.assertEqual(len(detectors), 2)
            self.assertTrue(all(detector.closed for detector in detectors))
            self.assertEqual(result["train_count"], 18)
            self.assertEqual(conflicts.call_count, 3)
            manifest_path = registered["output_root"] / "manifest.json"
            manifest = materializer.read_json(manifest_path)
            expected = materializer.build_manifest_payload(
                repo_root=root,
                plan_path=root / fixture["plan_path"],
                registered=registered,
                rows=manifest["pages"],
            )
            self.assertEqual(manifest, expected)
            self.assertFalse(registered["temporary_root"].exists())

    def test_partial_batch_failure_retains_completed_records(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_fixture(root, 3)
            registered = fixture["registered"]

            class FailingDetector(EmptyDetector):
                count = 0

                def predict(self, **kwargs: object) -> list[dict[str, object]]:
                    type(self).count += 1
                    if type(self).count == 3:
                        raise RuntimeError("synthetic third-page failure")
                    return super().predict(**kwargs)

            def run_batch(**kwargs: object) -> dict[str, float | int]:
                batch_runtime.materialize_batch_pages(
                    spec=kwargs["spec"],
                    items=kwargs["items"],
                    page_dir=kwargs["page_dir"],
                    detector_factory=lambda _spec: FailingDetector(),
                )
                raise AssertionError("failure should escape materialize_batch_pages")

            with mock.patch.object(
                materializer, "validate_registered_inputs", return_value=registered
            ):
                with self.assertRaisesRegex(
                    materializer.MaterializationError, "inference failed"
                ):
                    batch_runtime.materialize(
                        repo_root=root,
                        plan_path=fixture["plan_path"],
                        ledger_path=fixture["ledger_path"],
                        health_reader=safe_health,
                        lock_factory=noop_lock,
                        conflict_checker=lambda: None,
                        batch_runner=run_batch,
                    )

            _page_dir, _record_dir, completed = materializer.prepare_resume_state(
                repo_root=root,
                plan_file=root / fixture["plan_path"],
                registered=registered,
            )
            self.assertEqual(list(completed), fixture["file_names"][:2])
            self.assertFalse(registered["output_root"].exists())

    def test_materialize_rejects_worker_batch_and_device_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaisesRegex(materializer.MaterializationError, "fixed"):
                batch_runtime.materialize(repo_root=root, worker_count=2)
            with self.assertRaisesRegex(materializer.MaterializationError, "fixed"):
                batch_runtime.materialize(repo_root=root, batch_size=7)
            with self.assertRaisesRegex(
                materializer.MaterializationError, "cannot be disabled"
            ):
                batch_runtime.materialize(
                    repo_root=root, reject_booted_ios_simulators=False
                )

            fixture = make_fixture(root, 1)
            plan = json.loads((root / fixture["plan_path"]).read_text())
            plan["external_text_layout_materialization"]["device"] = "mps"
            (root / fixture["plan_path"]).write_text(json.dumps(plan))
            with mock.patch.object(
                materializer,
                "validate_registered_inputs",
                return_value=fixture["registered"],
            ):
                with self.assertRaisesRegex(materializer.MaterializationError, "CPU-only"):
                    batch_runtime.materialize(
                        repo_root=root,
                        plan_path=fixture["plan_path"],
                        ledger_path=fixture["ledger_path"],
                    )


if __name__ == "__main__":
    unittest.main()
