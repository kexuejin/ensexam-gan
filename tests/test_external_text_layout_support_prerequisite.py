from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import scripts.analysis.materialize_external_text_layout_support_train_only as materializer

from scripts.analysis.audit_external_text_layout_support import (
    ABLATION_CHANNELS,
    CHANNELS,
    AuditError,
    build_page,
    load_layout_npz,
    validate_materialization,
)
from scripts.analysis.materialize_external_text_layout_support_train_only import (
    MaterializationError,
    NPZ_KEYS,
    atomic_write_json,
    atomic_write_npz,
    enforce_health_limits,
    materialize_one,
    normalize_detections,
    prepare_resume_state,
    process_tree_rss_bytes,
    rasterize_layout,
    validate_plan,
    wait_for_page_process,
)


ROOT = Path(__file__).resolve().parents[1]


def write_rgb(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(values, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write image: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_health(_pid: int) -> dict[str, float | int]:
    return {
        "memory_free_percent": 80.0,
        "process_tree_rss_bytes": 100 * 1024**2,
        "swap_used_bytes": 0,
    }


def make_resume_fixture(root: Path, file_names: list[str]) -> dict[str, object]:
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
        json.dumps({"external_text_layout_materialization": spec}), encoding="utf-8"
    )
    ledger_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text("\n".join(file_names) + "\n", encoding="utf-8")
    sources: list[tuple[str, str, Path]] = []
    for index, file_name in enumerate(file_names):
        source = root / "sources" / file_name
        write_rgb(source, np.full((2, 2, 3), index, dtype=np.uint8))
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
        "ledger_path": ledger_path.relative_to(root),
        "plan_path": plan_path.relative_to(root),
        "registered": registered,
        "spec": spec,
    }


class ExternalTextLayoutSupportTest(unittest.TestCase):
    def test_registered_plan_is_strictly_valid(self) -> None:
        plan = json.loads(
            (ROOT / "docs/external-text-layout-support-prerequisite-v1.json").read_text(
                encoding="utf-8"
            )
        )
        validate_plan(plan)

    def test_normalize_detections_clips_and_sorts_by_frozen_key(self) -> None:
        polygons = np.asarray(
            [
                [[9.4, 9.4], [12.4, 9.4], [12.4, 12.4], [9.4, 12.4]],
                [[2.4, 1.4], [4.4, 1.4], [4.4, 3.4], [2.4, 3.4]],
            ],
            dtype=np.float32,
        )
        normalized, scores = normalize_detections(
            polygons, np.asarray([0.8, 0.9], dtype=np.float32), height=10, width=10
        )
        np.testing.assert_array_equal(
            normalized[0], np.asarray([[2, 1], [4, 1], [4, 3], [2, 3]], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            normalized[1], np.asarray([[9, 9], [9, 9], [9, 9], [9, 9]], dtype=np.int32)
        )
        np.testing.assert_array_equal(scores, np.asarray([0.9, 0.8], dtype=np.float32))

    def test_rasterize_layout_uses_binary_occupancy_and_max_confidence(self) -> None:
        polygons = np.asarray(
            [
                [[1, 1], [3, 1], [3, 3], [1, 3]],
                [[2, 2], [4, 2], [4, 4], [2, 4]],
            ],
            dtype=np.int32,
        )
        occupancy, confidence = rasterize_layout(
            polygons,
            np.asarray([0.4, 0.8], dtype=np.float32),
            height=6,
            width=6,
        )
        self.assertEqual(occupancy.dtype, np.uint8)
        self.assertEqual(confidence.dtype, np.float32)
        self.assertEqual(int(occupancy[2, 2]), 1)
        self.assertAlmostEqual(float(confidence[2, 2]), 0.8, places=6)
        self.assertTrue(np.isin(occupancy, [0, 1]).all())

    def test_malformed_detector_output_fails_closed(self) -> None:
        with self.assertRaises(MaterializationError):
            normalize_detections(
                np.zeros((1, 4, 3), dtype=np.float32),
                np.ones(1, dtype=np.float32),
                height=4,
                width=4,
            )
        with self.assertRaises(MaterializationError):
            normalize_detections(
                np.zeros((1, 4, 2), dtype=np.float32),
                np.asarray([1.1], dtype=np.float32),
                height=4,
                width=4,
            )

    def test_npz_contract_rejects_extra_key(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "page.npz"
            np.savez_compressed(
                path,
                polygons=np.empty((0, 4, 2), dtype=np.int32),
                scores=np.empty((0,), dtype=np.float32),
                text_confidence=np.zeros((2, 2), dtype=np.float32),
                text_occupancy=np.zeros((2, 2), dtype=np.uint8),
                forbidden=np.zeros(1),
            )
            with self.assertRaises(AuditError):
                load_layout_npz(path, expected_shape=(2, 2))

    def test_npz_contract_reconstructs_raster(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "page.npz"
            polygons = np.asarray(
                [[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.int32
            )
            scores = np.asarray([0.75], dtype=np.float32)
            occupancy, confidence = rasterize_layout(
                polygons, scores, height=2, width=2
            )
            np.savez_compressed(
                path,
                polygons=polygons,
                scores=scores,
                text_confidence=confidence,
                text_occupancy=occupancy,
            )
            arrays = load_layout_npz(path, expected_shape=(2, 2))
            self.assertEqual(set(arrays), NPZ_KEYS)

    def test_five_channel_page_features_share_ablation_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            second_dir = root / "second"
            label_dir = root / "labels"
            layout_dir = root / "layout"
            second = np.asarray(
                [
                    [[10, 20, 30], [40, 50, 60]],
                    [[70, 80, 90], [100, 110, 120]],
                ],
                dtype=np.uint8,
            )
            target = second.copy()
            target[0, 0] += 10
            target[1, 0] += 10
            write_rgb(second_dir / "page.png", second)
            write_rgb(label_dir / "page.png", target)
            polygons = np.asarray(
                [[[0, 0], [0, 0], [0, 1], [0, 1]]], dtype=np.int32
            )
            scores = np.asarray([0.75], dtype=np.float32)
            occupancy, confidence = rasterize_layout(
                polygons, scores, height=2, width=2
            )
            layout_dir.mkdir(parents=True)
            np.savez_compressed(
                layout_dir / "page.npz",
                polygons=polygons,
                scores=scores,
                text_confidence=confidence,
                text_occupancy=occupancy,
            )
            page = build_page(
                file_name="page.png",
                second_stage_dir=second_dir,
                label_dir=label_dir,
                layout_dir=layout_dir,
                layout_row={"height": 2, "width": 2},
                second_stage_row={
                    "base_edit_threshold": "12",
                    "second_delta_threshold": "32",
                    "dark_threshold": "0",
                },
                margin_gray=2.0,
                sample_cap=1,
            )
            self.assertEqual(page["features"].shape[1], len(CHANNELS))
            self.assertEqual(page["ablation_features"].shape[1], len(ABLATION_CHANNELS))
            np.testing.assert_array_equal(
                page["ablation_features"], page["features"][:, :3]
            )

    def test_target_path_rejected_before_any_detector_factory(self) -> None:
        import scripts.analysis.materialize_external_text_layout_support_train_only as materializer

        with mock.patch.object(materializer, "validate_registered_inputs", side_effect=MaterializationError("forbidden source path")):
            detector_factory = mock.Mock(side_effect=AssertionError("detector called"))
            with self.assertRaises(MaterializationError):
                materializer.materialize(
                    repo_root=ROOT,
                    detector_factory=detector_factory,
                )
            detector_factory.assert_not_called()

    def test_materializer_failure_preserves_only_validated_page_progress(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_resume_fixture(root, ["one.png", "two.png"])
            registered = fixture["registered"]

            class FailingDetector:
                calls = 0

                def predict(self, **_kwargs):
                    type(self).calls += 1
                    if type(self).calls == 2:
                        raise RuntimeError("synthetic detector failure")
                    return [{"dt_polys": [], "dt_scores": []}]

                def close(self):
                    pass

            with (
                mock.patch.object(
                    materializer, "validate_registered_inputs", return_value=registered
                ),
                mock.patch.object(materializer, "runtime_health", side_effect=safe_health),
                mock.patch.object(
                    materializer, "assert_no_conflicting_model_processes"
                ),
            ):
                with self.assertRaises(MaterializationError):
                    materializer.materialize(
                        repo_root=root,
                        plan_path=fixture["plan_path"],
                        ledger_path=fixture["ledger_path"],
                        detector_factory=lambda _spec: FailingDetector(),
                    )
            temporary_root = registered["temporary_root"]
            self.assertFalse(registered["output_root"].exists())
            self.assertTrue((temporary_root / "pages" / "one.npz").is_file())
            self.assertTrue((temporary_root / "records" / "one.json").is_file())
            self.assertFalse((temporary_root / "records" / "two.json").exists())

    def test_resume_reuses_completed_page_and_publishes_exact_surface(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_resume_fixture(root, ["one.png", "two.png"])
            registered = fixture["registered"]
            calls: list[str] = []

            class Detector:
                def predict(self, *, input: str, **_kwargs):
                    calls.append(Path(input).name)
                    return [{"dt_polys": [], "dt_scores": []}]

                def close(self):
                    pass

            page_dir, record_dir, _completed = prepare_resume_state(
                repo_root=root,
                plan_file=root / fixture["plan_path"],
                registered=registered,
            )
            one_source = registered["sources"][0][2]
            detector = Detector()
            row = materialize_one(
                detector=detector,
                file_name="one.png",
                source_path=one_source,
                spec=fixture["spec"],
                page_dir=page_dir,
            )
            atomic_write_json(record_dir / "one.json", row)
            calls.clear()
            with (
                mock.patch.object(
                    materializer, "validate_registered_inputs", return_value=registered
                ),
                mock.patch.object(materializer, "runtime_health", side_effect=safe_health),
                mock.patch.object(
                    materializer, "assert_no_conflicting_model_processes"
                ),
            ):
                result = materializer.materialize(
                    repo_root=root,
                    plan_path=fixture["plan_path"],
                    ledger_path=fixture["ledger_path"],
                    detector_factory=lambda _spec: Detector(),
                )
            self.assertEqual(calls, ["two.png"])
            self.assertEqual(result["train_count"], 2)
            output_root = registered["output_root"]
            self.assertEqual(
                {path.name for path in output_root.iterdir()}, {"manifest.json", "pages"}
            )
            self.assertEqual(
                {path.name for path in (output_root / "pages").iterdir()},
                {"one.npz", "two.npz"},
            )
            self.assertFalse(registered["temporary_root"].exists())

    def test_resume_after_publish_before_cleanup_revalidates_without_detector(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_resume_fixture(root, ["one.png"])
            registered = fixture["registered"]
            calls: list[str] = []

            class Detector:
                def predict(self, *, input: str, **_kwargs):
                    calls.append(Path(input).name)
                    return [{"dt_polys": [], "dt_scores": []}]

                def close(self):
                    pass

            real_rmtree = materializer.shutil.rmtree
            injected = False
            marker_path, cleanup_root = materializer.published_transaction_paths(
                registered["output_root"]
            )

            def fail_cleanup_once(path, *args, **kwargs):
                nonlocal injected
                if path == cleanup_root and not injected:
                    injected = True
                    raise OSError("synthetic crash after publish")
                return real_rmtree(path, *args, **kwargs)

            def enter_common(stack: ExitStack) -> None:
                stack.enter_context(
                    mock.patch.object(
                        materializer,
                        "validate_registered_inputs",
                        return_value=registered,
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        materializer, "runtime_health", side_effect=safe_health
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        materializer, "assert_no_conflicting_model_processes"
                    )
                )

            with ExitStack() as stack:
                enter_common(stack)
                stack.enter_context(
                    mock.patch.object(
                        materializer.shutil,
                        "rmtree",
                        side_effect=fail_cleanup_once,
                    )
                )
                with self.assertRaisesRegex(OSError, "after publish"):
                    materializer.materialize(
                        repo_root=root,
                        plan_path=fixture["plan_path"],
                        ledger_path=fixture["ledger_path"],
                        detector_factory=lambda _spec: Detector(),
                    )
            self.assertEqual(calls, ["one.png"])
            self.assertTrue(registered["output_root"].is_dir())
            self.assertFalse(registered["temporary_root"].exists())
            self.assertTrue(cleanup_root.is_dir())
            self.assertTrue(marker_path.is_file())
            calls.clear()
            with ExitStack() as stack:
                enter_common(stack)
                result = materializer.materialize(
                    repo_root=root,
                    plan_path=fixture["plan_path"],
                    ledger_path=fixture["ledger_path"],
                    detector_factory=lambda _spec: Detector(),
                )
            self.assertEqual(calls, [])
            self.assertEqual(result["train_count"], 1)
            self.assertFalse(registered["temporary_root"].exists())
            self.assertFalse(cleanup_root.exists())
            self.assertFalse(marker_path.exists())

    def test_resume_rejects_corrupt_committed_page(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_resume_fixture(root, ["one.png"])
            registered = fixture["registered"]
            page_dir, record_dir, _completed = prepare_resume_state(
                repo_root=root,
                plan_file=root / fixture["plan_path"],
                registered=registered,
            )
            source = registered["sources"][0][2]
            polygons = np.empty((0, 4, 2), dtype=np.int32)
            scores = np.empty((0,), dtype=np.float32)
            atomic_write_npz(
                page_dir / "one.npz",
                polygons=polygons,
                scores=scores,
                confidence=np.zeros((2, 2), dtype=np.float32),
                occupancy=np.zeros((2, 2), dtype=np.uint8),
            )
            row = {
                "confidence_max": 0.0,
                "confidence_mean": 0.0,
                "detection_count": 0,
                "file": "one.png",
                "height": 2,
                "npz_sha256": sha256_file(page_dir / "one.npz"),
                "occupancy_pixels": 0,
                "source_sha256": sha256_file(source),
                "width": 2,
            }
            atomic_write_json(record_dir / "one.json", row)
            with (page_dir / "one.npz").open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(MaterializationError, "NPZ changed"):
                prepare_resume_state(
                    repo_root=root,
                    plan_file=root / fixture["plan_path"],
                    registered=registered,
                )

    def test_resume_discards_uncommitted_orphan_page(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_resume_fixture(root, ["one.png"])
            registered = fixture["registered"]
            page_dir, _record_dir, _completed = prepare_resume_state(
                repo_root=root,
                plan_file=root / fixture["plan_path"],
                registered=registered,
            )
            atomic_write_npz(
                page_dir / "one.npz",
                polygons=np.empty((0, 4, 2), dtype=np.int32),
                scores=np.empty((0,), dtype=np.float32),
                confidence=np.zeros((2, 2), dtype=np.float32),
                occupancy=np.zeros((2, 2), dtype=np.uint8),
            )
            _page_dir, _record_dir, completed = prepare_resume_state(
                repo_root=root,
                plan_file=root / fixture["plan_path"],
                registered=registered,
            )
            self.assertEqual(completed, {})
            self.assertFalse((page_dir / "one.npz").exists())

    def test_resume_recovers_interrupted_atomic_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_resume_fixture(root, ["one.png"])
            registered = fixture["registered"]
            temporary_root = registered["temporary_root"]
            initializing_root = temporary_root.with_name(
                f"{temporary_root.name}.initializing"
            )
            initializing_root.mkdir(parents=True)
            (initializing_root / "partial").write_text("interrupted", encoding="utf-8")

            page_dir, record_dir, completed = prepare_resume_state(
                repo_root=root,
                plan_file=root / fixture["plan_path"],
                registered=registered,
            )

            self.assertEqual(completed, {})
            self.assertEqual(page_dir, temporary_root / "pages")
            self.assertEqual(record_dir, temporary_root / "records")
            self.assertFalse(initializing_root.exists())
            self.assertEqual(
                {path.name for path in temporary_root.iterdir()},
                {"pages", "progress.json", "records"},
            )

    def test_worker_count_above_one_fails_before_input_validation(self) -> None:
        with mock.patch.object(
            materializer,
            "validate_registered_inputs",
            side_effect=AssertionError("validation must not run"),
        ):
            with self.assertRaisesRegex(MaterializationError, "fixed at one"):
                materializer.materialize(repo_root=ROOT, worker_count=2)

    def test_external_materializer_lock_rejects_a_second_instance(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            lock_path = Path(raw) / "materializer.lock"
            with materializer.runtime.exclusive_run_lock(lock_path):
                with self.assertRaisesRegex(
                    MaterializationError, "already active"
                ):
                    with materializer.runtime.exclusive_run_lock(lock_path):
                        self.fail("second materializer acquired the exclusive lock")
        self.assertNotIn(
            str(ROOT), str(materializer.runtime.HOST_USER_RUN_LOCK_PATH)
        )
        self.assertIn(
            str(os.getuid()), materializer.runtime.HOST_USER_RUN_LOCK_PATH.name
        )

    def test_external_materializer_rejects_conflicting_model_processes(self) -> None:
        safe_rows = [(901, "python scripts/analysis/audit_external_layout.py")]
        materializer.runtime.assert_no_conflicting_model_processes(safe_rows)
        for command in (
            "python scripts/infer/run_primary_full_page.py --device cpu",
            "python scripts/run_second_stage_residual_repair.py --device mps",
            "python scripts/run_hybrid_second_stage_gate.py --device cpu",
            "python scripts/micro_train_region_probe.py",
            "python scripts/analysis/train_page_selector_ranker.py",
            "python scripts/analysis/train_region_component_ranker.py",
            "python scripts/train/train_monotonic_residual_erase.py",
            "python meta_train.py",
            "python train.py",
        ):
            with self.subTest(command=command):
                with self.assertRaisesRegex(
                    MaterializationError, "conflicting model process"
                ):
                    materializer.runtime.assert_no_conflicting_model_processes(
                        [(902, command)]
                    )

    def test_health_limits_fail_closed_for_rss_memory_and_swap(self) -> None:
        safe = {
            "memory_free_percent": 80.0,
            "process_tree_rss_bytes": 100,
            "swap_used_bytes": 0,
        }
        enforce_health_limits(safe)
        for changed in (
            {**safe, "process_tree_rss_bytes": materializer.MAX_DETECTOR_RSS_BYTES + 1},
            {**safe, "memory_free_percent": materializer.MIN_MEMORY_FREE_PERCENT - 0.1},
            {**safe, "swap_used_bytes": materializer.MAX_SWAP_USED_BYTES + 1},
        ):
            with self.assertRaises(MaterializationError):
                enforce_health_limits(changed)

    def test_process_tree_rss_includes_all_descendants(self) -> None:
        output = "1 0 10\n2 1 20\n3 2 30\n4 0 40\n"
        self.assertEqual(process_tree_rss_bytes(1, output), 60 * 1024)

    def test_memory_pressure_abort_terminates_page_process(self) -> None:
        class FakeProcess:
            pid = 123
            exitcode = None

            def is_alive(self) -> bool:
                return True

            def join(self, timeout=None) -> None:
                del timeout

        process = FakeProcess()
        unsafe = {
            "memory_free_percent": materializer.MIN_MEMORY_FREE_PERCENT - 1.0,
            "process_tree_rss_bytes": 100,
            "swap_used_bytes": 0,
        }
        with mock.patch.object(materializer.runtime, "terminate_page_process") as terminate:
            with self.assertRaisesRegex(MaterializationError, "memory safety"):
                wait_for_page_process(process, health_reader=lambda _pid: unsafe)
        terminate.assert_called_once_with(process)

    def test_resume_rejects_progress_provenance_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fixture = make_resume_fixture(root, ["one.png"])
            registered = fixture["registered"]
            prepare_resume_state(
                repo_root=root,
                plan_file=root / fixture["plan_path"],
                registered=registered,
            )
            progress = registered["temporary_root"] / "progress.json"
            payload = json.loads(progress.read_text(encoding="utf-8"))
            payload["expected_train_count"] = 2
            atomic_write_json(progress, payload)
            with self.assertRaisesRegex(MaterializationError, "provenance changed"):
                prepare_resume_state(
                    repo_root=root,
                    plan_file=root / fixture["plan_path"],
                    registered=registered,
                )


if __name__ == "__main__":
    unittest.main()
