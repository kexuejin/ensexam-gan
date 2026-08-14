from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


from scripts.analysis import probe_external_text_layout_runtime_safety as probe
from scripts.analysis.materialize_external_text_layout_support_train_only import (
    MaterializationError,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_sleep(_seconds: float) -> None:
    pass


def build_fixture(root: Path) -> tuple[Path, Path, dict[str, object]]:
    source = root / "data-links" / "samples" / "train" / "raw" / "page.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source-only")
    manifest = root / "hardcase_lists" / "train.txt"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "data-links/samples/train/raw/page.png\n", encoding="utf-8"
    )
    plan_path = root / "docs" / "plan.json"
    plan_path.parent.mkdir(parents=True)
    plan = {
        "data": {
            "manifest": {
                "path": "hardcase_lists/train.txt",
                "sha256": sha256_file(manifest),
            }
        },
        "evidence": {"runtime": {"python": "test"}},
        "external_text_layout_materialization": {
            "device": "cpu",
            "engine": "transformers",
            "model_name": "PP-OCRv6_medium_det",
            "output_root": "outputs/formal-materialization",
        },
        "planned_implementation": {
            "audit_output": "outputs/formal-audit/audit.json"
        },
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    ledger_path = root / "docs" / "ledger.json"
    ledger_path.write_text("{}\n", encoding="utf-8")
    return plan_path, ledger_path, plan


def fixture_contract(root: Path) -> dict[str, object]:
    source = root / "data-links" / "samples" / "train" / "raw" / "page.png"
    return {
        "frozen_inputs": {
            "page": {
                "path": str(source.relative_to(root)),
                "sha256": sha256_file(source),
            }
        }
    }


class ExternalTextLayoutRuntimeSafetyProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.thread_cap_patch = mock.patch.dict(
            probe.os.environ,
            {name: "1" for name in probe.THREAD_CAP_NAMES},
        )
        self.thread_cap_patch.start()
        self.addCleanup(self.thread_cap_patch.stop)

    def test_resource_failure_uses_lock_and_monitor_without_formal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path, ledger_path, _plan = build_fixture(root)
            result_path = Path("outputs/runtime-probe/result.json")
            lock_path = root / "host-user.lock"
            temporary_root = root / "probe-temporary"
            temporary_root.mkdir()
            lock_calls: list[Path] = []
            health_calls: list[int] = []

            @contextmanager
            def observed_lock(path: Path):
                lock_calls.append(path)
                yield

            def health(pid: int) -> dict[str, float | int]:
                health_calls.append(pid)
                return {
                    "memory_free_percent": 74.0,
                    "process_tree_rss_bytes": 0,
                    "swap_used_bytes": 0,
                }

            page_runner_calls: list[dict[str, object]] = []

            def resource_failure(**kwargs):
                page_runner_calls.append(kwargs)
                running = json.loads((root / result_path).read_text(encoding="utf-8"))
                self.assertEqual(running["terminal"], "RUNNING")
                self.assertEqual(running["attempt_count"], 1)
                raise probe.runtime.ResourceLimitError(
                    "system memory safety limit crossed: 31.0% free",
                    trigger_health={
                        "memory_free_percent": 31.0,
                        "process_tree_rss_bytes": 7 * 1024**3,
                        "swap_used_bytes": 600 * 1024**2,
                    },
                    observed_health={
                        "minimum_memory_free_percent": 31.0,
                        "peak_process_tree_rss_bytes": 7 * 1024**3,
                        "peak_swap_used_bytes": 600 * 1024**2,
                    },
                )

            with (
                mock.patch.object(probe.materializer, "validate_plan"),
                mock.patch.object(probe.materializer, "validate_authority"),
                mock.patch.object(
                    probe,
                    "validate_probe_contract",
                    return_value=fixture_contract(root),
                ),
                mock.patch.object(
                    probe.materializer,
                    "validate_runtime",
                    return_value={"python": "test"},
                ),
                mock.patch.object(
                    probe,
                    "validate_probe_detector",
                    return_value={"model_safetensors": "frozen"},
                ),
                mock.patch.object(
                    probe.materializer, "assert_no_conflicting_model_processes"
                ) as conflict_check,
                mock.patch.object(
                    probe.runtime, "exclusive_run_lock", side_effect=observed_lock
                ),
                mock.patch.object(
                    probe.tempfile, "mkdtemp", return_value=str(temporary_root)
                ),
            ):
                result = probe.run_runtime_probe(
                    repo_root=root,
                    plan_path=plan_path.relative_to(root),
                    ledger_path=ledger_path.relative_to(root),
                    result_path=result_path,
                    page_runner=resource_failure,
                    health_reader=health,
                    simulator_counter=lambda: 0,
                    sleeper=no_sleep,
                    lock_path=lock_path,
                )

            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertEqual(
                result["reason_code"], "runtime_resource_prerequisite_failed"
            )
            self.assertFalse(result["formal_evidence"])
            self.assertFalse(result["formal_outputs_written"])
            self.assertFalse(result["target_access"])
            self.assertFalse(result["label_access"])
            self.assertFalse(result["recognition"])
            self.assertEqual(
                result["detector_files"], {"model_safetensors": "frozen"}
            )
            self.assertEqual(
                result["failure_health"]["trigger_health"][
                    "process_tree_rss_bytes"
                ],
                7 * 1024**3,
            )
            self.assertEqual(
                result["failure_health"]["observed_health"],
                {
                    "minimum_memory_free_percent": 31.0,
                    "peak_process_tree_rss_bytes": 7 * 1024**3,
                    "peak_swap_growth_bytes": 600 * 1024**2,
                },
            )
            self.assertEqual(lock_calls, [lock_path])
            self.assertEqual(len(health_calls), 62)
            self.assertEqual(conflict_check.call_count, 3)
            self.assertEqual(len(page_runner_calls), 1)
            runner_call = page_runner_calls[0]
            self.assertEqual(
                runner_call["maximum_process_tree_rss_bytes"],
                probe.PROBE_MAX_PROCESS_TREE_RSS_BYTES,
            )
            self.assertEqual(
                runner_call["minimum_memory_free_percent"],
                probe.PROBE_MIN_RUNTIME_MEMORY_FREE_PERCENT,
            )
            self.assertEqual(
                runner_call["maximum_swap_used_bytes"],
                probe.PROBE_MAX_SWAP_GROWTH_BYTES,
            )
            self.assertTrue(callable(runner_call["health_reader"]))
            self.assertTrue(runner_call["reject_booted_ios_simulators"])
            self.assertTrue((root / result_path).is_file())
            self.assertFalse((root / "outputs/formal-materialization").exists())
            self.assertFalse((root / "outputs/formal-audit").exists())
            self.assertFalse(temporary_root.exists())

    def test_synthetic_pass_satisfies_terminal_acceptance_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path, ledger_path, _plan = build_fixture(root)
            result_path = Path("outputs/runtime-probe/result.json")

            def successful_page(**kwargs):
                page_dir = kwargs["page_dir"]
                record_path = kwargs["record_path"]
                source_path = kwargs["source_path"]
                npz_path = page_dir / "page.npz"
                np = probe.materializer.np
                probe.materializer.atomic_write_npz(
                    npz_path,
                    polygons=np.empty((0, 4, 2), dtype=np.int32),
                    scores=np.empty((0,), dtype=np.float32),
                    confidence=np.zeros((1, 1), dtype=np.float32),
                    occupancy=np.zeros((1, 1), dtype=np.uint8),
                )
                probe.materializer.atomic_write_json(
                    record_path,
                    {
                        "confidence_max": 0.0,
                        "confidence_mean": 0.0,
                        "detection_count": 0,
                        "file": "page.png",
                        "height": 1,
                        "npz_sha256": sha256_file(npz_path),
                        "occupancy_pixels": 0,
                        "source_sha256": sha256_file(source_path),
                        "width": 1,
                    },
                )
                monitored_health = kwargs["health_reader"](123)
                self.assertEqual(monitored_health["swap_used_bytes"], 0)
                return {
                    "minimum_memory_free_percent": 75.0,
                    "peak_process_tree_rss_bytes": 1024,
                    "peak_swap_used_bytes": 128 * 1024**2,
                }

            sleep_calls: list[float] = []
            launch_swap_baseline = 2 * 1024**3

            with (
                mock.patch.object(probe.materializer, "validate_plan"),
                mock.patch.object(probe.materializer, "validate_authority"),
                mock.patch.object(
                    probe,
                    "validate_probe_contract",
                    return_value=fixture_contract(root),
                ),
                mock.patch.object(
                    probe.materializer,
                    "validate_runtime",
                    return_value={"python": "test"},
                ),
                mock.patch.object(
                    probe,
                    "validate_probe_detector",
                    return_value={"model_safetensors": "frozen"},
                ),
                mock.patch.object(
                    probe.materializer, "assert_no_conflicting_model_processes"
                ),
            ):
                result = probe.run_runtime_probe(
                    repo_root=root,
                    plan_path=plan_path.relative_to(root),
                    ledger_path=ledger_path.relative_to(root),
                    result_path=result_path,
                    page_runner=successful_page,
                    health_reader=lambda _pid: {
                        "memory_free_percent": 80.0,
                        "process_tree_rss_bytes": 0,
                        "swap_used_bytes": launch_swap_baseline,
                    },
                    simulator_counter=lambda: 0,
                    sleeper=sleep_calls.append,
                    lock_path=root / "host-user.lock",
                )

            self.assertEqual(result["terminal"], "PASS")
            self.assertEqual(result["reason_code"], "runtime_safety_probe_passed")
            self.assertEqual(result["attempt_count"], 1)
            self.assertTrue(result["page_completed"])
            self.assertEqual(result["residual_model_process_count"], 0)
            self.assertFalse(result["temporary_page_outputs_retained"])
            self.assertEqual(
                result["launch_swap_baseline_bytes"], launch_swap_baseline
            )
            self.assertEqual(result["launch_health"]["sample_count"], 61)
            self.assertEqual(len(sleep_calls), 60)
            self.assertEqual(
                result["page_health"]["peak_swap_growth_bytes"],
                128 * 1024**2,
            )
            self.assertNotIn("peak_swap_used_bytes", result["page_health"])
            self.assertEqual(
                result["peak_swap_growth_bytes"], 128 * 1024**2
            )
            persisted = json.loads((root / result_path).read_text(encoding="utf-8"))
            self.assertEqual(persisted, result)

    def test_probe_reads_only_plan_manifest_and_raw_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path, ledger_path, _plan = build_fixture(root)
            forbidden = (
                root / "data-links" / "samples" / "train" / "targets" / "page.png"
            )
            forbidden.parent.mkdir(parents=True)
            forbidden.write_bytes(b"must-not-read")
            recognition = root / "recognition_text.json"
            recognition.write_text('{"page":"secret"}', encoding="utf-8")
            opened_paths: list[Path] = []
            original_open = Path.open

            def observed_open(path: Path, *args, **kwargs):
                opened_paths.append(path.resolve())
                return original_open(path, *args, **kwargs)

            with (
                mock.patch.object(probe.materializer, "validate_plan"),
                mock.patch.object(probe.materializer, "validate_authority"),
                mock.patch.object(
                    probe,
                    "validate_probe_contract",
                    return_value=fixture_contract(root),
                ),
                mock.patch.object(
                    probe.materializer,
                    "validate_runtime",
                    return_value={"python": "test"},
                ),
                mock.patch.object(
                    probe,
                    "validate_probe_detector",
                    return_value={"model_safetensors": "frozen"},
                ),
                mock.patch.object(
                    probe.materializer, "assert_no_conflicting_model_processes"
                ),
                mock.patch.object(Path, "open", observed_open),
            ):
                result = probe.run_runtime_probe(
                    repo_root=root,
                    plan_path=plan_path.relative_to(root),
                    ledger_path=ledger_path.relative_to(root),
                    result_path=Path("outputs/runtime-probe/result.json"),
                    page_runner=mock.Mock(
                        side_effect=MaterializationError("synthetic resource stop")
                    ),
                    health_reader=lambda _pid: {
                        "memory_free_percent": 80.0,
                        "process_tree_rss_bytes": 0,
                        "swap_used_bytes": 0,
                    },
                    simulator_counter=lambda: 0,
                    sleeper=no_sleep,
                    lock_path=root / "host-user.lock",
                )

            self.assertEqual(result["page"]["file"], "page.png")
            self.assertNotIn(forbidden.resolve(), opened_paths)
            self.assertNotIn(recognition.resolve(), opened_paths)
            self.assertFalse(any("target" in str(path).lower() for path in opened_paths))
            self.assertFalse(
                any("recognition" in str(path).lower() for path in opened_paths)
            )

    def test_probe_refuses_formal_materialization_or_audit_result_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path, ledger_path, _plan = build_fixture(root)
            with (
                mock.patch.object(probe.materializer, "validate_plan"),
                mock.patch.object(probe.materializer, "validate_authority"),
                mock.patch.object(
                    probe,
                    "validate_probe_contract",
                    return_value=fixture_contract(root),
                ),
                mock.patch.object(
                    probe.materializer,
                    "validate_runtime",
                    return_value={"python": "test"},
                ),
                mock.patch.object(
                    probe,
                    "validate_probe_detector",
                    return_value={"model_safetensors": "frozen"},
                ),
            ):
                for destination in (
                    Path("outputs/formal-materialization/probe.json"),
                    Path("outputs/formal-audit/probe.json"),
                ):
                    with self.subTest(destination=destination):
                        with self.assertRaisesRegex(
                            MaterializationError, "outside formal evidence"
                        ):
                            probe.run_runtime_probe(
                                repo_root=root,
                                plan_path=plan_path.relative_to(root),
                                ledger_path=ledger_path.relative_to(root),
                                result_path=destination,
                            )

    def test_probe_detector_rejects_frozen_file_hash_drift(self) -> None:
        artifact = {"external_path": "/frozen/file", "sha256": "0" * 64}
        plan = {
            "evidence": {
                "official_text_detector": {
                    "license": "Apache-2.0",
                    "model_name": "PP-OCRv6_medium_det",
                    **{
                        name: artifact
                        for name in probe.DETECTOR_ARTIFACT_NAMES
                    },
                }
            },
            "external_text_layout_materialization": {"model_dir": "/frozen"},
        }
        with mock.patch.object(
            probe.materializer,
            "validate_external_artifact",
            side_effect=MaterializationError("probe detector model_safetensors sha256 changed"),
        ):
            with self.assertRaisesRegex(MaterializationError, "sha256 changed"):
                probe.validate_probe_detector(plan)

    def test_preregistered_contract_and_source_are_hash_bound(self) -> None:
        contract = probe.validate_probe_contract(
            probe.ROOT,
            probe.PROBE_CONTRACT_PATH,
            probe.PLAN_PATH,
            probe.DEFAULT_RESULT_PATH,
        )
        self.assertEqual(
            sha256_file(probe.ROOT / probe.PROBE_CONTRACT_PATH),
            probe.EXPECTED_PROBE_CONTRACT_SHA256,
        )
        plan = probe.materializer.read_json(probe.ROOT / probe.PLAN_PATH)
        source = probe.select_probe_source(probe.ROOT, plan)
        probe.validate_probe_source_binding(probe.ROOT, source, contract)
        self.assertEqual(source.name, "hw5k_1011.jpg")
        self.assertEqual(probe.safety_limits()["launch_memory_free_percent_min"], 70.0)
        self.assertEqual(probe.safety_limits()["runtime_memory_free_percent_min"], 45.0)
        self.assertEqual(
            probe.safety_limits()["detector_process_tree_rss_bytes_max"],
            8 * 1024**3,
        )
        self.assertEqual(
            probe.safety_limits()["runtime_swap_growth_bytes_max"],
            512 * 1024**2,
        )
        self.assertEqual(
            probe.safety_limits()["launch_stability_window_seconds"], 60.0
        )
        probe.validate_thread_caps({name: "1" for name in probe.THREAD_CAP_NAMES})
        with self.assertRaisesRegex(MaterializationError, "thread caps"):
            probe.validate_thread_caps(
                {name: None for name in probe.THREAD_CAP_NAMES}
            )

    def test_booted_simulator_closes_probe_before_page_runner(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path, ledger_path, _plan = build_fixture(root)
            result_path = Path("outputs/runtime-probe/result.json")
            page_runner = mock.Mock(
                side_effect=AssertionError("page runner must not start")
            )
            with (
                mock.patch.object(probe.materializer, "validate_plan"),
                mock.patch.object(probe.materializer, "validate_authority"),
                mock.patch.object(
                    probe,
                    "validate_probe_contract",
                    return_value=fixture_contract(root),
                ),
                mock.patch.object(
                    probe.materializer,
                    "validate_runtime",
                    return_value={"python": "test"},
                ),
                mock.patch.object(
                    probe,
                    "validate_probe_detector",
                    return_value={"model_safetensors": "frozen"},
                ),
                mock.patch.object(
                    probe.materializer, "assert_no_conflicting_model_processes"
                ),
            ):
                result = probe.run_runtime_probe(
                    repo_root=root,
                    plan_path=plan_path.relative_to(root),
                    ledger_path=ledger_path.relative_to(root),
                    result_path=result_path,
                    page_runner=page_runner,
                    simulator_counter=lambda: 1,
                    lock_path=root / "host-user.lock",
                )

            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertEqual(result["attempt_count"], 0)
            self.assertIn("Booted iOS Simulator", result["reason"])
            page_runner.assert_not_called()
            self.assertFalse((root / result_path).exists())

    def test_existing_result_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path, ledger_path, _plan = build_fixture(root)
            result_path = Path("outputs/runtime-probe/result.json")
            destination = root / result_path
            destination.parent.mkdir(parents=True)
            destination.write_text('{"terminal":"RUNNING"}\n', encoding="utf-8")
            with (
                mock.patch.object(probe.materializer, "validate_plan"),
                mock.patch.object(probe.materializer, "validate_authority"),
                mock.patch.object(
                    probe,
                    "validate_probe_contract",
                    return_value=fixture_contract(root),
                ),
                mock.patch.object(
                    probe.materializer,
                    "validate_runtime",
                    return_value={"python": "test"},
                ),
                mock.patch.object(
                    probe,
                    "validate_probe_detector",
                    return_value={"model_safetensors": "frozen"},
                ),
                self.assertRaisesRegex(MaterializationError, "retry is prohibited"),
            ):
                probe.run_runtime_probe(
                    repo_root=root,
                    plan_path=plan_path.relative_to(root),
                    ledger_path=ledger_path.relative_to(root),
                    result_path=result_path,
                )
            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                '{"terminal":"RUNNING"}\n',
            )

    def test_high_but_stable_launch_swap_allows_page_runner(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path, ledger_path, _plan = build_fixture(root)
            result_path = Path("outputs/runtime-probe/result.json")
            page_runner = mock.Mock(
                side_effect=MaterializationError("synthetic attempted page")
            )
            launch_swap_baseline = 2 * 1024**3
            with (
                mock.patch.object(probe.materializer, "validate_plan"),
                mock.patch.object(probe.materializer, "validate_authority"),
                mock.patch.object(
                    probe,
                    "validate_probe_contract",
                    return_value=fixture_contract(root),
                ),
                mock.patch.object(
                    probe.materializer,
                    "validate_runtime",
                    return_value={"python": "test"},
                ),
                mock.patch.object(
                    probe,
                    "validate_probe_detector",
                    return_value={"model_safetensors": "frozen"},
                ),
                mock.patch.object(
                    probe.materializer, "assert_no_conflicting_model_processes"
                ),
            ):
                result = probe.run_runtime_probe(
                    repo_root=root,
                    plan_path=plan_path.relative_to(root),
                    ledger_path=ledger_path.relative_to(root),
                    result_path=result_path,
                    page_runner=page_runner,
                    health_reader=lambda _pid: {
                        "memory_free_percent": 80.0,
                        "process_tree_rss_bytes": 0,
                        "swap_used_bytes": launch_swap_baseline,
                    },
                    simulator_counter=lambda: 0,
                    sleeper=no_sleep,
                    lock_path=root / "host-user.lock",
                )

            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertEqual(result["attempt_count"], 1)
            self.assertEqual(
                result["launch_swap_baseline_bytes"], launch_swap_baseline
            )
            page_runner.assert_called_once()
            self.assertTrue((root / result_path).is_file())

    def test_launch_swap_growth_rejects_without_consuming_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path, ledger_path, _plan = build_fixture(root)
            result_path = Path("outputs/runtime-probe/result.json")
            page_runner = mock.Mock(
                side_effect=AssertionError("page runner must not start")
            )
            baseline = 2 * 1024**3
            health_calls = 0

            def growing_health(_pid: int) -> dict[str, float | int]:
                nonlocal health_calls
                health_calls += 1
                return {
                    "memory_free_percent": 80.0,
                    "process_tree_rss_bytes": 0,
                    "swap_used_bytes": baseline + (1 if health_calls > 1 else 0),
                }

            with (
                mock.patch.object(probe.materializer, "validate_plan"),
                mock.patch.object(probe.materializer, "validate_authority"),
                mock.patch.object(
                    probe,
                    "validate_probe_contract",
                    return_value=fixture_contract(root),
                ),
                mock.patch.object(
                    probe.materializer,
                    "validate_runtime",
                    return_value={"python": "test"},
                ),
                mock.patch.object(
                    probe,
                    "validate_probe_detector",
                    return_value={"model_safetensors": "frozen"},
                ),
                mock.patch.object(
                    probe.materializer, "assert_no_conflicting_model_processes"
                ),
            ):
                result = probe.run_runtime_probe(
                    repo_root=root,
                    plan_path=plan_path.relative_to(root),
                    ledger_path=ledger_path.relative_to(root),
                    result_path=result_path,
                    page_runner=page_runner,
                    health_reader=growing_health,
                    simulator_counter=lambda: 0,
                    sleeper=no_sleep,
                    lock_path=root / "host-user.lock",
                )

            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertEqual(result["attempt_count"], 0)
            self.assertIn("swap safety limit exceeded", result["reason"])
            page_runner.assert_not_called()
            self.assertFalse((root / result_path).exists())

    def test_relative_health_reader_subtracts_baseline_and_clamps_zero(self) -> None:
        samples = iter([900, 1000, 1500])
        reader = probe.relative_swap_health_reader(
            lambda _pid: {
                "memory_free_percent": 80.0,
                "process_tree_rss_bytes": 0,
                "swap_used_bytes": next(samples),
            },
            1000,
        )

        self.assertEqual(reader(1)["swap_used_bytes"], 0)
        self.assertEqual(reader(1)["swap_used_bytes"], 0)
        self.assertEqual(reader(1)["swap_used_bytes"], 500)

    def test_runtime_swap_growth_terminates_through_existing_monitor(self) -> None:
        class FakeProcess:
            pid = 123
            exitcode = None

            def is_alive(self) -> bool:
                return True

            def join(self, timeout=None) -> None:
                del timeout

        process = FakeProcess()
        baseline = 2 * 1024**3
        relative_reader = probe.relative_swap_health_reader(
            lambda _pid: {
                "memory_free_percent": 80.0,
                "process_tree_rss_bytes": 1024,
                "swap_used_bytes": (
                    baseline + probe.PROBE_MAX_SWAP_GROWTH_BYTES + 1
                ),
            },
            baseline,
        )

        with mock.patch.object(
            probe.runtime, "terminate_page_process"
        ) as terminate:
            with self.assertRaisesRegex(MaterializationError, "swap safety"):
                probe.materializer.wait_for_page_process(
                    process,
                    health_reader=relative_reader,
                    maximum_process_tree_rss_bytes=(
                        probe.PROBE_MAX_PROCESS_TREE_RSS_BYTES
                    ),
                    minimum_memory_free_percent=(
                        probe.PROBE_MIN_RUNTIME_MEMORY_FREE_PERCENT
                    ),
                    maximum_swap_used_bytes=probe.PROBE_MAX_SWAP_GROWTH_BYTES,
                )

        terminate.assert_called_once_with(process)


if __name__ == "__main__":
    unittest.main()
