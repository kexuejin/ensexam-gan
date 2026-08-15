from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterator
import unittest
from unittest import mock


from scripts.analysis import run_external_text_layout_recovered_materialization as launcher


@contextmanager
def noop_lock(_path: Path) -> Iterator[None]:
    yield


class RunExternalTextLayoutRecoveredMaterializationTest(unittest.TestCase):
    def test_repository_contract_and_shared_hashes_are_exact(self) -> None:
        contract = launcher.validate_repository_contract(launcher.ROOT)
        self.assertEqual(contract["schema_version"], 2)
        self.assertEqual(launcher.validate_v4_contract(launcher.ROOT)["schema_version"], 4)
        self.assertEqual(launcher.validate_v5_contract(launcher.ROOT)["schema_version"], 5)
        self.assertEqual(launcher.validate_v6_contract(launcher.ROOT)["schema_version"], 6)
        self.assertEqual(launcher.validate_v8_contract(launcher.ROOT)["schema_version"], 8)
        self.assertEqual(
            launcher.materializer.sha256_file(
                launcher.ROOT
                / "scripts/analysis/materialize_external_text_layout_support_train_only.py"
            ),
            launcher.EXPECTED_SHARED_MATERIALIZER_SHA256,
        )
        self.assertEqual(
            launcher.materializer.sha256_file(
                launcher.ROOT / "tests/test_external_text_layout_support_prerequisite.py"
            ),
            launcher.EXPECTED_SHARED_TEST_SHA256,
        )
        self.assertEqual(
            launcher.materializer.sha256_file(
                launcher.ROOT
                / "scripts/analysis/external_text_layout_materialization_runtime.py"
            ),
            launcher.EXPECTED_SHARED_RUNTIME_SHA256,
        )
        self.assertEqual(
            launcher.materializer.sha256_file(
                launcher.ROOT
                / "scripts/analysis/probe_external_text_layout_runtime_safety.py"
            ),
            launcher.EXPECTED_SAFETY_PROBE_SHA256,
        )

    def test_derived_plan_is_exact_and_preserves_original(self) -> None:
        contract = launcher.validate_repository_contract(launcher.ROOT)
        original_path = launcher.ROOT / contract["evidence"]["original_plan"]["path"]
        original = json.loads(original_path.read_text(encoding="utf-8"))

        derived, payload = launcher.build_derived_plan(contract, original)

        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            launcher.EXPECTED_DERIVED_PLAN_SHA256,
        )
        restored = json.loads(json.dumps(derived))
        del restored["recovered_input_overlay"]
        restored["evidence"]["second_stage_metrics"]["sha256"] = original[
            "evidence"
        ]["second_stage_metrics"]["sha256"]
        self.assertEqual(restored, original)

    def test_archive_inputs_match_terminal_publication(self) -> None:
        contract = launcher.validate_repository_contract(launcher.ROOT)
        identities = launcher.validate_archive_inputs(launcher.ROOT, contract)
        self.assertEqual(identities["primary"]["prediction_set"]["count"], 275)
        self.assertEqual(
            identities["second_stage"]["metrics_sha256"],
            "79fd61278e689a0003e37a5bdf20f856184b49c8fdb3af8ad9af03a3a13c451b",
        )

    def test_execution_authority_requires_v8_integration_pass(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = root / launcher.LEDGER_PATH
            ledger_path.parent.mkdir(parents=True)
            prerequisites = [
                {
                    "id": "external_text_layout_recovered_archive_publication",
                    "status": "passed",
                },
                {
                    "id": "external_text_layout_recovered_materializer_input_preregistration",
                    "status": "passed",
                },
                {
                    "id": "external_text_layout_recovered_materializer_launch_v2_preregistration",
                    "status": "passed",
                },
                {
                    "id": "external_text_layout_recovered_materializer_launch_v2_integration",
                    "status": "passed",
                },
                {
                    "id": "external_text_layout_recovered_materializer_baseline_relative_launch_v3_preregistration",
                    "status": "passed",
                },
                {
                    "id": "external_text_layout_recovered_materializer_baseline_relative_launch_v3_integration",
                    "status": "passed",
                },
                {
                    "id": "external_text_layout_recovered_materializer_formal_rss_launch_v4_preregistration",
                    "status": "passed",
                },
                {
                    "id": "external_text_layout_recovered_materializer_formal_rss_launch_v4_integration",
                    "status": "passed",
                },
                {
                    "id": "external_text_layout_recovered_materializer_formal_memory_launch_v5_preregistration",
                    "status": "passed",
                },
                {
                    "id": "external_text_layout_support_train_only_diagnostic",
                    "status": "pending",
                },
            ]
            ledger = {
                "active_iteration": {
                    "prerequisites": prerequisites,
                    "terminal": "PREREQUISITE_NEEDED",
                }
            }
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            with self.assertRaises(launcher.RecoveredMaterializationError):
                launcher.validate_execution_authority(root)
            prerequisites.append(
                {
                    "id": "external_text_layout_recovered_materializer_formal_memory_launch_v5_integration",
                    "status": "passed",
                }
            )
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            with self.assertRaises(launcher.RecoveredMaterializationError):
                launcher.validate_execution_authority(root)
            prerequisites.extend(
                [
                    {
                        "id": "external_text_layout_recovered_materializer_batched_launch_v6_preregistration",
                        "status": "passed",
                    },
                    {
                        "id": "external_text_layout_recovered_materializer_batched_launch_v6_integration",
                        "status": "passed",
                    },
                ]
            )
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            with self.assertRaises(launcher.RecoveredMaterializationError):
                launcher.validate_execution_authority(root)
            prerequisites.append(
                {
                    "id": "external_text_layout_recovered_materializer_bounded_rss_launch_v8_preregistration",
                    "status": "passed",
                }
            )
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            with self.assertRaises(launcher.RecoveredMaterializationError):
                launcher.validate_execution_authority(root)
            prerequisites.append(
                {
                    "id": "external_text_layout_recovered_materializer_bounded_rss_launch_v8_integration",
                    "status": "passed",
                }
            )
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            launcher.validate_execution_authority(root)

    def test_derived_plan_write_is_atomic_idempotent_and_fail_closed(self) -> None:
        payload = b'{"plan": true}\n'
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "plans" / "effective.json"
            self.assertEqual(
                launcher.write_or_validate_derived_plan(path, payload, expected),
                "written",
            )
            self.assertEqual(
                launcher.write_or_validate_derived_plan(path, payload, expected),
                "existing",
            )
            path.write_bytes(b"changed")
            with self.assertRaises(launcher.RecoveredMaterializationError):
                launcher.write_or_validate_derived_plan(path, payload, expected)

    def test_closed_authority_rejects_before_plan_write_or_materializer(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            original = root / "docs" / "plan.json"
            original.parent.mkdir(parents=True)
            original.write_text("{}\n", encoding="utf-8")
            payload = b"{}\n"
            contract = {
                "derived_plan": {
                    "expected_sha256": hashlib.sha256(payload).hexdigest(),
                    "path": "generated/effective-plan.json",
                },
                "evidence": {"original_plan": {"path": "docs/plan.json"}},
            }
            with (
                mock.patch.object(
                    launcher, "validate_repository_contract", return_value=contract
                ),
                mock.patch.object(launcher, "validate_artifact", return_value=original),
                mock.patch.object(
                    launcher, "validate_archive_inputs", return_value={}
                ),
                mock.patch.object(
                    launcher, "build_derived_plan", return_value=({}, payload)
                ),
                mock.patch.object(
                    launcher,
                    "validate_execution_authority",
                    side_effect=launcher.RecoveredMaterializationError("closed"),
                ),
                mock.patch.object(
                    launcher, "run_baseline_relative_materializer"
                ) as materialize,
            ):
                with self.assertRaises(launcher.RecoveredMaterializationError):
                    launcher.run_launcher(root)
            self.assertFalse((root / "generated/effective-plan.json").exists())
            materialize.assert_not_called()

    def test_launcher_calls_batched_materializer_with_exact_plan(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            original = root / "docs" / "plan.json"
            original.parent.mkdir(parents=True)
            original.write_text("{}\n", encoding="utf-8")
            payload = b"{}\n"
            expected_sha = hashlib.sha256(payload).hexdigest()
            contract = {
                "derived_plan": {
                    "expected_sha256": expected_sha,
                    "path": "generated/effective-plan.json",
                    "semantic_changes_from_original": {
                        "evidence.second_stage_metrics.sha256": {
                            "before": "b" * 64,
                            "after": "a" * 64,
                        }
                    },
                },
                "evidence": {"original_plan": {"path": "docs/plan.json"}},
            }
            summary = {
                "content_sha256": "c" * 64,
                "manifest": "outputs/materialization/manifest.json",
                "manifest_sha256": "d" * 64,
                "output_root": "outputs/materialization",
                "terminal": "PASS",
                "train_count": 275,
            }
            runtime_safety = launcher.recovered_runtime_safety()
            with (
                mock.patch.object(
                    launcher, "validate_repository_contract", return_value=contract
                ),
                mock.patch.object(launcher, "validate_artifact", return_value=original),
                mock.patch.object(
                    launcher,
                    "validate_archive_inputs",
                    return_value={"primary": {}, "second_stage": {}},
                ),
                mock.patch.object(
                    launcher, "build_derived_plan", return_value=({}, payload)
                ),
                mock.patch.object(launcher, "validate_execution_authority"),
                mock.patch.object(
                    launcher,
                    "run_baseline_relative_materializer",
                    return_value=({"terminal": "PASS"}, runtime_safety),
                ) as materialize,
                mock.patch.object(
                    launcher, "validate_materialization_output", return_value=summary
                ),
            ):
                result = launcher.run_launcher(root)
            materialize.assert_called_once_with(
                repo_root=root,
                derived_plan_path=root / "generated/effective-plan.json",
            )
            self.assertEqual(result["terminal"], "PASS")
            self.assertEqual(result["schema_version"], 8)
            self.assertEqual(result["runtime_safety"], runtime_safety)
            self.assertEqual(
                json.loads((root / launcher.RESULT_PATH).read_text()), result
            )

    def test_existing_terminal_result_bypasses_closed_authority(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            original = root / "docs" / "plan.json"
            original.parent.mkdir(parents=True)
            original.write_text("{}\n", encoding="utf-8")
            result_path = root / launcher.RESULT_PATH
            result_path.parent.mkdir(parents=True)
            result_path.write_text("{}\n", encoding="utf-8")
            contract = {
                "derived_plan": {
                    "expected_sha256": "a" * 64,
                    "path": "generated/effective-plan.json",
                },
                "evidence": {"original_plan": {"path": "docs/plan.json"}},
            }
            terminal = {"schema_version": 8, "terminal": "PASS"}
            with (
                mock.patch.object(
                    launcher, "validate_repository_contract", return_value=contract
                ),
                mock.patch.object(launcher, "validate_artifact", return_value=original),
                mock.patch.object(
                    launcher, "validate_archive_inputs", return_value={}
                ),
                mock.patch.object(
                    launcher, "build_derived_plan", return_value=({}, b"{}\n")
                ),
                mock.patch.object(
                    launcher, "validate_existing_result", return_value=terminal
                ),
                mock.patch.object(
                    launcher, "validate_execution_authority"
                ) as authority,
            ):
                self.assertEqual(launcher.run_launcher(root), terminal)
                authority.assert_not_called()

    def test_existing_final_materialization_recovers_missing_launcher_result(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            original = root / "docs" / "plan.json"
            original.parent.mkdir(parents=True)
            original.write_text("{}\n", encoding="utf-8")
            payload = b"{}\n"
            expected_sha = hashlib.sha256(payload).hexdigest()
            contract = {
                "derived_plan": {
                    "expected_sha256": expected_sha,
                    "path": "generated/effective-plan.json",
                    "semantic_changes_from_original": {
                        "evidence.second_stage_metrics.sha256": {
                            "before": "b" * 64,
                            "after": "a" * 64,
                        }
                    },
                },
                "evidence": {"original_plan": {"path": "docs/plan.json"}},
            }
            output_root = root / launcher.materializer.OUTPUT_ROOT
            output_root.mkdir(parents=True)
            summary = {
                "content_sha256": "c" * 64,
                "manifest": "outputs/materialization/manifest.json",
                "manifest_sha256": "d" * 64,
                "output_root": "outputs/materialization",
                "terminal": "PASS",
                "train_count": 275,
            }
            with (
                mock.patch.object(
                    launcher, "validate_repository_contract", return_value=contract
                ),
                mock.patch.object(launcher, "validate_artifact", return_value=original),
                mock.patch.object(
                    launcher,
                    "validate_archive_inputs",
                    return_value={"primary": {}, "second_stage": {}},
                ),
                mock.patch.object(
                    launcher, "build_derived_plan", return_value=({}, payload)
                ),
                mock.patch.object(launcher, "validate_execution_authority"),
                mock.patch.object(
                    launcher, "validate_materialization_output", return_value=summary
                ),
                mock.patch.object(
                    launcher, "run_baseline_relative_materializer"
                ) as materialize,
            ):
                result = launcher.run_launcher(root)
                materialize.assert_not_called()
            self.assertEqual(result["materialization"], summary)
            self.assertEqual(
                result["runtime_safety"], launcher.recovered_runtime_safety()
            )
            self.assertTrue((root / launcher.RESULT_PATH).is_file())

    def test_high_stable_swap_uses_explicit_batch_limits_without_global_mutation(
        self,
    ) -> None:
        baseline = 2 * 1024**3

        def health(_pid: int) -> dict[str, float | int]:
            return {
                "memory_free_percent": 80.0,
                "process_tree_rss_bytes": 1024,
                "swap_used_bytes": baseline,
            }

        materializer_result = {
            "minimum_memory_free_percent": 63.0,
            "peak_process_tree_rss_bytes": 10_334_617_600,
            "peak_swap_used_bytes": 0,
            "terminal": "PASS",
        }
        captured: dict[str, object] = {}

        def runner(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            self.assertEqual(
                kwargs["health_reader"](123)["swap_used_bytes"], 0
            )
            launcher.batch_runtime.enforce_recovered_health_limits(
                {
                    "memory_free_percent": 39.0,
                    "process_tree_rss_bytes": 8 * 1024**3,
                    "swap_used_bytes": 512 * 1024**2,
                },
                maximum_process_tree_rss_bytes=kwargs[
                    "maximum_process_tree_rss_bytes"
                ],
                minimum_memory_free_percent=kwargs[
                    "minimum_memory_free_percent"
                ],
                maximum_swap_used_bytes=kwargs["maximum_swap_used_bytes"],
            )
            with kwargs["lock_factory"](Path("/lock")):
                pass
            kwargs["conflict_checker"]()
            return materializer_result

        originals = (
            launcher.materializer.runtime.exclusive_run_lock,
            launcher.materializer.runtime.runtime_health,
            launcher.materializer.runtime.enforce_health_limits,
            launcher.batch_runtime.materialize,
        )
        conflicts = mock.Mock()
        simulators = mock.Mock(return_value=0)
        result, evidence = launcher.run_baseline_relative_materializer(
            repo_root=Path("/repo"),
            derived_plan_path=Path("/repo/effective-plan.json"),
            health_reader=health,
            sleeper=lambda _seconds: None,
            materialize_runner=runner,
            lock_factory=noop_lock,
            conflict_checker=conflicts,
            simulator_checker=simulators,
        )
        self.assertEqual(result, materializer_result)
        self.assertEqual(evidence["launch_swap_baseline_bytes"], baseline)
        self.assertEqual(evidence["launch_health"]["peak_swap_growth_bytes"], 0)
        self.assertEqual(
            evidence["materialization_health"]["peak_swap_growth_bytes"], 0
        )
        self.assertTrue(captured["reject_booted_ios_simulators"])
        self.assertEqual(captured["batch_size"], launcher.batch_runtime.BATCH_SIZE)
        self.assertEqual(
            captured["batch_timeout_seconds"],
            launcher.batch_runtime.BATCH_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            captured["monitor_interval_seconds"],
            launcher.FORMAL_MONITOR_INTERVAL_SECONDS,
        )
        self.assertEqual(
            captured["maximum_process_tree_rss_bytes"],
            launcher.RECOVERED_MAX_PROCESS_TREE_RSS_BYTES,
        )
        self.assertEqual(
            (
                launcher.materializer.runtime.exclusive_run_lock,
                launcher.materializer.runtime.runtime_health,
                launcher.materializer.runtime.enforce_health_limits,
                launcher.batch_runtime.materialize,
            ),
            originals,
        )
        self.assertEqual(conflicts.call_count, 4)
        self.assertEqual(simulators.call_count, 3)
        launcher.validate_runtime_safety(evidence)

    def test_launch_swap_growth_rejects_before_materializer(self) -> None:
        calls = 0
        baseline = 2 * 1024**3

        def health(_pid: int) -> dict[str, float | int]:
            nonlocal calls
            calls += 1
            return {
                "memory_free_percent": 80.0,
                "process_tree_rss_bytes": 1024,
                "swap_used_bytes": baseline if calls == 1 else baseline + 1,
            }

        runner = mock.Mock()
        with self.assertRaises(launcher.materializer.runtime.ResourceLimitError):
            launcher.run_baseline_relative_materializer(
                repo_root=Path("/repo"),
                derived_plan_path=Path("/repo/effective-plan.json"),
                health_reader=health,
                sleeper=lambda _seconds: None,
                materialize_runner=runner,
                lock_factory=noop_lock,
                conflict_checker=lambda: None,
                simulator_checker=lambda: 0,
            )
        runner.assert_not_called()

    def test_runtime_swap_growth_terminates_the_batch(self) -> None:
        baseline = 2 * 1024**3
        calls = 0

        def health(_pid: int) -> dict[str, float | int]:
            nonlocal calls
            calls += 1
            growth = 0 if calls <= 61 else 513 * 1024**2
            return {
                "memory_free_percent": 80.0,
                "process_tree_rss_bytes": 1024,
                "swap_used_bytes": baseline + growth,
            }

        class Process:
            pid = 12345
            exitcode = 0

            def is_alive(self) -> bool:
                return True

            def join(self, timeout: float | None = None) -> None:
                del timeout

        def runner(**kwargs: object) -> dict[str, object]:
            launcher.batch_runtime.wait_for_batch_process(
                Process(),
                health_reader=kwargs["health_reader"],
                maximum_process_tree_rss_bytes=kwargs[
                    "maximum_process_tree_rss_bytes"
                ],
                minimum_memory_free_percent=kwargs[
                    "minimum_memory_free_percent"
                ],
                maximum_swap_used_bytes=kwargs["maximum_swap_used_bytes"],
                batch_timeout_seconds=kwargs["batch_timeout_seconds"],
                monitor_interval_seconds=kwargs["monitor_interval_seconds"],
            )
            raise AssertionError("runtime growth should terminate the batch")

        with mock.patch.object(
            launcher.materializer.runtime, "terminate_page_process"
        ) as terminate:
            with self.assertRaises(launcher.materializer.runtime.ResourceLimitError):
                launcher.run_baseline_relative_materializer(
                    repo_root=Path("/repo"),
                    derived_plan_path=Path("/repo/effective-plan.json"),
                    health_reader=health,
                    sleeper=lambda _seconds: None,
                    materialize_runner=runner,
                    lock_factory=noop_lock,
                    conflict_checker=lambda: None,
                    simulator_checker=lambda: 0,
                )
            terminate.assert_called_once()

    def test_runtime_safety_validation_rejects_nonfinite_and_recovery_drift(self) -> None:
        evidence = {
            "evidence_state": "live_monitored_execution",
            "launch_health": {
                "minimum_memory_free_percent": 70.0,
                "peak_process_tree_rss_bytes": 0,
                "peak_swap_growth_bytes": 0,
                "sample_count": 61,
                "stability_sample_interval_seconds": 1.0,
                "stability_window_seconds": 60.0,
            },
            "launch_swap_baseline_bytes": 1024,
            "limits": launcher.launcher_safety_limits(),
            "materialization_health": {
                "minimum_memory_free_percent": 35.0,
                "peak_process_tree_rss_bytes": 8 * 1024**3,
                "peak_swap_growth_bytes": 512 * 1024**2,
            },
            "post_run_health": {
                "memory_free_percent": 35.0,
                "process_tree_rss_bytes": 0,
                "swap_growth_bytes": 0,
            },
            "terminal": "PASS",
        }
        launcher.validate_runtime_safety(evidence)
        nonfinite = json.loads(json.dumps(evidence))
        nonfinite["post_run_health"]["memory_free_percent"] = float("nan")
        with self.assertRaises(launcher.RecoveredMaterializationError):
            launcher.validate_runtime_safety(nonfinite)
        recovered = launcher.recovered_runtime_safety()
        recovered["unexpected"] = True
        with self.assertRaises(launcher.RecoveredMaterializationError):
            launcher.validate_runtime_safety(recovered)

    def test_formal_memory_floor_allows_39_and_terminates_below_35_percent(self) -> None:
        baseline = 2 * 1024**3
        calls = 0

        def health(_pid: int) -> dict[str, float | int]:
            nonlocal calls
            calls += 1
            free = 80.0 if calls <= 61 else 34.9
            return {
                "memory_free_percent": free,
                "process_tree_rss_bytes": 1024,
                "swap_used_bytes": baseline,
            }

        class Process:
            pid = 12345
            exitcode = 0

            def is_alive(self) -> bool:
                return True

            def join(self, timeout: float | None = None) -> None:
                del timeout

        def runner(**kwargs: object) -> dict[str, object]:
            launcher.batch_runtime.enforce_recovered_health_limits(
                {
                    "memory_free_percent": 39.0,
                    "process_tree_rss_bytes": 1024,
                    "swap_used_bytes": 0,
                },
                maximum_process_tree_rss_bytes=kwargs[
                    "maximum_process_tree_rss_bytes"
                ],
                minimum_memory_free_percent=kwargs[
                    "minimum_memory_free_percent"
                ],
                maximum_swap_used_bytes=kwargs["maximum_swap_used_bytes"],
            )
            launcher.batch_runtime.wait_for_batch_process(
                Process(),
                health_reader=kwargs["health_reader"],
                maximum_process_tree_rss_bytes=kwargs[
                    "maximum_process_tree_rss_bytes"
                ],
                minimum_memory_free_percent=kwargs[
                    "minimum_memory_free_percent"
                ],
                maximum_swap_used_bytes=kwargs["maximum_swap_used_bytes"],
                batch_timeout_seconds=kwargs["batch_timeout_seconds"],
                monitor_interval_seconds=kwargs["monitor_interval_seconds"],
            )
            raise AssertionError("free memory below the floor should terminate")

        with mock.patch.object(
            launcher.materializer.runtime, "terminate_page_process"
        ) as terminate:
            with self.assertRaises(launcher.materializer.runtime.ResourceLimitError):
                launcher.run_baseline_relative_materializer(
                    repo_root=Path("/repo"),
                    derived_plan_path=Path("/repo/effective-plan.json"),
                    health_reader=health,
                    sleeper=lambda _seconds: None,
                    materialize_runner=runner,
                    lock_factory=noop_lock,
                    conflict_checker=lambda: None,
                    simulator_checker=lambda: 0,
                )
            terminate.assert_called_once()

    def test_recovered_rss_limit_allows_observed_page_and_terminates_above_11_gib(self) -> None:
        baseline = 2 * 1024**3
        calls = 0

        def health(_pid: int) -> dict[str, float | int]:
            nonlocal calls
            calls += 1
            rss = (
                1024
                if calls <= 61
                else launcher.RECOVERED_MAX_PROCESS_TREE_RSS_BYTES + 1
            )
            return {
                "memory_free_percent": 80.0,
                "process_tree_rss_bytes": rss,
                "swap_used_bytes": baseline,
            }

        class Process:
            pid = 12345
            exitcode = 0

            def is_alive(self) -> bool:
                return True

            def join(self, timeout: float | None = None) -> None:
                del timeout

        def runner(**kwargs: object) -> dict[str, object]:
            launcher.batch_runtime.enforce_recovered_health_limits(
                {
                    "memory_free_percent": 74.0,
                    "process_tree_rss_bytes": 11_147_149_312,
                    "swap_used_bytes": 0,
                },
                maximum_process_tree_rss_bytes=kwargs[
                    "maximum_process_tree_rss_bytes"
                ],
                minimum_memory_free_percent=kwargs[
                    "minimum_memory_free_percent"
                ],
                maximum_swap_used_bytes=kwargs["maximum_swap_used_bytes"],
            )
            launcher.batch_runtime.wait_for_batch_process(
                Process(),
                health_reader=kwargs["health_reader"],
                maximum_process_tree_rss_bytes=kwargs[
                    "maximum_process_tree_rss_bytes"
                ],
                minimum_memory_free_percent=kwargs[
                    "minimum_memory_free_percent"
                ],
                maximum_swap_used_bytes=kwargs["maximum_swap_used_bytes"],
                batch_timeout_seconds=kwargs["batch_timeout_seconds"],
                monitor_interval_seconds=kwargs["monitor_interval_seconds"],
            )
            raise AssertionError("RSS above the recovered limit should terminate")

        with mock.patch.object(
            launcher.materializer.runtime, "terminate_page_process"
        ) as terminate:
            with self.assertRaises(launcher.materializer.runtime.ResourceLimitError):
                launcher.run_baseline_relative_materializer(
                    repo_root=Path("/repo"),
                    derived_plan_path=Path("/repo/effective-plan.json"),
                    health_reader=health,
                    sleeper=lambda _seconds: None,
                    materialize_runner=runner,
                    lock_factory=noop_lock,
                    conflict_checker=lambda: None,
                    simulator_checker=lambda: 0,
                )
            terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
