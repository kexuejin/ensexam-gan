from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


from scripts.analysis import reconstruct_external_text_layout_frozen_caches as reconstruction


def probe_gate() -> dict[str, object]:
    return {
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
            "page_timeout_seconds": reconstruction.runtime.PAGE_TIMEOUT_SECONDS,
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


def reconstruction_gate() -> dict[str, object]:
    return {
        "maximum_process_tree_rss_bytes": reconstruction.runtime.MAX_DETECTOR_RSS_BYTES,
        "maximum_swap_used_bytes": reconstruction.runtime.MAX_SWAP_USED_BYTES,
        "minimum_system_free_memory_percent": reconstruction.runtime.MIN_MEMORY_FREE_PERCENT,
        "probe_pass_required_before_helper_import": True,
        "stage_execution": (
            "one_model_process_at_a_time_under_the_external_layout_host_lock"
        ),
    }


def monitor_contract() -> dict[str, object]:
    return {"monitor": reconstruction.expected_runtime_monitor()}


def authority_ledger() -> dict[str, object]:
    return {
        "active_iteration": {
            "prerequisites": [
                {
                    "id": "materially_new_support_successor_preregistration_v4",
                    "status": "passed",
                },
                {
                    "id": (
                        "external_text_layout_tiled_probe_cache_reconstruction_"
                        "v2_preregistration"
                    ),
                    "status": "passed",
                },
                {
                    "id": (
                        "external_text_layout_cache_reconstruction_runtime_"
                        "monitor_preregistration"
                    ),
                    "status": "passed",
                },
                {
                    "id": "external_text_layout_support_train_only_diagnostic",
                    "status": "pending",
                },
            ],
            "terminal": "PREREQUISITE_NEEDED",
        },
        "program": {
            "product_default": "artifacts/current-primary",
            "promotion_state": "disabled",
            "reserved_blind_state": "disabled",
        },
    }


def safe_probe_result() -> dict[str, object]:
    safe_health = {
        "memory_free_percent": 80.0,
        "process_tree_rss_bytes": 1024,
        "swap_used_bytes": 0,
    }
    return {
        "attempt_count": 1,
        "booted_ios_simulator_count": 0,
        "contract": probe_gate()["contract"],
        "detector": {
            "device": "cpu",
            "engine": "transformers",
            "model_name": "PP-OCRv6_medium_det",
        },
        "formal_evidence": False,
        "formal_outputs_written": False,
        "initial_health": dict(safe_health),
        "label_access": False,
        "page": {
            "file": "hw5k_1011.jpg",
            "source_sha256": reconstruction.EXPECTED_PROBE_SOURCE_SHA256,
        },
        "page_completed": True,
        "page_health": {
            "minimum_memory_free_percent": 50.0,
            "peak_process_tree_rss_bytes": 2048,
            "peak_swap_used_bytes": 0,
        },
        "post_run_health": dict(safe_health),
        "probe": "external_text_layout_tiled_9x9_single_page_runtime_safety",
        "reason_code": "runtime_safety_probe_passed",
        "recognition": False,
        "residual_model_process_count": 0,
        "result_authority": "runtime_prerequisite_only",
        "routing_metadata_access": False,
        "safety_limits": probe_gate()["required_safety_limits"],
        "schema_version": 1,
        "target_access": False,
        "temporary_page_outputs_retained": False,
        "thread_caps": probe_gate()["required_thread_caps"],
        "terminal": "PASS",
    }


def write_result(root: Path, relative: str, result: dict[str, object]) -> None:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result), encoding="utf-8")


def write_probe(root: Path, result: dict[str, object]) -> None:
    write_result(root, str(probe_gate()["probe_result"]), result)


def write_cache(
    root: Path,
    relative: str,
    manifest_lines: list[str],
    marker: str,
) -> dict[str, object]:
    cache = root / relative
    predictions = cache / "pred"
    predictions.mkdir(parents=True)
    metrics = cache / "metrics.csv"
    metrics.write_text(f"file,value\n{marker},1\n", encoding="utf-8")
    names = reconstruction.expected_prediction_names(manifest_lines)
    for name in names:
        (predictions / name).write_bytes(f"{marker}:{name}".encode("utf-8"))
    rows = [
        f"{name} {reconstruction.sha256_file(predictions / name)}"
        for name in names
    ]
    return {
        "metrics_sha256": reconstruction.sha256_file(metrics),
        "prediction_set": {
            "content_sha256": reconstruction.sha256_rows(rows),
            "count": len(names),
            "filename_sha256": reconstruction.sha256_rows(names),
        },
    }


def cache_state(root: Path) -> dict[str, object]:
    del root
    manifest_lines = ["sources/page-a.jpg", "sources/page-b.jpeg"]
    return {
        "contract": {
            "build": {
                "archive_manifest": "archive/train.txt",
                "archive_primary": "archive/primary",
                "archive_second_stage": "archive/second-stage",
                "primary": "build/primary",
                "second_stage": "build/second-stage",
            },
            "expected_outputs": {},
            "historical_runtime": dict(
                reconstruction.EXPECTED_HISTORICAL_RUNTIME
            ),
            "probe_gate": probe_gate(),
            "reconstruction_gate": reconstruction_gate(),
        },
        "manifest_lines": manifest_lines,
        "monitor_contract": monitor_contract(),
        "plan": {
            "pipeline_preparation": {
                "primary": {"samples_file": "original/train.txt"},
                "second_stage": {
                    "primary_pred_dir": "build/primary/pred",
                    "samples_file": "original/train.txt",
                },
            }
        },
    }


@contextmanager
def unlocked(_path: Path):
    yield


class ExternalTextLayoutFrozenCacheReconstructionTest(unittest.TestCase):
    def test_authority_requires_v2_and_monitor_preregistrations(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = root / reconstruction.LEDGER_PATH
            ledger_path.parent.mkdir(parents=True)
            ledger = authority_ledger()
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            contract = {"authority": dict(reconstruction.EXPECTED_AUTHORITY)}
            reconstruction.validate_authority(root, contract)

            changed = copy.deepcopy(ledger)
            changed["active_iteration"]["prerequisites"][1]["status"] = "pending"
            ledger_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError,
                "external layout authority changed",
            ):
                reconstruction.validate_authority(root, contract)

            changed = copy.deepcopy(ledger)
            changed["active_iteration"]["prerequisites"][2]["status"] = "pending"
            ledger_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError,
                "external layout authority changed",
            ):
                reconstruction.validate_authority(root, contract)

    def test_repository_contract_agrees_with_frozen_boundaries_and_limits(self) -> None:
        contract_path = reconstruction.ROOT / reconstruction.CONTRACT_PATH
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertEqual(
            reconstruction.sha256_file(contract_path),
            reconstruction.EXPECTED_CONTRACT_SHA256,
        )
        reconstruction.validate_reconstruction_boundaries(contract)

        changed = copy.deepcopy(contract)
        changed["probe_gate"]["maximum_swap_used_bytes"] += 1
        with self.assertRaisesRegex(
            reconstruction.CacheReconstructionError, "tiled probe safety gate changed"
        ):
            reconstruction.validate_reconstruction_boundaries(changed)

        changed = copy.deepcopy(contract)
        changed["reconstruction_gate"]["maximum_swap_used_bytes"] += 1
        with self.assertRaisesRegex(
            reconstruction.CacheReconstructionError,
            "reconstruction safety gate changed",
        ):
            reconstruction.validate_reconstruction_boundaries(changed)

        monitor_path = reconstruction.ROOT / reconstruction.MONITOR_CONTRACT_PATH
        self.assertEqual(
            reconstruction.sha256_file(monitor_path),
            reconstruction.EXPECTED_MONITOR_CONTRACT_SHA256,
        )
        registered_monitor = reconstruction.validate_runtime_monitor_contract(
            reconstruction.ROOT
        )
        changed_monitor = copy.deepcopy(registered_monitor)
        changed_monitor["monitor"]["maximum_swap_used_bytes"] += 1
        with self.assertRaisesRegex(
            reconstruction.CacheReconstructionError,
            "runtime monitor changed",
        ):
            reconstruction.validate_runtime_monitor_settings(changed_monitor)

    def test_probe_must_exist_and_pass_with_complete_nonnegative_health(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = {"probe_gate": probe_gate()}
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError, "probe is missing"
            ):
                reconstruction.validate_probe_pass(root, contract)

            result = safe_probe_result()
            write_probe(root, result)
            self.assertEqual(
                reconstruction.validate_probe_pass(root, contract), result
            )

            old_result = copy.deepcopy(result)
            old_result["probe"] = "external_text_layout_single_page_runtime_safety"
            write_result(
                root,
                "outputs/external-text-layout-runtime-safety-probe-repaired-20260814/result.json",
                old_result,
            )
            (root / str(probe_gate()["probe_result"])).unlink()
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError,
                "tiled one-page runtime probe is missing",
            ):
                reconstruction.validate_probe_pass(root, contract)

            write_probe(root, result)
            old_identity = copy.deepcopy(result)
            old_identity["probe"] = "external_text_layout_single_page_runtime_safety"
            write_probe(root, old_identity)
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError,
                "tiled one-page runtime probe did not pass",
            ):
                reconstruction.validate_probe_pass(root, contract)

            old_schema = copy.deepcopy(result)
            old_schema["safety_limits"] = {
                "detector_process_tree_rss_bytes_max": (
                    reconstruction.runtime.MAX_DETECTOR_RSS_BYTES
                ),
                "memory_free_percent_min": reconstruction.runtime.MIN_MEMORY_FREE_PERCENT,
                "page_timeout_seconds": reconstruction.runtime.PAGE_TIMEOUT_SECONDS,
                "swap_used_bytes_max": reconstruction.runtime.MAX_SWAP_USED_BYTES,
            }
            write_probe(root, old_schema)
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError,
                "tiled one-page probe limits changed",
            ):
                reconstruction.validate_probe_pass(root, contract)

            write_probe(root, result)
            wrong_contract = copy.deepcopy(result)
            wrong_contract["contract"]["sha256"] = "0" * 64
            write_probe(root, wrong_contract)
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError,
                "tiled one-page runtime probe did not pass",
            ):
                reconstruction.validate_probe_pass(root, contract)

            for field in ("page_completed", "residual_model_process_count"):
                with self.subTest(field=field):
                    missing_completion = copy.deepcopy(result)
                    del missing_completion[field]
                    write_probe(root, missing_completion)
                    with self.assertRaisesRegex(
                        reconstruction.CacheReconstructionError,
                        "tiled one-page runtime probe did not pass",
                    ):
                        reconstruction.validate_probe_pass(root, contract)

            write_probe(root, result)
            missing_attempt = copy.deepcopy(result)
            del missing_attempt["attempt_count"]
            write_probe(root, missing_attempt)
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError,
                "tiled one-page runtime probe did not pass",
            ):
                reconstruction.validate_probe_pass(root, contract)

            write_probe(root, result)
            wrong_threads = copy.deepcopy(result)
            wrong_threads["thread_caps"]["OMP_NUM_THREADS"] = "2"
            write_probe(root, wrong_threads)
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError,
                "tiled one-page probe thread caps changed",
            ):
                reconstruction.validate_probe_pass(root, contract)

            missing = copy.deepcopy(result)
            del missing["page_health"]["peak_swap_used_bytes"]
            write_probe(root, missing)
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError,
                "probe page health lacks peak_swap_used_bytes",
            ):
                reconstruction.validate_probe_pass(root, contract)

            negative = copy.deepcopy(result)
            negative["post_run_health"]["process_tree_rss_bytes"] = -1
            write_probe(root, negative)
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError,
                "probe post_run_health has invalid process_tree_rss_bytes",
            ):
                reconstruction.validate_probe_pass(root, contract)

            unsafe = copy.deepcopy(result)
            unsafe["initial_health"]["swap_used_bytes"] = (
                reconstruction.runtime.MAX_SWAP_USED_BYTES + 1
            )
            write_probe(root, unsafe)
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError,
                "probe initial_health crossed a runtime limit",
            ):
                reconstruction.validate_probe_pass(root, contract)

    def test_probe_source_and_numeric_evidence_must_be_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = {"probe_gate": probe_gate()}
            result = safe_probe_result()

            for source_sha256 in (None, "0" * 64):
                with self.subTest(source_sha256=source_sha256):
                    changed = copy.deepcopy(result)
                    if source_sha256 is None:
                        del changed["page"]["source_sha256"]
                    else:
                        changed["page"]["source_sha256"] = source_sha256
                    write_probe(root, changed)
                    with self.assertRaisesRegex(
                        reconstruction.CacheReconstructionError,
                        "tiled one-page runtime probe did not pass",
                    ):
                        reconstruction.validate_probe_pass(root, contract)

            for field, value in (
                ("attempt_count", True),
                ("residual_model_process_count", False),
                ("booted_ios_simulator_count", False),
            ):
                with self.subTest(field=field):
                    changed = copy.deepcopy(result)
                    changed[field] = value
                    write_probe(root, changed)
                    with self.assertRaisesRegex(
                        reconstruction.CacheReconstructionError,
                        "tiled one-page runtime probe did not pass",
                    ):
                        reconstruction.validate_probe_pass(root, contract)

    def test_prediction_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            predictions = root / "pred"
            predictions.mkdir()
            (predictions / "page.png").write_bytes(b"prediction")
            expected = {
                "content_sha256": "0" * 64,
                "count": 1,
                "filename_sha256": reconstruction.sha256_rows(["page.png"]),
            }
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError,
                "prediction content changed",
            ):
                reconstruction.validate_prediction_set(
                    predictions, ["page.png"], expected
                )

    def test_historical_runtime_must_match_before_reconstruction(self) -> None:
        contract = {
            "historical_runtime": dict(
                reconstruction.EXPECTED_HISTORICAL_RUNTIME
            )
        }
        expected = dict(reconstruction.EXPECTED_HISTORICAL_RUNTIME)
        self.assertEqual(
            reconstruction.validate_reconstruction_runtime(
                contract, identity_reader=lambda: expected
            ),
            expected,
        )
        changed = {**expected, "torch": "different"}
        with self.assertRaisesRegex(
            reconstruction.CacheReconstructionError,
            "historical reconstruction runtime changed",
        ):
            reconstruction.validate_reconstruction_runtime(
                contract, identity_reader=lambda: changed
            )

    def test_commands_use_archive_manifest_and_original_build_paths(self) -> None:
        state = cache_state(Path("unused"))
        observed: dict[str, object] = {}

        class FakeHelper:
            @staticmethod
            def primary_command(repo_root, plan, output_dir):
                observed["primary"] = (repo_root, plan, output_dir)
                return [plan["pipeline_preparation"]["primary"]["samples_file"]]

            @staticmethod
            def second_stage_command(repo_root, plan, output_dir):
                observed["second_stage"] = (repo_root, plan, output_dir)
                config = plan["pipeline_preparation"]["second_stage"]
                return [config["samples_file"], config["primary_pred_dir"]]

        primary = reconstruction.build_stage_command(
            repo_root=Path("/repo"),
            state=state,
            stage="primary",
            output_dir=Path("/repo/build/.primary.materializing"),
            helper=FakeHelper,
        )
        second_stage = reconstruction.build_stage_command(
            repo_root=Path("/repo"),
            state=state,
            stage="second_stage",
            output_dir=Path("/repo/build/.second-stage.materializing"),
            helper=FakeHelper,
        )

        self.assertEqual(primary, ["archive/train.txt"])
        self.assertEqual(
            second_stage, ["archive/train.txt", "build/primary/pred"]
        )
        self.assertEqual(
            observed["primary"][2], Path("/repo/build/.primary.materializing")
        )
        self.assertEqual(
            observed["second_stage"][2],
            Path("/repo/build/.second-stage.materializing"),
        )

    def test_monitored_command_publishes_atomically_and_rewrites_metrics(self) -> None:
        from scripts.analysis import materialize_sign_separated_train_inputs as helper

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            final_dir = root / "build/cache"
            temporary = final_dir.with_name(f".{final_dir.name}.materializing")
            command = [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; import sys; "
                    "p=Path(sys.argv[1]); (p/'pred').mkdir(parents=True); "
                    "(p/'metrics.csv').write_text('path\\n'+str(p)+'\\n', "
                    "encoding='utf-8')"
                ),
                str(temporary),
            ]
            actual_command, _health = (
                reconstruction.run_monitored_atomic_directory_command(
                    repo_root=root,
                    final_dir=final_dir,
                    command_builder=lambda _temporary: command,
                    log_path=root / "control/success.log",
                    monitor_contract=monitor_contract(),
                    helper=helper,
                    health_reader=lambda _pid: {
                        "memory_free_percent": 80.0,
                        "process_tree_rss_bytes": 1024,
                        "swap_used_bytes": 0,
                    },
                )
            )

            self.assertEqual(actual_command, command)
            self.assertTrue(final_dir.is_dir())
            self.assertFalse(temporary.exists())
            metrics = (final_dir / "metrics.csv").read_text(encoding="utf-8")
            self.assertIn(str(final_dir), metrics)
            self.assertNotIn(str(temporary), metrics)

    def test_nonzero_monitored_command_never_publishes_final_cache(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            final_dir = root / "build/cache"
            with self.assertRaisesRegex(RuntimeError, r"failed \(7\)"):
                reconstruction.run_monitored_atomic_directory_command(
                    repo_root=root,
                    final_dir=final_dir,
                    command_builder=lambda _temporary: [
                        sys.executable,
                        "-c",
                        "raise SystemExit(7)",
                    ],
                    log_path=root / "control/nonzero.log",
                    monitor_contract=monitor_contract(),
                    helper=mock.Mock(),
                    health_reader=lambda _pid: {
                        "memory_free_percent": 80.0,
                        "process_tree_rss_bytes": 1024,
                        "swap_used_bytes": 0,
                    },
                )
            self.assertFalse(final_dir.exists())

    def test_monitor_failures_terminate_the_child_process_group(self) -> None:
        failures = (
            (
                "resource_limit",
                lambda _pid: {
                    "memory_free_percent": 80.0,
                    "process_tree_rss_bytes": (
                        reconstruction.runtime.MAX_DETECTOR_RSS_BYTES + 1
                    ),
                    "swap_used_bytes": 0,
                },
                reconstruction.CacheReconstructionError,
            ),
            (
                "health_reader",
                mock.Mock(side_effect=OSError("health reader failed")),
                OSError,
            ),
        )
        for name, health_reader, error_type in failures:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                final_dir = root / "build/cache"
                process = mock.Mock(pid=43210)
                process.poll.return_value = None
                process.wait.return_value = 0
                popen_factory = mock.Mock(return_value=process)
                with mock.patch.object(
                    reconstruction.os,
                    "killpg",
                    side_effect=[None, ProcessLookupError()],
                ) as killpg:
                    with self.assertRaises(error_type):
                        reconstruction.run_monitored_atomic_directory_command(
                            repo_root=root,
                            final_dir=final_dir,
                            command_builder=lambda _temporary: ["synthetic"],
                            log_path=root / f"control/{name}.log",
                            monitor_contract=monitor_contract(),
                            helper=mock.Mock(),
                            health_reader=health_reader,
                            popen_factory=popen_factory,
                        )
                popen_factory.assert_called_once()
                self.assertTrue(
                    popen_factory.call_args.kwargs["start_new_session"]
                )
                killpg.assert_has_calls(
                    [mock.call(process.pid, signal.SIGTERM), mock.call(process.pid, 0)]
                )
                process.wait.assert_called_once_with(timeout=5.0)
                self.assertFalse(final_dir.exists())

    def test_process_group_termination_escalates_to_sigkill(self) -> None:
        process = mock.Mock(pid=54321)
        process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="synthetic", timeout=5.0),
            0,
        ]
        with mock.patch.object(reconstruction.os, "killpg") as killpg:
            reconstruction.terminate_monitored_process_group(
                process,
                grace_seconds=5.0,
            )

        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(process.pid, signal.SIGTERM),
                mock.call(process.pid, 0),
                mock.call(process.pid, signal.SIGKILL),
            ],
        )
        self.assertEqual(
            process.wait.call_args_list,
            [mock.call(timeout=5.0), mock.call(timeout=5.0)],
        )

    def test_publication_is_relative_idempotent_and_preflights_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = cache_state(root)
            expected = state["contract"]["expected_outputs"]
            expected["primary"] = write_cache(
                root, "build/primary", state["manifest_lines"], "primary"
            )
            expected["second_stage"] = write_cache(
                root,
                "build/second-stage",
                state["manifest_lines"],
                "second-stage",
            )

            first = reconstruction.publish_caches(root, state)
            second = reconstruction.publish_caches(root, state)
            self.assertEqual(first, second)
            for link in (root / "archive/primary", root / "archive/second-stage"):
                self.assertTrue(link.is_symlink())
                self.assertFalse(Path(os.readlink(link)).is_absolute())
            self.assertEqual(
                reconstruction.verify_published_caches(root, state)["status"],
                "verified",
            )

            absolute = root / "archive/absolute-primary"
            absolute.symlink_to(root / "build/primary", target_is_directory=True)
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError,
                "not the registered relative symlink",
            ):
                reconstruction.validate_publication_destination(
                    absolute, root / "build/primary"
                )

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = cache_state(root)
            expected = state["contract"]["expected_outputs"]
            expected["primary"] = write_cache(
                root, "build/primary", state["manifest_lines"], "primary"
            )
            expected["second_stage"] = write_cache(
                root,
                "build/second-stage",
                state["manifest_lines"],
                "second-stage",
            )
            conflict = root / "archive/second-stage"
            conflict.mkdir(parents=True)
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError,
                "already has different content",
            ):
                reconstruction.publish_caches(root, state)
            self.assertFalse((root / "archive/primary").exists())

    def test_execution_gates_run_before_helper_loading(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = cache_state(root)
            state["contract"]["expected_outputs"] = {
                "primary": {},
                "second_stage": {},
            }
            helper_loader = mock.Mock()

            unsafe_probe = safe_probe_result()
            unsafe_probe["page_health"]["peak_swap_used_bytes"] = (
                reconstruction.runtime.MAX_SWAP_USED_BYTES + 1
            )
            write_probe(root, unsafe_probe)
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError,
                "probe page health crossed a runtime limit",
            ):
                reconstruction.reconstruct_stage(
                    repo_root=root,
                    state=state,
                    stage="primary",
                    helper_loader=helper_loader,
                )
            helper_loader.assert_not_called()

            write_probe(root, safe_probe_result())
            with (
                mock.patch.object(
                    reconstruction.runtime,
                    "exclusive_run_lock",
                    side_effect=unlocked,
                ),
                mock.patch.object(
                    reconstruction,
                    "validate_reconstruction_runtime",
                    return_value=dict(
                        reconstruction.EXPECTED_HISTORICAL_RUNTIME
                    ),
                ),
                mock.patch.object(
                    reconstruction,
                    "validate_current_launch_health",
                    side_effect=reconstruction.CacheReconstructionError(
                        "current launch health crossed a runtime limit"
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    reconstruction.CacheReconstructionError,
                    "current launch health crossed a runtime limit",
                ):
                    reconstruction.reconstruct_stage(
                        repo_root=root,
                        state=state,
                        stage="primary",
                        helper_loader=helper_loader,
                    )
            helper_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
