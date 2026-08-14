from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


from scripts.analysis import reconstruct_external_text_layout_frozen_caches as reconstruction


def runtime_gate() -> dict[str, object]:
    return {
        "maximum_process_tree_rss_bytes": reconstruction.runtime.MAX_DETECTOR_RSS_BYTES,
        "maximum_swap_used_bytes": reconstruction.runtime.MAX_SWAP_USED_BYTES,
        "minimum_system_free_memory_percent": reconstruction.runtime.MIN_MEMORY_FREE_PERCENT,
        "probe_page": "hw5k_1011.jpg",
        "probe_result": (
            "outputs/external-text-layout-runtime-safety-probe-repaired-20260814/"
            "result.json"
        ),
        "required_probe_reason_code": "runtime_safety_probe_passed",
        "required_probe_terminal": "PASS",
        "stage_execution": (
            "one_model_process_at_a_time_under_the_external_layout_host_lock"
        ),
    }


def safe_probe_result() -> dict[str, object]:
    safe_health = {
        "memory_free_percent": 80.0,
        "process_tree_rss_bytes": 1024,
        "swap_used_bytes": 0,
    }
    return {
        "detector": {"device": "cpu"},
        "formal_evidence": False,
        "formal_outputs_written": False,
        "initial_health": dict(safe_health),
        "label_access": False,
        "page": {"file": "hw5k_1011.jpg"},
        "page_health": {
            "minimum_memory_free_percent": 70.0,
            "peak_process_tree_rss_bytes": 2048,
            "peak_swap_used_bytes": 0,
        },
        "post_run_health": dict(safe_health),
        "probe": "external_text_layout_single_page_runtime_safety",
        "reason_code": "runtime_safety_probe_passed",
        "recognition": False,
        "result_authority": "runtime_prerequisite_only",
        "routing_metadata_access": False,
        "safety_limits": {
            "detector_process_tree_rss_bytes_max": (
                reconstruction.runtime.MAX_DETECTOR_RSS_BYTES
            ),
            "memory_free_percent_min": reconstruction.runtime.MIN_MEMORY_FREE_PERCENT,
            "page_timeout_seconds": reconstruction.runtime.PAGE_TIMEOUT_SECONDS,
            "swap_used_bytes_max": reconstruction.runtime.MAX_SWAP_USED_BYTES,
        },
        "schema_version": 1,
        "target_access": False,
        "temporary_page_outputs_retained": False,
        "terminal": "PASS",
    }


def write_probe(root: Path, result: dict[str, object]) -> None:
    destination = root / str(runtime_gate()["probe_result"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result), encoding="utf-8")


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
            "runtime_gate": runtime_gate(),
        },
        "manifest_lines": manifest_lines,
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
    def test_repository_contract_agrees_with_frozen_boundaries_and_limits(self) -> None:
        contract_path = reconstruction.ROOT / reconstruction.CONTRACT_PATH
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertEqual(
            reconstruction.sha256_file(contract_path),
            reconstruction.EXPECTED_CONTRACT_SHA256,
        )
        reconstruction.validate_reconstruction_boundaries(contract)

        changed = copy.deepcopy(contract)
        changed["runtime_gate"]["maximum_swap_used_bytes"] += 1
        with self.assertRaisesRegex(
            reconstruction.CacheReconstructionError, "runtime safety gate changed"
        ):
            reconstruction.validate_reconstruction_boundaries(changed)

    def test_probe_must_exist_and_pass_with_complete_nonnegative_health(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = {"runtime_gate": runtime_gate()}
            with self.assertRaisesRegex(
                reconstruction.CacheReconstructionError, "probe is missing"
            ):
                reconstruction.validate_probe_pass(root, contract)

            result = safe_probe_result()
            write_probe(root, result)
            self.assertEqual(
                reconstruction.validate_probe_pass(root, contract), result
            )

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
