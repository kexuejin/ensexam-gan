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


class ExternalTextLayoutRuntimeSafetyProbeTest(unittest.TestCase):
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

            def resource_failure(**_kwargs):
                raise MaterializationError(
                    "system memory safety limit crossed: 31.0% free"
                )

            with (
                mock.patch.object(probe.materializer, "validate_plan"),
                mock.patch.object(probe.materializer, "validate_authority"),
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
            self.assertEqual(lock_calls, [lock_path])
            self.assertEqual(len(health_calls), 2)
            conflict_check.assert_called_once_with()
            self.assertTrue((root / result_path).is_file())
            self.assertFalse((root / "outputs/formal-materialization").exists())
            self.assertFalse((root / "outputs/formal-audit").exists())
            self.assertFalse(temporary_root.exists())

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


if __name__ == "__main__":
    unittest.main()
