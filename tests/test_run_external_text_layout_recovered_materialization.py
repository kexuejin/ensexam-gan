from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


from scripts.analysis import run_external_text_layout_recovered_materialization as launcher


class RunExternalTextLayoutRecoveredMaterializationTest(unittest.TestCase):
    def test_repository_contract_and_shared_hashes_are_exact(self) -> None:
        contract = launcher.validate_repository_contract(launcher.ROOT)
        self.assertEqual(contract["schema_version"], 2)
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

    def test_execution_authority_requires_v2_integration_pass(self) -> None:
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
                    "id": "external_text_layout_recovered_materializer_launch_v2_integration",
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
                mock.patch.object(launcher.materializer, "materialize") as materialize,
            ):
                with self.assertRaises(launcher.RecoveredMaterializationError):
                    launcher.run_launcher(root)
            self.assertFalse((root / "generated/effective-plan.json").exists())
            materialize.assert_not_called()

    def test_launcher_calls_unchanged_materializer_with_exact_plan(self) -> None:
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
                    launcher.materializer,
                    "materialize",
                    return_value={"terminal": "PASS"},
                ) as materialize,
                mock.patch.object(
                    launcher, "validate_materialization_output", return_value=summary
                ),
            ):
                result = launcher.run_launcher(root)
            materialize.assert_called_once_with(
                repo_root=root,
                plan_path=Path("generated/effective-plan.json"),
                ledger_path=launcher.LEDGER_PATH,
                worker_count=1,
            )
            self.assertEqual(result["terminal"], "PASS")
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
            terminal = {"schema_version": 2, "terminal": "PASS"}
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
                mock.patch.object(launcher.materializer, "materialize") as materialize,
            ):
                result = launcher.run_launcher(root)
                materialize.assert_not_called()
            self.assertEqual(result["materialization"], summary)
            self.assertTrue((root / launcher.RESULT_PATH).is_file())


if __name__ == "__main__":
    unittest.main()
