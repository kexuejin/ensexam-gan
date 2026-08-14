import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.analysis.validate_monotonic_residual_erase_training_preflight import (
    LEDGER_PATH,
    TRAINING_PLAN_PATH,
    PreflightError,
    run_preflight,
    validate_outputs_absent,
)


ROOT = Path(__file__).resolve().parents[1]


class MonotonicResidualEraseTrainingPreflightTest(unittest.TestCase):
    def write_json(self, path: Path, value: dict) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def mutated_plan(self, root: Path) -> tuple[dict, Path]:
        plan = json.loads((ROOT / TRAINING_PLAN_PATH).read_text(encoding="utf-8"))
        return plan, root / "training-plan.json"

    def mutated_ledger(self, root: Path) -> tuple[dict, Path]:
        ledger = json.loads((ROOT / LEDGER_PATH).read_text(encoding="utf-8"))
        return ledger, root / "ledger.json"

    def test_current_registered_preflight_replays_after_materialization(self) -> None:
        result = run_preflight(repo_root=ROOT)
        self.assertEqual(result["terminal"], "PASS", result)
        self.assertTrue(result["runnable"])
        self.assertTrue(result["metadata_only"])
        self.assertEqual(result["train_files"]["effective_train_count"], 275)
        self.assertTrue(result["target_patch_materialized"])
        self.assertFalse(result["training_started"])
        self.assertFalse(result["checkpoint_generated"])

    def test_preflight_can_be_reverified_after_ledger_pass(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger, ledger_path = self.mutated_ledger(root)
            prerequisite = next(
                item
                for item in ledger["active_iteration"]["prerequisites"]
                if item["id"] == "monotonic_residual_erase_training_preflight"
            )
            prerequisite["status"] = "passed"
            self.write_json(ledger_path, ledger)
            result = run_preflight(repo_root=ROOT, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PASS", result)
            self.assertEqual(
                result["ledger_authority"]["training_preflight_ledger_status"],
                "passed",
            )

    def test_materialized_inputs_fail_closed_while_audit_is_pending(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger, ledger_path = self.mutated_ledger(root)
            prerequisite = next(
                item
                for item in ledger["active_iteration"]["prerequisites"]
                if item["id"]
                == "monotonic_residual_erase_train_materialization_audit"
            )
            prerequisite["status"] = "pending"
            self.write_json(ledger_path, ledger)
            result = run_preflight(repo_root=ROOT, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("planned output must be absent", result["reason"])

    def test_learning_rate_drift_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            plan, path = self.mutated_plan(root)
            plan["trainer"]["lr"] = 1.0e-4
            self.write_json(path, plan)
            result = run_preflight(repo_root=ROOT, training_plan_path=path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("trainer changed", result["reason"])

    def test_class_balance_drift_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            plan, path = self.mutated_plan(root)
            plan["trainer"]["class_balance"] = "global_pixel_mean"
            self.write_json(path, plan)
            result = run_preflight(repo_root=ROOT, training_plan_path=path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("trainer changed", result["reason"])

    def test_dark_or_route_patch_selection_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            plan, path = self.mutated_plan(root)
            plan["patch_builder"]["selection"] = "direction_balanced"
            self.write_json(path, plan)
            result = run_preflight(repo_root=ROOT, training_plan_path=path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("patch_builder changed", result["reason"])

    def test_trainer_hash_mismatch_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            plan, path = self.mutated_plan(root)
            plan["evidence"]["trainer"]["sha256"] = "0" * 64
            self.write_json(path, plan)
            result = run_preflight(repo_root=ROOT, training_plan_path=path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("artifact hash mismatch", result["reason"])

    def test_data_role_authority_must_remain_passed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger, path = self.mutated_ledger(root)
            prerequisite = next(
                item
                for item in ledger["active_iteration"]["prerequisites"]
                if item["id"] == "monotonic_residual_erase_data_role_preflight"
            )
            prerequisite["status"] = "pending"
            self.write_json(path, ledger)
            result = run_preflight(repo_root=ROOT, ledger_path=path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("required prerequisite", result["reason"])

    def test_missing_data_role_record_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger, path = self.mutated_ledger(root)
            ledger["records"] = [
                record
                for record in ledger["records"]
                if record.get("id") != "monotonic-residual-erase-data-role-preflight"
            ]
            self.write_json(path, ledger)
            result = run_preflight(repo_root=ROOT, ledger_path=path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("data-role PASS record", result["reason"])

    def test_existing_planned_output_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "already-there").mkdir()
            plan = {
                "planned_outputs_must_be_absent": {
                    "training_output_dir": "already-there"
                }
            }
            with self.assertRaisesRegex(PreflightError, "must be absent"):
                validate_outputs_absent(root, plan)


if __name__ == "__main__":
    unittest.main()
