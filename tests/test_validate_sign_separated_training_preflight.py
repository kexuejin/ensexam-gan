import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.analysis.validate_sign_separated_training_preflight import (
    LEDGER_PATH,
    TRAINING_PLAN_PATH,
    PreflightError,
    run_preflight,
    validate_outputs_absent,
)


ROOT = Path(__file__).resolve().parents[1]


class SignSeparatedTrainingPreflightTest(unittest.TestCase):
    def write_json(self, path: Path, value: dict) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def mutated_plan(self, root: Path) -> tuple[dict, Path]:
        plan = json.loads((ROOT / TRAINING_PLAN_PATH).read_text(encoding="utf-8"))
        path = root / "training-plan.json"
        return plan, path

    def mutated_ledger(self, root: Path) -> tuple[dict, Path]:
        ledger = json.loads((ROOT / LEDGER_PATH).read_text(encoding="utf-8"))
        ledger["active_iteration"] = {
            "id": "sign-separated-residual-repair",
            "prerequisites": [
                {
                    "id": "sign_separated_residual_synthetic_preflight",
                    "status": "passed",
                },
                {
                    "id": "sign_separated_residual_data_role_preflight",
                    "status": "passed",
                },
                {
                    "id": "sign_separated_residual_training_preflight",
                    "status": "passed",
                },
            ],
        }
        path = root / "ledger.json"
        return ledger, path

    def test_killed_family_cannot_reopen_training_preflight(self) -> None:
        result = run_preflight(repo_root=ROOT)
        self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED", result)
        self.assertFalse(result["runnable"])
        self.assertIn("active iteration", result["reason"])

    def test_training_preflight_can_be_reverified_after_ledger_pass(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger, ledger_path = self.mutated_ledger(root)
            prerequisite = next(
                item
                for item in ledger["active_iteration"]["prerequisites"]
                if item["id"] == "sign_separated_residual_training_preflight"
            )
            prerequisite["status"] = "passed"
            self.write_json(ledger_path, ledger)
            result = run_preflight(repo_root=ROOT, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PASS", result)
            self.assertEqual(
                result["ledger_authority"]["training_preflight_ledger_status"],
                "passed",
            )

    def test_learning_rate_drift_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            plan, plan_path = self.mutated_plan(root)
            plan["trainer"]["lr"] = 1.0e-4
            self.write_json(plan_path, plan)
            result = run_preflight(
                repo_root=ROOT, training_plan_path=plan_path
            )
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("trainer changed", result["reason"])

    def test_validation_enablement_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            plan, plan_path = self.mutated_plan(root)
            plan["trainer"]["validation_enabled"] = True
            self.write_json(plan_path, plan)
            result = run_preflight(
                repo_root=ROOT, training_plan_path=plan_path
            )
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("trainer changed", result["reason"])

    def test_trainer_hash_mismatch_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            plan, plan_path = self.mutated_plan(root)
            _ledger, ledger_path = self.mutated_ledger(root)
            plan["evidence"]["trainer"]["sha256"] = "0" * 64
            self.write_json(plan_path, plan)
            self.write_json(ledger_path, _ledger)
            result = run_preflight(
                repo_root=ROOT,
                training_plan_path=plan_path,
                ledger_path=ledger_path,
            )
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("artifact hash mismatch", result["reason"])

    def test_data_role_authority_must_remain_passed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger, ledger_path = self.mutated_ledger(root)
            prerequisite = next(
                item
                for item in ledger["active_iteration"]["prerequisites"]
                if item["id"] == "sign_separated_residual_data_role_preflight"
            )
            prerequisite["status"] = "pending"
            self.write_json(ledger_path, ledger)
            result = run_preflight(repo_root=ROOT, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("data-role prerequisite", result["reason"])

    def test_missing_data_role_record_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger, ledger_path = self.mutated_ledger(root)
            ledger["records"] = [
                record
                for record in ledger["records"]
                if record.get("id") != "sign-separated-residual-data-role-preflight"
            ]
            self.write_json(ledger_path, ledger)
            result = run_preflight(repo_root=ROOT, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("data-role PASS record", result["reason"])

    def test_existing_planned_output_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            existing = root / "already-there"
            existing.mkdir()
            plan = {
                "planned_outputs_must_be_absent": {
                    "training_output_dir": "already-there"
                }
            }
            with self.assertRaisesRegex(PreflightError, "must be absent"):
                validate_outputs_absent(root, plan)


if __name__ == "__main__":
    unittest.main()
