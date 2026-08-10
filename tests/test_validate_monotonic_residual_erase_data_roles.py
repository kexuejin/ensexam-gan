import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.analysis.validate_monotonic_residual_erase_data_roles import (
    EXPECTED_SUPERVISION,
    LEDGER_PATH,
    PLAN_PATH,
    PreflightError,
    pixel_decoder_imports,
    run_preflight,
    validate_outputs_absent,
    validate_training_cli_closed,
)


ROOT = Path(__file__).resolve().parents[1]


class MonotonicResidualEraseDataRolePreflightTest(unittest.TestCase):
    def write_json(self, path: Path, value: dict) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def mutated_plan(self, root: Path) -> tuple[dict, Path]:
        plan = json.loads((ROOT / PLAN_PATH).read_text(encoding="utf-8"))
        return plan, root / "role-plan.json"

    def mutated_ledger(self, root: Path) -> tuple[dict, Path]:
        ledger = json.loads((ROOT / LEDGER_PATH).read_text(encoding="utf-8"))
        return ledger, root / "ledger.json"

    def test_actual_preflight_reuses_frozen_roles_without_pixel_access(self) -> None:
        result = run_preflight(repo_root=ROOT)
        self.assertEqual(result["terminal"], "PASS", result)
        self.assertTrue(result["runnable"])
        self.assertTrue(result["metadata_only"])
        self.assertEqual(result["overlap_count"], 0)
        self.assertEqual(
            result["role_counts"],
            {
                "development_next120": 112,
                "development_train160": 156,
                "holdout40": 40,
                "inner_val15": 15,
                "reserved_blind": 0,
                "scut115": 115,
                "train": 275,
            },
        )
        self.assertEqual(
            result["train_domain_counts"],
            {"hw5k": 253, "scut": 22},
        )
        self.assertFalse(result["real_image_decode"])
        self.assertFalse(result["mask_decode"])
        self.assertFalse(result["target_decode"])
        self.assertFalse(result["training_started"])
        self.assertFalse(result["quality_gate_started"])
        self.assertFalse(result["promotion_enabled"])

    def test_supervision_maps_only_target_lighter_train_pixels_to_edit(self) -> None:
        result = run_preflight(repo_root=ROOT)
        self.assertEqual(result["terminal"], "PASS", result)
        self.assertEqual(result["supervision_contract"], EXPECTED_SUPERVISION)
        self.assertEqual(
            result["supervision_contract"]["target_access_roles"],
            ["train"],
        )
        self.assertIn(
            "target_darker",
            result["supervision_contract"]["preserve_negative_includes"],
        )

    def test_target_access_role_drift_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            plan, path = self.mutated_plan(root)
            plan["supervision_contract"]["target_access_roles"].append(
                "inner_val15"
            )
            self.write_json(path, plan)
            result = run_preflight(repo_root=ROOT, role_plan_path=path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("supervision contract changed", result["reason"])

    def test_model_hash_drift_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            plan, path = self.mutated_plan(root)
            plan["evidence"]["model"]["sha256"] = "0" * 64
            self.write_json(path, plan)
            result = run_preflight(repo_root=ROOT, role_plan_path=path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("artifact hash mismatch", result["reason"])

    def test_inactive_iteration_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger, path = self.mutated_ledger(root)
            ledger["active_iteration"]["id"] = "different-iteration"
            self.write_json(path, ledger)
            result = run_preflight(repo_root=ROOT, ledger_path=path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("active iteration", result["reason"])

    def test_missing_synthetic_record_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger, path = self.mutated_ledger(root)
            ledger["records"] = [
                record
                for record in ledger["records"]
                if record.get("id")
                != "monotonic-residual-erase-synthetic-prerequisite"
            ]
            self.write_json(path, ledger)
            result = run_preflight(repo_root=ROOT, ledger_path=path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("synthetic PASS record", result["reason"])

    def test_validator_and_trainers_keep_pixel_and_model_surfaces_closed(self) -> None:
        self.assertEqual(pixel_decoder_imports(), [])
        self.assertFalse(validate_training_cli_closed(ROOT))

    def test_existing_planned_output_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "docs/monotonic-residual-erase-training-plan.json"
            path.parent.mkdir(parents=True)
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(PreflightError, "must be absent"):
                validate_outputs_absent(root)


if __name__ == "__main__":
    unittest.main()
