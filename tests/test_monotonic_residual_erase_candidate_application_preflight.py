import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.analysis.validate_monotonic_residual_erase_candidate_application_preflight import (
    LEDGER_PATH,
    PLAN_PATH,
    run_preflight,
)
from scripts.infer.run_monotonic_residual_erase_candidate import (
    apply_monotonic_candidate_gate,
    build_parser,
    read_sample_paths,
)


ROOT = Path(__file__).resolve().parents[1]


class MonotonicResidualEraseCandidateApplicationPreflightTest(unittest.TestCase):
    def write_json(self, path: Path, value: dict) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def mutated_ledger(self, root: Path) -> tuple[dict, Path]:
        ledger = json.loads((ROOT / LEDGER_PATH).read_text(encoding="utf-8"))
        return ledger, root / "ledger.json"

    def test_registered_preflight_proves_reachable_monotonic_application(self) -> None:
        result = run_preflight(repo_root=ROOT)
        self.assertEqual(result["terminal"], "PASS", result)
        self.assertTrue(result["runnable"])
        self.assertTrue(result["checkpoint_killed"])
        self.assertEqual(result["training"]["learning_rate"], 0.0001)
        self.assertFalse(result["training"]["real_training_started"])
        reachability = result["synthetic_reachability"]
        self.assertLess(
            reachability["legacy_case"]["delta_max_gray"],
            12.0,
        )
        self.assertGreaterEqual(
            reachability["registered_case"]["delta_max_gray"],
            12.0,
        )
        self.assertTrue(
            all(
                case["exact_identity_output"]
                for case in reachability["preserve_cases"]
            )
        )
        application = result["candidate_application"]
        self.assertTrue(application["identity_noop"])
        self.assertTrue(application["reachable_brighten_applied"])
        self.assertTrue(application["darker_candidate_rejected"])
        self.assertFalse(result["checkpoint_generated"])
        self.assertFalse(result["quality_gate_started"])
        self.assertFalse(result["promotion_enabled"])

    def test_inference_surface_has_no_target_route_or_legacy_gate(self) -> None:
        options = set(build_parser()._option_string_actions)
        forbidden = {
            "--base-edit-threshold",
            "--second-delta-threshold",
            "--label-dir",
            "--target-dir",
            "--route-override",
        }
        self.assertFalse(options & forbidden)

    def test_gate_requires_probability_and_meaningful_brighten_delta(self) -> None:
        baseline = np.full((2, 2, 3), 100, dtype=np.uint8)
        candidate = np.full((2, 2, 3), 113, dtype=np.uint8)
        probability = np.array([[0.6, 0.4], [0.6, 0.6]], dtype=np.float32)
        candidate[1, 0] = 111
        merged, gate, _delta = apply_monotonic_candidate_gate(
            baseline,
            candidate,
            probability,
            edit_probability_threshold=0.5,
            minimum_delta_threshold=12.0,
        )
        self.assertTrue(gate[0, 0])
        self.assertFalse(gate[0, 1])
        self.assertFalse(gate[1, 0])
        self.assertTrue(gate[1, 1])
        self.assertTrue(np.array_equal(merged[0, 1], baseline[0, 1]))

    def test_gate_rejects_any_darker_channel(self) -> None:
        baseline = np.full((2, 2, 3), 100, dtype=np.uint8)
        candidate = np.full((2, 2, 3), 113, dtype=np.uint8)
        candidate[0, 0, 0] = 99
        with self.assertRaisesRegex(ValueError, "darkened"):
            apply_monotonic_candidate_gate(
                baseline,
                candidate,
                np.ones((2, 2), dtype=np.float32),
                edit_probability_threshold=0.5,
                minimum_delta_threshold=12.0,
            )

    def test_sample_list_rejects_label_paths(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "samples.txt"
            path.write_text("data/train/all_labels/a.jpg\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "target or label"):
                read_sample_paths(path)

    def test_plan_drift_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "plan.json"
            plan = json.loads((ROOT / PLAN_PATH).read_text(encoding="utf-8"))
            plan["candidate_application"]["minimum_delta_threshold"] = 1.0
            self.write_json(path, plan)
            result = run_preflight(repo_root=ROOT, plan_path=path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("plan hash changed", result["reason"])

    def test_materialization_authority_drift_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger, path = self.mutated_ledger(root)
            prerequisite = next(
                item
                for item in ledger["active_iteration"]["prerequisites"]
                if item["id"]
                == "monotonic_residual_erase_train_materialization_audit"
            )
            prerequisite["status"] = "pending"
            self.write_json(path, ledger)
            result = run_preflight(repo_root=ROOT, ledger_path=path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("materialization prerequisite", result["reason"])

    def test_passed_application_status_requires_registered_record(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger, path = self.mutated_ledger(root)
            prerequisite = next(
                item
                for item in ledger["active_iteration"]["prerequisites"]
                if item["id"]
                == "monotonic_residual_erase_candidate_application_preflight"
            )
            prerequisite["status"] = "passed"
            ledger["records"] = [
                item
                for item in ledger["records"]
                if item.get("id")
                != "monotonic-residual-erase-candidate-application-preflight"
            ]
            self.write_json(path, ledger)
            result = run_preflight(repo_root=ROOT, ledger_path=path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("application PASS record", result["reason"])


if __name__ == "__main__":
    unittest.main()
