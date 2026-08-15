import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.analysis.validate_external_text_layout_direct_support_residual_preflight import (
    LEDGER_PATH,
    PLAN_PATH,
    PreflightError,
    assert_exact_plan,
    run_preflight,
    run_synthetic_projection,
    score_to_delta_gray,
    validate_ledger_authority,
    validate_outputs_absent,
)


ROOT = Path(__file__).resolve().parents[1]


class ExternalTextLayoutDirectSupportResidualPreflightTest(unittest.TestCase):
    def write_json(self, path: Path, value: dict) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def registered_plan(self) -> dict:
        return json.loads((ROOT / PLAN_PATH).read_text(encoding="utf-8"))

    def registered_ledger(self) -> dict:
        return json.loads((ROOT / LEDGER_PATH).read_text(encoding="utf-8"))

    def test_registered_plan_contract_is_exact(self) -> None:
        plan = self.registered_plan()
        assert_exact_plan(plan)
        plan["direct_support_projection"]["delta_bound_gray"] = 19.0
        with self.assertRaisesRegex(PreflightError, "projection"):
            assert_exact_plan(plan)

    def test_score_projection_is_nonnegative_and_bounded(self) -> None:
        delta = score_to_delta_gray(
            np.asarray([-1.0, 0.0, 0.5, 1.0, 2.0]),
            preserve_center=0.0,
            positive_center=1.0,
            delta_bound_gray=20.4,
        )
        np.testing.assert_allclose(delta, [0.0, 0.0, 10.2, 20.4, 20.4])
        with self.assertRaisesRegex(ValueError, "exceed"):
            score_to_delta_gray(
                np.asarray([0.0]),
                preserve_center=1.0,
                positive_center=1.0,
                delta_bound_gray=20.4,
            )

    def test_synthetic_projection_reaches_gate_without_darkening(self) -> None:
        result = run_synthetic_projection(self.registered_plan())
        self.assertTrue(result["nonnegative"])
        self.assertEqual(result["delta_min_gray"], 0.0)
        self.assertEqual(result["delta_bound_gray"], 20.4)
        self.assertGreater(result["gate_count"], 0)

    def test_ledger_requires_support_pass_and_checkpoint_kill(self) -> None:
        ledger = self.registered_ledger()
        authority = validate_ledger_authority(ROOT, ledger)
        self.assertEqual(authority["conditioned_checkpoint_terminal"], "KILL")
        prerequisite = next(
            item
            for item in ledger["active_iteration"]["prerequisites"]
            if item["id"] == "external_text_layout_conditioned_monotonic_checkpoint_audit"
        )
        prerequisite["status"] = "pending"
        with self.assertRaisesRegex(PreflightError, "checkpoint"):
            validate_ledger_authority(ROOT, ledger)

    def test_registered_preflight_passes_without_opening_candidate_surface(self) -> None:
        result = run_preflight(repo_root=ROOT)
        self.assertEqual(result["terminal"], "PASS", result)
        self.assertFalse(result["model_training_started"])
        self.assertFalse(result["candidate_inference_started"])
        self.assertFalse(result["quality_gate_started"])
        self.assertFalse(result["promotion_enabled"])
        self.assertFalse(result["target_decode"])

    def test_existing_planned_output_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            existing = root / "candidate"
            existing.mkdir()
            plan = {
                "planned_outputs_must_be_absent": {
                    "first_gate_candidate": "candidate"
                }
            }
            with self.assertRaisesRegex(PreflightError, "must be absent"):
                validate_outputs_absent(root, plan)


if __name__ == "__main__":
    unittest.main()
