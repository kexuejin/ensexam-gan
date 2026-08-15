import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.analysis.validate_external_text_layout_incremental_support_residual_preflight import (
    LEDGER_PATH,
    OUTPUT_PATH,
    PLAN_PATH,
    PreflightError,
    assert_exact_plan,
    incremental_score_to_delta_gray,
    run_preflight,
    run_synthetic_projection,
    validate_ledger_authority,
    validate_outputs_absent,
)


ROOT = Path(__file__).resolve().parents[1]


class ExternalTextLayoutIncrementalSupportResidualPreflightTest(unittest.TestCase):
    def registered_plan(self) -> dict:
        return json.loads((ROOT / PLAN_PATH).read_text(encoding="utf-8"))

    def registered_ledger(self) -> dict:
        return json.loads((ROOT / LEDGER_PATH).read_text(encoding="utf-8"))

    def test_registered_plan_contract_is_exact(self) -> None:
        plan = self.registered_plan()
        assert_exact_plan(plan)
        plan["incremental_support_projection"]["score_formula"] = "full_score"
        with self.assertRaisesRegex(PreflightError, "projection"):
            assert_exact_plan(plan)

    def test_incremental_projection_is_nonnegative_and_bounded(self) -> None:
        delta = incremental_score_to_delta_gray(
            np.asarray([-1.0, 0.0, 0.5, 1.0, 2.0]),
            preserve_center=0.0,
            positive_center=1.0,
            delta_bound_gray=20.4,
        )
        np.testing.assert_allclose(delta, [0.0, 0.0, 10.2, 20.4, 20.4])
        with self.assertRaisesRegex(ValueError, "exceed"):
            incremental_score_to_delta_gray(
                np.asarray([0.0]),
                preserve_center=1.0,
                positive_center=1.0,
                delta_bound_gray=20.4,
            )

    def test_synthetic_projection_reaches_gate_without_darkening(self) -> None:
        result = run_synthetic_projection(self.registered_plan())
        self.assertEqual(
            result["score_formula"],
            "support_full_score_minus_rgb_ablation_score",
        )
        self.assertTrue(result["nonnegative"])
        self.assertEqual(result["delta_min_gray"], 0.0)
        self.assertEqual(result["delta_bound_gray"], 20.4)
        self.assertGreater(result["gate_count"], 0)

    def test_ledger_requires_support_pass_and_direct_support_kill(self) -> None:
        ledger = self.registered_ledger()
        authority = validate_ledger_authority(ROOT, ledger)
        self.assertEqual(authority["direct_support_terminal"], "KILL")
        prerequisite = next(
            item
            for item in ledger["active_iteration"]["prerequisites"]
            if item["id"] == "external_text_layout_direct_support_residual_reachability_diagnostic"
        )
        prerequisite["status"] = "pending"
        with self.assertRaisesRegex(PreflightError, "external_text_layout_direct_support"):
            validate_ledger_authority(ROOT, ledger)

    def test_registered_preflight_artifact_passed_without_opening_candidate_surface(self) -> None:
        result = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
        self.assertEqual(result["terminal"], "PASS", result)
        self.assertFalse(result["model_training_started"])
        self.assertFalse(result["candidate_inference_started"])
        self.assertFalse(result["quality_gate_started"])
        self.assertFalse(result["promotion_enabled"])
        self.assertFalse(result["target_decode"])

    def test_live_preflight_rerun_fails_closed_after_diagnostic_output_exists(self) -> None:
        result = run_preflight(repo_root=ROOT)
        self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED", result)
        self.assertIn("planned output must be absent", result["reason"])

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
