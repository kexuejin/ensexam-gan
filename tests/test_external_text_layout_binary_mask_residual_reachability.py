import json
from pathlib import Path
import unittest

from scripts.analysis.audit_external_text_layout_binary_mask_residual_reachability import (
    evaluate_acceptance,
    run_diagnostic,
)


ROOT = Path(__file__).resolve().parents[1]


class ExternalTextLayoutBinaryMaskResidualReachabilityTest(unittest.TestCase):
    def test_acceptance_uses_registered_reachability_gates(self) -> None:
        diagnostic = {
            "gate_threshold_gray": 12.0,
            "maximum_preserve_gate_ratio": 0.005,
            "minimum_positive_gate_ratio": 0.05,
            "minimum_positive_over_preserve_gate_margin": 0.04,
            "minimum_reachable_patch_ratio": 0.1,
            "required_patch_count": 256,
        }
        accepted = evaluate_acceptance(
            {
                "patch_count": 256,
                "positive_gate_ratio": 0.06,
                "positive_over_preserve_gate_margin": 0.055,
                "preserve_gate_ratio": 0.005,
                "reachable_patch_ratio": 0.1,
            },
            diagnostic,
        )
        self.assertTrue(accepted["passed"])
        rejected = evaluate_acceptance(
            {
                "patch_count": 256,
                "positive_gate_ratio": 0.06,
                "positive_over_preserve_gate_margin": 0.03,
                "preserve_gate_ratio": 0.005,
                "reachable_patch_ratio": 0.1,
            },
            diagnostic,
        )
        self.assertFalse(rejected["passed"])
        self.assertFalse(rejected["conditions"]["positive_over_preserve_gate_margin"])

    def test_registered_diagnostic_runs_train_only_without_candidate_surface(self) -> None:
        result = run_diagnostic(repo_root=ROOT)
        self.assertIn(result["terminal"], {"PASS", "KILL"})
        self.assertFalse(result["model_training_started"])
        self.assertFalse(result["candidate_inference_started"])
        self.assertFalse(result["quality_gate_started"])
        self.assertFalse(result["promotion_enabled"])
        self.assertEqual(result["target_decode_roles"], ["train"])
        self.assertEqual(result["summary"]["patch_count"], 256)
        self.assertEqual(result["mask_formula"], "external_text_occupancy_equals_one")
        self.assertEqual(
            set(result["validation_roles_forbidden"]),
            {"inner_val15", "scut115", "holdout40", "reserved_blind"},
        )
        self.assertEqual(
            result["patch_index"]["path"],
            "hardcase_lists/external-text-layout-conditioned-monotonic-train-patches-v1.csv",
        )
        self.assertEqual(
            result["support_diagnostic"]["path"],
            "outputs/external-text-layout-support-prerequisite-20260813/audit.json",
        )
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
