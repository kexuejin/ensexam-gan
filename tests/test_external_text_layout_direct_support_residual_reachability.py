import json
from pathlib import Path
import unittest

import numpy as np

from scripts.analysis.audit_external_text_layout_direct_support_residual_reachability import (
    ReachabilityError,
    evaluate_acceptance,
    fold_contracts,
    run_diagnostic,
    support_scores,
)


ROOT = Path(__file__).resolve().parents[1]


class ExternalTextLayoutDirectSupportResidualReachabilityTest(unittest.TestCase):
    def test_support_scores_apply_registered_standardized_linear_fit(self) -> None:
        features = np.asarray(
            [
                [2.0, 4.0],
                [4.0, 8.0],
            ],
            dtype=np.float32,
        )
        fit = {
            "feature_mean": [2.0, 2.0],
            "feature_scale": [2.0, 4.0],
            "weights": [1.0, -2.0, 0.5],
        }
        np.testing.assert_allclose(support_scores(features, fit), [-0.5, -1.5])
        bad = dict(fit)
        bad["weights"] = [1.0, 2.0]
        with self.assertRaisesRegex(ReachabilityError, "weights"):
            support_scores(features, bad)

    def test_fold_contracts_require_passed_ordered_support_fits(self) -> None:
        support_audit = {
            "terminal": "PASS",
            "folds": [
                {
                    "fold": fold,
                    "full_fit": {
                        "feature_mean": [0.0] * 5,
                        "feature_scale": [1.0] * 5,
                        "weights": [0.0] * 6,
                    },
                    "positive_score_mean": 1.0 + fold,
                    "preserve_score_mean": 0.0,
                }
                for fold in range(5)
            ],
        }
        contracts = fold_contracts(support_audit)
        self.assertEqual(set(contracts), set(range(5)))
        support_audit["folds"][2]["preserve_score_mean"] = 9.0
        with self.assertRaisesRegex(ReachabilityError, "centers"):
            fold_contracts(support_audit)

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
        self.assertEqual(result["fit_source"], "support_diagnostic_fold_fits_only")
        self.assertEqual(result["fold_fit_count"], 5)
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
