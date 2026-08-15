import json
from pathlib import Path
import unittest

import numpy as np

from scripts.analysis.audit_external_text_layout_incremental_support_residual_reachability import (
    IncrementalSupportError,
    build_fold_calibrations,
    evaluate_acceptance,
    fold_contracts,
    incremental_scores,
    run_diagnostic,
)


ROOT = Path(__file__).resolve().parents[1]


class ExternalTextLayoutIncrementalSupportResidualReachabilityTest(unittest.TestCase):
    def test_incremental_scores_subtract_rgb_ablation_fit(self) -> None:
        features = np.asarray(
            [
                [2.0, 4.0, 6.0, 0.0, 0.0],
                [4.0, 8.0, 12.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        full_fit = {
            "feature_mean": [0.0] * 5,
            "feature_scale": [1.0] * 5,
            "weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        ablation_fit = {
            "feature_mean": [0.0] * 3,
            "feature_scale": [1.0] * 3,
            "weights": [0.25, 0.0, 0.0, 0.0],
        }
        np.testing.assert_allclose(
            incremental_scores(features, full_fit, ablation_fit),
            [1.5, 3.0],
        )

    def test_fold_contracts_require_passed_full_and_ablation_fits(self) -> None:
        support_audit = {
            "terminal": "PASS",
            "folds": [
                {
                    "ablation_fit": {
                        "feature_mean": [0.0] * 3,
                        "feature_scale": [1.0] * 3,
                        "weights": [0.0] * 4,
                    },
                    "auc_margin": 0.03,
                    "fold": fold,
                    "full_fit": {
                        "feature_mean": [0.0] * 5,
                        "feature_scale": [1.0] * 5,
                        "weights": [0.0] * 6,
                    },
                }
                for fold in range(5)
            ],
        }
        contracts = fold_contracts(support_audit)
        self.assertEqual(set(contracts), set(range(5)))
        del support_audit["folds"][2]["ablation_fit"]
        with self.assertRaisesRegex(IncrementalSupportError, "ablation"):
            fold_contracts(support_audit)

    def test_fold_calibrations_hold_out_current_fold(self) -> None:
        fold_sums = {
            fold: {
                "positive_count": 10.0,
                "positive_sum": float(20 + fold),
                "preserve_count": 10.0,
                "preserve_sum": float(fold),
            }
            for fold in range(5)
        }
        calibrations = build_fold_calibrations(fold_sums)
        self.assertTrue(all(row["ordered_centers"] for row in calibrations.values()))
        self.assertAlmostEqual(calibrations[0]["positive_center"], 2.25)
        self.assertAlmostEqual(calibrations[0]["preserve_center"], 0.25)
        self.assertEqual(calibrations[0]["calibration_positive_pixels"], 40)
        self.assertEqual(calibrations[0]["calibration_preserve_pixels"], 40)

    def test_acceptance_uses_registered_reachability_and_center_gates(self) -> None:
        diagnostic = {
            "gate_threshold_gray": 12.0,
            "maximum_preserve_gate_ratio": 0.005,
            "minimum_ordered_center_folds": 5,
            "minimum_positive_gate_ratio": 0.05,
            "minimum_positive_over_preserve_gate_margin": 0.04,
            "minimum_reachable_patch_ratio": 0.1,
            "required_patch_count": 256,
        }
        accepted = evaluate_acceptance(
            {
                "ordered_center_folds": 5,
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
                "ordered_center_folds": 4,
                "patch_count": 256,
                "positive_gate_ratio": 0.06,
                "positive_over_preserve_gate_margin": 0.055,
                "preserve_gate_ratio": 0.005,
                "reachable_patch_ratio": 0.1,
            },
            diagnostic,
        )
        self.assertFalse(rejected["passed"])
        self.assertFalse(rejected["conditions"]["ordered_center_folds"])

    def test_registered_diagnostic_runs_train_only_without_candidate_surface(self) -> None:
        result = run_diagnostic(repo_root=ROOT)
        self.assertIn(result["terminal"], {"PASS", "KILL"})
        self.assertFalse(result["model_training_started"])
        self.assertFalse(result["candidate_inference_started"])
        self.assertFalse(result["quality_gate_started"])
        self.assertFalse(result["promotion_enabled"])
        self.assertEqual(result["target_decode_roles"], ["train"])
        self.assertEqual(result["summary"]["patch_count"], 256)
        self.assertEqual(
            result["fit_source"],
            "support_diagnostic_full_and_rgb_ablation_fold_fits_only",
        )
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
