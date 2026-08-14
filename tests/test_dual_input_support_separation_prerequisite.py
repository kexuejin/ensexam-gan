import json
from pathlib import Path
import unittest

import numpy as np

from scripts.analysis.audit_dual_input_support_separation import (
    AuditError,
    auc_rank,
    balanced_indices,
    fit_closed_form_ridge,
    fold_for_name,
    validate_authority,
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]


class DualInputSupportSeparationPrerequisiteTest(unittest.TestCase):
    def test_registered_plan_and_ledger_keep_training_closed(self) -> None:
        plan = json.loads(
            (ROOT / "docs/dual-input-support-separation-prerequisite-v1.json")
            .read_text(encoding="utf-8")
        )
        ledger = json.loads(
            (ROOT / "docs/current-primary-quality-loop-ledger.json").read_text(
                encoding="utf-8"
            )
        )
        validate_plan(plan)
        validate_authority(ledger)
        self.assertFalse(plan["authorization"]["model_training"])
        self.assertFalse(plan["authorization"]["candidate_inference"])
        self.assertEqual(plan["representation"]["feature_count"], 13)

    def test_coordinate_sampling_is_balanced_and_deterministic(self) -> None:
        mask = np.zeros((32, 32), dtype=bool)
        mask[:, ::2] = True
        first = balanced_indices(mask, "page.png", 64)
        second = balanced_indices(mask, "page.png", 64)
        self.assertEqual(len(first[0]), 64)
        self.assertEqual(len(first[1]), 64)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertTrue(mask.reshape(-1)[first[0]].all())
        self.assertFalse(mask.reshape(-1)[first[1]].any())

    def test_coordinate_sampling_rejects_empty_class(self) -> None:
        with self.assertRaisesRegex(AuditError, "lacks both support classes"):
            balanced_indices(np.ones((8, 8), dtype=bool), "page.png", 8)

    def test_auc_uses_average_ranks_for_ties(self) -> None:
        labels = np.asarray([-1, 1, -1, 1])
        self.assertEqual(auc_rank(labels, np.ones(4)), 0.5)
        self.assertEqual(auc_rank(labels, np.asarray([0.0, 1.0, 0.0, 1.0])), 1.0)

    def test_closed_form_ridge_separates_registered_signal(self) -> None:
        train = np.asarray(
            [[-2.0, 0.0], [-1.0, 0.5], [1.0, 0.5], [2.0, 0.0]]
        )
        labels = np.asarray([-1, -1, 1, 1])
        test = np.asarray([[-1.5, 0.2], [1.5, 0.2]])
        scores, evidence = fit_closed_form_ridge(train, labels, test, 1.0)
        self.assertLess(scores[0], scores[1])
        self.assertEqual(len(evidence["weights"]), 3)

    def test_fold_assignment_is_stable(self) -> None:
        self.assertEqual(fold_for_name("hw5k_1011.jpg"), fold_for_name("hw5k_1011.jpg"))
        self.assertIn(fold_for_name("scut_1.jpg"), range(5))

    def test_plan_drift_fails_closed(self) -> None:
        plan = json.loads(
            (ROOT / "docs/dual-input-support-separation-prerequisite-v1.json")
            .read_text(encoding="utf-8")
        )
        plan["diagnostic"]["lambda"] = 0.1
        with self.assertRaisesRegex(AuditError, "diagnostic field changed"):
            validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
