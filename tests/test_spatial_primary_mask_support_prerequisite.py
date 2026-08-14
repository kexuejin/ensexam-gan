import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.analysis.audit_spatial_primary_mask_support import (
    AuditError,
    MASK_CHANNELS,
    auc_rank,
    balanced_indices,
    evaluate_fold,
    fit_closed_form_ridge,
    fold_for_name,
    validate_authority as validate_audit_authority,
    validate_plan,
)
from scripts.analysis.materialize_primary_masks_train_only import (
    MaterializationError,
    assert_source_image_path,
    read_manifest,
    validate_authority,
)


ROOT = Path(__file__).resolve().parents[1]


class SpatialPrimaryMaskSupportPrerequisiteTest(unittest.TestCase):
    def test_completed_diagnostic_closes_repeat_execution(self) -> None:
        plan = json.loads(
            (ROOT / "docs/spatial-primary-mask-support-prerequisite-v1.json")
            .read_text(encoding="utf-8")
        )
        ledger = json.loads(
            (ROOT / "docs/current-primary-quality-loop-ledger.json").read_text(
                encoding="utf-8"
            )
        )
        validate_plan(plan)
        with self.assertRaisesRegex(AuditError, "diagnostic is not pending"):
            validate_audit_authority(ledger)
        with self.assertRaisesRegex(
            MaterializationError, "diagnostic is not pending"
        ):
            validate_authority(ledger)
        self.assertEqual(ledger["program"]["product_default"], "artifacts/current-primary")
        self.assertEqual(plan["representation"]["channels"], list(MASK_CHANNELS))
        self.assertFalse(plan["authorization"]["model_training"])

    def test_coordinate_sampling_is_balanced_and_deterministic(self) -> None:
        mask = np.zeros((32, 32), dtype=bool)
        mask[:, ::2] = True
        first = balanced_indices(mask, "page.png", 64)
        second = balanced_indices(mask, "page.png", 64)
        self.assertEqual(len(first[0]), 64)
        self.assertEqual(len(first[1]), 64)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_registered_probe_math_is_stable(self) -> None:
        labels = np.asarray([-1, 1, -1, 1])
        self.assertEqual(auc_rank(labels, np.ones(4)), 0.5)
        train = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        test = np.asarray([[-1.5], [1.5]])
        scores, evidence = fit_closed_form_ridge(
            train, np.asarray([-1, -1, 1, 1]), test, 1.0
        )
        self.assertLess(scores[0], scores[1])
        self.assertEqual(len(evidence["weights"]), 2)
        self.assertIn(fold_for_name("scut_1.jpg"), range(5))

    def test_plan_drift_fails_closed(self) -> None:
        plan = json.loads(
            (ROOT / "docs/spatial-primary-mask-support-prerequisite-v1.json")
            .read_text(encoding="utf-8")
        )
        plan["diagnostic"]["lambda"] = 0.1
        with self.assertRaisesRegex(AuditError, "diagnostic field changed"):
            validate_plan(plan)

    def test_registered_fold_evaluation_covers_full_and_ablation(self) -> None:
        pages = []
        labels = np.asarray([-1, -1, 1, 1], dtype=np.int8)
        signal = np.asarray([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
        for fold in range(5):
            for page_index in range(2):
                features = np.column_stack(
                    [signal, signal * 0.5, signal * 0.5, signal * signal]
                )
                pages.append(
                    {
                        "file": f"fold-{fold}-page-{page_index}.jpg",
                        "fold": fold,
                        "features": features,
                        "ablation_features": np.zeros((4, 3), dtype=np.float32),
                        "labels": labels,
                        "samples_per_class": 2,
                    }
                )
        result = evaluate_fold(pages, 0, 1.0)
        self.assertEqual(result["test_page_count"], 2)
        self.assertGreater(result["full_auc"], result["ablation_auc"])
        self.assertEqual(set(result["mask_channel_strata"]), set(MASK_CHANNELS))

    def test_target_paths_are_rejected_before_decode(self) -> None:
        with self.assertRaisesRegex(MaterializationError, "target"):
            assert_source_image_path(Path("data/all_labels/page.jpg"))

    def test_manifest_rejects_duplicate_or_missing_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "manifest.txt"
            manifest.write_text("a.jpg\na.jpg\n", encoding="utf-8")
            with self.assertRaisesRegex(MaterializationError, "duplicate"):
                read_manifest(root, manifest)

    def test_manifest_requires_exact_train275_role(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "manifest.txt"
            manifest.write_text("a.jpg\n", encoding="utf-8")
            (root / "a.jpg").write_bytes(b"source")
            with self.assertRaisesRegex(MaterializationError, "expected 275"):
                read_manifest(root, manifest)


if __name__ == "__main__":
    unittest.main()
