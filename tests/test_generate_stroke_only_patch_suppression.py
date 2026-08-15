import unittest

import numpy as np

from scripts.analysis.generate_stroke_only_patch_suppression import (
    VARIANTS,
    materialize_candidate,
    stroke_only_mask,
    validate_allowed_splits,
)


class StrokeOnlyPatchSuppressionTest(unittest.TestCase):
    def test_changes_only_source_dark_lifted_strokes_not_bright_patch_background(self) -> None:
        source = np.full((40, 40, 3), 240, np.uint8)
        baseline = np.full((40, 40, 3), 230, np.uint8)
        candidate = baseline.copy()

        source[10:20, 10:12] = 70
        baseline[10:20, 10:12] = 130
        candidate[10:20, 10:12] = 240

        candidate[8:24, 8:24] = 245

        mask, metrics = stroke_only_mask(source, baseline, candidate, VARIANTS[0])
        repaired = materialize_candidate(baseline, candidate, mask, VARIANTS[0].blend)

        self.assertGreater(metrics["mask_px"], 0)
        self.assertTrue(mask[12, 10])
        self.assertFalse(mask[8, 8])
        self.assertEqual(int(repaired[8, 8, 0]), int(baseline[8, 8, 0]))
        self.assertEqual(int(repaired[12, 10, 0]), int(candidate[12, 10, 0]))

    def test_allowed_split_guard_rejects_validation_surfaces_by_default(self) -> None:
        rows = [
            {"split": "train160", "file": "ok.jpg"},
            {"split": "scut115", "file": "blocked.jpg"},
        ]

        with self.assertRaisesRegex(ValueError, "scut115"):
            validate_allowed_splits(rows, {"train", "train160"})

    def test_allowed_split_guard_accepts_train_rows(self) -> None:
        rows = [
            {"split": "train", "file": "a.jpg"},
            {"split": "train160", "file": "b.jpg"},
        ]

        validate_allowed_splits(rows, {"train", "train160"})


if __name__ == "__main__":
    unittest.main()
