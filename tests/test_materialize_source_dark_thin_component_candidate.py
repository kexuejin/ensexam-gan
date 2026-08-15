from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from scripts.analysis.materialize_source_dark_thin_component_candidate import (
    DEFAULT_SOURCE_BUCKET,
    DEFAULT_SOURCE_CANDIDATE,
    ThinComponentLiftVariant,
    process_variant_row,
    select_rows,
    source_dark_thin_component_lift,
)


class SourceDarkThinComponentCandidateTest(unittest.TestCase):
    def test_lifts_small_source_dark_component(self) -> None:
        source = np.full((64, 64, 3), 245, np.uint8)
        baseline = np.full((64, 64, 3), 230, np.uint8)
        source[30:34, 30:34] = 60
        baseline[30:34, 30:34] = 120
        variant = ThinComponentLiftVariant(
            name="test",
            source_dark=90,
            baseline_dark=190,
            local_dark_delta=20,
            seed_lift_floor=10,
            max_lift=32,
            alpha=0.5,
            min_component_area=2,
            max_component_area=40,
            local_median_kernel=31,
        )

        candidate, metrics = source_dark_thin_component_lift(source, baseline, variant)

        self.assertGreater(metrics["mask_px"], 0)
        self.assertGreater(int(candidate[31, 31, 0]), int(baseline[31, 31, 0]))
        self.assertEqual(int(candidate[10, 10, 0]), int(baseline[10, 10, 0]))

    def test_rejects_large_components(self) -> None:
        source = np.full((64, 64, 3), 245, np.uint8)
        baseline = np.full((64, 64, 3), 230, np.uint8)
        source[10:40, 10:40] = 60
        baseline[10:40, 10:40] = 120
        variant = ThinComponentLiftVariant(
            name="test",
            source_dark=90,
            baseline_dark=190,
            local_dark_delta=20,
            seed_lift_floor=10,
            max_lift=32,
            alpha=0.5,
            min_component_area=2,
            max_component_area=40,
            local_median_kernel=31,
        )

        candidate, metrics = source_dark_thin_component_lift(source, baseline, variant)

        self.assertEqual(metrics["mask_px"], 0)
        self.assertEqual(int(candidate[20, 20, 0]), int(baseline[20, 20, 0]))

    def test_select_rows_rejects_validation_splits(self) -> None:
        rows = [
            {
                "split": "holdout40",
                "file": "466.jpg",
                "bucket": DEFAULT_SOURCE_BUCKET,
                "candidate": DEFAULT_SOURCE_CANDIDATE,
            }
        ]

        with self.assertRaisesRegex(ValueError, "outside train-only authority"):
            select_rows(
                rows,
                source_bucket=DEFAULT_SOURCE_BUCKET,
                source_candidate=DEFAULT_SOURCE_CANDIDATE,
                allowed_splits={"train", "train160"},
            )

    def test_process_row_generation_does_not_require_existing_target_before_scoring(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            source_path = root / "source.png"
            baseline_path = root / "baseline.png"
            target_path = root / "target.png"
            source = np.full((64, 64, 3), 245, np.uint8)
            baseline = np.full((64, 64, 3), 230, np.uint8)
            target = np.full((64, 64, 3), 245, np.uint8)
            source[30:34, 30:34] = 60
            baseline[30:34, 30:34] = 120
            self.assertTrue(cv2.imwrite(str(source_path), source))
            self.assertTrue(cv2.imwrite(str(baseline_path), baseline))
            self.assertTrue(cv2.imwrite(str(target_path), target))
            variant = ThinComponentLiftVariant(
                name="test",
                source_dark=90,
                baseline_dark=190,
                local_dark_delta=20,
                seed_lift_floor=10,
                max_lift=32,
                alpha=0.5,
                min_component_area=2,
                max_component_area=40,
                local_median_kernel=31,
            )

            review_row, diagnostic = process_variant_row(
                {
                    "split": "train160",
                    "file": "166.jpg",
                    "bucket": DEFAULT_SOURCE_BUCKET,
                    "candidate": DEFAULT_SOURCE_CANDIDATE,
                    "source_input": str(source_path),
                    "baseline_pred": str(baseline_path),
                    "target": str(target_path),
                },
                output_dir=root / "out",
                output_bucket="bucket",
                output_candidate="candidate",
                variant=variant,
                change_threshold=12,
                eval_threshold=12,
            )

            self.assertTrue(Path(str(review_row["candidate_pred"])).is_file())
            self.assertFalse("target" in diagnostic and diagnostic["target"] == str(target_path))
            self.assertGreater(int(diagnostic["changed_px"]), 0)


if __name__ == "__main__":
    unittest.main()
