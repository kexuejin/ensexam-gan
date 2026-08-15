from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from scripts.analysis.materialize_source_achroma_primary_edit_candidate import (
    DEFAULT_SOURCE_BUCKET,
    DEFAULT_SOURCE_CANDIDATE,
    SourceAchromaPrimaryEditLiftConfig,
    process_row,
    select_rows,
    source_achroma_primary_edit_lift,
)


class SourceAchromaPrimaryEditCandidateTest(unittest.TestCase):
    def test_lifts_only_low_chroma_residual_already_edited_by_primary(self) -> None:
        source = np.full((96, 96, 3), 245, np.uint8)
        baseline = np.full((96, 96, 3), 230, np.uint8)
        source[40:48, 40:48] = (35, 35, 35)
        baseline[40:48, 40:48] = (170, 170, 170)
        source[12:20, 12:20] = (20, 20, 210)
        baseline[12:20, 12:20] = (170, 170, 170)
        source[70:76, 70:76] = (35, 35, 35)
        baseline[70:76, 70:76] = source[70:76, 70:76]

        candidate, metrics = source_achroma_primary_edit_lift(
            source,
            baseline,
            SourceAchromaPrimaryEditLiftConfig(local_median_kernel=31),
        )

        self.assertGreater(metrics["mask_px"], 0)
        self.assertGreater(int(candidate[44, 44, 0]), int(baseline[44, 44, 0]))
        self.assertEqual(int(candidate[16, 16, 0]), int(baseline[16, 16, 0]))
        self.assertEqual(int(candidate[72, 72, 0]), int(baseline[72, 72, 0]))

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

    def test_process_row_writes_candidate_and_scores_train_only_metrics(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            source_path = root / "source.png"
            baseline_path = root / "baseline.png"
            target_path = root / "target.png"
            source = np.full((64, 64, 3), 245, np.uint8)
            baseline = np.full((64, 64, 3), 245, np.uint8)
            target = np.full((64, 64, 3), 245, np.uint8)
            source[30:34, 30:34] = (35, 35, 35)
            baseline[30:34, 30:34] = (232, 232, 232)
            self.assertTrue(cv2.imwrite(str(source_path), source))
            self.assertTrue(cv2.imwrite(str(baseline_path), baseline))
            self.assertTrue(cv2.imwrite(str(target_path), target))

            review_row, diagnostic = process_row(
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
                config=SourceAchromaPrimaryEditLiftConfig(local_median_kernel=31),
                change_threshold=12,
                eval_threshold=12,
            )

            self.assertTrue(Path(str(review_row["candidate_pred"])).is_file())
            self.assertGreater(int(diagnostic["changed_px"]), 0)
            self.assertGreater(float(diagnostic["residual_gain"]), 0.0)
            self.assertIn("overerase_delta", diagnostic)


if __name__ == "__main__":
    unittest.main()
