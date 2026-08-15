from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from scripts.analysis.materialize_source_edge_primary_edit_candidate import (
    DEFAULT_SOURCE_BUCKET,
    DEFAULT_SOURCE_CANDIDATE,
    SourceEdgePrimaryEditLiftConfig,
    process_row,
    select_rows,
    source_edge_primary_edit_lift,
)


class SourceEdgePrimaryEditCandidateTest(unittest.TestCase):
    def test_lifts_only_source_edge_residual_already_edited_by_primary(self) -> None:
        source = np.full((96, 96, 3), 245, np.uint8)
        baseline = np.full((96, 96, 3), 245, np.uint8)
        source[40:56, 40:56] = (35, 35, 35)
        baseline[40:56, 40:56] = (170, 170, 170)
        source[68:84, 68:84] = (35, 35, 35)
        baseline[68:84, 68:84] = source[68:84, 68:84]

        candidate, metrics = source_edge_primary_edit_lift(
            source,
            baseline,
            SourceEdgePrimaryEditLiftConfig(local_median_kernel=31),
        )

        self.assertGreater(metrics["mask_px"], 0)
        self.assertGreater(int(candidate[40, 44, 0]), int(baseline[40, 44, 0]))
        self.assertEqual(int(candidate[48, 48, 0]), int(baseline[48, 48, 0]))
        self.assertEqual(int(candidate[68, 72, 0]), int(baseline[68, 72, 0]))

    def test_flat_primary_edit_region_does_not_change_without_source_edge(self) -> None:
        source = np.full((64, 64, 3), 35, np.uint8)
        baseline = np.full((64, 64, 3), 170, np.uint8)

        candidate, metrics = source_edge_primary_edit_lift(
            source,
            baseline,
            SourceEdgePrimaryEditLiftConfig(local_median_kernel=31),
        )

        self.assertEqual(metrics["mask_px"], 0)
        self.assertTrue(np.array_equal(candidate, baseline))

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
            source[30:38, 30:38] = (35, 35, 35)
            baseline[30:38, 30:38] = (232, 232, 232)
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
                config=SourceEdgePrimaryEditLiftConfig(local_median_kernel=31),
                change_threshold=12,
                eval_threshold=12,
            )

            self.assertTrue(Path(str(review_row["candidate_pred"])).is_file())
            self.assertGreater(int(diagnostic["changed_px"]), 0)
            self.assertGreater(float(diagnostic["residual_gain"]), 0.0)
            self.assertIn("overerase_delta", diagnostic)


if __name__ == "__main__":
    unittest.main()
