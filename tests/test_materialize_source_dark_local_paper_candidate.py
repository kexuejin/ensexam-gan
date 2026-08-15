import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from scripts.analysis.materialize_source_dark_local_paper_candidate import (
    DEFAULT_OUTPUT_BUCKET,
    DEFAULT_OUTPUT_CANDIDATE,
    DEFAULT_SOURCE_BUCKET,
    DEFAULT_SOURCE_CANDIDATE,
    LocalPaperLiftConfig,
    process_row,
    select_rows,
    source_dark_local_paper_lift,
)


class SourceDarkLocalPaperCandidateTest(unittest.TestCase):
    def test_lifts_only_source_dark_local_residual_not_clean_paper(self) -> None:
        source = np.full((96, 96, 3), 245, np.uint8)
        baseline = np.full((96, 96, 3), 230, np.uint8)
        baseline[44:52, 44:48] = 120
        source[44:52, 44:48] = 60
        baseline[10:20, 10:20] = 220

        candidate, metrics = source_dark_local_paper_lift(
            source,
            baseline,
            LocalPaperLiftConfig(local_median_kernel=31),
        )

        self.assertGreater(metrics["mask_px"], 0)
        self.assertGreater(int(candidate[46, 46, 0]), int(baseline[46, 46, 0]))
        self.assertEqual(int(candidate[12, 12, 0]), int(baseline[12, 12, 0]))
        self.assertEqual(int(candidate[30, 30, 0]), int(baseline[30, 30, 0]))

    def test_select_rows_rejects_validation_splits(self) -> None:
        rows = [
            {
                "split": "scut115",
                "file": "17.jpg",
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

    def test_process_row_does_not_read_target_pixels(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            source_path = root / "source.png"
            baseline_path = root / "baseline.png"
            source = np.full((96, 96, 3), 245, np.uint8)
            baseline = np.full((96, 96, 3), 230, np.uint8)
            source[44:52, 44:48] = 60
            baseline[44:52, 44:48] = 120
            self.assertTrue(cv2.imwrite(str(source_path), source))
            self.assertTrue(cv2.imwrite(str(baseline_path), baseline))

            row = {
                "split": "train160",
                "file": "166.jpg",
                "bucket": DEFAULT_SOURCE_BUCKET,
                "candidate": DEFAULT_SOURCE_CANDIDATE,
                "source_input": str(source_path),
                "baseline_pred": str(baseline_path),
                "candidate_pred": "missing-historical-candidate.png",
                "target": str(root / "target-does-not-exist.png"),
                "notes": "synthetic",
            }
            review_row, diagnostic = process_row(
                row,
                output_dir=root / "out",
                output_bucket=DEFAULT_OUTPUT_BUCKET,
                output_candidate=DEFAULT_OUTPUT_CANDIDATE,
                config=LocalPaperLiftConfig(local_median_kernel=31),
            )

            self.assertEqual(review_row["target"], row["target"])
            self.assertTrue(Path(str(review_row["candidate_pred"])).is_file())
            self.assertGreater(int(diagnostic["changed_px"]), 0)


if __name__ == "__main__":
    unittest.main()
