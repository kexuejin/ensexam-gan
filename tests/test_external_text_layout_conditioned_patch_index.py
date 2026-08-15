import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from scripts.analysis.build_external_text_layout_conditioned_monotonic_patch_index import (
    CHANNEL_ORDER,
    SELECTION,
    build_conditioned_candidates,
    layout_patch_metrics,
    select_top_brighten,
    summarize_rows,
)


class ExternalTextLayoutConditionedPatchIndexTest(unittest.TestCase):
    def write_image(self, path: Path, bgr: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.assertTrue(cv2.imwrite(str(path), bgr))

    def write_layout(
        self,
        path: Path,
        occupancy: np.ndarray,
        confidence: np.ndarray,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            polygons=np.zeros((0, 4, 2), dtype=np.int32),
            scores=np.zeros((0,), dtype=np.float32),
            text_occupancy=occupancy,
            text_confidence=confidence,
        )

    def test_layout_patch_metrics_are_bounded_and_occupied_mean_is_local(self) -> None:
        occupancy = np.array([[1, 0], [1, 0]], dtype=np.float32)
        confidence = np.array([[0.8, 0.1], [0.6, 0.2]], dtype=np.float32)
        metrics = layout_patch_metrics(occupancy, confidence)
        self.assertEqual(metrics["text_occupancy_ratio"], 0.5)
        self.assertAlmostEqual(metrics["text_confidence_mean"], 0.425, places=6)
        self.assertAlmostEqual(
            metrics["text_confidence_occupied_mean"],
            0.7,
            places=6,
        )

    def test_build_candidates_binds_layout_hashes_and_support_metrics(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            input_dir = root / "pred"
            label_dir = root / "labels"
            layout_dir = root / "layout"
            baseline = np.full((4, 4, 3), 100, dtype=np.uint8)
            target = baseline.copy()
            target[:2, :2] = 130
            target[2:, 2:] = 140
            occupancy = np.zeros((4, 4), dtype=np.uint8)
            occupancy[:2, :2] = 1
            confidence = occupancy.astype(np.float32) * 0.75
            self.write_image(input_dir / "a.png", baseline)
            self.write_image(label_dir / "a.png", target)
            self.write_layout(layout_dir / "a.npz", occupancy, confidence)

            candidates, hashes = build_conditioned_candidates(
                filenames=["a.png"],
                label_dir=label_dir,
                input_dir=input_dir,
                layout_dir=layout_dir,
                tile_size=2,
                overlap=0,
                luminance_margin_gray=2.0,
                min_positive_ratio=0.001,
            )
        self.assertEqual(len(candidates), 2)
        self.assertIn("layout_content_sha256", hashes)
        selected = select_top_brighten(candidates, 1)
        self.assertEqual(selected[0]["file"], "a.png")
        self.assertEqual(float(selected[0]["edit_positive_ratio"]), 1.0)
        self.assertIn("text_occupancy_ratio", selected[0])

    def test_layout_shape_drift_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            input_dir = root / "pred"
            label_dir = root / "labels"
            layout_dir = root / "layout"
            baseline = np.full((4, 4, 3), 100, dtype=np.uint8)
            target = baseline.copy()
            target[:2, :2] = 130
            self.write_image(input_dir / "a.png", baseline)
            self.write_image(label_dir / "a.png", target)
            self.write_layout(
                layout_dir / "a.npz",
                np.zeros((2, 2), dtype=np.uint8),
                np.zeros((2, 2), dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "occupancy"):
                build_conditioned_candidates(
                    filenames=["a.png"],
                    label_dir=label_dir,
                    input_dir=input_dir,
                    layout_dir=layout_dir,
                    tile_size=2,
                    overlap=0,
                    luminance_margin_gray=2.0,
                    min_positive_ratio=0.001,
                )

    def test_summary_records_channel_order_and_content_hashes(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            csv_path = root / "patches.csv"
            rows = [
                {
                    "edit_positive_score": 30.0,
                    "file": "a.png",
                    "x1": 0,
                    "y1": 0,
                    "x2": 2,
                    "y2": 2,
                    "edit_positive_ratio": 1.0,
                    "edit_positive_mean_delta": 30.0,
                    "preserve_negative_ratio": 0.0,
                    "text_occupancy_ratio": 0.5,
                    "text_confidence_mean": 0.25,
                    "text_confidence_occupied_mean": 0.5,
                }
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary = summarize_rows(
                filenames=["a.png"],
                candidates=rows,
                rows=rows,
                output_csv=csv_path,
                content_hashes={
                    "input_content_sha256": "a" * 64,
                    "label_content_sha256": "b" * 64,
                    "layout_content_sha256": "c" * 64,
                },
            )
        self.assertEqual(summary["selection"], SELECTION)
        self.assertEqual(summary["channel_order"], CHANNEL_ORDER)
        self.assertEqual(summary["input_channels"], 5)
        self.assertEqual(summary["layout_content_sha256"], "c" * 64)


if __name__ == "__main__":
    unittest.main()
