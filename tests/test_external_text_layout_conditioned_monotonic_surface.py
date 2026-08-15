import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import torch

from scripts.infer.monotonic_residual_erase import (
    MODEL_TYPE,
    MonotonicResidualEraseCleanupNet,
)
from scripts.infer.run_external_text_layout_conditioned_monotonic_candidate import (
    build_parser as build_candidate_parser,
    infer_conditioned_candidate_components,
    load_candidate,
)
from scripts.train.train_external_text_layout_conditioned_monotonic import (
    CONDITIONED_INPUT_CHANNELS,
    MASK_SOURCE,
    ExternalTextLayoutConditionedPatchDataset,
    build_model,
    build_parser as build_training_parser,
    checkpoint_payload,
    compute_conditioned_loss_terms,
    load_layout_grids,
)


class ExternalTextLayoutConditionedMonotonicSurfaceTest(unittest.TestCase):
    def write_image(self, path: Path, rgb: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
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

    def fixture(self, root: Path) -> dict[str, Path]:
        data_root = root / "data"
        input_dir = root / "pred"
        layout_dir = root / "layout"
        patch_index = root / "patches.csv"
        rgb = np.full((4, 4, 3), 120, dtype=np.uint8)
        target = rgb.copy()
        target[:2, :2] = 140
        occupancy = np.zeros((4, 4), dtype=np.uint8)
        occupancy[:2, :2] = 1
        confidence = occupancy.astype(np.float32) * 0.75
        self.write_image(input_dir / "sample.png", rgb)
        self.write_image(data_root / "train" / "all_labels" / "sample.png", target)
        self.write_layout(layout_dir / "sample.npz", occupancy, confidence)
        with patch_index.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["file", "x1", "y1", "x2", "y2"])
            writer.writeheader()
            writer.writerow({"file": "sample.png", "x1": 0, "y1": 0, "x2": 4, "y2": 4})
        return {
            "data_root": data_root,
            "input_dir": input_dir,
            "layout_dir": layout_dir,
            "patch_index": patch_index,
        }

    def loss_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            luminance_margin_gray=2.0,
            magnitude_weight=1.0,
            preserve_delta_weight=1.0,
            support_positive_weight=1.0,
            support_preserve_weight=1.0,
        )

    def test_dataset_stacks_rgb_occupancy_and_confidence(self) -> None:
        with TemporaryDirectory() as raw:
            paths = self.fixture(Path(raw))
            dataset = ExternalTextLayoutConditionedPatchDataset(
                data_root=paths["data_root"],
                split="train",
                input_dir=paths["input_dir"],
                layout_dir=paths["layout_dir"],
                patch_index_file=paths["patch_index"],
                tile_size=4,
            )
            features, target = dataset[0]
        self.assertEqual(features.shape, (CONDITIONED_INPUT_CHANNELS, 4, 4))
        self.assertEqual(target.shape, (3, 4, 4))
        self.assertTrue(torch.equal(features[3, :2, :2], torch.ones((2, 2))))
        self.assertTrue(torch.equal(features[3, 2:, 2:], torch.zeros((2, 2))))
        self.assertAlmostEqual(float(features[4, 0, 0]), 0.75, places=6)

    def test_layout_loader_fails_closed_on_shape_and_value_drift(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            good = root / "good.npz"
            self.write_layout(
                good,
                np.zeros((2, 2), dtype=np.uint8),
                np.zeros((2, 2), dtype=np.float32),
            )
            load_layout_grids(good, expected_shape=(2, 2))
            bad_shape = root / "bad-shape.npz"
            self.write_layout(
                bad_shape,
                np.zeros((1, 2), dtype=np.uint8),
                np.zeros((1, 2), dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "occupancy"):
                load_layout_grids(bad_shape, expected_shape=(2, 2))
            bad_confidence = root / "bad-confidence.npz"
            self.write_layout(
                bad_confidence,
                np.zeros((2, 2), dtype=np.uint8),
                np.full((2, 2), 1.5, dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                load_layout_grids(bad_confidence, expected_shape=(2, 2))

    def test_conditioned_loss_uses_rgb_target_and_five_channel_model(self) -> None:
        torch.manual_seed(20260815)
        model = build_model(0.08)
        features = torch.full((1, CONDITIONED_INPUT_CHANNELS, 4, 4), 0.5)
        features[:, 3, :2, :2] = 1.0
        features[:, 4, :2, :2] = 0.75
        target = features[:, :3].clone()
        target[:, :, :2, :2] += 0.04
        terms = compute_conditioned_loss_terms(model, features, target, self.loss_args())
        terms["loss"].backward()
        self.assertGreater(float(terms["support_positive_bce"].detach()), 0.0)
        self.assertEqual(model.input_channels, CONDITIONED_INPUT_CHANNELS)
        with self.assertRaisesRegex(ValueError, "5 channels"):
            compute_conditioned_loss_terms(model, features[:, :3], target, self.loss_args())

    def test_training_surface_is_exact_and_checkpoint_is_conditioned(self) -> None:
        parser = build_training_parser()
        options = set(parser._option_string_actions)
        self.assertIn("--layout-dir", options)
        self.assertNotIn("--init-checkpoint", options)
        self.assertNotIn("--model-type", options)
        self.assertFalse(any(option.startswith("--val-") for option in options))
        args = parser.parse_args(
            [
                "--data-root",
                "data",
                "--input-dir",
                "pred",
                "--layout-dir",
                "layout",
                "--patch-index-file",
                "patches.csv",
                "--output-dir",
                "output",
            ]
        )
        payload = checkpoint_payload(build_model(0.08), args, 80)
        self.assertEqual(payload["args"]["model_type"], MODEL_TYPE)
        self.assertEqual(payload["args"]["mask_source"], MASK_SOURCE)
        self.assertEqual(payload["args"]["input_channels"], CONDITIONED_INPUT_CHANNELS)
        self.assertEqual(payload["args"]["data_root"], "data")
        self.assertNotIsInstance(payload["args"]["data_root"], Path)
        self.assertFalse(payload["args"]["validation_enabled"])

    def test_candidate_surface_keeps_target_free_layout_required_contract(self) -> None:
        options = set(build_candidate_parser()._option_string_actions)
        self.assertIn("--layout-dir", options)
        forbidden = {
            "--base-edit-threshold",
            "--second-delta-threshold",
            "--label-dir",
            "--target-dir",
            "--route-override",
        }
        self.assertFalse(options & forbidden)

    def test_conditioned_candidate_inference_preserves_identity_output(self) -> None:
        model = MonotonicResidualEraseCleanupNet(input_channels=5).eval()
        baseline = np.full((4, 4, 3), 123, dtype=np.uint8)
        occupancy = np.zeros((4, 4), dtype=np.float32)
        occupancy[:2, :2] = 1.0
        confidence = occupancy * 0.75
        candidate, probability = infer_conditioned_candidate_components(
            model,
            baseline,
            occupancy,
            confidence,
            torch.device("cpu"),
            tile_size=4,
            stride=4,
        )
        self.assertTrue(np.array_equal(candidate, baseline))
        self.assertTrue(np.allclose(probability, 0.5))

    def test_candidate_loader_rejects_unconditioned_checkpoint(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "unconditioned.pt"
            model = MonotonicResidualEraseCleanupNet(input_channels=3)
            torch.save(
                {
                    "args": {
                        "model_type": MODEL_TYPE,
                        "residual_delta_bound": 0.08,
                    },
                    "model": model.state_dict(),
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "conditioned"):
                load_candidate(path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
