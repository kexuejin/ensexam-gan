import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import torch

from scripts.analysis.build_sign_separated_residual_patch_index import (
    select_direction_balanced,
    training_filename,
)
from scripts.infer.patch_cleanup_erasemap import (
    SignSeparatedResidualDeltaCleanupNet,
)
from scripts.train.train_sign_separated_residual_probe import (
    LOSS_TERM_NAMES,
    MASK_SOURCE,
    MODEL_TYPE,
    TargetDifferencePatchDataset,
    build_parser,
    compute_sign_separated_loss_terms,
)


def sign_args() -> argparse.Namespace:
    return argparse.Namespace(
        sign_direction_margin=2.0,
        route_loss_weight=1.0,
        bright_magnitude_weight=1.0,
        dark_magnitude_weight=1.0,
        identity_delta_weight=1.0,
    )


class SignSeparatedResidualTrainingTest(unittest.TestCase):
    def test_dedicated_cli_has_only_registered_training_surface(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--data-root",
                "data",
                "--input-dir",
                "input",
                "--patch-index-file",
                "patches.csv",
                "--output-dir",
                "output",
            ]
        )
        self.assertEqual(MODEL_TYPE, "sign_separated_residual_delta")
        self.assertEqual(MASK_SOURCE, "target_delta")
        self.assertEqual(args.max_steps, 80)
        self.assertEqual(args.lr, 2e-5)
        self.assertEqual(
            LOSS_TERM_NAMES,
            (
                "loss",
                "route_ce",
                "bright_magnitude_l1",
                "dark_magnitude_l1",
                "identity_delta_l1",
            ),
        )
        self.assertNotIn("--model-type", parser._option_string_actions)
        self.assertNotIn("--mask-source", parser._option_string_actions)
        self.assertNotIn("--init-checkpoint", parser._option_string_actions)
        self.assertFalse(
            any(option.startswith("--val-") for option in parser._option_string_actions)
        )

    def branch_gradient_case(self, direction: int) -> dict[str, float]:
        torch.manual_seed(23)
        model = SignSeparatedResidualDeltaCleanupNet(residual_delta_bound=0.08)
        inp = torch.full((1, 3, 8, 8), 0.5)
        target = inp + direction * 0.04
        terms = compute_sign_separated_loss_terms(
            model, inp, target, sign_args()
        )
        terms["loss"].backward()
        bright_grad = float(
            model.bright_magnitude_head[-1].bias.grad.abs().sum()
        )
        dark_grad = float(model.dark_magnitude_head[-1].bias.grad.abs().sum())
        route_grad = float(model.route_head[-1].bias.grad.abs().sum())
        return {
            "bright": bright_grad,
            "dark": dark_grad,
            "route": route_grad,
        }

    def test_branch_losses_keep_opposite_magnitude_gradient_zero(self) -> None:
        bright = self.branch_gradient_case(1)
        dark = self.branch_gradient_case(-1)
        self.assertGreater(bright["bright"], 0.0)
        self.assertEqual(bright["dark"], 0.0)
        self.assertGreater(bright["route"], 0.0)
        self.assertGreater(dark["dark"], 0.0)
        self.assertEqual(dark["bright"], 0.0)
        self.assertGreater(dark["route"], 0.0)

    def test_one_training_step_moves_in_registered_direction(self) -> None:
        for direction in (1, -1):
            with self.subTest(direction=direction):
                torch.manual_seed(29)
                model = SignSeparatedResidualDeltaCleanupNet(
                    residual_delta_bound=0.08
                )
                inp = torch.full((1, 3, 8, 8), 0.5)
                target = inp + direction * 0.04
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                terms = compute_sign_separated_loss_terms(
                    model, inp, target, sign_args()
                )
                optimizer.zero_grad(set_to_none=True)
                terms["loss"].backward()
                optimizer.step()
                delta = (model(inp)[0] - inp).detach()
                if direction > 0:
                    self.assertGreater(float(delta.max()), 0.0)
                    self.assertGreaterEqual(float(delta.min()), 0.0)
                else:
                    self.assertLess(float(delta.min()), 0.0)
                    self.assertLessEqual(float(delta.max()), 0.0)
                self.assertLessEqual(float(delta.abs().max()), 0.08 + 1e-7)

    def test_target_delta_dataset_does_not_require_explicit_mask(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            label_dir = root / "data/train/all_labels"
            input_dir = root / "input"
            label_dir.mkdir(parents=True)
            input_dir.mkdir()
            inp = np.full((8, 8, 3), 128, dtype=np.uint8)
            target = inp.copy()
            target[2:4, 3:5] = 160
            self.assertTrue(cv2.imwrite(str(input_dir / "001.png"), inp))
            self.assertTrue(cv2.imwrite(str(label_dir / "001.jpg"), target))
            patch_index = root / "patches.csv"
            with patch_index.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["file", "x1", "y1", "x2", "y2"]
                )
                writer.writeheader()
                writer.writerow(
                    {"file": "001.jpg", "x1": 0, "y1": 0, "x2": 8, "y2": 8}
                )
            dataset = TargetDifferencePatchDataset(
                data_root=root / "data",
                split="train",
                input_dir=input_dir,
                patch_index_file=patch_index,
                tile_size=8,
            )
            loaded_input, loaded_target = dataset[0]
            self.assertGreater(
                float((loaded_target - loaded_input).abs().sum()), 0.0
            )
            self.assertFalse((root / "data/train/all_masks").exists())

    def test_direction_balanced_selection_is_deterministic(self) -> None:
        candidates = [
            {
                "file": "scut_1.jpg",
                "x1": index,
                "y1": 0,
                "x2": index + 1,
                "y2": 1,
                "brighten_ratio": bright,
                "darken_ratio": dark,
                "brighten_score": bright * 10,
                "darken_score": dark * 10,
            }
            for index, (bright, dark) in enumerate(
                [(0.3, 0.0), (0.2, 0.4), (0.0, 0.5)]
            )
        ]
        first = select_direction_balanced(candidates, top_k_per_direction=1)
        second = select_direction_balanced(candidates, top_k_per_direction=1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(
            {row["selected_for"] for row in first}, {"brighten", "darken"}
        )

    def test_training_filename_is_domain_aware(self) -> None:
        self.assertEqual(training_filename("scut/train/129.jpg"), "scut_129.jpg")
        self.assertEqual(training_filename("hw5k/train/4779.jpg"), "hw5k_4779.jpg")
        with self.assertRaises(ValueError):
            training_filename("scut/test/129.jpg")


if __name__ == "__main__":
    unittest.main()
