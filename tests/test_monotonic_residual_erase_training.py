import argparse
import math
from pathlib import Path
import unittest

import torch

from scripts.analysis.build_monotonic_residual_erase_patch_index import (
    select_top_brighten,
    support_metrics,
)
from scripts.train.train_monotonic_residual_erase import (
    MASK_SOURCE,
    build_model,
    build_parser,
    checkpoint_payload,
    compute_monotonic_loss_terms,
)


class MonotonicResidualEraseTrainingTest(unittest.TestCase):
    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            luminance_margin_gray=2.0,
            support_positive_weight=1.0,
            support_preserve_weight=1.0,
            magnitude_weight=1.0,
            preserve_delta_weight=1.0,
        )

    def test_sparse_positive_and_preserve_support_are_class_balanced(self) -> None:
        model = build_model(0.08)
        inp = torch.full((1, 3, 8, 8), 0.5)
        target = inp.clone()
        target[:, :, 0, 0] += 0.04
        terms = compute_monotonic_loss_terms(model, inp, target, self.args())
        self.assertAlmostEqual(
            float(terms["support_positive_bce"].detach()),
            math.log(2.0),
            places=6,
        )
        self.assertAlmostEqual(
            float(terms["support_preserve_bce"].detach()),
            math.log(2.0),
            places=6,
        )

    def test_target_darker_is_preserve_supervision(self) -> None:
        model = build_model(0.08)
        inp = torch.full((1, 3, 8, 8), 0.5)
        terms = compute_monotonic_loss_terms(
            model, inp, inp - 0.04, self.args()
        )
        terms["loss"].backward()
        self.assertEqual(float(terms["support_positive_bce"].detach()), 0.0)
        self.assertGreater(float(terms["support_preserve_bce"].detach()), 0.0)
        self.assertEqual(float(terms["bright_magnitude_l1"].detach()), 0.0)
        self.assertEqual(
            float(model.bright_magnitude_head[-1].bias.grad.abs().sum()), 0.0
        )

    def test_checkpoint_identifies_closed_model_and_mask_source(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--data-root", "data",
                "--input-dir", "pred",
                "--patch-index-file", "patches.csv",
                "--output-dir", "output",
            ]
        )
        payload = checkpoint_payload(build_model(0.08), args, 80)
        self.assertEqual(payload["args"]["model_type"], "monotonic_residual_erase")
        self.assertEqual(payload["args"]["mask_source"], MASK_SOURCE)
        self.assertFalse(payload["args"]["validation_enabled"])

    def test_patch_selection_is_brighten_only_and_deterministic(self) -> None:
        rows = [
            {
                "file": "b.png",
                "x1": 0,
                "y1": 0,
                "edit_positive_score": 2.0,
            },
            {
                "file": "a.png",
                "x1": 0,
                "y1": 0,
                "edit_positive_score": 2.0,
            },
            {
                "file": "c.png",
                "x1": 0,
                "y1": 0,
                "edit_positive_score": 1.0,
            },
        ]
        selected = select_top_brighten(rows, 2)
        self.assertEqual([row["file"] for row in selected], ["a.png", "b.png"])

    def test_support_metrics_assign_all_nonpositive_pixels_to_preserve(self) -> None:
        delta = torch.tensor([[4.0, 0.0], [-4.0, 1.0]]).numpy()
        metrics = support_metrics(delta, 2.0)
        self.assertEqual(metrics["edit_positive_ratio"], 0.25)
        self.assertEqual(metrics["preserve_negative_ratio"], 0.75)

    def test_training_parser_has_no_resume_or_validation_path(self) -> None:
        options = set(build_parser()._option_string_actions)
        self.assertNotIn("--init-checkpoint", options)
        self.assertNotIn("--model-type", options)
        self.assertFalse(any(option.startswith("--val-") for option in options))


if __name__ == "__main__":
    unittest.main()
