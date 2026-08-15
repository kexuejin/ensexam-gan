import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from scripts.analysis.audit_monotonic_residual_erase_prerequisite import (
    HISTORICAL_TRAINER_SHA256,
    compute_synthetic_loss_terms,
    optimization_case,
    run_audit,
)
from scripts.infer.patch_cleanup_erasemap import (
    EraseMapCleanupNet,
    ResidualDeltaCleanupNet,
    SignSeparatedResidualDeltaCleanupNet,
    build_model,
)
from scripts.infer.monotonic_residual_erase import (
    MODEL_TYPE,
    MonotonicResidualEraseCleanupNet,
    build_monotonic_residual_erase_model,
    load_monotonic_residual_erase_model,
)


BOUND = 0.08


def final_bias_gradient(model, head_name: str) -> float:
    gradient = getattr(model, head_name)[-1].bias.grad
    return 0.0 if gradient is None else float(gradient.detach().abs().sum())


class MonotonicResidualErasePrerequisiteTest(unittest.TestCase):
    def make_model(self) -> MonotonicResidualEraseCleanupNet:
        torch.manual_seed(20260810)
        return MonotonicResidualEraseCleanupNet(BOUND)

    def test_model_is_exact_identity_without_competing_branch(self) -> None:
        model = self.make_model().eval()
        x = torch.rand(2, 3, 16, 16) * 0.8 + 0.1
        with torch.no_grad():
            pred, alpha, clean_candidate = model(x)
            components = model.forward_components(x)

        self.assertTrue(torch.equal(pred, x))
        self.assertTrue(torch.equal(clean_candidate, x))
        self.assertTrue(torch.equal(components["signed_delta"], torch.zeros_like(x)))
        self.assertTrue(torch.equal(alpha, torch.full_like(alpha, 0.5)))
        self.assertEqual(set(inspect.signature(model.forward).parameters), {"x"})
        self.assertFalse(
            any(
                "dark" in name or "route" in name
                for name, _parameter in model.named_parameters()
            )
        )

    def test_layout_conditioned_input_preserves_rgb_only_output(self) -> None:
        torch.manual_seed(20260815)
        model = MonotonicResidualEraseCleanupNet(
            BOUND,
            input_channels=5,
        ).eval()
        rgb = torch.rand(1, 3, 16, 16) * 0.8 + 0.1
        layout = torch.rand(1, 2, 16, 16)
        conditioned = torch.cat([rgb, layout], dim=1)
        with torch.no_grad():
            pred, alpha, clean_candidate = model(conditioned)
            components = model.forward_components(conditioned)

        self.assertEqual(pred.shape, rgb.shape)
        self.assertTrue(torch.equal(pred, rgb))
        self.assertTrue(torch.equal(clean_candidate, rgb))
        self.assertTrue(
            torch.equal(components["signed_delta"], torch.zeros_like(rgb))
        )
        self.assertTrue(torch.equal(alpha, torch.full_like(alpha, 0.5)))
        self.assertEqual(model.input_channels, 5)
        with self.assertRaisesRegex(ValueError, "5 channels"):
            model(torch.zeros(1, 3, 16, 16))

    def test_layout_conditioned_checkpoint_roundtrip_preserves_channel_count(
        self,
    ) -> None:
        torch.manual_seed(20260815)
        model = MonotonicResidualEraseCleanupNet(
            BOUND,
            input_channels=5,
        ).eval()
        features = torch.rand(1, 5, 16, 16) * 0.8 + 0.1
        with torch.no_grad():
            model.edit_support_head[-1].bias.fill_(1.0)
            model.bright_magnitude_head[-1].bias.fill_(0.25)
            expected = model(features)[0]

        with TemporaryDirectory() as raw:
            checkpoint_path = Path(raw) / "conditioned.pt"
            torch.save(
                {
                    "args": {
                        "input_channels": 5,
                        "model_type": MODEL_TYPE,
                        "residual_delta_bound": BOUND,
                    },
                    "model": model.state_dict(),
                },
                checkpoint_path,
            )
            restored = load_monotonic_residual_erase_model(
                checkpoint_path,
                torch.device("cpu"),
            )
            with torch.no_grad():
                actual = restored(features)[0]

        self.assertEqual(restored.input_channels, 5)
        self.assertTrue(torch.equal(actual, expected))

    def test_dedicated_builder_preserves_the_shared_model_factory(self) -> None:
        self.assertIsInstance(
            build_monotonic_residual_erase_model(),
            MonotonicResidualEraseCleanupNet,
        )
        self.assertEqual(
            build_monotonic_residual_erase_model(input_channels=5).input_channels,
            5,
        )
        with self.assertRaisesRegex(ValueError, "Unsupported cleanup model type"):
            build_model("monotonic_residual_erase")
        self.assertIsInstance(build_model("erasemap"), EraseMapCleanupNet)
        self.assertIsInstance(
            build_model("residual_delta"),
            ResidualDeltaCleanupNet,
        )
        self.assertIsInstance(
            build_model("sign_separated_residual_delta"),
            SignSeparatedResidualDeltaCleanupNet,
        )
        with self.assertRaisesRegex(ValueError, "residual_delta_bound"):
            MonotonicResidualEraseCleanupNet(0.0)
        with self.assertRaisesRegex(ValueError, "input_channels"):
            MonotonicResidualEraseCleanupNet(BOUND, input_channels=2)

    def test_forced_output_is_nonnegative_and_bounded(self) -> None:
        model = self.make_model().eval()
        x = torch.full((1, 3, 8, 8), 0.5)
        with torch.no_grad():
            model.edit_support_head[-1].bias.fill_(100.0)
            model.bright_magnitude_head[-1].bias.fill_(100.0)
            pred, _alpha, clean_candidate = model(x)
        for output in (pred, clean_candidate):
            delta = output - x
            self.assertGreater(float(delta.max()), 0.0)
            self.assertGreaterEqual(float(delta.min()), 0.0)
            self.assertLessEqual(float(delta.max()), BOUND + 1e-7)

    def test_brighten_target_has_live_support_and_magnitude_gradients(self) -> None:
        model = self.make_model().train()
        x = torch.full((1, 3, 8, 8), 0.5)
        terms = compute_synthetic_loss_terms(model, x, x + 0.05)
        terms["loss"].backward()

        self.assertEqual(int(terms["edit_mask_count"]), 64)
        self.assertEqual(int(terms["preserve_mask_count"]), 0)
        self.assertGreater(final_bias_gradient(model, "edit_support_head"), 0.0)
        self.assertGreater(
            final_bias_gradient(model, "bright_magnitude_head"),
            0.0,
        )

    def test_darker_and_identity_targets_are_preserve_negatives(self) -> None:
        x = torch.full((1, 3, 8, 8), 0.5)
        for target_delta in (-0.05, 0.0):
            with self.subTest(target_delta=target_delta):
                model = self.make_model().train()
                terms = compute_synthetic_loss_terms(
                    model,
                    x,
                    x + target_delta,
                )
                terms["loss"].backward()
                self.assertEqual(int(terms["edit_mask_count"]), 0)
                self.assertEqual(int(terms["preserve_mask_count"]), 64)
                self.assertGreater(
                    final_bias_gradient(model, "edit_support_head"),
                    0.0,
                )
                self.assertEqual(
                    final_bias_gradient(model, "bright_magnitude_head"),
                    0.0,
                )
                optimized = optimization_case(target_delta)
                self.assertTrue(optimized["exact_noop"])
                self.assertEqual(optimized["bright_magnitude_abs_max"], 0.0)

    def test_brighten_optimization_moves_only_positive(self) -> None:
        result = optimization_case(0.05)
        self.assertFalse(result["exact_noop"])
        self.assertEqual(result["negative_pixel_count"], 0)
        self.assertGreater(result["delta_max"], 0.0)
        self.assertLessEqual(result["delta_max"], BOUND + 1e-7)

    def test_loss_rejects_shape_mismatch(self) -> None:
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            compute_synthetic_loss_terms(
                model,
                torch.zeros(1, 3, 8, 8),
                torch.zeros(1, 3, 4, 4),
            )

    def test_fail_closed_audit_is_deterministic_and_synthetic_only(self) -> None:
        first = run_audit()
        second = run_audit()
        self.assertEqual(first, second)
        self.assertEqual(first["terminal"], "PASS")
        self.assertTrue(first["exact_identity_init"])
        self.assertTrue(first["zero_output_projection_init"])
        self.assertFalse(first["has_competing_route_or_dark_branch"])
        self.assertFalse(first["training_cli_enabled"])
        self.assertFalse(first["training_authorized"])
        self.assertFalse(first["real_data_access"])
        self.assertFalse(first["quality_gate_started"])
        self.assertFalse(first["promotion_enabled"])
        self.assertEqual(
            first["historical_trainer_sha256"],
            HISTORICAL_TRAINER_SHA256,
        )
        self.assertEqual(set(first["existing_models"]), {
            "erasemap",
            "residual_delta",
            "sign_separated_residual_delta",
        })
        self.assertTrue(first["serialization_exact"])


if __name__ == "__main__":
    unittest.main()
