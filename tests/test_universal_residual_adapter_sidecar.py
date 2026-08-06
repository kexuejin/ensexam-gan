import inspect
import tempfile
import unittest
from pathlib import Path

import torch

from networks.discriminator import Discriminator
from networks.generator import Generator, UniversalResidualAdapterSidecar
from train import (
    apply_generator_trainable_patterns,
    build_training_checkpoint,
    load_initial_checkpoint,
)


def sidecar_cfg(**overrides):
    cfg = {
        "coarse_in_channels": 3,
        "refine_in_channels": 7,
        "cbam_reduction": 16,
        "universal_residual_adapter_sidecar": {
            "enabled": True,
            "adapter_count": 3,
            "hidden_channels": 16,
            "residual_bound": 12.0 / 255.0,
        },
    }
    cfg["universal_residual_adapter_sidecar"].update(overrides)
    return cfg


class UniversalResidualAdapterSidecarTest(unittest.TestCase):
    def test_generator_rejects_non_admitted_sidecar_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "adapter_count"):
            Generator(sidecar_cfg(adapter_count=4))
        with self.assertRaisesRegex(ValueError, "residual_bound"):
            Generator(sidecar_cfg(residual_bound=0.05))

    def test_default_generator_has_no_sidecar_params_or_shape_change(self) -> None:
        generator = Generator().eval()
        self.assertFalse(generator.universal_residual_adapter_sidecar_enabled)
        self.assertFalse(
            any("universal_residual_adapter_sidecar" in key for key in generator.state_dict())
        )
        input_image = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            outputs = generator(input_image)
            telemetry_outputs = generator(
                input_image,
                return_universal_sidecar_telemetry=True,
            )
        self.assertEqual(len(outputs), 7)
        self.assertEqual(len(telemetry_outputs), 2)
        self.assertEqual(len(telemetry_outputs[0]), 7)
        self.assertEqual(float(telemetry_outputs[1]["ura_enabled"]), 0.0)

    def test_zero_init_sidecar_matches_same_baseline_generator(self) -> None:
        torch.manual_seed(7)
        baseline = Generator().eval()
        enabled = Generator(sidecar_cfg()).eval()
        incompatible = enabled.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(
                key.startswith("universal_residual_adapter_sidecar.")
                for key in incompatible.missing_keys
            )
        )
        input_image = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            baseline_outputs = baseline(input_image)
            enabled_outputs, telemetry = enabled(
                input_image,
                return_universal_sidecar_telemetry=True,
            )
        self.assertEqual(len(enabled_outputs), 7)
        for expected, actual in zip(baseline_outputs, enabled_outputs):
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(float(telemetry["ura_fallback_code"]), 0.0)

    def test_zero_output_sidecar_keeps_final_projection_gradients_alive(self) -> None:
        torch.manual_seed(11)
        sidecar = UniversalResidualAdapterSidecar(feature_channels=16).train()
        feature = torch.randn(2, 16, 8, 8)
        baseline = torch.zeros(2, 3, 8, 8)
        target = torch.full_like(baseline, 0.25)

        candidate, telemetry = sidecar(feature, baseline)
        self.assertTrue(torch.equal(candidate.detach(), baseline))
        self.assertEqual(float(telemetry["ura_fallback_code"]), 0.0)
        self.assertGreater(
            float(torch.tanh(sidecar.global_residual_scale).detach().abs()),
            0.0,
        )

        loss = torch.nn.functional.mse_loss(candidate, target)
        loss.backward()

        final_bias_grads = [
            adapter[-1].bias.grad.detach().abs().sum()
            for adapter in sidecar.adapters
        ]
        self.assertTrue(
            all(float(grad_sum) > 0.0 for grad_sum in final_bias_grads),
            "zero-output sidecar must keep final projection bias gradients alive",
        )

    def test_sidecar_trainable_pattern_freezes_base_generator(self) -> None:
        generator = Generator(sidecar_cfg())
        summary = apply_generator_trainable_patterns(
            generator,
            ["^universal_residual_adapter_sidecar\\."],
        )
        trainable_names = [
            name for name, parameter in generator.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable_names)
        self.assertTrue(
            all(
                name.startswith("universal_residual_adapter_sidecar.")
                for name in trainable_names
            )
        )
        self.assertGreater(summary["frozen_tensors"], 0)
        self.assertGreater(summary["trainable_tensors"], 0)

    def test_initial_checkpoint_allows_sidecar_only_missing_keys(self) -> None:
        baseline_generator = Generator()
        baseline_discriminator = Discriminator()
        enabled_generator = Generator(sidecar_cfg())
        enabled_discriminator = Discriminator()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "baseline.pth"
            torch.save(
                {
                    "G_state_dict": baseline_generator.state_dict(),
                    "D_state_dict": baseline_discriminator.state_dict(),
                },
                checkpoint_path,
            )
            missing, unexpected = load_initial_checkpoint(
                enabled_generator,
                enabled_discriminator,
                str(checkpoint_path),
                torch.device("cpu"),
            )

        self.assertFalse(unexpected)
        self.assertTrue(missing)
        self.assertTrue(
            all(
                key.startswith("universal_residual_adapter_sidecar.")
                for key in missing
            )
        )

    def test_checkpoint_payload_can_omit_optimizer_and_scheduler_state(self) -> None:
        generator = Generator(sidecar_cfg())
        discriminator = Discriminator()
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=1)
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=1)

        checkpoint = build_training_checkpoint(
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            scheduler_g,
            scheduler_d,
            epoch=1,
            avg_loss_G=1.0,
            avg_loss_D=2.0,
            val_loss=None,
            save_optimizer_state=False,
            save_scheduler_state=False,
        )

        self.assertIn("G_state_dict", checkpoint)
        self.assertNotIn("optimizer_G", checkpoint)
        self.assertNotIn("optimizer_D", checkpoint)
        self.assertNotIn("scheduler_G", checkpoint)
        self.assertNotIn("scheduler_D", checkpoint)

    def test_public_forward_surface_has_no_routing_arguments(self) -> None:
        forbidden = ("domain", "source", "caller", "route", "expert", "path")
        params = inspect.signature(Generator.forward).parameters
        for name in params:
            with self.subTest(name=name):
                self.assertFalse(any(token in name.lower() for token in forbidden))

    def test_gate_is_continuous_simplex(self) -> None:
        generator = Generator(sidecar_cfg()).eval()
        input_image = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            _, reconstruction_feature = generator(
                input_image,
                return_reconstruction_feature=True,
            )
            weights = torch.softmax(
                generator.universal_residual_adapter_sidecar.gate(
                    reconstruction_feature
                ),
                dim=1,
            )
        self.assertEqual(weights.shape, (2, 3))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(
            torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6, rtol=0.0)
        )

    def test_residual_bound_and_fallback(self) -> None:
        sidecar = UniversalResidualAdapterSidecar(
            feature_channels=16,
            residual_bound=0.02,
            fallback_residual_abs_max=0.01,
        )
        with torch.no_grad():
            sidecar.global_residual_scale.fill_(10.0)
            for adapter in sidecar.adapters:
                adapter[-1].bias.fill_(100.0)
        feature = torch.zeros(1, 16, 8, 8)
        baseline = torch.randn(1, 3, 8, 8).clamp(-0.5, 0.5)
        candidate, telemetry = sidecar(feature, baseline)
        self.assertTrue(torch.equal(candidate, baseline))
        self.assertEqual(float(telemetry["ura_fallback_code"]), 1.0)


if __name__ == "__main__":
    unittest.main()
