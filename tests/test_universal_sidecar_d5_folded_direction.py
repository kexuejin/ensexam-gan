import inspect
import unittest

import torch

from networks.generator import Generator, UniversalResidualAdapterSidecar
from train import validate_universal_sidecar_config


FOLDED_MODE = "primary_edit_direction_folded"


def sidecar_model_config() -> dict:
    return {
        "coarse_in_channels": 3,
        "refine_in_channels": 7,
        "cbam_reduction": 16,
        "universal_residual_adapter_sidecar": {
            "enabled": True,
            "adapter_count": 3,
            "hidden_channels": 16,
            "residual_bound": 12.0 / 255.0,
            "residual_parameterization": FOLDED_MODE,
        },
    }


def training_config() -> dict:
    return {
        "model": sidecar_model_config(),
        "train": {
            "resume": False,
            "resume_path": "",
            "init_checkpoint": (
                "./artifacts/current-primary/micro_region_probe_step0001.pth"
            ),
            "trainable_generator_patterns": [
                "^universal_residual_adapter_sidecar\\."
            ],
            "freeze_generator_batchnorm_stats": True,
            "save_optimizer_state": False,
            "save_scheduler_state": False,
            "seed": 20260806,
            "reproducibility_mode": "strict",
        },
    }


class FoldedDirectionSidecarTest(unittest.TestCase):
    def make_sidecar(self, *, residual_bound: float = 12.0 / 255.0):
        return UniversalResidualAdapterSidecar(
            feature_channels=16,
            residual_bound=residual_bound,
            residual_parameterization=FOLDED_MODE,
        )

    def test_folded_mode_is_registered_without_changing_public_forward(self) -> None:
        validate_universal_sidecar_config(training_config())
        sidecar = self.make_sidecar()
        self.assertTrue(all(adapter[-1].out_channels == 1 for adapter in sidecar.adapters))
        self.assertEqual(
            set(inspect.signature(Generator.forward).parameters),
            {
                "self",
                "Iin",
                "return_reconstruction_feature",
                "return_universal_sidecar_telemetry",
            },
        )

    def test_positive_and_negative_raw_values_fold_to_same_direction(self) -> None:
        sidecar = self.make_sidecar(residual_bound=0.02).eval()
        feature = torch.zeros(1, 16, 4, 4)
        input_image = torch.zeros(1, 3, 4, 4)
        baseline = torch.empty_like(input_image)
        baseline[:, 0] = 0.4
        baseline[:, 1] = -0.2
        baseline[:, 2] = 0.1

        deltas = []
        with torch.no_grad():
            sidecar.global_residual_scale.fill_(10.0)
            for raw_value in (2.0, -2.0):
                for adapter in sidecar.adapters:
                    adapter[-1].bias.fill_(raw_value)
                candidate, telemetry = sidecar(feature, baseline, input_image)
                self.assertEqual(float(telemetry["ura_fallback_code"]), 0.0)
                deltas.append(candidate - baseline)

        self.assertTrue(torch.allclose(deltas[0], deltas[1], atol=1e-7, rtol=0.0))
        primary_edit = baseline - input_image
        self.assertEqual(int((deltas[1] * primary_edit < -1e-8).sum()), 0)
        self.assertGreater(float(deltas[1].abs().max()), 0.0)
        self.assertLessEqual(float(deltas[1].abs().max()), 0.02 + 1e-7)

    def test_zero_init_and_two_step_gradients_survive_either_raw_sign(self) -> None:
        for expected_raw_sign in (1, -1):
            with self.subTest(expected_raw_sign=expected_raw_sign):
                torch.manual_seed(20260809)
                sidecar = self.make_sidecar().train()
                initial_scale = float(sidecar.global_residual_scale.detach())
                optimizer = torch.optim.SGD(sidecar.parameters(), lr=0.25)
                feature = torch.ones(1, 16, 4, 4)
                input_image = torch.zeros(1, 3, 4, 4)
                baseline = torch.empty_like(input_image)
                baseline[:, 0] = 0.4
                baseline[:, 1] = -0.2
                baseline[:, 2] = 0.1
                primary_direction = baseline / baseline.abs().amax(
                    dim=1, keepdim=True
                )
                target = baseline + expected_raw_sign * 0.2 * primary_direction

                candidate, _ = sidecar(feature, baseline, input_image)
                self.assertTrue(torch.equal(candidate.detach(), baseline))
                loss = torch.nn.functional.mse_loss(candidate, target)
                optimizer.zero_grad()
                loss.backward()
                first_bias_gradients = [
                    float(adapter[-1].bias.grad.detach().abs().sum())
                    for adapter in sidecar.adapters
                ]
                self.assertTrue(all(value > 0.0 for value in first_bias_gradients))
                self.assertEqual(float(sidecar.global_residual_scale.grad), 0.0)
                optimizer.step()

                with torch.no_grad():
                    mixed_after_first = torch.stack(
                        [adapter(feature) for adapter in sidecar.adapters], dim=1
                    ).mul(
                        torch.softmax(sidecar.gate(feature), dim=1).view(
                            1, sidecar.adapter_count, 1, 1, 1
                        )
                    ).sum(dim=1)
                self.assertTrue(
                    torch.all(torch.sign(mixed_after_first) == expected_raw_sign)
                )
                self.assertGreater(float(mixed_after_first.abs().sum()), 0.0)

                candidate, _ = sidecar(feature, baseline, input_image)
                second_loss = torch.nn.functional.mse_loss(candidate, target)
                optimizer.zero_grad()
                second_loss.backward()
                second_bias_gradients = [
                    float(adapter[-1].bias.grad.detach().abs().sum())
                    for adapter in sidecar.adapters
                ]
                self.assertTrue(all(value > 0.0 for value in second_bias_gradients))
                self.assertGreater(
                    float(sidecar.global_residual_scale.grad.detach().abs()), 0.0
                )
                optimizer.step()
                self.assertNotEqual(
                    float(sidecar.global_residual_scale.detach()),
                    initial_scale,
                )

    def test_zero_primary_edit_is_noop(self) -> None:
        sidecar = self.make_sidecar().eval()
        with torch.no_grad():
            sidecar.global_residual_scale.fill_(10.0)
            for adapter in sidecar.adapters:
                adapter[-1].bias.fill_(-100.0)
        feature = torch.zeros(1, 16, 4, 4)
        baseline = torch.randn(1, 3, 4, 4).clamp(-0.5, 0.5)
        candidate, _ = sidecar(feature, baseline, baseline.clone())
        self.assertTrue(torch.equal(candidate, baseline))

    def test_d4_negative_raw_value_remains_zero(self) -> None:
        sidecar = UniversalResidualAdapterSidecar(
            feature_channels=16,
            residual_parameterization="primary_edit_direction",
        ).eval()
        with torch.no_grad():
            sidecar.global_residual_scale.fill_(10.0)
            for adapter in sidecar.adapters:
                adapter[-1].bias.fill_(-100.0)
        feature = torch.zeros(1, 16, 4, 4)
        input_image = torch.zeros(1, 3, 4, 4)
        baseline = torch.full_like(input_image, 0.25)
        candidate, _ = sidecar(feature, baseline, input_image)
        self.assertTrue(torch.equal(candidate, baseline))


if __name__ == "__main__":
    unittest.main()
