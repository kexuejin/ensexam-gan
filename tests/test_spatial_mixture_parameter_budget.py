"""Parameter-budget tests for the spatial reconstruction mixture core.

Enforces the frozen "equal active reconstruction capacity" contract:
  trunk   = 89,728
  head    = 111,235 per expert (E1/E2 independent, identical channel plan)
  B_recon = 89,728 + 2 * 111,235 = 312,198
  gate    = 28,723

Learned controls own exactly 312,198 active reconstruction parameters; the
spatial control adds exactly the 28,723-parameter gate (total 340,921). No
dummy, disconnected, zero-multiplied, or optimizer-owned-but-out-of-forward
parameters are permitted.
"""
from __future__ import annotations

import torch
import unittest

from networks.spatial_reconstruction_mixture import (
    GATE_PARAMS,
    HEAD_PARAMS,
    RECON_PARAMS,
    TRUNK_PARAMS,
    MODE_BASELINE,
    MODE_SINGLE_HEAD,
    MODE_SPATIAL_MIXTURE,
    MODE_UNIFORM_TWO_EXPERT,
    SpatialContinuousReconstructionMixture,
    SpatialSoftGate,
)


class FrozenComponentCountTest(unittest.TestCase):
    def test_component_parameter_counts_match_frozen_budget(self) -> None:
        mixture = SpatialContinuousReconstructionMixture(mode=MODE_UNIFORM_TWO_EXPERT)
        self.assertEqual(mixture.trunk.param_count(), TRUNK_PARAMS)
        self.assertEqual(mixture.expert1.param_count(), HEAD_PARAMS)
        self.assertEqual(mixture.expert2.param_count(), HEAD_PARAMS)

    def test_experts_have_identical_channel_plan(self) -> None:
        mixture = SpatialContinuousReconstructionMixture(mode=MODE_UNIFORM_TWO_EXPERT)
        self.assertEqual(mixture.expert1.param_count(), mixture.expert2.param_count())

    def test_gate_parameter_count_is_frozen(self) -> None:
        gate = SpatialSoftGate()
        self.assertEqual(gate.param_count(), GATE_PARAMS)


class ControlBudgetTest(unittest.TestCase):
    def test_each_learnt_control_has_correct_total_trainable(self) -> None:
        expectations = {
            MODE_BASELINE: 0,
            MODE_SINGLE_HEAD: RECON_PARAMS,
            MODE_UNIFORM_TWO_EXPERT: RECON_PARAMS,
            MODE_SPATIAL_MIXTURE: RECON_PARAMS + GATE_PARAMS,
        }
        for mode, expected in expectations.items():
            with self.subTest(mode=mode, expected=expected):
                mixture = SpatialContinuousReconstructionMixture(mode=mode)
                self.assertEqual(mixture.param_count(), expected)
                self.assertEqual(mixture.expected_budget(), expected)
                self.assertEqual(mixture.mode_param_counts()["total_enabled_trainable"], expected)

    def test_active_reconstruction_budget_is_equal_across_controls(self) -> None:
        for mode in (MODE_SINGLE_HEAD, MODE_UNIFORM_TWO_EXPERT, MODE_SPATIAL_MIXTURE):
            with self.subTest(mode=mode):
                mixture = SpatialContinuousReconstructionMixture(mode=mode)
                self.assertEqual(mixture.active_reconstruction_params(), RECON_PARAMS)
        baseline = SpatialContinuousReconstructionMixture(mode=MODE_BASELINE)
        self.assertEqual(baseline.active_reconstruction_params(), 0)

    def test_spatial_differs_from_uniform_by_exactly_gate_params(self) -> None:
        uniform = SpatialContinuousReconstructionMixture(mode=MODE_UNIFORM_TWO_EXPERT)
        spatial = SpatialContinuousReconstructionMixture(mode=MODE_SPATIAL_MIXTURE)
        self.assertEqual(
            spatial.param_count() - uniform.param_count(),
            GATE_PARAMS,
        )
        # Same reconstruction capacity; only the gate accounts for the difference.
        self.assertEqual(
            spatial.active_reconstruction_params(),
            uniform.active_reconstruction_params(),
        )

    def test_trunk_heads_gate_sum_to_spatial_total(self) -> None:
        spatial = SpatialContinuousReconstructionMixture(mode=MODE_SPATIAL_MIXTURE)
        counts = spatial.mode_param_counts()
        self.assertEqual(
            counts["trunk"] + counts["expert1"] + counts["expert2"] + counts["gate"],
            spatial.param_count(),
        )
        self.assertEqual(spatial.param_count(), RECON_PARAMS + GATE_PARAMS)


class NoDummyOrStaleParameterTest(unittest.TestCase):
    def test_every_trainable_parameter_receives_gradient_after_perturbation(self) -> None:
        """After perturbing the zero-init point, every learnable parameter must
        receive a finite, non-zero gradient (synthetic erase/repair liveness).

        At the exact zero-init point gradients are structurally zero by design;
        this test perturbs the heads/gate so gradients demonstrably flow
        through every parameter of every learned control.
        """
        torch.manual_seed(0)

        def _build(mode: str):
            model = SpatialContinuousReconstructionMixture(mode=mode)
            with torch.no_grad():
                for expert in (model.expert1, model.expert2):
                    torch.nn.init.normal_(expert.correction.weight, std=0.05)
                    torch.nn.init.normal_(expert.correction.bias, std=0.05)
                if mode == MODE_SPATIAL_MIXTURE:
                    torch.nn.init.normal_(model.gate.ghead.weight, std=0.1)
            return model

        for mode in (MODE_SINGLE_HEAD, MODE_UNIFORM_TWO_EXPERT, MODE_SPATIAL_MIXTURE):
            with self.subTest(mode=mode):
                model = _build(mode)
                feats = torch.randn(4, 27, 24, 24)
                y0 = 0.05 * torch.randn(4, 3, 24, 24)
                candidate, _, _ = model.mixture_output(feats, y0)
                candidate.pow(2).mean().backward()

                trainable = [p for p in model.parameters() if p.requires_grad]
                self.assertTrue(trainable)
                dead = [
                    name
                    for name, p in model.named_parameters()
                    if p.requires_grad
                    and (p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum().item() == 0)
                ]
                self.assertEqual(dead, [], f"{mode}: dead parameters {dead}")

    def test_gate_parameters_are_reachable_from_forward(self) -> None:
        """The gate head parameters are functionally connected (zero-init, then
        a synthetic forward must be able to move them and appear in the output).
        """
        torch.manual_seed(0)
        model = SpatialContinuousReconstructionMixture(mode=MODE_SPATIAL_MIXTURE)
        feats = torch.randn(1, 27, 16, 16)
        y0 = torch.zeros(1, 3, 16, 16)
        with torch.no_grad():
            torch.nn.init.normal_(model.gate.ghead.weight, std=0.2)
        candidate, _, gate_weights = model.mixture_output(feats, y0)
        self.assertIsNotNone(gate_weights)
        w = gate_weights["spatial"]
        self.assertFalse(
            torch.allclose(w, w.new_full(w.shape, 1.0 / 3.0), atol=1e-6),
            "gate must learn to leave the uniform mixture once perturbed",
        )


if __name__ == "__main__":
    unittest.main()
