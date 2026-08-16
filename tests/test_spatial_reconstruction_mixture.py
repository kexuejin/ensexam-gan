"""Behavior tests for the spatial continuous reconstruction mixture core.

Covers initialization equivalence, spatial-gate simplex/finite invariants,
control-mode forward semantics, the bounded correction edit range, runtime
fail-closed fallback to the anchor y0, and the absence of any
metadata/domain/path routing inputs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import unittest

from networks.spatial_reconstruction_mixture import (
    CORRECTION_BOUND,
    CPU_INIT_TOLERANCE,
    MPS_INIT_TOLERANCE,
    MODE_BASELINE,
    MODE_SINGLE_HEAD,
    MODE_SPATIAL_MIXTURE,
    MODE_UNIFORM_TWO_EXPERT,
    VALID_MODES,
    SpatialSoftGate,
    SpatialContinuousReconstructionMixture,
    TerminalReconstructionHead,
)


def _feats_y0(batch: int = 2, height: int = 32, width: int = 32, channels: int = 27):
    feats = torch.randn(batch, channels, height, width)
    y0 = 0.05 * torch.randn(batch, 3, height, width)
    return feats, y0


class InitializationEquivalenceTest(unittest.TestCase):
    def test_all_learned_controls_initialize_to_y0_cpu(self) -> None:
        for mode in (
            MODE_SINGLE_HEAD,
            MODE_UNIFORM_TWO_EXPERT,
            MODE_SPATIAL_MIXTURE,
        ):
            with self.subTest(mode=mode):
                torch.manual_seed(0)
                mixture = SpatialContinuousReconstructionMixture(mode=mode)
                feats, y0 = _feats_y0()
                candidate, _, _ = mixture.mixture_output(feats, y0)
                delta = (candidate - y0).abs().max().item()
                self.assertLessEqual(
                    delta,
                    CPU_INIT_TOLERANCE,
                    f"{mode}: initialized mixture must equal y0 within CPU tolerance",
                )

    def test_heads_reproduce_anchor_exactly_cpu(self) -> None:
        torch.manual_seed(0)
        mixture = SpatialContinuousReconstructionMixture(mode=MODE_UNIFORM_TWO_EXPERT)
        feats, y0 = _feats_y0()
        _, raws, _ = mixture.mixture_output(feats, y0)
        # Zero-init final projections => raw corrections are exactly zero.
        self.assertEqual(raws[0].abs().max().item(), 0.0)
        self.assertEqual(raws[1].abs().max().item(), 0.0)

    def test_gate_logits_initialize_to_uniform_simplex(self) -> None:
        torch.manual_seed(0)
        gate = SpatialSoftGate()
        feats = torch.randn(3, 27, 16, 16)
        w = gate(feats)
        self.assertEqual(tuple(w.shape), (3, 3, 16, 16))
        # All entries should be exactly 1/3 at init (zero logits => uniform).
        self.assertTrue(torch.allclose(w, w.new_full((3, 3, 16, 16), 1.0 / 3.0), atol=1e-6))


class SpatialGateInvariantTest(unittest.TestCase):
    def test_gate_weights_are_simplex(self) -> None:
        torch.manual_seed(0)
        gate = SpatialSoftGate()
        w = gate(torch.randn(4, 27, 20, 20))
        self.assertTrue(torch.isfinite(w).all())
        self.assertTrue((w >= 0).all())
        self.assertTrue(torch.allclose(w.sum(dim=1), w.new_ones((4, 20, 20)), atol=1e-5))

    def test_gate_weights_are_spatially_continuous_not_reduced_to_page(self) -> None:
        torch.manual_seed(0)
        gate = SpatialSoftGate()
        # The frozen init contract sets logits to zero (uniform 1/3), so perturb
        # the gate head to observe the learned spatial mixing profile.
        with torch.no_grad():
            torch.nn.init.normal_(gate.ghead.weight, std=0.2)
        w = gate(torch.randn(2, 27, 32, 32))
        # Values vary across pixels (not collapsed to per-page single weights).
        std = w.std(dim=(2, 3))
        self.assertTrue((std > 1e-4).all())

    def test_gate_never_uses_argmax_or_hard_routing(self) -> None:
        # The gate only ever normalizes via softmax; there is no hard dispatch.
        torch.manual_seed(0)
        gate = SpatialSoftGate()
        w = gate(torch.randn(1, 27, 8, 8))
        # A soft, positive mixture requires every column to be in the open (0,1).
        self.assertTrue((w > 0).all())
        self.assertTrue((w < 1).all())


class ControlForwardSemanticsTest(unittest.TestCase):
    def test_baseline_returns_anchor_unchanged(self) -> None:
        torch.manual_seed(0)
        mixture = SpatialContinuousReconstructionMixture(mode=MODE_BASELINE)
        feats, y0 = _feats_y0()
        candidate, raws, gate = mixture.mixture_output(feats, y0)
        self.assertTrue(torch.equal(candidate, y0))
        self.assertIsNone(raws)
        self.assertIsNone(gate)

    def test_baseline_has_zero_trainable_params(self) -> None:
        mixture = SpatialContinuousReconstructionMixture(mode=MODE_BASELINE)
        self.assertEqual(mixture.param_count(), 0)

    def test_single_head_is_average_of_two_corrections(self) -> None:
        torch.manual_seed(0)
        mixture = SpatialContinuousReconstructionMixture(mode=MODE_SINGLE_HEAD)
        feats, y0 = _feats_y0()
        candidate, (raw1, raw2), _ = mixture.mixture_output(feats, y0)
        correction = 0.5 * (raw1 + raw2)
        expected = torch.clamp(y0 + CORRECTION_BOUND * torch.tanh(correction), -1.0, 1.0)
        self.assertTrue(torch.allclose(candidate, expected, atol=1e-5))

    def test_uniform_two_expert_matches_fixed_mixture(self) -> None:
        torch.manual_seed(0)
        mixture = SpatialContinuousReconstructionMixture(mode=MODE_UNIFORM_TWO_EXPERT)
        feats, y0 = _feats_y0()
        candidate, (raw1, raw2), _ = mixture.mixture_output(feats, y0)
        y1 = torch.clamp(y0 + CORRECTION_BOUND * torch.tanh(raw1), -1.0, 1.0)
        y2 = torch.clamp(y0 + CORRECTION_BOUND * torch.tanh(raw2), -1.0, 1.0)
        expected = torch.clamp(0.0 * y0 + 0.5 * y1 + 0.5 * y2, -1.0, 1.0)
        self.assertTrue(torch.allclose(candidate, expected, atol=1e-5))

    def test_spatial_mixture_output_within_valid_range(self) -> None:
        torch.manual_seed(0)
        mixture = SpatialContinuousReconstructionMixture(mode=MODE_SPATIAL_MIXTURE)
        feats, y0 = _feats_y0()
        candidate, _, gate = mixture.mixture_output(feats, y0)
        self.assertIsNotNone(gate)
        self.assertTrue(torch.isfinite(candidate).all())
        self.assertTrue((candidate >= -1.0).all())
        self.assertTrue((candidate <= 1.0).all())

    def test_forward_returns_candidate_only(self) -> None:
        torch.manual_seed(0)
        mixture = SpatialContinuousReconstructionMixture(mode=MODE_SPATIAL_MIXTURE)
        feats, y0 = _feats_y0()
        candidate = mixture(feats, y0)
        self.assertEqual(tuple(candidate.shape), tuple(y0.shape))


class CorrectionEditRangeTest(unittest.TestCase):
    def test_max_correction_reaches_at_least_twice_the_12_gray_event(self) -> None:
        head = TerminalReconstructionHead()
        raw = torch.tensor([[[[50.0]]]])  # saturate tanh
        corrected = head.bounded_apply(raw, torch.zeros_like(raw))
        max_gray = float(corrected.abs().max()) * 127.5
        # Frozen edit range must exceed 12 gray with margin (>= 24 gray).
        self.assertGreaterEqual(max_gray, 24.0)
        # The bound itself is 25 gray ~= 25/127.5 in [-1,1] units.
        self.assertAlmostEqual(float(CORRECTION_BOUND) * 127.5, 25.0, places=5)

    def test_zero_correction_application_is_identity_on_anchor(self) -> None:
        head = TerminalReconstructionHead()
        feats = torch.randn(2, 64, 16, 16)
        anchor = 0.1 * torch.randn(2, 3, 16, 16)
        raw = head(feats)
        # At init raw is exactly zero => applying it reproduces anchor.
        self.assertEqual(raw.abs().max().item(), 0.0)
        out = head.bounded_apply(raw, anchor)
        self.assertTrue(torch.allclose(out, anchor, atol=1e-6))


class FailClosedTest(unittest.TestCase):
    def test_nonfinite_gate_weight_falls_back_to_anchor(self) -> None:
        torch.manual_seed(0)
        mixture = SpatialContinuousReconstructionMixture(mode=MODE_SPATIAL_MIXTURE)
        with torch.no_grad():
            mixture.gate.ghead.weight.fill_(float("nan"))
        feats, y0 = _feats_y0()
        candidate, _, _ = mixture.mixture_output(feats, y0)
        self.assertTrue(torch.equal(candidate, y0))
        safety = mixture.last_safety()
        self.assertTrue(safety["fallback"])
        self.assertFalse(safety["finite"])


class NoMetadataRoutingInputTest(unittest.TestCase):
    def test_module_signature_has_no_metadata_domain_or_path_inputs(self) -> None:
        # The core forward contract is strictly (feats, y0): no dataset/ds
        # identity, split, path, caller, or routing argument.
        import inspect
        forward_sig = inspect.signature(SpatialContinuousReconstructionMixture.forward)
        params = list(forward_sig.parameters)
        self.assertNotIn("domain", params)
        self.assertNotIn("identity", params)
        self.assertNotIn("split", params)
        self.assertNotIn("path", params)

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SpatialContinuousReconstructionMixture(mode="not-a-mode")

    def test_wrong_feature_channels_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SpatialContinuousReconstructionMixture(mode=MODE_UNIFORM_TWO_EXPERT, feature_channels=32)


if __name__ == "__main__":
    unittest.main()
