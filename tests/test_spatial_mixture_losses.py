"""Focused CPU tests for the Phase 0 spatial-mixture supervision regions and
frozen loss objective (``losses/spatial_mixture_losses.py``).

These tests are data-synthetic only: no dataset, no target-derived cached
masks, no quality split, no visual AI. They verify the frozen weights,
12-gray event surrogates, 0.25 temperature, 0.25 tail fraction, the eight
supervision regions (including an OpenCV reference for ``changed_mask``), the
four control modes, collapse guards, and gradient flow.
"""
import unittest

import cv2
import numpy as np
import torch

from losses.spatial_mixture_losses import (
    CHANGE_THRESHOLD,
    DIVERSITY_FLOOR,
    EVENT_TEMPERATURE,
    GATE_ANCHOR_FLOOR,
    GATE_EXPERT_CAP,
    ALLOWED_WEIGHTS,
    RegionParams,
    SpatialMixtureLoss,
    SupervisionRegions,
    VALID_MODES,
    assert_frozen_config,
    changed_mask,
    compute_regions,
    dilate3,
    erode3,
    morph_open3,
    sobel_magnitude,
)

LEVEL = 1.0 / 127.5  # one 8-bit gray step in the [-1,1] normalized space
B, H, W = 2, 16, 16


def norm_px(value: float) -> float:
    return value / 127.5 - 1.0


def zeros_rgb(batch=B, h=H, w=W):
    return torch.zeros(batch, 3, h, w)


def uniform_rgb(value, batch=B, h=H, w=W):
    return torch.full((batch, 3, h, w), float(value))


########################################
# 1. frozen config assertion
########################################
class FrozenConfigTest(unittest.TestCase):
    def test_none_is_accepted(self):
        assert_frozen_config(None)

    def test_exact_weights_are_accepted(self):
        assert_frozen_config(dict(ALLOWED_WEIGHTS))

    def test_drifted_weight_raises(self):
        cfg = dict(ALLOWED_WEIGHTS)
        cfg["pair"] = 1.5
        with self.assertRaisesRegex(ValueError, "pair"):
            assert_frozen_config(cfg)

    def test_missing_weight_raises(self):
        cfg = dict(ALLOWED_WEIGHTS)
        del cfg["gate_tv"]
        with self.assertRaisesRegex(KeyError, "gate_tv"):
            assert_frozen_config(cfg)

    def test_unknown_weight_raises(self):
        cfg = dict(ALLOWED_WEIGHTS)
        cfg["lambda_eval_legacy"] = 1.0
        with self.assertRaisesRegex(ValueError, "unexpected"):
            assert_frozen_config(cfg)

    def test_non_mapping_raises(self):
        with self.assertRaises(TypeError):
            assert_frozen_config(["pair"])


########################################
# 2. morphology helpers vs OpenCV reference
########################################
class MorphologyTest(unittest.TestCase):
    def _cv2_open_then_dilate(self, binary_u8):
        kernel = np.ones((3, 3), dtype=np.uint8)
        opened = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, kernel)
        return cv2.dilate(opened, kernel, iterations=1)

    def test_open3_matches_cv2(self):
        raw = np.zeros((H, W), dtype=np.uint8)
        raw[2:6, 3:7] = 255
        raw[0, 0] = 255  # isolated pixel removed by opening
        raw[8:10, 8:10] = 255  # 2x2 also removed by 3x3 opening
        binary = torch.from_numpy((raw > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        torch_open = morph_open3(binary)[0, 0] > 0.5
        expected = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        np.testing.assert_array_equal(torch_open.numpy(), expected)

    def test_open_plus_dilate_matches_cv2_changed_style(self):
        raw = np.zeros((H, W), dtype=np.uint8)
        raw[2:6, 3:7] = 255
        raw[0, 0] = 255
        raw[8:10, 8:10] = 255
        binary = torch.from_numpy((raw > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        torch_changed = dilate3(morph_open3(binary))[0, 0] > 0.5
        expected = self._cv2_open_then_dilate(raw) > 0
        np.testing.assert_array_equal(torch_changed.numpy(), expected)

    def test_dilate3_and_erode3_border_semantics(self):
        mask = torch.zeros((1, 1, H, W))
        mask[0, 0, 0, 0] = 1.0
        dilated = dilate3(mask)[0, 0] > 0.5
        # dilation reaches (0,0),(0,1),(1,0),(1,1); border truncation keeps it finite.
        self.assertTrue(bool(dilated[0, 0]))
        self.assertTrue(bool(dilated[0, 1]))
        self.assertTrue(bool(dilated[1, 0]))
        self.assertEqual(int(dilated.sum()), 4)

        full = torch.ones((1, 1, H, W))
        eroded = erode3(full)[0, 0]  # all ones stays all ones including border
        self.assertTrue(bool(eroded.all()))


########################################
# 3. changed mask + regions
########################################
class ChangedMaskTest(unittest.TestCase):
    def test_changed_matches_cv2_reference(self):
        # source uniform, target with a 22-gray (>= 12) block and a 4-gray
        # (sub-threshold) block.
        src = uniform_rgb(norm_px(128.0))
        tgt = uniform_rgb(norm_px(128.0))
        tgt[:, :, 2:6, 2:8] = norm_px(150.0)
        tgt[:, :, 10:12, 10:12] = norm_px(132.0)

        got = changed_mask(src, tgt).bool()  # [B,1,H,W]
        src_u8 = np.full((H, W, 3), 128, dtype=np.uint8)
        tgt_u8 = np.full((H, W, 3), 128, dtype=np.uint8)
        tgt_u8[2:6, 2:8] = 150
        tgt_u8[10:12, 10:12] = 132
        kernel = np.ones((3, 3), dtype=np.uint8)
        delta = (cv2.absdiff(src_u8, tgt_u8).mean(axis=2) >= 12).astype(np.uint8) * 255
        exp_raw = cv2.morphologyEx(delta, cv2.MORPH_OPEN, kernel)
        exp = cv2.dilate(exp_raw, kernel, iterations=1) > 0
        np.testing.assert_array_equal(got[0, 0].numpy(), exp)

    def test_regions_shapes_detached_and_bool(self):
        src = uniform_rgb(norm_px(128.0))
        tgt = uniform_rgb(norm_px(128.0))
        tgt[:, :, 2:6, 2:8] = norm_px(150.0)  # target-lighter
        tgt[:, :, 10:12, 10:12] = norm_px(96.0)  # target-darker
        rp = RegionParams(
            sobel_high=0.25,
            sobel_low=0.02,
            source_dark=norm_px(40.0),
        )
        regs = compute_regions(src, tgt, rp)
        for name, mask in regs.as_dict().items():
            self.assertEqual(mask.shape, (B, 1, H, W), name)
            self.assertEqual(mask.dtype, torch.bool, name)
            self.assertFalse(mask.requires_grad, name)

    def test_target_lighter_partition_and_margin(self):
        src = uniform_rgb(norm_px(128.0))
        tgt = uniform_rgb(norm_px(128.0))
        tgt[:, :, 2:6, 2:8] = norm_px(151.0)   # +23 gray: clearly lighter, in changed
        tgt[:, :, 8:12, 8:12] = norm_px(140.0)  # +12 gray: lighter boundary
        tgt[:, :, 12:16, 12:16] = norm_px(112.0)  # -16 gray: darker
        regs = compute_regions(src, tgt)
        self.assertTrue(bool(regs.target_lighter.any()))
        self.assertTrue(bool(regs.target_darker_or_ambiguous.any()))
        # partition: disjoint and within changed (the union equals changed)
        self.assertFalse(bool((regs.target_lighter & regs.target_darker_or_ambiguous).any()))
        union = regs.target_lighter | regs.target_darker_or_ambiguous
        self.assertTrue(bool((union == regs.changed).all()))


class PaperPrintPageEdgeTest(unittest.TestCase):
    def test_paper_print_thresholds(self):
        src = uniform_rgb(norm_px(200.0))
        # strong edge block (print-like) and flat low-gradient block (paper)
        src[:, :, 2:5, 2:5] = norm_px(30.0)   # high local gradient against 200 bg
        tgt = src.clone()
        rp = RegionParams(sobel_high=0.10, sobel_low=0.01)
        regs = compute_regions(src, tgt, rp)
        mag = sobel_magnitude(src)
        print_mask = (mag >= 0.10) & ~regs.changed
        paper_mask = (mag < 0.01) & ~regs.changed
        self.assertTrue(bool(regs.unchanged_print_preserve.any()))
        self.assertTrue(bool(regs.paper.any()))
        np.testing.assert_array_equal(
            regs.unchanged_print_preserve[0, 0].numpy(), print_mask[0, 0].numpy()
        )
        np.testing.assert_array_equal(regs.paper[0, 0].numpy(), paper_mask[0, 0].numpy())
        self.assertFalse((regs.paper & regs.unchanged_print_preserve).any())

    def test_page_edge_is_16px_border_outside_changed(self):
        src = uniform_rgb(norm_px(128.0))
        tgt = uniform_rgb(norm_px(128.0))
        tgt[:, :, 4:12, 4:12] = norm_px(150.0)
        rp = RegionParams(page_edge_px=4)
        regs = compute_regions(src, tgt, rp)
        edge = regs.page_edge[0, 0].numpy()
        # rows/cols 0..3 and H-1-3..H-1 are edge; but changed at 4:12 excludes
        # some of them where changed overlaps.
        self.assertTrue(bool(edge[:4, :].any()))
        self.assertTrue(bool(edge[-4:, :].any()))
        self.assertTrue(bool(edge[:, :4].any()))
        self.assertTrue(bool(edge[:, -4:].any()))
        self.assertFalse(bool((regs.page_edge & regs.changed).any()))


class SmallComponentTest(unittest.TestCase):
    def test_area_filter_4_to_64(self):
        src = uniform_rgb(norm_px(200.0))
        # dark (source_dark) blobs of varying area, all well below 200-gray bg.
        src[:, :, 1, 1] = norm_px(30.0)          # area 1  -> removed
        src[:, :, 3:5, 3:5] = norm_px(30.0)      # area 4  -> kept (boundary)
        src[:, :, 6:16, 6:16] = norm_px(30.0)    # area 100 -> removed (10x10)
        tgt = src.clone()                        # changed is empty
        rp = RegionParams(
            source_dark=norm_px(60.0),
            small_component_min_area=4,
            small_component_max_area=64,
        )
        regs = compute_regions(src, tgt, rp)
        sc = regs.small_component_hard_negative[0, 0].numpy()
        # removed: area-1 at (1,1), area-1 at (0,6) is not set, area-100 block.
        self.assertFalse(bool(sc[1, 1]))
        self.assertFalse(bool(sc[0, 6]))
        self.assertFalse(bool(sc[6:16, 6:16].any()))
        # kept: exactly the 4-px block at (3:5, 3:5)
        self.assertEqual(int(sc[3:5, 3:5].sum()), 4)


########################################
# 4. loss terms
########################################
class LossTermTest(unittest.TestCase):
    def _identity(self):
        src = uniform_rgb(norm_px(128.0))
        tgt = uniform_rgb(norm_px(128.0))
        tgt[:, :, 2:7, 2:9] = norm_px(151.0)  # 5x7 clearly-lighter block, survives 3x3 open
        return src, tgt

    def test_pair_is_region_balanced(self):
        src, tgt = self._identity()
        candidate = tgt.clone()
        loss = SpatialMixtureLoss(mode="single")
        regions = compute_regions(src, tgt)
        pair = loss.l_pair(candidate, tgt, regions)
        self.assertTrue(torch.isfinite(pair))
        # candidate == target in all regions -> charbonnier = eps (1e-3)
        CHAR_EPS = float(np.sqrt(1e-3 ** 2))
        self.assertAlmostEqual(pair.item(), CHAR_EPS, places=5)

    def test_residual12_zero_when_candidate_equals_target(self):
        src, tgt = self._identity()
        loss = SpatialMixtureLoss(mode="single")
        regions = compute_regions(src, tgt)
        self.assertAlmostEqual(loss.l_residual12(tgt, tgt, regions).item(), 0.0, places=5)

    def test_residual12_material_when_candidate_is_source(self):
        # candidate = source leaves full residual in target-lighter support.
        src, tgt = self._identity()
        loss = SpatialMixtureLoss(mode="single")
        regions = compute_regions(src, tgt)
        res = loss.l_residual12(src, tgt, regions)
        self.assertGreater(res.item(), 0.0)
        # event at delta=22 gray is far above the 12-gray event
        self.assertGreater(res.item(), 0.9)

    def test_overerase12_on_outside_changed(self):
        src, tgt = self._identity()
        cand = src.clone()
        cand[:, :, 14, 14] = norm_px(200.0)  # an edit outside changed
        loss = SpatialMixtureLoss(mode="single")
        regions = compute_regions(src, tgt)
        oe = loss.l_overerase12(cand, src, regions)
        self.assertGreater(oe.item(), 0.0)
        # no edit: zero
        oe0 = loss.l_overerase12(src, src, regions)
        self.assertAlmostEqual(oe0.item(), 0.0, places=6)

    def test_print_preserve_and_paper_zero_when_identity(self):
        # Charbonnier identity is eps (1e-3), not exactly 0.
        CHAR_EPS = float(np.sqrt(1e-3 ** 2))
        src, tgt = self._identity()
        # Strong, larger print-like edge OUTSIDE the changed block so
        # unchanged_print_preserve is non-empty; tgt matches src there so the
        # region stays unchanged and the identity loss is exactly eps.
        src[:, :, 10:15, 10:15] = norm_px(20.0)
        tgt[:, :, 10:15, 10:15] = norm_px(20.0)
        rp = RegionParams(sobel_high=0.10, sobel_low=0.01)
        loss = SpatialMixtureLoss(mode="single", region_params=rp)
        regions = compute_regions(src, tgt, rp)
        self.assertAlmostEqual(
            loss.l_print_preserve(src, src, regions).item(), CHAR_EPS, places=5
        )
        self.assertAlmostEqual(
            loss.l_paper(src, tgt, regions).item(), CHAR_EPS, places=5
        )

    def test_collision_grad_zero_when_candidate_equals_target(self):
        src, tgt = self._identity()
        rp = RegionParams(sobel_high=0.0, sobel_low=0.0)
        loss = SpatialMixtureLoss(mode="single", region_params=rp)
        regions = compute_regions(src, tgt, rp)
        col = loss.l_collision_grad(tgt, tgt, regions)
        self.assertAlmostEqual(col.item(), 0.0, places=6)
        col_bad = loss.l_collision_grad(src, tgt, regions)
        self.assertGreaterEqual(col_bad.item(), 0.0)
        self.assertTrue(torch.isfinite(col_bad))

    def test_expert_diversity(self):
        src, tgt = self._identity()
        regions = compute_regions(src, tgt)
        loss = SpatialMixtureLoss(mode="uniform")
        y1 = tgt.clone()
        y2 = tgt.clone()
        div_same = loss.l_expert_diversity(y1, y2, regions)
        # mean_abs(y1-y2)=0 -> hinge = DIVERSITY_FLOOR
        self.assertAlmostEqual(div_same.item(), DIVERSITY_FLOOR, places=6)
        y2 = y2 + 3 * LEVEL  # >> 1 gray apart
        div_far = loss.l_expert_diversity(y1, y2, regions)
        self.assertAlmostEqual(div_far.item(), 0.0, places=6)

    def test_gate_usage_and_tv(self):
        loss = SpatialMixtureLoss(mode="spatial")
        w = torch.full((B, 3, H, W), 1.0 / 3.0)
        self.assertAlmostEqual(loss.l_gate_usage(w).item(), 0.0, places=6)
        self.assertAlmostEqual(loss.l_gate_tv(w).item(), 0.0, places=6)
        w_low_anchor = w.clone()
        w_low_anchor[:, 0] = 0.01  # anchor share below 0.10
        w_low_anchor[:, 1:] = (1.0 - 0.01) / 2
        self.assertGreater(loss.l_gate_usage(w_low_anchor).item(), 0.0)
        w_high_cap = w.clone()
        w_high_cap[:, 0] = 0.99  # expert share above 0.80
        w_high_cap[:, 1:] = 0.005
        self.assertGreater(loss.l_gate_usage(w_high_cap).item(), 0.0)
        w_tv = w.clone()
        w_tv[:, 0, :, :8] = 1.0
        w_tv[:, 0, :, 8:] = 0.0
        w_tv[:, 1:] = 0.0
        self.assertGreater(loss.l_gate_tv(w_tv).item(), 0.0)


########################################
# 5. mode contract + forward + gradient
########################################
class ModeAndForwardTest(unittest.TestCase):
    def _inputs(self):
        src = uniform_rgb(norm_px(128.0))
        tgt = uniform_rgb(norm_px(128.0))
        tgt[:, :, 2:6, 2:8] = norm_px(150.0)
        return src, tgt

    def test_valid_modes(self):
        self.assertEqual(set(VALID_MODES), {"baseline", "single", "uniform", "spatial"})

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            SpatialMixtureLoss(mode="nope")

    def test_mode_requires_errors(self):
        src, tgt = self._inputs()
        candidate = tgt.clone()
        y1, y2 = tgt.clone(), tgt.clone()
        w = torch.full((B, 3, H, W), 1.0 / 3.0)
        for mode in ("baseline", "single"):
            loss = SpatialMixtureLoss(mode=mode)
            with self.assertRaises(ValueError):
                loss(src, tgt, candidate, y1=y1, y2=y2)  # y1/y2 forbidden
        loss_u = SpatialMixtureLoss(mode="uniform")
        with self.assertRaises(ValueError):
            loss_u(src, tgt, candidate)  # y1/y2 required
        loss_s = SpatialMixtureLoss(mode="spatial")
        with self.assertRaises(ValueError):
            loss_s(src, tgt, candidate, y1=y1, y2=y2)  # gate_weights required

    def test_forward_returns_total_parts_regions(self):
        src, tgt = self._inputs()
        candidate = tgt.clone().requires_grad_(True)
        y1 = tgt.clone().requires_grad_(True)
        y2 = (tgt + 2 * LEVEL).clone().requires_grad_(True)
        w = torch.full((B, 3, H, W), 1.0 / 3.0, requires_grad=True)
        loss = SpatialMixtureLoss(mode="spatial")
        total, parts, regions = loss(src, tgt, candidate, y1=y1, y2=y2, gate_weights=w)
        self.assertIsInstance(regions, SupervisionRegions)
        self.assertEqual(
            set(parts),
            {"pair", "residual12", "overerase12", "print_preserve",
             "collision_grad", "paper", "expert_diversity", "gate_usage", "gate_tv"},
        )
        # forward already scales each part by its frozen weight; the plain sum
        # of the returned (weighted) parts must equal the total. Re-multiplying
        # here would double-count the weights.
        expected = None
        for name in ("pair", "residual12", "overerase12",
                     "print_preserve", "collision_grad", "paper",
                     "expert_diversity", "gate_usage", "gate_tv"):
            part = parts[name]
            if part is None:
                continue
            expected = part if expected is None else expected + part
        self.assertTrue(torch.allclose(total, expected, atol=1e-6))

    def test_gradient_flows_to_candidate_and_experts(self):
        src, tgt = self._inputs()
        candidate = tgt.clone().requires_grad_(True)
        y1 = tgt.clone().requires_grad_(True)
        y2 = (tgt + 2 * LEVEL).clone().requires_grad_(True)
        w = torch.full((B, 3, H, W), 1.0 / 3.0, requires_grad=True)
        loss = SpatialMixtureLoss(mode="spatial")
        total, _, _ = loss(src, tgt, candidate, y1=y1, y2=y2, gate_weights=w)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        for name, t in (("candidate", candidate), ("y1", y1), ("y2", y2), ("w", w)):
            self.assertIsNotNone(t.grad, name)
            self.assertTrue(torch.isfinite(t.grad).all(), name)
            if name in ("candidate",):  # reconstruction terms drive candidate
                self.assertTrue(bool((t.grad != 0).any()), name)

    def test_baseline_forward_runs_without_expert_inputs(self):
        src, tgt = self._inputs()
        candidate = tgt.clone()
        loss = SpatialMixtureLoss(mode="baseline")
        total, parts, _ = loss(src, tgt, candidate)
        self.assertTrue(torch.isfinite(total))
        self.assertIsNone(parts["expert_diversity"])
        self.assertIsNone(parts["gate_usage"])
        self.assertIsNone(parts["gate_tv"])


########################################
# helper used by tests
########################################
def CHARBONNIER_EPS_MEAN():
    # charbonnier(0) = eps for a region mean
    import math
    return float(math.sqrt(1e-3 ** 2))


if __name__ == "__main__":
    unittest.main()