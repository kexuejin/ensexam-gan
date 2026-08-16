"""Phase 0 supervision regions and frozen objective for the spatial continuous
reconstruction mixture program.

This module implements the four control modes (``baseline``, ``single``,
``uniform``, ``spatial``) and the exact loss weights, 12-gray event surrogates,
0.25-gray temperature, 0.25 tail fraction, and region definitions frozen in
``docs/plans/2026-08-16-spatial-continuous-reconstruction-mixture-implementation.md``.

Contract (from the implementation plan):

- No GAN, discriminator, VGG perceptual, style, or legacy mask losses. No
  checkpoints, dataset identity, split identity, domain label, file path, or
  caller hint reach the model or any term.
- Region geometry is deterministic and local over the source/target pair and the
  frozen train-pool materialization thresholds. It never touches target-derived
  masks cached from other roles, never reads the dataset, and never opens any
  quality split. All region masks are detached (data-derived constants), so the
  only differentiable path is the candidate, expert, and gate tensors.
- ``SpatialMixtureLoss`` asserts the frozen weights/thresholds explicitly and
  fails closed on any config that drifts from the preregistered values. The
  legacy ``lambda_eval_*`` keys were never wired into ``train.py``/``losses.py``,
  so this module consumes and asserts its own frozen values rather than rely on
  any legacy wiring.

Normalization convention: inputs are the project's ``[-1, 1]`` tensors
(``x*127.5 + 127.5`` recovers 0..255 gray). A gray level ``g`` is ``g / 127.5``
in this space, matching ``EnsExamLoss`` (``12/127.5``) and the eval metric. The
``changed`` mask reuses the exact 3x3 open-then-dilate morphology of
``scripts/eval/eval_hardcase_worst_pages.py::build_changed_mask`` via an
equivalent deterministic tensor implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Frozen numeric constants (preregistered; do not drift).
# --------------------------------------------------------------------------- #
GRAY_DIVISOR = 127.5
CHANGE_THRESHOLD_GRAY = 12.0
EVENT_TEMPERATURE_GRAY = 0.25
TAIL_FRACTION = 0.25
CHANGE_THRESHOLD = CHANGE_THRESHOLD_GRAY / GRAY_DIVISOR
EVENT_TEMPERATURE = EVENT_TEMPERATURE_GRAY / GRAY_DIVISOR
TARGET_LIGHTER_MARGIN = 2.0 / GRAY_DIVISOR

PAGE_EDGE_PX = 16
COLLISION_BAND_PX = 2
SMALL_COMPONENT_MIN_AREA = 4
SMALL_COMPONENT_MAX_AREA = 64

CHARBONNIER_EPS = 1e-3

# L_total weights (frozen).
W_PAIR = 1.00
W_RESIDUAL = 2.15
W_OVERERASE = 4.75
W_PRINT_PRESERVE = 2.00
W_COLLISION_GRAD = 1.00
W_PAPER = 0.50
W_EXPERT_DIVERSITY = 0.05
W_GATE_USAGE = 0.02
W_GATE_TV = 0.05

# Anti-collapse floors (normalized-space thresholds).
DIVERSITY_FLOOR = 1.0 / GRAY_DIVISOR      # 1 gray in [-1,1] units
GATE_ANCHOR_FLOOR = 0.10
GATE_EXPERT_CAP = 0.80

ALLOWED_WEIGHTS = {
    "pair": W_PAIR,
    "residual": W_RESIDUAL,
    "overerase": W_OVERERASE,
    "print_preserve": W_PRINT_PRESERVE,
    "collision_grad": W_COLLISION_GRAD,
    "paper": W_PAPER,
    "expert_diversity": W_EXPERT_DIVERSITY,
    "gate_usage": W_GATE_USAGE,
    "gate_tv": W_GATE_TV,
}

VALID_MODES = ("baseline", "single", "uniform", "spatial")


def assert_frozen_config(config):
    """Fail closed unless a provided loss config matches the frozen weights.

    ``None`` means the frozen defaults are used (no-op). The plan declares the
    phase-0 objective is frozen and one-shot, so any drifted weight must
    terminate the run rather than silently change the objective.
    """
    if config is None:
        return
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping or None")
    for key, expected in ALLOWED_WEIGHTS.items():
        if key not in config:
            raise KeyError(f"missing frozen loss weight '{key}'")
        try:
            actual = float(config[key])
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"loss weight '{key}' must be numeric") from exc
        if abs(actual - expected) > 1e-12:
            raise ValueError(
                f"frozen loss weight '{key}' drifted: expected {expected}, got {actual}"
            )
    unknown = sorted(set(config) - set(ALLOWED_WEIGHTS))
    if unknown:
        raise ValueError(f"unexpected loss config keys: {unknown}")


# --------------------------------------------------------------------------- #
# Deterministic morphology helpers (equivalent to cv2 3x3 square kernel).
# --------------------------------------------------------------------------- #
def dilate3(binary):
    """3x3 max-dilation with out-of-bounds neighbours ignored (cv2 semantics).

    ``binary`` is a 4-D float tensor in ``{0,1}``. Padding with 0 is neutral for
    a max/dilation, so a border pixel is 1 iff any in-bounds neighbour is 1.
    """
    padded = F.pad(binary, (1, 1, 1, 1), mode="constant", value=0.0)
    return F.max_pool2d(padded, kernel_size=3, stride=1)


def erode3(binary):
    """3x3 min-erosion with out-of-bounds neighbours ignored (cv2 semantics).

    Padding with 1 is neutral for a min/erosion, so a border pixel stays 1 iff
    every in-bounds neighbour is 1.
    """
    padded = F.pad(binary, (1, 1, 1, 1), mode="constant", value=1.0)
    return -F.max_pool2d(-padded, kernel_size=3, stride=1)


def morph_open3(binary):
    """3x3 opening (erode then dilate), matching cv2 MORPH_OPEN + ones(3,3)."""
    return dilate3(erode3(binary))


def changed_mask(source, target):
    """Mean-RGB-abs changed mask with an open-then-dilate morphology.

    Mirrors ``build_changed_mask`` (threshold >= 12 gray, 3x3 open, then one
    3x3 dilation) in deterministic tensor form. Returns a float ``{0,1}``
    tensor of the same shape as the input.
    """
    delta = (source - target).abs().mean(dim=1, keepdim=True)
    binary = (delta >= CHANGE_THRESHOLD).to(dtype=source.dtype)
    opened = morph_open3(binary)
    return dilate3(opened)


def sobel_magnitude(image):
    """Sobel gradient magnitude on mean-channel luminance.

    Returns a 4-D float tensor in the same normalized space as ``image``. The
    threshold percentiles materialized from the train pool use this exact
    function so the loss and the materialization thresholds agree.
    """
    luma = image.mean(dim=1, keepdim=True)
    device, dtype = luma.device, luma.dtype
    wx = torch.tensor([[-1.0, 0.0, 1.0],
                       [-2.0, 0.0, 2.0],
                       [-1.0, 0.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    wy = torch.tensor([[-1.0, -2.0, -1.0],
                       [0.0, 0.0, 0.0],
                       [1.0, 2.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    dx = F.conv2d(luma, wx, padding=1)
    dy = F.conv2d(luma, wy, padding=1)
    return (dx.square() + dy.square()).sqrt()


# --------------------------------------------------------------------------- #
# Supervision regions.
# --------------------------------------------------------------------------- #
@dataclass
class RegionParams:
    """Frozen train-pool materialization thresholds.

    ``sobel_high`` / ``sobel_low`` / ``source_dark`` are computed once from the
    eligible train pool and frozen in the master manifest before any run. They
    are passed here so the loss's regions match the materialized thresholds.
    ``sobel_high``/``sobel_low`` are in the units of :func:`sobel_magnitude`;
    ``source_dark`` is a mean-luminance gray threshold divided by
    ``GRAY_DIVISOR``.
    """

    sobel_high: float = 0.0
    sobel_low: float = 0.0
    source_dark: float = 0.0
    page_edge_px: int = PAGE_EDGE_PX
    collision_band_px: int = COLLISION_BAND_PX
    small_component_min_area: int = SMALL_COMPONENT_MIN_AREA
    small_component_max_area: int = SMALL_COMPONENT_MAX_AREA


@dataclass
class SupervisionRegions:
    """Deterministic train-only supervision regions (detached bool tensors)."""

    changed: torch.Tensor
    target_lighter: torch.Tensor
    target_darker_or_ambiguous: torch.Tensor
    unchanged_print_preserve: torch.Tensor
    collision_boundary: torch.Tensor
    paper: torch.Tensor
    page_edge: torch.Tensor
    small_component_hard_negative: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "changed": self.changed,
            "target_lighter": self.target_lighter,
            "target_darker_or_ambiguous": self.target_darker_or_ambiguous,
            "unchanged_print_preserve": self.unchanged_print_preserve,
            "collision_boundary": self.collision_boundary,
            "paper": self.paper,
            "page_edge": self.page_edge,
            "small_component_hard_negative": self.small_component_hard_negative,
        }


def _connected_components_small(dark_binary, *, min_area, max_area):
    """Exact connected-component filter for the small-component hard negative.

    Operates on a detached ``[B,1,H,W]`` float ``{0,1}`` tensor. Uses OpenCV's
    connected-component labelling (8-connectivity) on detached data, which is
    deterministic, local, and identical to the repo's metric tooling. No visual
    AI and no dataset access. Returns a detached bool ``[B,1,H,W]`` mask
    selecting only blobs whose area lies within ``[min_area, max_area]``.
    """
    import cv2
    import numpy as np

    if dark_binary.ndim != 4 or dark_binary.shape[1] != 1:
        raise ValueError("expected [B,1,H,W] binary input")
    if min_area < 1 or max_area < min_area:
        raise ValueError("small-component area bounds are invalid")

    batch, _, h, w = dark_binary.shape
    arr = (dark_binary.detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    out = np.zeros((batch, h, w), dtype=np.bool_)
    for b in range(batch):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            arr[b, 0], connectivity=8
        )
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if min_area <= area <= max_area:
                out[b][labels == label] = True
    return torch.from_numpy(out).view(batch, 1, h, w).to(device=dark_binary.device)


def compute_regions(source, target, region_params=None):
    """Compute all Phase 0 supervision regions from a source/target pair.

    ``source``/``target`` are ``[B,3,H,W]`` normalized tensors. All returned
    masks are detached ``[B,1,H,W]`` booleans on the source device.
    """
    if source.shape != target.shape:
        raise ValueError(f"source/target shape mismatch: {source.shape} vs {target.shape}")
    if source.ndim != 4 or source.shape[1] != 3:
        raise ValueError("source/target must be [B,3,H,W]")
    rp = region_params or RegionParams()
    device = source.device
    batch, _, h, w = source.shape

    changed = changed_mask(source, target).bool()

    luma_src = source.mean(dim=1, keepdim=True)
    luma_tgt = target.mean(dim=1, keepdim=True)
    target_lighter = changed & ((luma_tgt - luma_src) > TARGET_LIGHTER_MARGIN)
    target_darker_or_ambiguous = changed & ~target_lighter

    mag = sobel_magnitude(source)
    unchanged_print_preserve = (mag >= rp.sobel_high) & ~changed
    paper = (mag < rp.sobel_low) & ~changed

    band = target_lighter.float()
    for _ in range(max(rp.collision_band_px, 1)):
        band = dilate3(band)
    collision_boundary = unchanged_print_preserve & band.bool()

    row = torch.arange(h, device=device).view(-1, 1)
    col = torch.arange(w, device=device).view(1, -1)
    is_row_edge = (row < rp.page_edge_px) | ((h - row - 1) < rp.page_edge_px)
    is_col_edge = (col < rp.page_edge_px) | ((w - col - 1) < rp.page_edge_px)
    page_edge = changed.new_zeros(batch, 1, h, w, dtype=torch.bool)
    page_edge[:, :, is_row_edge & is_col_edge] = True
    page_edge = page_edge & ~changed

    dark_map = (luma_src < rp.source_dark).float()
    small_component_hard_negative = (
        _connected_components_small(
            dark_map,
            min_area=rp.small_component_min_area,
            max_area=rp.small_component_max_area,
        )
        & ~changed
    )

    return SupervisionRegions(
        changed=changed,
        target_lighter=target_lighter,
        target_darker_or_ambiguous=target_darker_or_ambiguous,
        unchanged_print_preserve=unchanged_print_preserve,
        collision_boundary=collision_boundary,
        paper=paper,
        page_edge=page_edge,
        small_component_hard_negative=small_component_hard_negative,
    )


# --------------------------------------------------------------------------- #
# Frozen loss objective.
# --------------------------------------------------------------------------- #
def _charbonnier(diff):
    """Charbonnier penalty ``sqrt(d^2 + eps^2)`` with eps = 1e-3."""
    return (diff.square() + CHARBONNIER_EPS ** 2).sqrt()


def _sample_means(loss_pixel, mask):
    """Per-sample mean of pixel losses over a mask, then mean over valid samples.

    ``loss_pixel`` is ``[B,C,H,W]``; ``mask`` is ``[B,1,H,W]`` bool. The
    per-sample value is the pixel mean across all channels, and the returned
    scalar is the mean over samples whose mask has at least one pixel (so an
    empty-support sample contributes nothing and does not zero out the batch).
    """
    cnt = mask.sum(dim=(1, 2, 3))
    denom = cnt.clamp_min(1.0) * loss_pixel.shape[1]
    per_sample = (loss_pixel * mask).sum(dim=(1, 2, 3)) / denom
    valid = cnt > 0
    if not bool(valid.any()):
        return loss_pixel.sum() * 0.0
    return per_sample[valid].mean()


def _sample_tail_mean(event_pixel, mask, fraction):
    """Within-page top-``fraction`` tail mean over support, then batch mean.

    For each sample, gathers the differentiable event values on the masked
    support, takes the top ``fraction`` by value, and averages them; the result
    is the mean over samples whose support is non-empty. Matches the plan's
    "top 25% per-page tail mean then batch mean".
    """
    batch = mask.shape[0]
    vals = []
    for b in range(batch):
        mask_b = mask[b].squeeze(0)          # [H, W] bool
        event_b = event_pixel[b].squeeze(0)  # [H, W] float
        sel = event_b[mask_b].reshape(-1)
        if sel.numel() == 0:
            continue
        n = sel.numel()
        k = max(1, int(round(n * float(fraction))))
        vals.append(torch.topk(sel, k=k, largest=True).values.mean())
    if not vals:
        return event_pixel.sum() * 0.0
    return torch.stack(vals).mean()


def _gate_total_variation(weights):
    """Anisotropic TV of the three gate channels, normalized and batch-mean."""
    _, _, h, w = weights.shape
    dx = (weights[..., 1:, :] - weights[..., :-1, :]).abs().sum()
    dy = (weights[..., :, 1:] - weights[..., :, :-1]).abs().sum()
    return (dx + dy) / (h * w)


class SpatialMixtureLoss(nn.Module):
    """Frozen Phase 0 objective for one control mode.

    ``mode`` selects which collapse terms are active:

    - ``baseline`` / ``single``: reconstruction terms only.
    - ``uniform``: adds ``L_expert_diversity`` using ``y1``/``y2``.
    - ``spatial``: adds ``L_expert_diversity`` plus ``L_gate_usage`` and
      ``L_gate_TV`` using ``gate_weights``.

    All weights and the frozen thresholds are asserted in ``__init__`` through
    ``assert_frozen_config`` (fails closed on any drift).
    """

    def __init__(self, mode="spatial", config=None, region_params=None):
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")
        assert_frozen_config(config)
        self.mode = mode
        self.region_params = region_params or RegionParams()
        self.frozen = {
            "pair": W_PAIR,
            "residual": W_RESIDUAL,
            "overerase": W_OVERERASE,
            "print_preserve": W_PRINT_PRESERVE,
            "collision_grad": W_COLLISION_GRAD,
            "paper": W_PAPER,
            "expert_diversity": W_EXPERT_DIVERSITY,
            "gate_usage": W_GATE_USAGE,
            "gate_tv": W_GATE_TV,
        }

    # --- reconstruction terms ---
    def l_pair(self, candidate, target, regions):
        region_masks = [
            regions.target_lighter,
            regions.target_darker_or_ambiguous,
            regions.unchanged_print_preserve,
            regions.paper,
            ~(regions.target_lighter
              | regions.target_darker_or_ambiguous
              | regions.unchanged_print_preserve
              | regions.paper),
        ]
        diffs = _charbonnier(candidate - target)
        term_sum = None
        active = 0
        for mask in region_masks:
            if not bool(mask.any()):
                continue
            m = _sample_means(diffs, mask)
            term_sum = m if term_sum is None else (term_sum + m)
            active += 1
        if active == 0:
            return diffs.sum() * 0.0
        return term_sum / active

    def l_residual12(self, candidate, target, regions):
        residual_delta = (candidate - target).abs().mean(dim=1, keepdim=True)
        event = torch.sigmoid((residual_delta - CHANGE_THRESHOLD) / EVENT_TEMPERATURE)
        return _sample_tail_mean(event, regions.target_lighter, TAIL_FRACTION)

    def l_overerase12(self, candidate, source, regions):
        outside = ~regions.changed
        edit_delta = (candidate - source).abs().mean(dim=1, keepdim=True)
        event = torch.sigmoid((edit_delta - CHANGE_THRESHOLD) / EVENT_TEMPERATURE)
        return _sample_tail_mean(event, outside, TAIL_FRACTION)

    def l_print_preserve(self, candidate, source, regions):
        return _sample_means(
            _charbonnier(candidate - source), regions.unchanged_print_preserve
        )

    def l_collision_grad(self, candidate, target, regions):
        gx = sobel_magnitude(candidate)
        gy = sobel_magnitude(target)
        return _sample_means((gx - gy).abs(), regions.collision_boundary)

    def l_paper(self, candidate, target, regions):
        return _sample_means(_charbonnier(candidate - target), regions.paper)

    # --- collapse guards ---
    def l_expert_diversity(self, y1, y2, regions):
        mean_abs = (y1 - y2).abs().mean(dim=1, keepdim=True)
        hinge = F.relu(DIVERSITY_FLOOR - mean_abs)
        return _sample_means(hinge, regions.target_lighter)

    def l_gate_usage(self, gate_weights):
        means = gate_weights.mean(dim=(2, 3))  # [B, 3] per-expert pixel share
        anchor_shortfall = F.relu(GATE_ANCHOR_FLOOR - means[:, 0:1].mean())
        expert_cap = F.relu(means - GATE_EXPERT_CAP).sum(dim=1).mean()
        return anchor_shortfall + expert_cap

    def l_gate_tv(self, gate_weights):
        return _gate_total_variation(gate_weights)

    # --- forward ---
    def forward(
        self,
        source,
        target,
        candidate,
        *,
        y1=None,
        y2=None,
        gate_weights=None,
        regions=None,
    ):
        """Compute L_total and per-part losses.

        ``source``/``target``/``candidate`` are ``[B,3,H,W]`` normalized tensors.
        ``regions`` may be precomputed (:class:`SupervisionRegions`); if None it
        is derived deterministically from ``source``/``target``. ``y1``/``y2``
        and ``gate_weights`` are required for the modes whose collapse terms use
        them and are asserted here. Returns ``(L_total, parts_dict, regions)``.
        """
        if regions is None:
            regions = compute_regions(source, target, self.region_params)

        valid_mode_requires = {
            "baseline": (y1 is None and y2 is None and gate_weights is None),
            "single": (y1 is None and y2 is None and gate_weights is None),
            "uniform": (y1 is not None and y2 is not None and gate_weights is None),
            "spatial": (y1 is not None and y2 is not None and gate_weights is not None),
        }
        if not valid_mode_requires[self.mode]:
            raise ValueError(
                f"mode {self.mode!r} requires/forbids y1/y2/gate_weights inconsistently"
            )

        L_pair = self.l_pair(candidate, target, regions) * self.frozen["pair"]
        L_residual = self.l_residual12(candidate, target, regions) * self.frozen["residual"]
        L_overerase = self.l_overerase12(candidate, source, regions) * self.frozen["overerase"]
        L_print = self.l_print_preserve(candidate, source, regions) * self.frozen["print_preserve"]
        L_collision = self.l_collision_grad(candidate, target, regions) * self.frozen["collision_grad"]
        L_paper = self.l_paper(candidate, target, regions) * self.frozen["paper"]

        L_diversity = None
        L_gate_usage = None
        L_gate_tv = None
        if self.mode in ("uniform", "spatial"):
            L_diversity = self.l_expert_diversity(y1, y2, regions) * self.frozen["expert_diversity"]
        if self.mode == "spatial":
            L_gate_usage = self.l_gate_usage(gate_weights) * self.frozen["gate_usage"]
            L_gate_tv = self.l_gate_tv(gate_weights) * self.frozen["gate_tv"]

        parts = {
            "pair": L_pair,
            "residual12": L_residual,
            "overerase12": L_overerase,
            "print_preserve": L_print,
            "collision_grad": L_collision,
            "paper": L_paper,
            "expert_diversity": L_diversity,
            "gate_usage": L_gate_usage,
            "gate_tv": L_gate_tv,
        }

        L_total = (
            L_pair
            + L_residual
            + L_overerase
            + L_print
            + L_collision
            + L_paper
            + (L_diversity if L_diversity is not None else candidate.sum() * 0.0)
            + (L_gate_usage if L_gate_usage is not None else candidate.sum() * 0.0)
            + (L_gate_tv if L_gate_tv is not None else candidate.sum() * 0.0)
        )
        return L_total, parts, regions


def compute_spatial_mixture_loss(
    *,
    y: torch.Tensor,
    y0: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    telemetry: dict | None = None,
    mode: str | None = None,
) -> tuple[torch.Tensor, dict]:
    """Frozen trainer-facing objective for one control mode.

    ``y`` is the candidate, ``y0`` the current-primary anchor, ``source`` the
    input image and ``target`` the ground truth (all ``[B,3,H,W]`` normalized
    tensors). ``telemetry`` carries optional ``y1``/``y2`` and
    ``gate_weights`` produced by the mixture adapter.

    ``mode`` defaults to ``"spatial"`` but is auto-selected from ``telemetry``
    when given: ``"spatial"`` if ``gate_weights`` present, ``"uniform"`` if only
    ``y1``/``y2`` present, else ``"single"``. Returns ``(L_total, parts)`` where
    ``parts`` mirrors ``SpatialMixtureLoss.forward`` (weighted scalar terms).
    """
    telemetry = telemetry or {}
    if mode is None:
        if telemetry.get("gate_weights") is not None:
            mode = "spatial"
        elif telemetry.get("y1") is not None and telemetry.get("y2") is not None:
            mode = "uniform"
        else:
            mode = "single"
    loss = SpatialMixtureLoss(mode=mode)
    total, parts, _regions = loss(
        source,
        target,
        y,
        y1=telemetry.get("y1"),
        y2=telemetry.get("y2"),
        gate_weights=telemetry.get("gate_weights"),
    )
    return total, parts
