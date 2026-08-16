"""
Spatial continuous reconstruction mixture network core.

Frozen by:
    docs/plans/2026-08-16-spatial-continuous-reconstruction-mixture-implementation.md

Three-output soft mixture anchored on the immutable current-primary output y0:
  E0 (anchor, exact y0, never trainable)
  E1 (erase reconstruction head, zero-init correction on y0)
  E2 (repair reconstruction head, zero-init correction on y0)
plus an optional spatial soft gate G(x -> [B,3,H,W]) mixing w0,w1,w2.

Enabled controls:
  baseline       -> y = y0 exactly (no trainable parameters here)
  single_head    -> shared trunk + two terminal heads whose corrections are
                    averaged into one RGB output added to y0
  uniform_two_expert -> shared trunk + E1/E2 with fixed weights (0, 0.5, 0.5)
  spatial_mixture    -> shared trunk + E1/E2 + spatial soft gate

This module ONLY implements the mixture core. It does not integrate with
`networks/generator.py`, does not load current-primary, and takes the
inference-available 27-channel feature bundle as a plain tensor. Defaults are
structurally fail-closed to the anchor y0.

Parameter contract (frozen F.REVISION):
  trunk  = 89,728
  head   = 111,235  (E1/E2 independent, identical channel plan)
  B_recon= trunk + 2*head = 312,198
  gate   = 28,723
  spatial total = 340,921

The frozen per-head count 111,235 requires the head middle
`Conv2d(64,64,3)` to use `bias=True` (73,984 ResBlock + 36,928 conv + 128 BN +
195 final = 111,235), exactly as the plan states. The trunk and gate first
convolutions also use `bias=True`. The `DownSample`/`UpSample` internal convs
keep `bias=False` per `networks/blocks.py`. Baseline mode constructs zero
learnable parameters (0-trainable, pure y0 pass-through).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.blocks import DownSample, ResBlock, UpSample

# Frozen capacity constants (see module docstring).
TRUNK_PARAMS = 89_728
HEAD_PARAMS = 111_235
RECON_PARAMS = TRUNK_PARAMS + 2 * HEAD_PARAMS  # 312_198
GATE_PARAMS = 28_723

# 25 eight-bit gray expressed in the [-1, 1] model tensor space.
CORRECTION_BOUND = 25.0 / 127.5

# Default structural safety tolerance on the initialized anchor equivalence.
CPU_INIT_TOLERANCE = 1e-7
MPS_INIT_TOLERANCE = 1e-6

# Control mode identifiers, matching the frozen configuration surface.
MODE_BASELINE = "baseline"
MODE_SINGLE_HEAD = "single_head"
MODE_UNIFORM_TWO_EXPERT = "uniform_two_expert"
MODE_SPATIAL_MIXTURE = "spatial_mixture"
VALID_MODES = {
    MODE_BASELINE,
    MODE_SINGLE_HEAD,
    MODE_UNIFORM_TWO_EXPERT,
    MODE_SPATIAL_MIXTURE,
}


def _verify_revision_defaults(in_channels: int, feature_channels: int) -> None:
    """Freeze the core channel plan to the approved revision."""
    if in_channels != 27:
        raise ValueError(
            f"frozen feature bundle must be 27 channels, got {in_channels}"
        )
    if feature_channels != 64:
        raise ValueError(
            f"frozen trunk/head feature width must be 64, got {feature_channels}"
        )


class SharedReconstructionTrunk(nn.Module):
    """Trainable-once trunk over the 27-channel feature bundle.

    Layer plan (exact):
        Conv2d(27, 64, 3, padding=1, bias=True) + BatchNorm + ReLU
        ResBlock(64, 64, stride=1)

    Frozen trainable count: 89,728.
    """

    def __init__(self, in_channels: int = 27, feature_channels: int = 64):
        super().__init__()
        _verify_revision_defaults(in_channels, feature_channels)
        self.in_channels = in_channels
        self.feature_channels = feature_channels
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )
        self.feature_res = ResBlock(feature_channels, feature_channels, stride=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.head_conv(feats)
        return self.feature_res(x)

    def param_count(self, *, requires_grad: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if (not requires_grad or p.requires_grad)
        )


class TerminalReconstructionHead(nn.Module):
    """Zero-init RGB correction head onto y0.

    Layer plan (exact):
        ResBlock(64, 64, stride=1)
        Conv2d(64, 64, 3, padding=1, bias=False) + BatchNorm + ReLU
        Conv2d(64, 3, 1, bias=True)   # zero-initialized weight and bias

    The final 1x1 projection is zero-initialized so the head outputs a
    zero correction at initialization, making E1/E2 reproduce y0 exactly.

    Frozen trainable count per head: 111,235.
    """

    def __init__(self, feature_channels: int = 64):
        super().__init__()
        self.feature_channels = feature_channels
        self.body_res = ResBlock(feature_channels, feature_channels, stride=1)
        self.body_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )
        self.correction = nn.Conv2d(feature_channels, 3, 1, bias=True)
        # Exact zero-initialization of the final RGB projection (weight + bias).
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Return the raw (unbounded) per-expert RGB correction tensor."""
        x = self.body_res(feats)
        x = self.body_conv(x)
        return self.correction(x)

    @torch.no_grad()
    def force_correction(self, feats: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        """Override the final projection to an explicit raw tensor (test only)."""
        self.correction.weight.zero_()
        self.correction.bias.zero_()
        # Not meaningful for arbitrary shapes; documented for the edit-range probe.
        raise NotImplementedError(
            "use a synthetic forward override, not in-place weight mutation"
        )

    def bounded_apply(self, correction: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Apply bounded correction c = BOUND*tanh(raw) onto y0 and clamp."""
        bounded = CORRECTION_BOUND * torch.tanh(correction)
        return torch.clamp(anchor + bounded, -1.0, 1.0)

    def param_count(self, *, requires_grad: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if (not requires_grad or p.requires_grad)
        )

    def zero_correction(self) -> None:
        """Force the final projection to exact zero (used for the edit probe)."""
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)


class SpatialSoftGate(nn.Module):
    """Multi-scale spatial soft gate producing [B,3,H,W] simplex weights.

    Layer plan (exact):
        Conv2d(27, 16, 3, padding=1, bias=True) + BN + ReLU
        DownSample(16, 16)
        DownSample(16, 32)
        UpSample(32, 16)
        (skip-connect: add first-downsample features after the first upsample)
        UpSample(16, 16)
        Conv2d(16, 3, 1, bias=True)
        softmax(dim=1)

    Logits initialize to ~0 (weight init near zero, default Conv init plus zero
    bias), so per-pixel weights converge to the uniform 1/3 simplex at start.
    """

    def __init__(self, in_channels: int = 27, hidden_channels: int = 16):
        super().__init__()
        if in_channels != 27:
            raise ValueError(f"frozen gate input must be 27, got {in_channels}")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.g0 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.d0 = DownSample(hidden_channels, hidden_channels)
        self.d1 = DownSample(hidden_channels, 2 * hidden_channels)
        self.u1 = UpSample(2 * hidden_channels, hidden_channels)
        self.u0 = UpSample(hidden_channels, hidden_channels)
        self.ghead = nn.Conv2d(hidden_channels, 3, 1, bias=True)
        # Zero the gate head (weight + bias) so logits initialize to exactly
        # zero, giving the frozen per-pixel uniform simplex w=(1/3,1/3,1/3).
        nn.init.zeros_(self.ghead.weight)
        nn.init.zeros_(self.ghead.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x0 = self.g0(feats)
        d0 = self.d0(x0)          # H/2
        d1 = self.d1(d0)          # H/4
        u1 = self.u1(d1)          # H/2
        u1 = u1 + d0              # skip-connect (same spatial size, same channels)
        u0 = self.u0(u1)          # H
        logits = self.ghead(u0)
        del u1, d0, d1, x0
        return F.softmax(logits, dim=1)

    def param_count(self, *, requires_grad: bool = True) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if (not requires_grad or p.requires_grad)
        )


class SpatialContinuousReconstructionMixture(nn.Module):
    """Anchored soft-mixture host.

    Encapsulates the shared trunk, independent E1/E2 heads, the optional spatial
    gate, and a control-mode assembly. The anchor `y0` is passed in as a plain
    tensor (never a module parameter) and travels unchanged through every mode,
    making the initialized mixture structurally equal to `y0`.

    Structural failures at runtime return `y0` and a numeric failure reason.
    """

    def __init__(
        self,
        mode: str = MODE_SPATIAL_MIXTURE,
        in_channels: int = 27,
        feature_channels: int = 64,
        gate_hidden_channels: int = 16,
    ):
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"unknown mixture mode {mode!r}")
        self.mode = mode
        if mode != MODE_BASELINE:
            _verify_revision_defaults(in_channels, feature_channels)

        self.trunk = None
        self.expert1 = None
        self.expert2 = None
        self.gate = None
        if mode == MODE_BASELINE:
            # Pure anchor pass-through: zero learnable parameters.
            return

        self.trunk = SharedReconstructionTrunk(
            in_channels=in_channels, feature_channels=feature_channels
        )
        self.expert1 = TerminalReconstructionHead(feature_channels=feature_channels)
        self.expert2 = TerminalReconstructionHead(feature_channels=feature_channels)

        if mode == MODE_SPATIAL_MIXTURE:
            self.gate = SpatialSoftGate(
                in_channels=in_channels, hidden_channels=gate_hidden_channels
            )

        self._numerics = {
            "reason": 0,
            "finite": True,
            "simplex_ok": True,
            "range_ok": True,
            "fallback": False,
        }

    # ── Forward ────────────────────────────────────────────────────────────
    def forward(self, feats: torch.Tensor, y0: torch.Tensor):
        """Emit (candidate)."""
        return self.mixture_output(feats, y0)[0]

    def mixture_output(self, feats: torch.Tensor, y0: torch.Tensor):
        """Emit (candidate, per-expert raw corrections, gate weights dict).

        `y0` is passed in rather than stored so the anchor can never be trained
        or mutated by this module. The output is structurally fail-closed to
        `y0` (reason code and fallback flag set) on any non-finite value.
        """
        # Sanitize the input channel contract only as a guard.
        if not torch.is_floating_point(feats):
            raise TypeError("feats must be a floating-point tensor")

        # Reset per-forward safety telemetry.
        self._numerics = {
            "reason": 0,
            "finite": True,
            "simplex_ok": True,
            "range_ok": True,
            "fallback": False,
        }

        if self.mode == MODE_BASELINE:
            return y0, None, None

        raw1 = self.expert1.forward(self.trunk(feats))
        raw2 = self.expert2.forward(self.trunk(feats))

        if self.mode == MODE_SINGLE_HEAD:
            # One learned RGB output: average the two corrections.
            correction = 0.5 * (raw1 + raw2)
            candidate = self._bounded_correction(correction, y0)
            return candidate, (raw1, raw2), None

        y1 = self.expert1.bounded_apply(raw1, y0)
        y2 = self.expert2.bounded_apply(raw2, y0)

        if self.mode == MODE_UNIFORM_TWO_EXPERT:
            w = y0.new_tensor([0.0, 0.5, 0.5])
            candidate = self._mix_anchor(w, y0, y1, y2)
            return candidate, (raw1, raw2), None

        if self.mode == MODE_SPATIAL_MIXTURE:
            w = self.gate(feats)  # [B,3,H,W]
            candidate = self._mix_anchor(w, y0, y1, y2)
            return candidate, (raw1, raw2), {"spatial": w}

        raise RuntimeError(f"unreachable mode {self.mode}")

    # ── Helpers ────────────────────────────────────────────────────────────
    def _bounded_correction(self, correction: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        """Single-head mode: one bounded correction added to y0."""
        return self.expert1.bounded_apply(correction, y0)

    def _mix_anchor(self, w, y0: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        """Linearly mix y0,y1,y2 with spatial (or fixed scalar) weights.

        `w` may be a spatial [B,3,H,W] tensor or a length-3 tensor. Anchors the
        output to y0 and is fail-closed to y0 on non-finite or out-of-simplex
        weights.
        """
        if w.shape[-1] == 3 and w.ndim != 4:
            # Scalar per-expert weights (uniform mode): broadcast over pixels.
            w0, w1, w2 = w[..., None, None].to(y0.dtype)
        else:
            w0, w1, w2 = w[:, 0:1], w[:, 1:2], w[:, 2:3]

        finite = (
            torch.isfinite(w0).all()
            and torch.isfinite(w1).all()
            and torch.isfinite(w2).all()
        )
        if not finite:
            self._numerics.update(finite=False, fallback=True, reason=2)
            return y0

        # simplex check (per-pixel sum ~= 1, within numeric tolerance)
        tol = 1e-5
        if w0.ndim == 4:
            simplex_ok = (
                (w0 + w1 + w2 - w0.new_ones(w0.shape)).abs().lt(tol).all()
                and (w0 >= 0).all()
                and (w1 >= 0).all()
                and (w2 >= 0).all()
            )
        else:
            simplex_ok = True
        if not simplex_ok:
            self._numerics.update(simplex_ok=False, fallback=True, reason=3)
            return y0

        y = w0 * y0 + w1 * y1 + w2 * y2
        if not torch.isfinite(y).all():
            self._numerics.update(finite=False, fallback=True, reason=4)
            return y0
        return torch.clamp(y, -1.0, 1.0)

    def last_safety(self) -> dict:
        """Numeric telemetry snapshot for the last forward."""
        return dict(self._numerics)

    def param_count(self, *, requires_grad: bool = True) -> int:
        if self.mode == MODE_BASELINE:
            return 0
        total = self.trunk.param_count(requires_grad=requires_grad)
        total += self.expert1.param_count(requires_grad=requires_grad)
        total += self.expert2.param_count(requires_grad=requires_grad)
        if self.gate is not None:
            total += self.gate.param_count(requires_grad=requires_grad)
        return total

    def active_reconstruction_params(self) -> int:
        """Cover the frozen 'equal active reconstruction capacity' budget."""
        if self.mode in (MODE_BASELINE,):
            return 0
        # Every learned mode uses the trunk + two heads as active reconstruction
        # capacity; the spatial gate is the separate mechanism under test.
        return RECON_PARAMS

    def expected_budget(self) -> int:
        """Frozen expected params for this mode (trainable count)."""
        if self.mode == MODE_BASELINE:
            return 0
        if self.mode == MODE_SINGLE_HEAD or self.mode == MODE_UNIFORM_TWO_EXPERT:
            return RECON_PARAMS
        if self.mode == MODE_SPATIAL_MIXTURE:
            return RECON_PARAMS + GATE_PARAMS
        raise RuntimeError(f"unreachable mode {self.mode}")

    def mode_param_counts(self) -> dict:
        """Per-component trainable parameter counts (telemetry)."""
        if self.mode == MODE_BASELINE:
            return {
                "mode": self.mode,
                "trunk": 0,
                "expert1": 0,
                "expert2": 0,
                "gate": 0,
                "total_enabled_trainable": 0,
                "active_reconstruction": 0,
            }
        return {
            "trunk": self.trunk.param_count(),
            "expert1": self.expert1.param_count(),
            "expert2": self.expert2.param_count(),
            "gate": self.gate.param_count() if self.gate is not None else 0,
            "mode": self.mode,
            "total_enabled_trainable": self.param_count(),
            "active_reconstruction": self.active_reconstruction_params(),
        }

    def zero_expert_outputs(self) -> None:
        """Freeze-and-zero E1/E2 final projections (used by test and Gate A)."""
        if self.mode != MODE_BASELINE:
            self.expert1.zero_correction()
            self.expert2.zero_correction()

    def initialized_anchor_delta(self, feats: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        """Absolute delta between the initialized mixture and y0."""
        candidate, _, _ = self.mixture_output(feats, y0)
        return (candidate - y0).abs()

    def mixture_parameter_snapshot(self) -> dict:
        """Structural snapshot used by Gate A budget custody."""
        counts = self.mode_param_counts()
        counts["trunk_expect"] = TRUNK_PARAMS
        counts["head_expect"] = HEAD_PARAMS
        counts["recon_expect"] = RECON_PARAMS
        counts["gate_expect"] = GATE_PARAMS
        return counts


class SpatialMixtureTrainerAdapter(nn.Module):
    """Trainer-facing adapter over the frozen mixture core.

    The bounded Phase 0 trainer supplies the six inference-available tensors
    separately; this adapter constructs the exact 27-channel feature bundle in
    the frozen channel order ``(Iin, y0, Ms, Mb, Ic1, reconstruction_feature)``
    and returns ``(candidate, telemetry)``. It deliberately accepts no
    domain/path/split/caller hints, matching the plan's routing-free contract.
    """

    def __init__(self, control: str):
        super().__init__()
        if control not in (MODE_SINGLE_HEAD, MODE_UNIFORM_TWO_EXPERT, MODE_SPATIAL_MIXTURE):
            raise ValueError(f"unknown learned control for trainer adapter: {control!r}")
        self.control = control
        self.core = SpatialContinuousReconstructionMixture(mode=control)

    def forward(
        self,
        *,
        x: torch.Tensor,
        y0: torch.Tensor,
        Ms: torch.Tensor,
        Mb: torch.Tensor,
        Ic1: torch.Tensor,
        feature: torch.Tensor,
    ):
        feats = torch.cat([x, y0, Ms, Mb, Ic1, feature], dim=1)
        candidate, raws, gate = self.core.mixture_output(feats, y0)
        safety = self.core.last_safety()
        telemetry = {
            "safety_reason": candidate.new_tensor(float(safety.get("reason", 0))),
            "fallback": candidate.new_tensor(float(bool(safety.get("fallback", False)))),
        }
        if raws is not None and self.control != MODE_SINGLE_HEAD:
            y1 = self.core.expert1.bounded_apply(raws[0], y0)
            y2 = self.core.expert2.bounded_apply(raws[1], y0)
            telemetry["y1"] = y1
            telemetry["y2"] = y2
            telemetry["e1_e2_mean_abs_disagreement"] = (y1 - y2).abs().mean().detach()
        if gate is not None and "spatial" in gate:
            w = gate["spatial"]
            telemetry["gate_weights"] = w
            telemetry["anchor_share"] = w[:, 0:1].mean().detach()
            telemetry["gate_spatial_std"] = w.std(dim=1).mean().detach()
            telemetry["expert_pixel_share_p99_proxy"] = torch.maximum(
                w[:, 1:2], w[:, 2:3]
            ).detach().amax()
        return candidate, telemetry


def build_control(control: str, base=None, device=None) -> SpatialMixtureTrainerAdapter:
    """Build a Phase 0 learned control for the bounded trainer.

    ``base`` is accepted only for the trainer seam and intentionally ignored:
    the current-primary anchor ``y0`` is passed at forward time and is never a
    trainable parameter of the mixture. ``device`` applies a ``.to`` relocation
    when provided.
    """
    model = SpatialMixtureTrainerAdapter(control)
    if device is not None:
        model = model.to(device)
    return model
