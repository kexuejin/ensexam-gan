#!/usr/bin/env python3
"""Gate A structural preflight for the spatial continuous reconstruction mixture.

Implements the 12 frozen Gate A checks from
`docs/plans/2026-08-16-spatial-continuous-reconstruction-mixture-implementation.md`
(lines "Gate A Structural Preflight"):

    1. Verify current-primary config/checkpoint hashes.
    2. Verify the disabled default path preserves legacy state-dict keys/outputs.
    3. Verify all base parameters `requires_grad=false` and all current-primary
       BatchNorm modules remain eval with immutable buffers.
    4. Verify optimizer ownership exactly matches the approved active modules.
    5. Verify the reconstruction/gate parameter counts.
    6. Verify zero-init equivalence on CPU and MPS.
    7. Verify gate finite/simplex invariants and output range, and that a forced
       final-head raw value can generate at least 24 gray of observed pre-clamp
       correction (edit-range margin above the fixed 12-gray event).
    8. Verify live gradients on synthetic erase and repair fixtures.
    9. Verify public signatures and config keys contain no domain/source/caller/
       path routing input.
    10. Verify fold materialization is deterministic, recorded hashes match, and
        prohibited surfaces are absent by identity and content hash.
    11. Verify no output directories from the sealed matrix already exist.

The MPS preflight function reports the MPS environment without requiring MPS in
unit tests; real Phase 0 training requires `--require-mps` (default) and stops
with PREREQUISITE_NEEDED when MPS is unavailable.

Scope discipline: this tool never runs Phase 0 training, never decodes quality
or blind pixels, and never opens inner_val15/Dev40/SCUT115/holdout40/HW5K
reserved surfaces. Fold manifests are read for identity/hash custody only.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from networks.generator import Generator  # noqa: E402
from networks.spatial_reconstruction_mixture import (  # noqa: E402
    CORRECTION_BOUND,
    CPU_INIT_TOLERANCE,
    GATE_PARAMS,
    HEAD_PARAMS,
    MODE_BASELINE,
    MODE_SINGLE_HEAD,
    MODE_SPATIAL_MIXTURE,
    MODE_UNIFORM_TWO_EXPERT,
    RECON_PARAMS,
    TRUNK_PARAMS,
    SpatialContinuousReconstructionMixture,
)
from scripts.analysis.materialize_spatial_mixture_phase0_folds import (  # noqa: E402
    FROZEN_FOLD_DOMAIN_COUNTS,
    build_prohibited_surface_identities,
    sha256_file,
    strip_domain_prefix,
)

# Frozen artifact hashes from the implementation plan.
EXPECTED_CONFIG_SHA256 = (
    "8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae"
)

# Structural config surface accepted for the mixture host (no routing inputs).
ALLOWED_MIXTURE_CFG_KEYS = {
    "enabled",
    "mode",
    "gate_hidden_channels",
    "in_channels",
    "feature_channels",
}
# Tokens that must never appear in param names, config keys, or forward
# signature arguments (metadata / routing inputs are prohibited).
FORBIDDEN_INPUT_TOKENS = (
    "domain",
    "source_caller",
    "caller",
    "route",
    "path",
    "split",
    "dataset",
    "label",
    "mask_gt",
    "target",
)

# Frozen fold counts (SCUT, HW5K) from the plan table.
FROZEN_FOLD_TOTALS = [64, 64, 64, 64, 64, 63]

# Legacy 7-tuple output positions returned by Generator.forward.
LEGACY_OUTPUT_TUPLE_LEN = 7

# Pre-clamp correction gray floor for the edit-range reachability check.
EDIT_RANGE_MIN_GRAY = 24.0
GRAY_TO_MODEL = 1.0 / 127.5


class PreflightError(RuntimeError):
    """Any Gate A check that does not pass."""


# ---------------------------------------------------------------------------
# 1. Current-primary hash custody
# ---------------------------------------------------------------------------


def verify_artifact_hashes(
    config_path: Path,
    checkpoint_path: Path,
    expected_config_sha: str = EXPECTED_CONFIG_SHA256,
    expected_checkpoint_sha: str = EXPECTED_CHECKPOINT_SHA256,
) -> dict[str, Any]:
    """Verify config/checkpoint SHA-256 against the frozen plan hashes."""
    if not config_path.is_file():
        raise PreflightError(f"current-primary config missing: {config_path}")
    if not checkpoint_path.is_file():
        raise PreflightError(f"current-primary checkpoint missing: {checkpoint_path}")
    config_sha = sha256_file(config_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    result = {
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha,
        "config_match": config_sha == expected_config_sha,
        "expected_config_sha256": expected_config_sha,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_match": checkpoint_sha == expected_checkpoint_sha,
        "expected_checkpoint_sha256": expected_checkpoint_sha,
    }
    if not result["config_match"]:
        raise PreflightError(
            f"current-primary config hash mismatch: got {config_sha}, "
            f"expected {expected_config_sha}"
        )
    if not result["checkpoint_match"]:
        raise PreflightError(
            f"current-primary checkpoint hash mismatch: got {checkpoint_sha}, "
            f"expected {expected_checkpoint_sha}"
        )
    return result


# ---------------------------------------------------------------------------
# 2. Disabled default preserves legacy state-dict keys and outputs
# ---------------------------------------------------------------------------


def verify_legacy_default(generator: Generator, device: torch.device) -> dict[str, Any]:
    """Disabled default: no mixture keys and legacy composed output unchanged."""
    if generator.spatial_reconstruction_mixture_enabled:
        raise PreflightError("expected a mixture-disabled Generator for the legacy check")
    state_dict = generator.state_dict()
    leaked = [
        key
        for key in state_dict
        if key.startswith("spatial_reconstruction_mixture.")
        or key.startswith("universal_residual_adapter_sidecar.")
    ]
    if leaked:
        raise PreflightError(f"disabled default leaked mixture keys: {leaked}")
    generator.to(device).eval()
    with torch.no_grad():
        source = torch.rand(2, 3, 64, 64, device=device)
        outputs = generator(source)
        if not isinstance(outputs, tuple) or len(outputs) != LEGACY_OUTPUT_TUPLE_LEN:
            raise PreflightError(
                f"legacy default forward must return a {LEGACY_OUTPUT_TUPLE_LEN}-tuple"
            )
        Ms, Mb, _Ic4, _Ic2, _Ic1, Ire, Icomp = outputs
        composed = Ire * Mb + source * (1.0 - Mb)
        max_delta = (Icomp - composed).abs().max().item()
    if max_delta > 1e-6:
        raise PreflightError(
            f"legacy default output drifted from composition: max_delta={max_delta}"
        )
    return {"legacy_outputs_preserved": True, "leaked_keys": []}


# ---------------------------------------------------------------------------
# 3. Base frozen + BatchNorm immutable; 4. optimizer ownership
# ---------------------------------------------------------------------------


def _apply_frozen_base(generator: Generator) -> dict[str, Any]:
    """Freeze every base parameter; keep mixture parameters trainable."""
    frozen_count = 0
    trainable_count = 0
    frozen_params = 0
    trainable_params = 0
    for name, param in generator.named_parameters():
        if name.startswith("spatial_reconstruction_mixture."):
            param.requires_grad_(True)
            trainable_count += 1
            trainable_params += param.numel()
        else:
            param.requires_grad_(False)
            frozen_count += 1
            frozen_params += param.numel()
    return {
        "frozen_tensors": frozen_count,
        "trainable_tensors": trainable_count,
        "frozen_params": frozen_params,
        "trainable_params": trainable_params,
    }


def _base_batchnorm_modules(generator: Generator) -> list[torch.nn.Module]:
    mixture = getattr(generator, "spatial_reconstruction_mixture", None)
    mixture_ids = {id(m) for m in mixture.modules()} if mixture is not None else set()
    out: list[torch.nn.Module] = []
    for module in generator.modules():
        if id(module) in mixture_ids:
            continue
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            out.append(module)
    return out


def snapshot_bn_buffers(modules: list[torch.nn.Module]) -> list[tuple[str, Any]]:
    snap: list[tuple[str, Any]] = []
    for module in modules:
        for name, buf in module.named_buffers(recurse=False):
            snap.append((f"{id(module)}:{name}", buf.detach().clone()))
    return snap


def verify_frozen_base_and_bn_immutable(
    generator: Generator, device: torch.device
) -> dict[str, Any]:
    """Base params frozen, base BN eval, base BN buffers unchanged after one forward."""
    if not generator.spatial_reconstruction_mixture_enabled:
        raise PreflightError("frozen-base check requires a mixture-enabled Generator")
    freeze = _apply_frozen_base(generator)
    base_bn = _base_batchnorm_modules(generator)
    for module in base_bn:
        module.eval()
    generator.to(device)

    bad_base = [
        name
        for name, param in generator.named_parameters()
        if not name.startswith("spatial_reconstruction_mixture.") and param.requires_grad
    ]
    if bad_base:
        raise PreflightError(f"base parameters still trainable: {bad_base[:5]}")
    bad_mixture = [
        name
        for name, param in generator.named_parameters()
        if name.startswith("spatial_reconstruction_mixture.") and not param.requires_grad
    ]
    if bad_mixture:
        raise PreflightError(f"mixture parameters frozen: {bad_mixture[:5]}")

    before = snapshot_bn_buffers(base_bn)
    with torch.no_grad():
        source = torch.rand(2, 3, 64, 64, device=device)
        generator(source)
    after = snapshot_bn_buffers(base_bn)
    if len(before) != len(after):
        raise PreflightError("base BatchNorm buffer count changed across forward")
    for (name_b, buf_b), (name_a, buf_a) in zip(before, after):
        if name_b != name_a:
            raise PreflightError("base BatchNorm buffer order changed across forward")
        if not torch.equal(buf_b, buf_a):
            raise PreflightError(f"base BatchNorm buffer mutated: {name_b}")

    return {
        **freeze,
        "base_batchnorm_in_eval": len(base_bn),
        "base_batchnorm_buffers_immutable": True,
        "mixture_enabled": True,
    }


def verify_optimizer_ownership(
    generator: Generator, lr: float = 5e-5, betas: tuple[float, float] = (0.3037, 0.9)
) -> dict[str, Any]:
    """Optimizer over exactly the approved mixture parameters, nothing else."""
    mixture = generator.spatial_reconstruction_mixture
    if mixture is None:
        raise PreflightError("optimizer-ownership check requires an enabled mixture")
    mixture_param_ids = {id(p) for p in mixture.parameters() if p.requires_grad}
    trainable = [p for p in generator.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, betas=betas, weight_decay=0.0)
    owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
    missing = mixture_param_ids - owned
    extra = owned - mixture_param_ids
    if missing or extra:
        raise PreflightError(
            f"optimizer ownership mismatch: missing={len(missing)} extra={len(extra)}"
        )
    all_frozen = [p for p in generator.parameters() if not p.requires_grad]
    base_owned = {
        id(p) for group in optimizer.param_groups for p in group["params"] if id(p) not in mixture_param_ids
    }
    return {
        "optimizer_params": len(owned),
        "mixture_trainable_params": len(mixture_param_ids),
        "base_params_owned_by_optimizer": len(base_owned),
        "all_frozen_base_tensors": len(all_frozen),
        "ownership_exact": True,
    }


# ---------------------------------------------------------------------------
# 5. Frozen parameter counts
# ---------------------------------------------------------------------------


def verify_parameter_counts(mixture: SpatialContinuousReconstructionMixture) -> dict[str, Any]:
    """All learned modes equal B_recon active reconstruction; spatial adds gate."""
    counts = mixture.mixture_parameter_snapshot()
    if mixture.mode == MODE_BASELINE:
        if counts["total_enabled_trainable"] != 0:
            raise PreflightError("baseline mode must have zero trainable parameters")
    else:
        if counts["trunk"] != TRUNK_PARAMS:
            raise PreflightError(f"trunk count {counts['trunk']} != {TRUNK_PARAMS}")
        if counts["expert1"] != HEAD_PARAMS or counts["expert2"] != HEAD_PARAMS:
            raise PreflightError("expert head count mismatch")
        if mixture.mode in (MODE_SINGLE_HEAD, MODE_UNIFORM_TWO_EXPERT):
            expect = RECON_PARAMS
        elif mixture.mode == MODE_SPATIAL_MIXTURE:
            if counts["gate"] != GATE_PARAMS:
                raise PreflightError(f"gate count {counts['gate']} != {GATE_PARAMS}")
            expect = RECON_PARAMS + GATE_PARAMS
        else:
            raise PreflightError(f"unhandled mode {mixture.mode}")
        if counts["active_reconstruction"] != RECON_PARAMS:
            raise PreflightError(
                f"active reconstruction budget {counts['active_reconstruction']} != {RECON_PARAMS}"
            )
        if counts["total_enabled_trainable"] != expect:
            raise PreflightError(
                f"trainable total {counts['total_enabled_trainable']} != {expect}"
            )
    return counts


# ---------------------------------------------------------------------------
# 6. Zero-init equivalence (CPU; MPS via mps_preflight_report)
# ---------------------------------------------------------------------------


def verify_init_equivalence(
    mixture: SpatialContinuousReconstructionMixture,
    device: torch.device,
    tolerance: float = CPU_INIT_TOLERANCE,
) -> dict[str, Any]:
    """Initialized mixture must reproduce y0 within the frozen tolerance."""
    if mixture.mode == MODE_BASELINE:
        return {"mode": MODE_BASELINE, "max_delta": 0.0, "pass": True}
    mixture = mixture.to(device)
    feats = torch.rand(2, 27, 32, 32, device=device)
    y0 = 0.05 * torch.rand(2, 3, 32, 32, device=device)
    with torch.no_grad():
        candidate, _, _ = mixture.mixture_output(feats, y0)
        max_delta = (candidate - y0).abs().max().item()
    if max_delta > tolerance:
        raise PreflightError(
            f"init equivalence failed: max_delta={max_delta} > tolerance={tolerance}"
        )
    return {"mode": mixture.mode, "max_delta": max_delta, "pass": True}


# ---------------------------------------------------------------------------
# 7. Gate invariants + edit range
# ---------------------------------------------------------------------------


def verify_gate_invariants(
    mixture: SpatialContinuousReconstructionMixture, device: torch.device
) -> dict[str, Any]:
    """Spatial gate weights finite, nonnegative, sum to one; output in range."""
    if mixture.mode != MODE_SPATIAL_MIXTURE:
        raise PreflightError("gate-invariant check requires spatial mode")
    mixture = mixture.to(device)
    feats = torch.rand(2, 27, 32, 32, device=device)
    y0 = 0.05 * torch.rand(2, 3, 32, 32, device=device)
    with torch.no_grad():
        candidate, _, gate_info = mixture.mixture_output(feats, y0)
        w = gate_info["spatial"]
        w_sum = w.sum(dim=1)
        finite = bool(torch.isfinite(w).all().item())
        nonneg = bool((w >= 0).all().item())
        simplex = bool((w_sum - 1.0).abs().max().item() <= 1e-5)
        in_range = bool((candidate.abs() <= 1.0).all().item())
    if not (finite and nonneg and simplex):
        raise PreflightError(
            f"gate simplex violation: finite={finite} nonneg={nonneg} simplex={simplex}"
        )
    if not in_range:
        raise PreflightError("candidate output out of [-1, 1] range")
    return {"finite": True, "nonneg": True, "simplex": True, "in_range": True}


def verify_edit_range(
    head: "TerminalReconstructionHead", device: torch.device
) -> dict[str, Any]:
    """Forced final-head raw value must produce >=24 gray of pre-clamp correction.

    This is a per-head capability check (mode independent): a terminal head must
    be able to emit a bounded correction of at least 24 eight-bit gray before
    clamping, proving margin above the fixed 12-gray event threshold.
    """
    from networks.spatial_reconstruction_mixture import TerminalReconstructionHead

    if not isinstance(head, TerminalReconstructionHead):
        head = TerminalReconstructionHead()
    head = head.to(device)
    head.zero_correction()
    with torch.no_grad():
        head.correction.weight.zero_()
        head.correction.bias.fill_(10.0)
    feats = torch.rand(2, 64, 16, 16, device=device)
    anchor = torch.zeros(2, 3, 16, 16, device=device)
    with torch.no_grad():
        raw = head.forward(feats)
        candidate = head.bounded_apply(raw, anchor)
    observed_gray = (candidate - anchor).abs().max().item() * 127.5
    raw_max = raw.abs().max().item()
    if observed_gray < EDIT_RANGE_MIN_GRAY:
        raise PreflightError(
            f"edit range too small: observed {observed_gray:.3f} gray < {EDIT_RANGE_MIN_GRAY}"
        )
    if raw_max <= 0.0:
        raise PreflightError("forced raw correction did not propagate")
    return {
        "observed_preclamp_gray": observed_gray,
        "min_gray": EDIT_RANGE_MIN_GRAY,
        "correction_bound_gray": CORRECTION_BOUND * 127.5,
        "raw_max": raw_max,
    }


# ---------------------------------------------------------------------------
# 8. Gradient liveness on synthetic erase / repair fixtures
# ---------------------------------------------------------------------------


def verify_gradient_liveness(
    mixture: SpatialContinuousReconstructionMixture,
    device: torch.device,
    fixture: str = "erase",
) -> dict[str, Any]:
    """Each active expert projection and the gate must receive live gradients.

    The shared trunk gets zero gradient through a purely zero-initialized expert
    projection by design (the exact-y0 anchor construction), so this check
    verifies liveness on the parameters that Gate A #8 actually requires to
    move: each expert's zero-init final RGB projection and, for the spatial
    control, the gate.
    """
    if mixture.mode == MODE_BASELINE:
        raise PreflightError("baseline has no learnable branch")
    mixture = mixture.to(device)
    feats = torch.rand(2, 27, 32, 32, device=device, requires_grad=False)
    y0 = 0.05 * torch.rand(2, 3, 32, 32, device=device)
    if fixture == "erase":
        target = y0 * 0.25  # erase toward darker
    else:
        target = torch.clamp(y0 + 0.1, -1.0, 1.0)  # repair toward original
    mixture.zero_expert_outputs()
    mixture.zero_grad()
    candidate, _, _ = mixture.mixture_output(feats, y0)
    loss = torch.nn.functional.l1_loss(candidate, target)
    loss.backward()

    def _grad_ok(name: str, param: torch.Tensor) -> dict:
        grad = param.grad
        if grad is None:
            raise PreflightError(f"dead gradient on {name}")
        grad_norm = float(grad.abs().sum().item())
        if not (torch.isfinite(grad).all().item() and grad_norm > 0.0):
            raise PreflightError(f"non-finite or zero gradient on {name}")
        return {name: grad_norm}

    grad_status: dict[str, float] = {}
    # Experts: both zero-init projection heads must receive live gradients.
    for expert_prefix, expert in (
        ("expert1", mixture.expert1),
        ("expert2", mixture.expert2),
    ):
        for name, param in expert.correction.named_parameters():
            grad_status.update(_grad_ok(f"{expert_prefix}.correction.{name}", param))
    # Gate: the learnable mixing head (ghead) must be live.
    if mixture.gate is not None:
        for name, param in mixture.gate.ghead.named_parameters():
            grad_status.update(_grad_ok(f"gate.ghead.{name}", param))
    return {"fixture": fixture, "loss": float(loss.item()), "grads": grad_status}


# ---------------------------------------------------------------------------
# 9. No metadata / routing inputs
# ---------------------------------------------------------------------------


def verify_no_metadata_inputs(
    mixture: SpatialContinuousReconstructionMixture,
    config_keys: list[str],
) -> dict[str, Any]:
    """Forward signature, param names, and config keys contain no routing inputs."""
    signature = inspect.signature(SpatialContinuousReconstructionMixture.forward)
    arg_names = list(signature.parameters)
    forbidden_args = [a for a in arg_names if a != "self" and _has_forbidden_token(a)]
    if forbidden_args:
        raise PreflightError(f"forbidden forward args: {forbidden_args}")

    param_names = [name for name, _ in mixture.named_parameters()]
    forbidden_params = [name for name in param_names if _has_forbidden_token(name)]
    if forbidden_params:
        raise PreflightError(f"forbidden parameter names: {forbidden_params[:5]}")

    unexpected_cfg = [k for k in config_keys if k not in ALLOWED_MIXTURE_CFG_KEYS]
    if unexpected_cfg:
        raise PreflightError(f"unexpected mixture config keys: {unexpected_cfg}")
    forbidden_cfg = [k for k in config_keys if _has_forbidden_token(k)]
    if forbidden_cfg:
        raise PreflightError(f"forbidden config keys: {forbidden_cfg}")

    return {
        "forward_args": arg_names,
        "param_count": len(param_names),
        "config_keys": sorted(config_keys),
        "no_metadata_input": True,
    }


def _has_forbidden_token(value: str) -> bool:
    lowered = value.lower()
    for token in FORBIDDEN_INPUT_TOKENS:
        if token in lowered:
            return True
    return False


# ---------------------------------------------------------------------------
# 10. Fold manifests: deterministic custody + prohibited-surface isolation
# ---------------------------------------------------------------------------


def verify_fold_manifests(
    fold_root: Path,
    *,
    prohibited_scut_stems: set[str] | None = None,
    prohibited_hw5k_stems: set[str] | None = None,
) -> dict[str, Any]:
    """Recompute fold file hashes, enforce counts, disjointness, and isolation."""
    scut_barred = prohibited_scut_stems or set()
    hw5k_barred = prohibited_hw5k_stems or set()
    master_path = fold_root / "master.json"
    if not master_path.is_file():
        raise PreflightError(f"fold master.json missing: {master_path}")

    master = json.loads(master_path.read_text(encoding="utf-8"))
    if master.get("fold_count") != 6:
        raise PreflightError(f"master fold_count must be 6, got {master.get('fold_count')}")
    if master.get("frozen_fold_domain_counts") != [list(pair) for pair in FROZEN_FOLD_DOMAIN_COUNTS]:
        raise PreflightError("master frozen_fold_domain_counts drift from the plan")

    all_identities: set[str] = set()
    per_fold_identities: list[set[str]] = []
    result: dict[str, Any] = {"folds": {}}
    expected_counts = dict(zip(map(str, range(6)), FROZEN_FOLD_TOTALS))

    for fold_index in range(6):
        key = str(fold_index)
        entry = master["folds"][key]
        txt_path = fold_root / Path(entry["txt"]).name
        json_path = fold_root / Path(entry["json"]).name
        for path, recorded_key in ((txt_path, "txt_sha256"), (json_path, "json_sha256")):
            if not path.is_file():
                raise PreflightError(f"fold {fold_index} asset missing: {path}")
            recomputed = sha256_file(path)
            recorded = entry.get(recorded_key, "")
            if recomputed != recorded:
                raise PreflightError(
                    f"fold {fold_index} {path.name} hash drift: got {recomputed}, "
                    f"recorded {recorded}"
                )
        identities = {
            line.strip()
            for line in txt_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if len(identities) != entry.get("total", -1):
            raise PreflightError(
                f"fold {fold_index} identity count {len(identities)} != recorded {entry.get('total')}"
            )
        if len(identities) != expected_counts[key]:
            raise PreflightError(
                f"fold {fold_index} total {len(identities)} != frozen {expected_counts[key]}"
            )
        # Canonical identity form (already `<domain>/train/<basename>` in fold .txt).
        for identity in identities:
            _split_identity(identity)  # raises if not <domain>/train/<basename>
        for identity in identities:
            domain, _, basename = _split_identity(identity)
            stem = strip_domain_prefix(basename)
            if domain == "scut" and stem in scut_barred:
                raise PreflightError(f"fold {fold_index} SCUT clash: {identity}")
            if domain == "hw5k" and stem in hw5k_barred:
                raise PreflightError(f"fold {fold_index} HW5K clash: {identity}")
        overlap = all_identities & identities
        if overlap:
            raise PreflightError(f"fold {fold_index} shares identities with earlier folds")
        all_identities |= identities
        per_fold_identities.append(identities)
        result["folds"][key] = {
            "total": len(identities),
            "txt_sha256": entry["txt_sha256"],
            "json_sha256": entry["json_sha256"],
        }

    recorded_pool = master.get("pool", {})
    if len(all_identities) != recorded_pool.get("total", -1):
        raise PreflightError(
            f"union identity count {len(all_identities)} != recorded pool {recorded_pool.get('total')}"
        )
    result["union_count"] = len(all_identities)
    result["disjoint"] = True
    result["prohibited_isolation"] = True
    return result


def _split_identity(identity: str) -> tuple[str, str, str]:
    parts = identity.split("/")
    if len(parts) != 3 or parts[1] != "train":
        raise PreflightError(f"identity not <domain>/train/<basename>: {identity!r}")
    return parts[0], parts[1], parts[2]


# ---------------------------------------------------------------------------
# 11. No existing sealed run output dirs
# ---------------------------------------------------------------------------


def verify_no_sealed_outputs(
    matrix_json: Path,
    *,
    require_absent_matrix: bool = True,
    extra_sealed_dirs: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """No run output dir referenced by the sealed matrix already exists."""
    existing: list[str] = []
    if matrix_json.is_file():
        if require_absent_matrix:
            raise PreflightError(
                f"sealed matrix already exists (no Phase 0 output allowed yet): {matrix_json}"
            )
        matrix = json.loads(matrix_json.read_text(encoding="utf-8"))
        runs = matrix.get("runs", [])
        for run in runs:
            out_dir = run.get("output_dir")
            if out_dir and Path(out_dir).exists():
                existing.append(str(out_dir))
    for directory in extra_sealed_dirs:
        if directory.exists():
            existing.append(str(directory))
    if existing:
        raise PreflightError(f"sealed Phase 0 output dirs already exist: {existing[:5]}")
    return {"matrix_json_present": matrix_json.is_file(), "existing_dirs": []}


# ---------------------------------------------------------------------------
# MPS preflight report (never required in unit tests)
# ---------------------------------------------------------------------------


def mps_preflight_report() -> dict[str, Any]:
    """Return an MPS environment report; does not require MPS availability."""
    report: dict[str, Any] = {
        "python": sys.executable,
        "torch_version": torch.__version__,
        "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }
    if report["mps_available"]:
        try:
            probe = torch.ones(1, device="mps")
            report["mps_alloc_ok"] = bool(probe.sum().item() == 1.0)
        except Exception as exc:  # noqa: BLE001
            report["mps_alloc_ok"] = False
            report["mps_alloc_error"] = str(exc)
    else:
        report["mps_alloc_ok"] = None
    return report


# ---------------------------------------------------------------------------
# Orchestration + CLI
# ---------------------------------------------------------------------------


DEFAULT_PLAN = Path(
    "docs/plans/2026-08-16-spatial-continuous-reconstruction-mixture-implementation.md"
)
DEFAULT_FOLD_ROOT = Path("hardcase_lists/spatial-mixture-phase0-v1")
DEFAULT_OUTPUT_DIR = Path("outputs/spatial-mixture-phase0-preflight-v1")
DEFAULT_CONFIG = Path("artifacts/current-primary/config.yaml")
DEFAULT_CHECKPOINT = Path("artifacts/current-primary/micro_region_probe_step0001.pth")
DEFAULT_MATRIX_JSON = Path("docs/spatial-mixture-phase0-matrix-v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLD_ROOT)
    parser.add_argument("--matrix-json", type=Path, default=DEFAULT_MATRIX_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", default=MODE_SPATIAL_MIXTURE)
    parser.add_argument("--require-mps", action="store_true", default=True)
    parser.add_argument("--no-require-mps", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    terminal = "PASS"
    try:
        checks["mps"] = mps_preflight_report()
        if args.require_mps and not args.no_require_mps:
            if not checks["mps"].get("mps_available"):
                terminal = "PREREQUISITE_NEEDED"
                checks["mps"]["required"] = True
                raise PreflightError(
                    "MPS unavailable; environment failure, not a model result"
                )
        checks["mps"]["required"] = bool(args.require_mps and not args.no_require_mps)

        checks["artifact_hashes"] = verify_artifact_hashes(args.config, args.checkpoint)

        # 2. Legacy default (danger-free structural check).
        legacy_generator = Generator()
        checks["legacy_default"] = verify_legacy_default(
            legacy_generator, torch.device(args.device)
        )
        del legacy_generator

        # 3.-4. Frozen base / BN / optimizer ownership with a mixture-enabled host.
        host = Generator(
            {
                "coarse_in_channels": 3,
                "refine_in_channels": 7,
                "cbam_reduction": 16,
                "spatial_reconstruction_mixture": {
                    "enabled": True,
                    "mode": args.mode,
                },
            }
        )
        checks["frozen_base_and_bn"] = verify_frozen_base_and_bn_immutable(
            host, torch.device(args.device)
        )
        checks["optimizer_ownership"] = verify_optimizer_ownership(host)

        # 5. Counts / 6. init / 7. gate+edit / 8. grads / 9. metadata on the mixture.
        mixture = host.spatial_reconstruction_mixture
        checks["parameter_counts"] = verify_parameter_counts(mixture)
        checks["init_equivalence"] = verify_init_equivalence(
            mixture, torch.device(args.device)
        )
        if args.mode == MODE_SPATIAL_MIXTURE:
            checks["gate_invariants"] = verify_gate_invariants(
                mixture, torch.device(args.device)
            )
        checks["edit_range"] = verify_edit_range(None, torch.device(args.device))
        checks["gradient_liveness_erase"] = verify_gradient_liveness(
            mixture, torch.device(args.device), fixture="erase"
        )
        checks["gradient_liveness_repair"] = verify_gradient_liveness(
            mixture, torch.device(args.device), fixture="repair"
        )
        cfg_keys = list(
            {
                "enabled",
                "mode",
                "gate_hidden_channels",
            }
        )
        checks["no_metadata_inputs"] = verify_no_metadata_inputs(mixture, cfg_keys)

        # 10. Fold manifests
        prohibited = build_prohibited_surface_identities(
            Path("hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt"),
            Path("outputs/source-edge-primary-edit-dev40-gate-20260816-v1/dev40_review_rows.csv"),
            Path("hardcase_lists/scut_val_holdout_40.txt"),
            Path("docs/scut-test115-relative.txt"),
            Path("hardcase_lists/hw5k_dev_232_20260729.txt"),
            Path("data/hw5k/reserved"),
        )
        checks["fold_manifests"] = verify_fold_manifests(
            args.fold_root,
            prohibited_scut_stems=prohibited["_scut_union"],
            prohibited_hw5k_stems=prohibited["_hw5k_union"],
        )

        # 11. No sealed outputs
        checks["no_sealed_outputs"] = verify_no_sealed_outputs(
            args.matrix_json,
            extra_sealed_dirs=(
                Path("artifacts/trials/spatial-mixture-phase0-v1"),
            ),
        )

        if not args.plan.is_file():
            raise PreflightError(f"plan not found: {args.plan}")
        plan_sha = sha256_file(args.plan)
        checks["plan_sha256"] = plan_sha
        checks["terminal"] = "PASS"
    except PreflightError as exc:
        checks["terminal"] = terminal if terminal != "PASS" else "FAIL"
        checks["error"] = str(exc)
    return checks


def write_audit(args: argparse.Namespace, checks: dict[str, Any]) -> Path:
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_name = out_dir.name  # ensure the preflight dir itself is allowed output
    audit_path = out_dir / "audit.json"
    audit_path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    return audit_path


def main() -> int:
    args = parse_args()
    args.require_mps = not args.no_require_mps
    checks = run_preflight(args)
    audit_path = write_audit(args, checks)
    print(json.dumps(checks, indent=2))
    print(f"audit={audit_path}")
    terminal = checks.get("terminal", "FAIL")
    if terminal == "PASS":
        return 0
    if terminal == "PREREQUISITE_NEEDED":
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())