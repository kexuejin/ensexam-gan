#!/usr/bin/env python3
"""Bounded Phase 0 trainer skeleton for the spatial continuous reconstruction
mixture program.

This runner owns:
  - control-mode assembly for the three learned controls;
  - frozen current-primary loading and base/BatchNorm immutability custody;
  - fail-closed MPS gating for real runs (no silent CPU fallback);
  - the frozen AdamW / float32 / 640-step / seed / batch budget;
  - sealed fold-master-manifest and control-config reading with custody checks;
  - a CPU one-step smoke path explicitly marked as NOT a model result;
  - checkpoint custody with no validation/quality-split access and no
    checkpoint selection.

Dependencies owned by other plan files are consumed through lazy imports that
fail closed with ``PREREQUISITE_NEEDED`` until those modules land:

  - ``networks.spatial_reconstruction_mixture.build_control(...)``;
  - ``losses.spatial_mixture_losses.compute_spatial_mixture_loss(...)``.

The production entry point (``main``) uses those defaults. Unit tests inject
the same seams through ``run_probe`` so every path is exercised without the
not-yet-implemented modules and without requiring MPS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_config, save_config  # noqa: E402
from data.dataset import EnsExamRealDataset  # noqa: E402
from networks.generator import Generator  # noqa: E402
from train import freeze_batchnorm_running_stats  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Frozen execution budget (implementation plan section "Frozen Optimizer").
# ─────────────────────────────────────────────────────────────────────────────

FROZEN_LR = 5e-5
FROZEN_BETAS = (0.3037, 0.9)
FROZEN_WEIGHT_DECAY = 0.0
FROZEN_BATCH_SIZE = 4
FROZEN_MAX_STEPS = 640
FROZEN_SEEDS = (42, 31415, 27182)
FROZEN_FOLDS = (0, 1, 2, 3, 4, 5)
FROZEN_IMG_SIZE = 256
FROZEN_OVERLAP = 96
FROZEN_MASK_THRESHOLD = 12

ALL_CONTROLS = ("baseline",)
LEARNED_CONTROLS = ("single_head", "uniform_two_expert", "spatial_mixture")

DEFAULT_CURRENT_PRIMARY_CONFIG = "artifacts/current-primary/config.yaml"
DEFAULT_CURRENT_PRIMARY_CHECKPOINT = (
    "artifacts/current-primary/micro_region_probe_step0001.pth"
)

_SMOKE_MARKER = "cpu-smoke-not-a-model-result"

_MIXTURE_MODULE = "networks.spatial_reconstruction_mixture"
_LOSS_MODULE = "losses.spatial_mixture_losses"


class Phase0PreconditionError(RuntimeError):
    """Fail-closed precondition missing (module, MPS, manifest, or budget)."""


class Phase0RunError(RuntimeError):
    """A run-time failure that terminates this exact Phase 0 unit."""


def _load_module(fqname: str, human_name: str):
    """Lazy import a plan-owned module; fail closed with a clear message."""
    try:
        return importlib.import_module(fqname)
    except ImportError as exc:  # pragma: no cover - exercised via CLI test
        raise Phase0PreconditionError(
            "PREREQUISITE_NEEDED: "
            f"{fqname} is not implemented yet; "
            f"refusing to run Phase 0 training without the {human_name} module."
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# CLI and run spec
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunSpec:
    control: str
    fold: int
    seed: int
    config_path: str
    master_manifest_path: str
    output_dir: str
    checkpoint_path: str = DEFAULT_CURRENT_PRIMARY_CHECKPOINT
    expected_checkpoint_sha256: str = ""
    device: str = "auto"
    smoke: bool = False
    max_steps: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    betas: tuple[float, float] | None = None

    @property
    def effective_max_steps(self) -> int:
        return 1 if self.smoke else FROZEN_MAX_STEPS

    @property
    def effective_batch_size(self) -> int:
        return 1 if self.smoke else FROZEN_BATCH_SIZE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control",
        required=True,
        choices=LEARNED_CONTROLS,
        help=(
            "Learned control mode; baseline replay has no optimizer and is owned "
            "by the repeatability/evaluation path, not this trainer."
        ),
    )
    parser.add_argument("--fold", type=int, required=True, help="Held-out fold (0..5).")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", required=True, help="Sealed control config YAML.")
    parser.add_argument(
        "--master-manifest",
        required=True,
        help="Sealed Phase 0 master manifest JSON with fold identity lists.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CURRENT_PRIMARY_CHECKPOINT,
        help="Frozen current-primary checkpoint (.pth with G_state_dict).",
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default="",
        help="Optional expected SHA-256 of the checkpoint; verified when non-empty.",
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "CPU one-step control-flow smoke; the output directory basename must "
            "contain 'cpu-smoke-not-a-model-result' and the result is NOT a model result."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--betas", default=None, help="Comma-separated float pair.")
    return parser.parse_args(argv)


def spec_from_args(args: argparse.Namespace) -> RunSpec:
    betas: tuple[float, float] | None = None
    if args.betas:
        raw = args.betas.split(",")
        if len(raw) != 2:
            raise Phase0RunError("--betas must be a comma-separated float pair")
        betas = (float(raw[0]), float(raw[1]))
    return RunSpec(
        control=args.control,
        fold=args.fold,
        seed=args.seed,
        config_path=args.config,
        master_manifest_path=args.master_manifest,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        device=args.device,
        smoke=args.smoke,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        betas=betas,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Budget / control / device validation
# ─────────────────────────────────────────────────────────────────────────────


def validate_spec(spec: RunSpec) -> None:
    if spec.control not in LEARNED_CONTROLS:
        raise Phase0RunError(
            f"control {spec.control!r} is not a learned control "
            f"(learned={LEARNED_CONTROLS}); baseline replay is not a trainer unit"
        )
    if spec.fold not in FROZEN_FOLDS:
        raise Phase0RunError(
            f"hold-out fold {spec.fold} outside frozen folds {FROZEN_FOLDS}"
        )
    if not spec.smoke and spec.seed not in FROZEN_SEEDS:
        raise Phase0RunError(
            f"seed {spec.seed} outside frozen seed family {FROZEN_SEEDS}"
        )
    if spec.smoke and _SMOKE_MARKER not in Path(spec.output_dir).name:
        raise Phase0RunError(
            "smoke output directory basename must contain "
            f"'cpu-smoke-not-a-model-result' (got {Path(spec.output_dir).name!r})"
        )

    budget = {
        "lr": spec.lr if spec.lr is not None else FROZEN_LR,
        "betas": spec.betas if spec.betas is not None else FROZEN_BETAS,
        "max_steps": (
            spec.max_steps if spec.max_steps is not None else spec.effective_max_steps
        ),
        "batch_size": (
            spec.batch_size
            if spec.batch_size is not None
            else spec.effective_batch_size
        ),
    }
    expected = {
        "lr": FROZEN_LR,
        "betas": FROZEN_BETAS,
        "max_steps": 1 if spec.smoke else FROZEN_MAX_STEPS,
        "batch_size": 1 if spec.smoke else FROZEN_BATCH_SIZE,
    }
    if budget["lr"] != expected["lr"]:
        raise Phase0RunError(
            f"lr {budget['lr']} violates frozen budget {expected['lr']}"
        )
    if tuple(budget["betas"]) != tuple(expected["betas"]):
        raise Phase0RunError(
            f"betas {budget['betas']} violates frozen budget {expected['betas']}"
        )
    if budget["max_steps"] != expected["max_steps"]:
        raise Phase0RunError(
            f"max_steps {budget['max_steps']} violates frozen budget "
            f"{expected['max_steps']}"
        )
    if budget["batch_size"] != expected["batch_size"]:
        raise Phase0RunError(
            f"batch_size {budget['batch_size']} violates frozen budget "
            f"{expected['batch_size']}"
        )


def mps_preflight(probe=None) -> tuple[bool, str]:
    """Return (ok, reason). Real runs require built + available + allocatable MPS."""
    if probe is not None:
        return probe()
    try:
        built = bool(torch.backends.mps.is_built())
        available = bool(torch.backends.mps.is_available())
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"mps probe raised: {exc}"
    if not built:
        return False, "torch.backends.mps.is_built() is False"
    if not available:
        return False, "torch.backends.mps.is_available() is False"
    try:
        _ = torch.ones(1, device="mps")
    except Exception as exc:
        return False, f"mps tensor allocation failed: {exc}"
    print(f"mps_preflight executable={sys.executable}", flush=True)
    print(f"mps_preflight torch={torch.__version__} built={built} available={available}", flush=True)
    return True, "mps preflight OK"


def resolve_device(spec: RunSpec, probe=None) -> tuple[torch.device, dict]:
    if not spec.smoke:
        ok, reason = mps_preflight(probe)
        if not ok:
            raise Phase0PreconditionError(
                "PREREQUISITE_NEEDED: MPS unavailable for a real Phase 0 run "
                f"({reason}); no silent CPU fallback is permitted."
            )
        if spec.device == "cpu":
            raise Phase0RunError(
                "--device cpu is prohibited for a real Phase 0 run; CPU is allowed "
                "only for the one-step smoke path."
            )
        return torch.device("mps"), {"device": "mps", "mps_preflight": reason}
    if spec.device == "mps":
        ok, reason = mps_preflight(probe)
        if not ok:
            raise Phase0PreconditionError(
                f"PREREQUISITE_NEEDED: MPS unavailable for smoke ({reason})."
            )
        return torch.device("mps"), {"device": "mps", "mps_preflight": reason}
    return torch.device("cpu"), {"device": "cpu", "mps_preflight": ""}


# ─────────────────────────────────────────────────────────────────────────────
# Fold master manifest custody
# ─────────────────────────────────────────────────────────────────────────────

_EXPECTED_FOLD_COUNTS = (64, 64, 64, 64, 64, 63)


def read_master_manifest(path: Path) -> dict:
    if not path.is_file():
        raise Phase0PreconditionError(
            f"PREREQUISITE_NEEDED: master manifest missing: {path}"
        )
    try:
        master = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise Phase0PreconditionError(
            f"PREREQUISITE_NEEDED: master manifest unreadable: {path} ({exc})"
        ) from exc
    if not isinstance(master.get("folds"), dict):
        raise Phase0PreconditionError(
            "PREREQUISITE_NEEDED: master manifest lacks a folds mapping"
        )
    if master.get("train_only") is not True:
        raise Phase0PreconditionError(
            "PREREQUISITE_NEEDED: master manifest train_only must be true"
        )
    for key in (str(i) for i in FROZEN_FOLDS):
        if key not in master["folds"]:
            raise Phase0PreconditionError(
                f"PREREQUISITE_NEEDED: master manifest missing fold {key}"
            )
    folds: dict[int, list[str]] = {}
    seen: dict[str, int] = {}
    for key in (str(i) for i in FROZEN_FOLDS):
        ids = list(master["folds"][key])
        if not ids:
            raise Phase0PreconditionError(
                f"PREREQUISITE_NEEDED: fold {key} has no identities"
            )
        folds[int(key)] = ids
        for identity in ids:
            if identity in seen:
                raise Phase0PreconditionError(
                    f"PREREQUISITE_NEEDED: identity {identity!r} appears in "
                    f"folds {seen[identity]} and {key}"
                )
            seen[identity] = key
            pieces = identity.split("/")
            if len(pieces) != 3 or pieces[1] != "train":
                raise Phase0PreconditionError(
                    "PREREQUISITE_NEEDED: identity must be "
                    f"<domain>/train/<basename>, got {identity!r}"
                )
            domain = pieces[0].lower()
            if domain not in ("scut", "hw5k"):
                raise Phase0PreconditionError(
                    f"PREREQUISITE_NEEDED: unknown domain {domain!r} in {identity!r}"
                )
            for token in master.get("prohibited", []):
                if token.lower() in Path(pieces[2]).name.lower():
                    raise Phase0PreconditionError(
                        f"PREREQUISITE_NEEDED: identity {identity!r} collides with "
                        f"prohibited surface token {token!r}"
                    )
    for key in (str(i) for i in FROZEN_FOLDS):
        expected = master.get("fold_counts", {}).get(key, _EXPECTED_FOLD_COUNTS[int(key)])
        if len(folds[int(key)]) != expected:
            raise Phase0PreconditionError(
                f"PREREQUISITE_NEEDED: fold {key} has {len(folds[int(key)])} ids, "
                f"expected {expected}"
            )
    return {
        "folds": folds,
        "train_only": True,
        "prohibited": master.get("prohibited", []),
        "salt": master.get("salt", ""),
        "variant": master.get("variant", ""),
    }


def split_train_holdout(
    master: dict, holdout_fold: int
) -> tuple[list[str], list[str]]:
    train_ids: list[str] = []
    for fold_idx in FROZEN_FOLDS:
        if fold_idx == holdout_fold:
            continue
        train_ids.extend(master["folds"][fold_idx])
    holdout_ids = list(master["folds"][holdout_fold])
    train_set = set(train_ids)
    if any(identity in train_set for identity in holdout_ids):
        raise Phase0PreconditionError(
            "PREREQUISITE_NEEDED: held-out fold overlaps training folds"
        )
    return train_ids, holdout_ids


# ─────────────────────────────────────────────────────────────────────────────
# Frozen base / mixture / loss / optimizer assembly
# ─────────────────────────────────────────────────────────────────────────────


def load_frozen_base(
    cfg: dict, checkpoint_path: str, expected_sha256: str, device: torch.device
) -> tuple[Generator, dict]:
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        raise Phase0PreconditionError(
            f"PREREQUISITE_NEEDED: checkpoint missing: {ckpt_file}"
        )
    actual_sha = sha256_file(ckpt_file)
    if expected_sha256 and actual_sha != expected_sha256:
        raise Phase0PreconditionError(
            f"PREREQUISITE_NEEDED: checkpoint hash mismatch: "
            f"expected {expected_sha256} actual {actual_sha}"
        )
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if "G_state_dict" not in ckpt:
        raise Phase0PreconditionError(
            "PREREQUISITE_NEEDED: checkpoint lacks G_state_dict"
        )
    model_cfg = cfg.get("model", {})
    G = Generator(cfg=model_cfg).to(device)
    missing, unexpected = G.load_state_dict(ckpt["G_state_dict"], strict=False)
    if unexpected:
        raise Phase0PreconditionError(
            f"PREREQUISITE_NEEDED: base checkpoint has unexpected keys: {unexpected[:8]}"
        )
    if missing:
        raise Phase0PreconditionError(
            f"PREREQUISITE_NEEDED: base checkpoint missing generator keys: {missing[:8]}"
        )
    for param in G.parameters():
        if param.requires_grad:
            param.requires_grad_(False)
    frozen_bn = freeze_batchnorm_running_stats(G)
    G.eval()
    buffer_snapshot = {
        name: buffer.detach().clone().cpu()
        for name, buffer in G.named_buffers()
    }
    param_snapshot = {
        name: param.detach().clone().cpu()
        for name, param in G.named_parameters()
    }
    print(f"frozen_base params={sum(p.numel() for p in G.parameters())} "
          f"bn_modules_eval={frozen_bn}", flush=True)
    return G, {
        "checkpoint_sha256": actual_sha,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "buffer_snapshot": buffer_snapshot,
        "param_snapshot": param_snapshot,
    }


def build_optimizer(model: torch.nn.Module) -> tuple[torch.optim.Optimizer, list[str]]:
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise Phase0RunError("mixture model exposes no trainable parameters")
    names = [name for name, _ in model.named_parameters() if _.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=FROZEN_LR,
        betas=FROZEN_BETAS,
        weight_decay=FROZEN_WEIGHT_DECAY,
    )
    return optimizer, names


def forward_base_with_features(
    base: Generator, source: torch.Tensor, device: torch.device
):
    """Run the frozen base and return (y0, masks, ic1, reconstruction feature)."""
    with torch.no_grad():
        outputs, reconstruction_feature = base(
            source, return_reconstruction_feature=True
        )
    Ms, Mb, _Ic4, _Ic2, Ic1, _Ire, Icomp = outputs
    return Icomp, Ms, Mb, Ic1, reconstruction_feature


# ─────────────────────────────────────────────────────────────────────────────
# Step loop
# ─────────────────────────────────────────────────────────────────────────────


def telemetry_numbers(telemetry: dict | None, device: torch.device) -> dict:
    result: dict[str, float] = {}
    for key, value in (telemetry or {}).items():
        try:
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            result[key] = float(tensor.detach().to(device="cpu", dtype=torch.float64))
        except Exception:
            continue
    return result


def run_probe(
    spec: RunSpec,
    *,
    mixture_builder=None,
    loss_fn=None,
    mps_probe=None,
    dataset_factory=None,
) -> dict:
    """Execute one bounded Phase 0 unit with injectable seams for tests.

    ``mixture_builder(control, base, device) -> nn.Module`` with forward
    ``(x, y0, Ms, Mb, Ic1, feature) -> (y, telemetry)`` and
    ``loss_fn(y, y0, source, target, telemetry) -> (total, terms_dict)``.
    """
    validate_spec(spec)

    cfg = load_config(spec.config_path)
    if cfg.get("data", {}).get("augmentation"):
        raise Phase0RunError(
            "Phase 0 forbids augmentation; sealed config enables data.augmentation"
        )
    if cfg.get("train", {}).get("scheduler", {}).get("enabled", False):
        raise Phase0RunError("Phase 0 forbids LR schedules")
    if cfg.get("early_stopping", {}).get("enabled", False):
        raise Phase0RunError("Phase 0 forbids early stopping")

    master_path = Path(spec.master_manifest_path)
    master = read_master_manifest(master_path)
    train_ids, holdout_ids = split_train_holdout(master, spec.fold)
    train_basenames = sorted(
        {Path(identity).name for identity in train_ids}
    )

    # Resolve plan-owned modules eagerly, before heavy base loading, so a
    # missing-owner module (or a module lacking its entry point) fails fast and
    # closed with PREREQUISITE_NEEDED.
    if mixture_builder is None:
        mixture_module = _load_module(_MIXTURE_MODULE, "spatial reconstruction mixture")
        if not callable(getattr(mixture_module, "build_control", None)):
            raise Phase0PreconditionError(
                "PREREQUISITE_NEEDED: "
                f"{_MIXTURE_MODULE} exists but does not yet expose "
                "build_control(control, base, device)"
            )

        def builder(control_, base_, device_):
            return mixture_module.build_control(
                control=control_, base=base_, device=device_
            )
    else:
        builder = mixture_builder

    if loss_fn is None:
        loss_module = _load_module(_LOSS_MODULE, "spatial mixture losses")
        if not callable(getattr(loss_module, "compute_spatial_mixture_loss", None)):
            raise Phase0PreconditionError(
                "PREREQUISITE_NEEDED: "
                f"{_LOSS_MODULE} exists but does not yet expose "
                "compute_spatial_mixture_loss(y, y0, source, target, telemetry)"
            )

        def loss(y, y0, source, target, telemetry):
            return loss_module.compute_spatial_mixture_loss(
                y=y, y0=y0, source=source, target=target, telemetry=telemetry
            )
    else:
        loss = loss_fn

    device, device_info = resolve_device(spec, mps_probe)

    output_dir = Path(spec.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise Phase0RunError(f"refusing existing non-empty output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, str(output_dir / "sealed-config.yaml"))
    (output_dir / "master-manifest.json").write_text(
        json.dumps(master, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    base, base_info = load_frozen_base(
        cfg, spec.checkpoint_path, spec.expected_checkpoint_sha256, device
    )

    model = builder(spec.control, base, device)
    model = model.to(device)
    for param in model.parameters():
        if param.dtype != torch.float32:
            raise Phase0RunError(
                f"mixture param {param.dtype} is not float32; "
                "Phase 0 is float32-only"
            )

    optimizer, owned_names = build_optimizer(model)
    base_trainable = sum(
        1 for name, p in base.named_parameters() if p.requires_grad
    )
    if base_trainable != 0:
        raise Phase0RunError(f"{base_trainable} base parameters are trainable")

    # Dataset over training folds only; held-out and quality surfaces are never
    # decoded here.
    data_cfg = cfg["data"]
    dataset_factory = dataset_factory or EnsExamRealDataset
    dataset = dataset_factory(
        data_root=data_cfg["data_root"],
        img_size=data_cfg.get("img_size", FROZEN_IMG_SIZE),
        is_train=True,
        overlap=data_cfg.get("overlap", FROZEN_OVERLAP),
        mask_threshold=data_cfg.get("mask_threshold", FROZEN_MASK_THRESHOLD),
        aug_cfg=None,
        file_list=train_basenames,
        phase="train",
        box_class_mode="all",
        box_preserve_mode="none",
    )
    loader = DataLoader(
        dataset,
        batch_size=spec.effective_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=torch.Generator().manual_seed(spec.seed),
    )

    # Seeded determinism for batch sampling; model init handled by the module.
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)

    trace_path = output_dir / "step_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss_total", "is_model_result", "terms_json"])

    start = time.time()
    data_iter = iter(loader)
    model.train()

    for step in range(1, spec.effective_max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        if len(batch) == 8:
            Iin, _Ms_gt, _Mb_gt, _Box_preserve_gt, _Igt4, _Igt2, _Igt1, Igt = batch
        else:
            Iin, _Ms_gt, _Mb_gt, _Igt4, _Igt2, _Igt1, Igt = batch
        Iin = Iin.to(device)
        Igt = Igt.to(device)

        optimizer.zero_grad(set_to_none=True)
        y0, Ms, Mb, Ic1, reconstruction_feature = forward_base_with_features(
            base, Iin, device
        )
        y, telemetry = model(
            x=Iin, y0=y0, Ms=Ms, Mb=Mb, Ic1=Ic1, feature=reconstruction_feature
        )
        total, terms = loss(y=y, y0=y0, source=Iin, target=Igt, telemetry=telemetry)
        if not torch.isfinite(total):
            raise Phase0RunError(f"non-finite loss at step {step}")
        total.backward()
        optimizer.step()

        telemetry_rec = telemetry_numbers(telemetry, device)
        term_rec = {}
        for key, value in terms.items():
            try:
                term_rec[key] = float(
                    value.detach().to(device="cpu", dtype=torch.float64)
                )
            except Exception:
                continue
        with trace_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                step,
                f"{float(total.detach().cpu()):.9f}",
                "false" if spec.smoke else "true",
                json.dumps({"terms": term_rec, "telemetry": telemetry_rec}),
            ])
        if step == 1 or step % 64 == 0 or step == spec.effective_max_steps:
            print(
                f"step={step}/{spec.effective_max_steps} "
                f"loss={float(total.detach().cpu()):.6f} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
        del Iin, Igt, y0, Ms, Mb, Ic1, reconstruction_feature, y, total, terms

    # Immutability custody: base params and BatchNorm buffers must be unchanged.
    for name, buffer in base.named_buffers():
        if not torch.equal(buffer.detach().cpu(), base_info["buffer_snapshot"][name]):
            raise Phase0RunError(f"base BatchNorm buffer changed: {name}")
    for name, param in base.named_parameters():
        if not torch.equal(param.detach().cpu(), base_info["param_snapshot"][name]):
            raise Phase0RunError(f"base parameter changed: {name}")

    mixture_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoint_name = (
        f"smoke_cpu_{spec.control}_fold{spec.fold}_seed{spec.seed}_nonresult.pt"
        if spec.smoke
        else f"{spec.control}_fold{spec.fold}_seed{spec.seed}_phase0_final.pth"
    )
    final_checkpoint = output_dir / checkpoint_name
    torch.save(
        {
            "control": spec.control,
            "fold": spec.fold,
            "seed": spec.seed,
            "micro_steps": spec.effective_max_steps,
            "is_model_result": not spec.smoke,
            "smoke": spec.smoke,
            "mixture_state_dict": model.state_dict(),
            "base_state_dict": base.state_dict(),
            "optimizer": optimizer.state_dict(),
            "source_checkpoint": spec.checkpoint_path,
            "config": cfg,
        },
        final_checkpoint,
    )

    result = {
        "control": spec.control,
        "fold": spec.fold,
        "seed": spec.seed,
        "terminal": "SMOKE" if spec.smoke else "COMPLETED",
        "is_model_result": not spec.smoke,
        "micro_steps": spec.effective_max_steps,
        "device": device_info["device"],
        "output_dir": str(output_dir),
        "final_checkpoint": str(final_checkpoint),
        "mixture_trainable_params": mixture_trainable,
        "base_trainable_params": 0,
        "train_fold_count": len(train_ids),
        "holdout_fold_count": len(holdout_ids),
        "checkpoint_sha256": base_info["checkpoint_sha256"],
        "config_sha256": sha256_file(Path(spec.config_path)),
        "master_manifest_sha256": sha256_file(master_path),
        "optimizer_owned_param_names": owned_names,
        "prohibited_surfaces_accessed": False,
        "base_frozen_and_buffers_unchanged": True,
        "no_checkpoint_selection": True,
        "no_early_stopping": True,
        "no_scheduler": True,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = spec_from_args(args)
    try:
        run_probe(spec)
    except (Phase0PreconditionError, Phase0RunError) as exc:
        print(f"FAILED_CLOSED: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())