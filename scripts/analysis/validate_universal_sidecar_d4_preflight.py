#!/usr/bin/env python3
"""Fail-closed preflight for the preregistered universal-sidecar D4 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from networks.discriminator import Discriminator  # noqa: E402
from networks.generator import Generator  # noqa: E402
from scripts.analysis.audit_universal_sidecar_structure import run_audit  # noqa: E402
try:  # noqa: E402
    from train import (  # type: ignore
        apply_generator_trainable_patterns,
        freeze_batchnorm_running_stats,
        load_initial_checkpoint,
        validate_universal_sidecar_config,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment fallback
    if exc.name not in {"torchvision", "albumentations"}:
        raise

    def apply_generator_trainable_patterns(
        model,
        patterns,
    ) -> dict[str, object]:
        if not patterns:
            total = sum(parameter.numel() for parameter in model.parameters())
            return {
                "enabled": False,
                "patterns": [],
                "trainable_tensors": sum(1 for _ in model.parameters()),
                "frozen_tensors": 0,
                "trainable_params": total,
                "total_params": total,
            }

        compiled = [re.compile(pattern) for pattern in patterns]
        trainable_tensors = 0
        frozen_tensors = 0
        trainable_params = 0
        total_params = 0
        matched_patterns = {pattern: 0 for pattern in patterns}
        for name, parameter in model.named_parameters():
            total_params += parameter.numel()
            matched = False
            for raw_pattern, compiled_pattern in zip(patterns, compiled):
                if compiled_pattern.search(name):
                    matched = True
                    matched_patterns[raw_pattern] += 1
            parameter.requires_grad_(matched)
            if matched:
                trainable_tensors += 1
                trainable_params += parameter.numel()
            else:
                frozen_tensors += 1
        unused = [pattern for pattern, count in matched_patterns.items() if count == 0]
        if unused:
            raise ValueError(
                "train.trainable_generator_patterns matched no parameters: "
                f"{unused}"
            )
        if trainable_tensors == 0:
            raise ValueError(
                "train.trainable_generator_patterns froze every generator parameter"
            )
        return {
            "enabled": True,
            "patterns": list(patterns),
            "trainable_tensors": trainable_tensors,
            "frozen_tensors": frozen_tensors,
            "trainable_params": trainable_params,
            "total_params": total_params,
        }

    def freeze_batchnorm_running_stats(model) -> int:
        batchnorm_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
        )
        frozen = 0
        for module in model.modules():
            if isinstance(module, batchnorm_types):
                module.eval()
                frozen += 1
        return frozen

    def load_initial_checkpoint(generator, discriminator, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "G_state_dict" not in checkpoint:
            raise KeyError("initial checkpoint must contain G_state_dict")
        incompatible = generator.load_state_dict(
            checkpoint["G_state_dict"], strict=False
        )
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        allowed_missing = {
            key
            for key in generator.state_dict()
            if key.startswith("universal_residual_adapter_sidecar.")
        }
        if unexpected or any(key not in allowed_missing for key in missing):
            raise RuntimeError(
                "initial generator checkpoint mismatch: "
                f"missing={missing} unexpected={unexpected}"
            )
        if "D_state_dict" in checkpoint:
            discriminator.load_state_dict(checkpoint["D_state_dict"])
        return missing, unexpected

    def validate_universal_sidecar_config(cfg: dict) -> None:
        model_cfg = cfg.get("model", {})
        sidecar_cfg = model_cfg.get("universal_residual_adapter_sidecar", {})
        if isinstance(sidecar_cfg, bool):
            sidecar_cfg = {"enabled": sidecar_cfg}
        if not isinstance(sidecar_cfg, dict) or not sidecar_cfg.get("enabled", False):
            return
        if int(sidecar_cfg.get("adapter_count", 3)) != 3:
            raise ValueError("universal sidecar requires adapter_count=3")
        residual_parameterization = str(
            sidecar_cfg.get("residual_parameterization", "free_rgb")
        )
        if residual_parameterization not in {"free_rgb", "primary_edit_direction"}:
            raise ValueError(
                "universal sidecar residual_parameterization must be free_rgb or "
                "primary_edit_direction"
            )
        residual_bound = float(sidecar_cfg.get("residual_bound", 12.0 / 255.0))
        if residual_bound <= 0.0 or residual_bound > 12.0 / 255.0:
            raise ValueError("universal sidecar residual_bound must be in (0, 12/255]")
        train_cfg = cfg.get("train", {})
        if train_cfg.get("resume", False) or train_cfg.get("resume_path"):
            raise ValueError("universal sidecar cannot resume optimizer state")
        if (
            str(train_cfg.get("init_checkpoint", ""))
            != "./artifacts/current-primary/micro_region_probe_step0001.pth"
        ):
            raise ValueError("universal sidecar requires current-primary initialization")
        patterns = train_cfg.get("trainable_generator_patterns") or []
        if not patterns:
            raise ValueError(
                "universal sidecar requires sidecar-only trainable_generator_patterns"
            )
        parameter_names = [name for name, _ in Generator(cfg=model_cfg).named_parameters()]
        for pattern in patterns:
            compiled = re.compile(str(pattern))
            matched = [name for name in parameter_names if compiled.search(name)]
            if not matched:
                raise ValueError(
                    "universal sidecar trainable pattern must match sidecar params"
                )
            if any(
                not name.startswith("universal_residual_adapter_sidecar.")
                for name in matched
            ):
                raise ValueError(
                    "universal sidecar trainable pattern must not match base params"
                )
        if train_cfg.get("freeze_generator_batchnorm_stats") is not True:
            raise ValueError(
                "universal sidecar requires BatchNorm freeze: "
                "freeze_generator_batchnorm_stats=true"
            )
        if train_cfg.get("save_optimizer_state", True):
            raise ValueError("universal sidecar must not save optimizer state")
        if train_cfg.get("save_scheduler_state", True):
            raise ValueError("universal sidecar must not save scheduler state")
        if "seed" not in train_cfg or train_cfg.get("reproducibility_mode") != "strict":
            raise ValueError("universal sidecar requires strict reproducibility")


D2_CONFIG = Path(
    "configs/local/"
    "config.local-universal-sidecar-d2-d1-mixed-scut130-hw5k260-step80-mps.yaml"
)
D4_CONFIG = Path(
    "configs/local/"
    "config.local-universal-sidecar-d4-d1-mixed-scut130-hw5k260-step80-"
    "primary-edit-direction-mps.yaml"
)
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
EXPECTED_DIFFERENCES = {
    "model.universal_residual_adapter_sidecar.residual_parameterization": (
        "primary_edit_direction"
    ),
    "train.save_dir": (
        "./artifacts/trials/"
        "universal-sidecar-d4-d1-mixed-scut130-hw5k260-step80-"
        "primary-edit-direction-20260809"
    ),
}
FORBIDDEN_D3_FIELDS = (
    "loss.lambda_cached_baseline_tail_nonregress",
    "loss.cached_baseline_tail_residual_alpha",
    "loss.cached_baseline_tail_overerase_alpha",
    "loss.cached_baseline_tail_fraction",
    "loss.cached_baseline_tail_residual_threshold_px",
    "loss.cached_baseline_tail_edit_threshold_px",
    "loss.cached_baseline_tail_event_temperature_px",
    "data.cached_baseline_tail_dir",
)


class PreflightError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PreflightError(f"missing config: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreflightError(f"config must be a mapping: {path}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PreflightError(f"missing json: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreflightError(f"json must be an object: {path}")
    return value


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    output: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        output.update(flatten(child, path))
    return output


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def read_name_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise PreflightError(f"missing sample manifest: {path}")
    names = [
        Path(line.strip()).name
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(names) != len(set(names)):
        raise PreflightError(f"duplicate names in sample manifest: {path}")
    return names


def assert_exact_config_delta(d2: dict[str, Any], d4: dict[str, Any]) -> None:
    left = flatten(d2)
    right = flatten(d4)
    changed = {
        key: right.get(key)
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }
    if changed != EXPECTED_DIFFERENCES:
        raise PreflightError(
            "D4/D2 semantic differences do not match preregistration: "
            f"{changed}"
        )


def assert_no_forbidden_d3_fields(d4: dict[str, Any]) -> None:
    flattened = flatten(d4)
    present = sorted(key for key in FORBIDDEN_D3_FIELDS if key in flattened)
    if present:
        raise PreflightError(f"D3 cache/baseline-tail fields must be absent: {present}")


def assert_gate_isolation(d4: dict[str, Any]) -> None:
    evaluation = d4.get("evaluation", {})
    if evaluation.get("skip_validation") is not True:
        raise PreflightError("D4 must keep validation disabled during training")
    if evaluation.get("skip_final_test") is not True:
        raise PreflightError("D4 must keep final test disabled during training")
    if evaluation.get("standalone_test_mode") not in {None, "none"}:
        raise PreflightError("standalone test gate is enabled")
    if evaluation.get("final_test_mode") not in {None, "none"}:
        raise PreflightError("final test gate is enabled")
    flattened = flatten(d4)
    forbidden_tokens = ("reserved_blind", "scut115", "holdout40", "promotion")
    enabled_forbidden = [
        key
        for key, value in flattened.items()
        if any(token in key.lower() for token in forbidden_tokens)
        and value not in {None, False, "", "none", 0}
    ]
    if enabled_forbidden:
        raise PreflightError(f"later gates enabled in D4: {enabled_forbidden}")


def assert_output_dir_missing(repo_root: Path, d4: dict[str, Any]) -> Path:
    output_dir = resolve_repo_path(repo_root, d4["train"]["save_dir"])
    if output_dir.exists():
        raise PreflightError(f"D4 save_dir must not exist yet: {output_dir}")
    return output_dir


def assert_train_manifest_matches_d2(repo_root: Path, d2: dict[str, Any], d4: dict[str, Any]) -> dict[str, Any]:
    d2_manifest = resolve_repo_path(repo_root, d2["data"]["train_file_list"])
    d4_manifest = resolve_repo_path(repo_root, d4["data"]["train_file_list"])
    d2_names = read_name_manifest(d2_manifest)
    d4_names = read_name_manifest(d4_manifest)
    if d4_names != d2_names:
        raise PreflightError("D4 train manifest must match D2 exactly")
    inner_val = repo_root / "hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt"
    inner_names = set(read_name_manifest(inner_val))
    overlap = sorted(set(d4_names) & inner_names)
    if overlap:
        raise PreflightError(f"D4 train manifest overlaps inner-val15: {overlap[:5]}")
    return {
        "path": str(d4_manifest),
        "sha256": sha256_file(d4_manifest),
        "sample_count": len(d4_names),
        "inner_val_overlap": 0,
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise PreflightError(f"checkpoint must be a mapping: {path}")
    if "G_state_dict" not in checkpoint or "D_state_dict" not in checkpoint:
        raise PreflightError(f"checkpoint lacks generator/discriminator state: {path}")
    return checkpoint


def assert_frozen_current_primary_matches_ledger(
    repo_root: Path,
    ledger: dict[str, Any],
    d4: dict[str, Any],
) -> dict[str, Any]:
    baseline = ledger.get("baseline", {})
    config_info = baseline.get("config", {})
    checkpoint_info = baseline.get("checkpoint", {})
    config_path = resolve_repo_path(repo_root, str(config_info.get("path", "")))
    checkpoint_path = resolve_repo_path(repo_root, str(checkpoint_info.get("path", "")))
    expected_config_sha = str(config_info.get("sha256", ""))
    expected_checkpoint_sha = str(checkpoint_info.get("sha256", ""))
    if not expected_config_sha or not expected_checkpoint_sha:
        raise PreflightError("ledger baseline hashes are missing")
    if sha256_file(config_path) != expected_config_sha:
        raise PreflightError("current-primary config hash mismatch")
    if sha256_file(checkpoint_path) != expected_checkpoint_sha:
        raise PreflightError("current-primary weights hash mismatch")

    baseline_cfg = read_yaml(config_path)
    baseline_model_cfg = baseline_cfg.get("model", baseline_cfg)
    baseline_generator = Generator(cfg=baseline_model_cfg).eval()
    baseline_discriminator = Discriminator().eval()
    checkpoint = _load_checkpoint(checkpoint_path)
    try:
        baseline_generator.load_state_dict(checkpoint["G_state_dict"], strict=True)
        baseline_discriminator.load_state_dict(checkpoint["D_state_dict"], strict=True)
    except RuntimeError as exc:
        raise PreflightError(
            f"current-primary checkpoint is not strict-load compatible: {exc}"
        ) from exc

    sidecar_generator = Generator(cfg=d4["model"]).eval()
    sidecar_discriminator = Discriminator().eval()
    try:
        missing, unexpected = load_initial_checkpoint(
            sidecar_generator,
            sidecar_discriminator,
            str(checkpoint_path),
            torch.device("cpu"),
        )
    except (KeyError, RuntimeError) as exc:
        raise PreflightError(
            f"D4 current-primary initialization compatibility failed: {exc}"
        ) from exc
    return {
        "baseline_checkpoint": str(checkpoint_path),
        "baseline_config": str(config_path),
        "baseline_strict_load": True,
        "sidecar_missing_keys": len(missing),
        "sidecar_unexpected_keys": len(unexpected),
    }


def assert_structure_and_training_scope(d4: dict[str, Any]) -> dict[str, Any]:
    model_cfg = d4.get("model", {})
    train_cfg = d4.get("train", {})
    if (
        model_cfg.get("universal_residual_adapter_sidecar", {})
        .get("residual_parameterization")
        != "primary_edit_direction"
    ):
        raise PreflightError(
            "D4 must use primary_edit_direction residual_parameterization"
        )
    try:
        validate_universal_sidecar_config(d4)
    except (TypeError, ValueError) as exc:
        raise PreflightError(f"runtime config validation failed: {exc}") from exc
    structure = run_audit(
        model_cfg=model_cfg,
        trainable_patterns=train_cfg.get("trainable_generator_patterns"),
    )
    generator = Generator(cfg=model_cfg)
    trainable = apply_generator_trainable_patterns(
        generator,
        train_cfg.get("trainable_generator_patterns"),
    )
    frozen_batchnorm = freeze_batchnorm_running_stats(generator)
    return {
        "structure_audit": structure,
        "trainable_tensors": trainable["trainable_tensors"],
        "frozen_tensors": trainable["frozen_tensors"],
        "trainable_params": trainable["trainable_params"],
        "frozen_batchnorm_layers": frozen_batchnorm,
    }


def run_preflight(
    *,
    repo_root: Path = ROOT,
    d2_config: Path | None = None,
    d4_config: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    d2_path = d2_config or (repo_root / D2_CONFIG)
    d4_path = d4_config or (repo_root / D4_CONFIG)
    resolved_ledger = ledger_path or (repo_root / LEDGER_PATH)
    try:
        d2 = read_yaml(d2_path)
        d4 = read_yaml(d4_path)
        ledger = read_json(resolved_ledger)
        assert_no_forbidden_d3_fields(d4)
        assert_gate_isolation(d4)
        assert_exact_config_delta(d2, d4)
        output_dir = assert_output_dir_missing(repo_root, d4)
        manifest = assert_train_manifest_matches_d2(repo_root, d2, d4)
        checkpoint = assert_frozen_current_primary_matches_ledger(
            repo_root, ledger, d4
        )
        structure = assert_structure_and_training_scope(d4)
    except (KeyError, OSError, PreflightError, TypeError, ValueError) as exc:
        return {
            "first_gate": "scut_inner_val15",
            "reason": str(exc),
            "runnable": False,
            "terminal": "PREREQUISITE_NEEDED",
        }
    return {
        "checkpoint_audit": checkpoint,
        "config": str(d4_path),
        "first_gate": "scut_inner_val15",
        "output_dir": str(output_dir),
        "runnable": True,
        "structure_audit": structure,
        "terminal": "PASS",
        "train_manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_preflight(repo_root=args.repo_root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if result["terminal"] == "PASS" else 2)


if __name__ == "__main__":
    main()
