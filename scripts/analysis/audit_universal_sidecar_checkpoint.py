#!/usr/bin/env python3
"""Audit a Universal Residual Adapter Sidecar training checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SIDECAR_PREFIX = "universal_residual_adapter_sidecar."
OPTIMIZER_KEYS = ("optimizer_G", "optimizer_D")
SCHEDULER_KEYS = ("scheduler_G", "scheduler_D")


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    value = tensor.detach().float().cpu()
    return {
        "norm": float(value.norm()),
        "maxabs": float(value.abs().max()) if value.numel() else 0.0,
    }


def audit_checkpoint(
    *,
    candidate_checkpoint: str | Path,
    baseline_checkpoint: str | Path,
) -> dict[str, Any]:
    candidate_checkpoint = Path(candidate_checkpoint)
    baseline_checkpoint = Path(baseline_checkpoint)
    candidate = torch.load(candidate_checkpoint, map_location="cpu", weights_only=False)
    baseline = torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
    candidate_g = candidate["G_state_dict"]
    baseline_g = baseline["G_state_dict"]

    sidecar_keys = sorted(k for k in candidate_g if k.startswith(SIDECAR_PREFIX))
    base_changed_keys = []
    base_missing_keys = []
    for key, baseline_value in baseline_g.items():
        if key.startswith(SIDECAR_PREFIX):
            continue
        candidate_value = candidate_g.get(key)
        if candidate_value is None:
            base_missing_keys.append(key)
            continue
        if not torch.equal(candidate_value.detach().cpu(), baseline_value.detach().cpu()):
            base_changed_keys.append(key)

    unexpected_non_sidecar_keys = sorted(
        key for key in candidate_g
        if not key.startswith(SIDECAR_PREFIX) and key not in baseline_g
    )
    final_projection_stats = {
        key: _tensor_stats(candidate_g[key])
        for key in sidecar_keys
        if key.endswith(".2.weight") or key.endswith(".2.bias")
    }
    moved_final_projection_keys = sorted(
        key for key, stats in final_projection_stats.items()
        if stats["maxabs"] > 0.0
    )
    global_scale_key = SIDECAR_PREFIX + "global_residual_scale"
    global_scale_stats = (
        _tensor_stats(candidate_g[global_scale_key])
        if global_scale_key in candidate_g
        else {"norm": 0.0, "maxabs": 0.0}
    )
    has_optimizer_state = any(candidate.get(key) is not None for key in OPTIMIZER_KEYS)
    has_scheduler_state = any(candidate.get(key) is not None for key in SCHEDULER_KEYS)

    status = "pass"
    failures: list[str] = []
    if not moved_final_projection_keys:
        failures.append("sidecar_final_projection_not_moved")
    if base_changed_keys or base_missing_keys or unexpected_non_sidecar_keys:
        failures.append("base_generator_changed")
    if has_optimizer_state:
        failures.append("optimizer_state_present")
    if has_scheduler_state:
        failures.append("scheduler_state_present")
    if failures:
        status = "fail"

    return {
        "status": status,
        "candidate_checkpoint": str(candidate_checkpoint),
        "baseline_checkpoint": str(baseline_checkpoint),
        "epoch": candidate.get("epoch"),
        "avg_loss_G": candidate.get("avg_loss_G"),
        "avg_loss_D": candidate.get("avg_loss_D"),
        "sidecar_key_count": len(sidecar_keys),
        "moved_final_projection_key_count": len(moved_final_projection_keys),
        "moved_final_projection_keys": moved_final_projection_keys,
        "global_residual_scale": global_scale_stats,
        "base_changed_count_vs_current_primary": len(base_changed_keys),
        "base_missing_count_vs_current_primary": len(base_missing_keys),
        "unexpected_non_sidecar_key_count": len(unexpected_non_sidecar_keys),
        "has_optimizer_state": has_optimizer_state,
        "has_scheduler_state": has_scheduler_state,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument(
        "--baseline-checkpoint",
        default="./artifacts/current-primary/micro_region_probe_step0001.pth",
    )
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = audit_checkpoint(
        candidate_checkpoint=args.candidate_checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
