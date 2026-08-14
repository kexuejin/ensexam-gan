#!/usr/bin/env python3
"""Audit the registered monotonic v2 checkpoint before any quality gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.validate_monotonic_residual_erase_candidate_application_preflight import (  # noqa: E402
    ACTIVE_ITERATION_ID,
    APPLICATION_OUTCOME,
    APPLICATION_RECORD_ID,
    APPLICATION_PREREQUISITE_ID,
    MATERIALIZATION_PREREQUISITE_ID,
    MATERIALIZATION_RECORD_ID,
    MATERIALIZATION_OUTCOME,
    LEDGER_PATH,
    PLAN_PATH,
    PLAN_SHA256,
    read_json,
    repo_path,
    sha256_file,
    validate_artifact,
    validate_plan,
)
from scripts.infer.monotonic_residual_erase import (  # noqa: E402
    MonotonicResidualEraseCleanupNet,
)
from scripts.train.train_monotonic_residual_erase import (  # noqa: E402
    MonotonicTargetDifferencePatchDataset,
)


CHECKPOINT_NAME = "monotonic_residual_erase_probe.pt"
HISTORY_NAME = "monotonic_loss_history.csv"
GATE_THRESHOLD_GRAY = 12.0
OUTPUT_BOUND_GRAY = 20.4
KILL_OUTCOME = "real_train_patch_gate_collapsed_to_subthreshold_noop"


class CheckpointAuditError(RuntimeError):
    pass


def validate_ledger_authority(
    repo_root: Path,
    ledger: dict[str, Any],
) -> dict[str, str]:
    active = ledger.get("active_iteration", {})
    if active.get("id") != ACTIVE_ITERATION_ID:
        raise CheckpointAuditError("active iteration changed")
    prerequisites = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if prerequisites.get(MATERIALIZATION_PREREQUISITE_ID) != "passed":
        raise CheckpointAuditError("materialization prerequisite is not passed")
    if prerequisites.get(APPLICATION_PREREQUISITE_ID) != "passed":
        raise CheckpointAuditError("candidate application prerequisite is not passed")

    materialization_records = [
        item
        for item in ledger.get("records", [])
        if isinstance(item, dict) and item.get("id") == MATERIALIZATION_RECORD_ID
    ]
    if len(materialization_records) != 1:
        raise CheckpointAuditError("materialization PASS record count changed")
    materialization = materialization_records[0]
    if (
        materialization.get("terminal") != "PASS"
        or materialization.get("outcome") != MATERIALIZATION_OUTCOME
    ):
        raise CheckpointAuditError("materialization PASS authority changed")
    for item in materialization.get("evidence", []):
        validate_artifact(repo_root, item, "materialization evidence")

    application_records = [
        item
        for item in ledger.get("records", [])
        if isinstance(item, dict) and item.get("id") == APPLICATION_RECORD_ID
    ]
    if len(application_records) != 1:
        raise CheckpointAuditError("candidate application PASS record count changed")
    application = application_records[0]
    if (
        application.get("terminal") != "PASS"
        or application.get("outcome") != APPLICATION_OUTCOME
    ):
        raise CheckpointAuditError("candidate application PASS authority changed")
    for item in application.get("evidence", []):
        validate_artifact(repo_root, item, "candidate application evidence")
    return {
        "materialization": "passed",
        "candidate_application": "passed",
    }


def checkpoint_path(repo_root: Path, plan: dict[str, Any]) -> Path:
    output_dir = repo_path(
        repo_root,
        plan["trainer"]["output_dir"],
        "training output directory",
    )
    path = output_dir / CHECKPOINT_NAME
    if not path.is_file():
        raise CheckpointAuditError(f"checkpoint is missing: {path}")
    return path


def validate_checkpoint_args(
    checkpoint: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    args = checkpoint.get("args")
    if not isinstance(args, dict):
        raise CheckpointAuditError("checkpoint args are missing")
    trainer = plan["trainer"]
    expected = {
        "model_type": "monotonic_residual_erase",
        "mask_source": "target_luma_delta",
        "residual_delta_bound": trainer["residual_delta_bound"],
        "split": trainer["split"],
        "tile_size": trainer["tile_size"],
        "max_steps": trainer["max_steps"],
        "batch_size": trainer["batch_size"],
        "lr": trainer["lr"],
        "seed": trainer["seed"],
        "log_every": trainer["log_every"],
        "save_every": trainer["save_every"],
        "luminance_margin_gray": trainer["luminance_margin_gray"],
        "support_positive_weight": trainer["support_positive_weight"],
        "support_preserve_weight": trainer["support_preserve_weight"],
        "magnitude_weight": trainer["magnitude_weight"],
        "preserve_delta_weight": trainer["preserve_delta_weight"],
        "validation_enabled": False,
    }
    for key, value in expected.items():
        if args.get(key) != value:
            raise CheckpointAuditError(f"checkpoint arg changed: {key}")
    if checkpoint.get("step") != trainer["max_steps"]:
        raise CheckpointAuditError("checkpoint step is not the registered final step")


def audit_train_patches(
    repo_root: Path,
    plan: dict[str, Any],
    model: MonotonicResidualEraseCleanupNet,
    device: torch.device,
) -> dict[str, Any]:
    trainer = plan["trainer"]
    base_plan = read_json(
        repo_path(
            repo_root,
            plan["evidence"]["base_training_plan"]["path"],
            "base training plan",
        )
    )
    dataset = MonotonicTargetDifferencePatchDataset(
        data_root=repo_path(repo_root, base_plan["data"]["data_root"], "data root"),
        split=trainer["split"],
        input_dir=repo_path(repo_root, trainer["input_dir"], "training input dir"),
        patch_index_file=repo_path(
            repo_root,
            trainer["patch_index_file"],
            "training patch index",
        ),
        tile_size=trainer["tile_size"],
    )
    if len(dataset) != 256:
        raise CheckpointAuditError(f"checkpoint audit patch count changed: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    counts = {
        "positive": 0,
        "preserve": 0,
        "positive_support": 0,
        "preserve_support": 0,
        "positive_gate": 0,
        "preserve_gate": 0,
    }
    sums = {
        "positive_delta": 0.0,
        "preserve_delta": 0.0,
        "positive_alpha": 0.0,
        "preserve_alpha": 0.0,
    }
    max_delta = 0.0
    negative_delta_pixels = 0
    patches_with_gate = 0
    patches_all_preserve_route = 0
    with torch.no_grad():
        for inp, target in loader:
            inp = inp.to(device)
            target = target.to(device)
            components = model.forward_components(inp)
            target_delta = target.mean(1, keepdim=True) - inp.mean(1, keepdim=True)
            positive = target_delta > (trainer["luminance_margin_gray"] / 255.0)
            preserve = ~positive
            delta = components["signed_delta"].mean(1, keepdim=True) * 255.0
            alpha = components["edit_alpha"]
            gate = (alpha >= 0.5) & (delta >= GATE_THRESHOLD_GRAY)
            pos_count = int(positive.sum().cpu())
            preserve_count = int(preserve.sum().cpu())
            counts["positive"] += pos_count
            counts["preserve"] += preserve_count
            counts["positive_support"] += int(((alpha >= 0.5) & positive).sum().cpu())
            counts["preserve_support"] += int(((alpha >= 0.5) & preserve).sum().cpu())
            counts["positive_gate"] += int((gate & positive).sum().cpu())
            counts["preserve_gate"] += int((gate & preserve).sum().cpu())
            sums["positive_delta"] += float((delta * positive).sum().cpu())
            sums["preserve_delta"] += float((delta * preserve).sum().cpu())
            sums["positive_alpha"] += float((alpha * positive).sum().cpu())
            sums["preserve_alpha"] += float((alpha * preserve).sum().cpu())
            max_delta = max(max_delta, float(delta.max().cpu()))
            negative_delta_pixels += int((delta < -1e-7).sum().cpu())
            patches_with_gate += int(bool(gate.any().cpu()))
            patches_all_preserve_route += int(not bool((alpha >= 0.5).any().cpu()))
    if negative_delta_pixels:
        raise CheckpointAuditError("checkpoint produced a negative monotonic delta")

    positive_count = counts["positive"]
    preserve_count = counts["preserve"]
    summary = {
        "patch_count": len(dataset),
        "positive_pixels": positive_count,
        "preserve_pixels": preserve_count,
        "positive_alpha_mean": sums["positive_alpha"] / positive_count,
        "preserve_alpha_mean": sums["preserve_alpha"] / preserve_count,
        "positive_support_ratio": counts["positive_support"] / positive_count,
        "preserve_support_ratio": counts["preserve_support"] / preserve_count,
        "positive_delta_mean_gray": sums["positive_delta"] / positive_count,
        "preserve_delta_mean_gray": sums["preserve_delta"] / preserve_count,
        "positive_gate_ratio": counts["positive_gate"] / positive_count,
        "preserve_gate_ratio": counts["preserve_gate"] / preserve_count,
        "max_delta_gray": max_delta,
        "negative_delta_pixel_count": negative_delta_pixels,
        "patches_with_gate": patches_with_gate,
        "patches_all_preserve_route": patches_all_preserve_route,
    }
    if (
        summary["positive_gate_ratio"] == 0.0
        and summary["preserve_gate_ratio"] == 0.0
    ):
        summary["terminal"] = "KILL"
        summary["outcome"] = KILL_OUTCOME
        return summary
    raise CheckpointAuditError(
        "checkpoint unexpectedly reached a nonzero gate; structural audit must be extended"
    )


def run_audit(
    *,
    repo_root: Path = ROOT,
    plan_path: Path | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_plan = plan_path or (repo_root / PLAN_PATH)
    resolved_ledger = ledger_path or (repo_root / LEDGER_PATH)
    try:
        if sha256_file(resolved_plan) != PLAN_SHA256:
            raise CheckpointAuditError("candidate plan hash changed")
        plan = validate_plan(resolved_plan)
        ledger = read_json(resolved_ledger)
        authority = validate_ledger_authority(repo_root, ledger)
        checkpoint = checkpoint_path(repo_root, plan)
        history = checkpoint.parent / HISTORY_NAME
        if not history.is_file():
            raise CheckpointAuditError(f"training history is missing: {history}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise CheckpointAuditError("checkpoint payload is not a mapping")
        validate_checkpoint_args(state, plan)
        model = MonotonicResidualEraseCleanupNet(
            float(state["args"]["residual_delta_bound"])
        )
        model.load_state_dict(state["model"])
        model = model.to(torch.device("mps"))
        model.eval()
        structural = {
            "model_type": state["args"]["model_type"],
            "residual_delta_bound": model.residual_delta_bound,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "final_step": state["step"],
            "checkpoint_loader": "trusted_weights_only_false_due_to_Path_args",
        }
        if structural["parameter_count"] != 384578:
            raise CheckpointAuditError("monotonic checkpoint parameter count changed")
        patch_summary = audit_train_patches(
            repo_root,
            plan,
            model,
            torch.device("mps"),
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, CheckpointAuditError) as exc:
        return {
            "terminal": "PREREQUISITE_NEEDED",
            "runnable": False,
            "reason": str(exc),
        }
    return {
        "status": "kill",
        "terminal": "KILL",
        "runnable": True,
        "outcome": KILL_OUTCOME,
        "repeat_policy": "do_not_repeat_exact_v2_training",
        "plan": str(resolved_plan),
        "plan_sha256": sha256_file(resolved_plan),
        "authority": authority,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "history": str(history),
        "history_sha256": sha256_file(history),
        "structural": structural,
        "patch_summary": patch_summary,
        "training_started": True,
        "candidate_inference_started": False,
        "quality_gate_started": False,
        "promotion_enabled": False,
        "reserved_blind_state": "unavailable",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = run_audit(
        repo_root=args.repo_root,
        plan_path=args.plan,
        ledger_path=args.ledger,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["terminal"] == "KILL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
