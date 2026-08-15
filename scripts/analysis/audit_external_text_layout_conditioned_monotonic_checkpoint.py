#!/usr/bin/env python3
"""Audit the conditioned monotonic checkpoint before any candidate gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.validate_external_text_layout_conditioned_preflight import (  # noqa: E402
    ACTIVE_ITERATION_ID,
    LEDGER_PATH,
    PATCH_MATERIALIZATION_ID,
    PLAN_PATH,
    SUPPORT_PREREQUISITE_ID,
    assert_exact_plan,
    read_json,
    repo_path,
    sha256_file,
    validate_artifact,
)
from scripts.infer.monotonic_residual_erase import (  # noqa: E402
    MODEL_TYPE,
    load_monotonic_residual_erase_model,
)
from scripts.train.train_external_text_layout_conditioned_monotonic import (  # noqa: E402
    CONDITIONED_INPUT_CHANNELS,
    LOSS_TERM_NAMES,
    MASK_SOURCE,
    ExternalTextLayoutConditionedPatchDataset,
)


DATA_ROOT = Path("data-links/samples/SCUT-HW5K-mixed-20260729")
CHECKPOINT_NAME = "external_text_layout_conditioned_monotonic.pt"
HISTORY_NAME = "conditioned_monotonic_loss_history.csv"
OUTPUT_PATH = Path(
    "outputs/external-text-layout-conditioned-monotonic-checkpoint-audit-20260815/audit.json"
)
GATE_THRESHOLD_GRAY = 12.0
OUTPUT_BOUND_GRAY = 20.4
EXPECTED_PATCH_COUNT = 256
EXPECTED_PARAMETER_COUNT = 385154
CHECKPOINT_PREREQUISITE_ID = (
    "external_text_layout_conditioned_monotonic_checkpoint_audit"
)
CHECKPOINT_RECORD_ID = "external-text-layout-conditioned-monotonic-checkpoint-audit"
KILL_OUTCOME = "conditioned_checkpoint_subthreshold_noop_before_candidate_gate"


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
    required = {
        SUPPORT_PREREQUISITE_ID: "passed",
        "external_text_layout_conditioned_monotonic_preflight": "passed",
        "external_text_layout_conditioned_monotonic_surface_integration": "passed",
        PATCH_MATERIALIZATION_ID: "passed",
    }
    for prerequisite, expected in required.items():
        if prerequisites.get(prerequisite) != expected:
            raise CheckpointAuditError(f"required prerequisite is not {expected}: {prerequisite}")

    records = [
        item
        for item in ledger.get("records", [])
        if isinstance(item, dict)
        and item.get("id") == "external-text-layout-conditioned-monotonic-patch-materialization"
    ]
    if len(records) != 1:
        raise CheckpointAuditError("conditioned patch materialization record count changed")
    materialization = records[0]
    if materialization.get("terminal") != "PASS":
        raise CheckpointAuditError("conditioned patch materialization is not PASS")
    for item in materialization.get("evidence", []):
        validate_artifact(repo_root, item, "patch materialization evidence")
    return {
        "patch_materialization": "passed",
        "preflight": "passed",
        "support": "passed",
        "surface": "passed",
    }


def checkpoint_path(repo_root: Path, plan: dict[str, Any]) -> Path:
    output_dir = repo_path(
        repo_root,
        plan["trainer"]["output_dir"],
        "training output directory",
    )
    path = output_dir / CHECKPOINT_NAME
    if not path.is_file():
        raise CheckpointAuditError(f"conditioned checkpoint is missing: {path}")
    return path


def validate_checkpoint_args(checkpoint: dict[str, Any], plan: dict[str, Any]) -> None:
    args = checkpoint.get("args")
    if not isinstance(args, dict):
        raise CheckpointAuditError("checkpoint args are missing")
    if any(isinstance(value, Path) for value in args.values()):
        raise CheckpointAuditError("checkpoint args contain non-portable Path objects")
    trainer = plan["trainer"]
    conditioned = plan["conditioned_input"]
    expected = {
        "batch_size": trainer["batch_size"],
        "data_root": str(DATA_ROOT),
        "device": trainer["device"],
        "input_channels": CONDITIONED_INPUT_CHANNELS,
        "input_dir": conditioned["rgb_root"],
        "layout_dir": conditioned["layout_root"],
        "layout_source": "external_text_occupancy_confidence",
        "log_every": trainer["log_every"],
        "lr": trainer["lr"],
        "mask_source": MASK_SOURCE,
        "max_steps": trainer["max_steps"],
        "model_type": MODEL_TYPE,
        "output_dir": trainer["output_dir"],
        "patch_index_file": trainer["patch_index_file"],
        "residual_delta_bound": trainer["residual_delta_bound"],
        "save_every": trainer["save_every"],
        "seed": trainer["seed"],
        "split": "train",
        "tile_size": trainer["tile_size"],
        "validation_enabled": False,
    }
    optional_defaults = {
        "luminance_margin_gray": 2.0,
        "support_positive_weight": 1.0,
        "support_preserve_weight": 1.0,
        "magnitude_weight": 1.0,
        "preserve_delta_weight": 1.0,
    }
    expected.update(optional_defaults)
    for key, value in expected.items():
        if args.get(key) != value:
            raise CheckpointAuditError(f"checkpoint arg changed: {key}")
    if checkpoint.get("step") != trainer["max_steps"]:
        raise CheckpointAuditError("checkpoint step is not the registered final step")


def summarize_history(history_path: Path, max_steps: int) -> dict[str, float | int]:
    if not history_path.is_file():
        raise CheckpointAuditError(f"training history is missing: {history_path}")
    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != max_steps:
        raise CheckpointAuditError(f"history row count changed: {len(rows)}")
    expected_fields = ["step", *LOSS_TERM_NAMES]
    if rows[0].keys() != dict.fromkeys(expected_fields).keys():
        raise CheckpointAuditError("history columns changed")
    first = rows[0]
    final = rows[-1]
    if int(first["step"]) != 1 or int(final["step"]) != max_steps:
        raise CheckpointAuditError("history step sequence changed")
    for row in rows:
        for name in LOSS_TERM_NAMES:
            value = float(row[name])
            if not math.isfinite(value) or value < 0.0:
                raise CheckpointAuditError("history contains invalid loss values")
    first_loss = float(first["loss"])
    final_loss = float(final["loss"])
    return {
        "final_bright_magnitude_l1": float(final["bright_magnitude_l1"]),
        "final_loss": final_loss,
        "final_preserve_delta_l1": float(final["preserve_delta_l1"]),
        "final_step": int(final["step"]),
        "first_loss": first_loss,
        "loss_delta": final_loss - first_loss,
        "row_count": len(rows),
    }


def classify_patch_summary(summary: dict[str, float | int]) -> dict[str, str]:
    if summary["negative_delta_pixel_count"]:
        raise CheckpointAuditError("checkpoint produced a negative monotonic delta")
    if summary["max_delta_gray"] > OUTPUT_BOUND_GRAY + 1e-6:
        raise CheckpointAuditError("checkpoint exceeded the monotonic output bound")
    if summary["patch_count"] != EXPECTED_PATCH_COUNT:
        raise CheckpointAuditError("checkpoint audit patch count changed")
    if (
        summary["positive_gate_ratio"] == 0.0
        and summary["preserve_gate_ratio"] == 0.0
        and summary["patches_with_gate"] == 0
        and summary["max_delta_gray"] < GATE_THRESHOLD_GRAY
    ):
        return {"outcome": KILL_OUTCOME, "terminal": "KILL"}
    raise CheckpointAuditError(
        "conditioned checkpoint reached the candidate gate; extend audit before inference"
    )


def audit_train_patches(
    repo_root: Path,
    plan: dict[str, Any],
    model: torch.nn.Module,
) -> dict[str, float | int | str]:
    trainer = plan["trainer"]
    conditioned = plan["conditioned_input"]
    dataset = ExternalTextLayoutConditionedPatchDataset(
        data_root=repo_root / DATA_ROOT,
        split="train",
        input_dir=repo_path(repo_root, conditioned["rgb_root"], "rgb root"),
        layout_dir=repo_path(repo_root, conditioned["layout_root"], "layout root"),
        patch_index_file=repo_path(
            repo_root,
            trainer["patch_index_file"],
            "patch index",
        ),
        tile_size=trainer["tile_size"],
    )
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
    min_delta = float("inf")
    negative_delta_pixels = 0
    patches_with_gate = 0
    with torch.no_grad():
        for features, target in loader:
            components = model.forward_components(features)
            target_delta = target.mean(1, keepdim=True) - features[:, :3].mean(
                1,
                keepdim=True,
            )
            positive = target_delta > (2.0 / 255.0)
            preserve = ~positive
            delta = components["signed_delta"].mean(1, keepdim=True) * 255.0
            alpha = components["edit_alpha"]
            gate = (alpha >= 0.5) & (delta >= GATE_THRESHOLD_GRAY)
            positive_count = int(positive.sum().cpu())
            preserve_count = int(preserve.sum().cpu())
            counts["positive"] += positive_count
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
            min_delta = min(min_delta, float(delta.min().cpu()))
            negative_delta_pixels += int((delta < -1e-7).sum().cpu())
            patches_with_gate += int(bool(gate.any().cpu()))
    positive_count = counts["positive"]
    preserve_count = counts["preserve"]
    summary: dict[str, float | int | str] = {
        "max_delta_gray": max_delta,
        "min_delta_gray": min_delta,
        "negative_delta_pixel_count": negative_delta_pixels,
        "patch_count": len(dataset),
        "patches_with_gate": patches_with_gate,
        "positive_alpha_mean": sums["positive_alpha"] / positive_count,
        "positive_delta_mean_gray": sums["positive_delta"] / positive_count,
        "positive_gate_ratio": counts["positive_gate"] / positive_count,
        "positive_pixels": positive_count,
        "positive_support_ratio": counts["positive_support"] / positive_count,
        "preserve_alpha_mean": sums["preserve_alpha"] / preserve_count,
        "preserve_delta_mean_gray": sums["preserve_delta"] / preserve_count,
        "preserve_gate_ratio": counts["preserve_gate"] / preserve_count,
        "preserve_pixels": preserve_count,
        "preserve_support_ratio": counts["preserve_support"] / preserve_count,
    }
    summary.update(classify_patch_summary(summary))
    return summary


def validate_forbidden_outputs(repo_root: Path, plan: dict[str, Any]) -> list[str]:
    forbidden = ["first_gate_candidate", "first_gate_score"]
    absent: list[str] = []
    outputs = plan["planned_outputs_must_be_absent"]
    for label in forbidden:
        path = repo_path(repo_root, outputs[label], label)
        if path.exists():
            raise CheckpointAuditError(f"quality output exists before audit close: {path}")
        absent.append(str(path))
    return absent


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
        plan = read_json(resolved_plan)
        assert_exact_plan(plan)
        ledger = read_json(resolved_ledger)
        authority = validate_ledger_authority(repo_root, ledger)
        checkpoint = checkpoint_path(repo_root, plan)
        history = checkpoint.parent / HISTORY_NAME
        state = torch.load(checkpoint, map_location="cpu")
        if not isinstance(state, dict):
            raise CheckpointAuditError("checkpoint payload is not a mapping")
        validate_checkpoint_args(state, plan)
        model = load_monotonic_residual_erase_model(checkpoint, torch.device("cpu"))
        if model.input_channels != CONDITIONED_INPUT_CHANNELS:
            raise CheckpointAuditError("checkpoint is not five-channel conditioned")
        structural = {
            "checkpoint_loader": "torch_default_weights_only_portable",
            "final_step": state["step"],
            "input_channels": model.input_channels,
            "model_type": state["args"]["model_type"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "residual_delta_bound": model.residual_delta_bound,
        }
        if structural["parameter_count"] != EXPECTED_PARAMETER_COUNT:
            raise CheckpointAuditError("conditioned checkpoint parameter count changed")
        history_summary = summarize_history(history, plan["trainer"]["max_steps"])
        patch_summary = audit_train_patches(repo_root, plan, model.eval())
        absent = validate_forbidden_outputs(repo_root, plan)
    except (
        CheckpointAuditError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        return {
            "reason": str(exc),
            "runnable": False,
            "terminal": "PREREQUISITE_NEEDED",
        }
    return {
        "absent_quality_outputs": absent,
        "authority": authority,
        "candidate_inference_started": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "history": str(history),
        "history_sha256": sha256_file(history),
        "history_summary": history_summary,
        "outcome": KILL_OUTCOME,
        "patch_summary": patch_summary,
        "plan": str(resolved_plan),
        "plan_sha256": sha256_file(resolved_plan),
        "promotion_enabled": False,
        "quality_gate_started": False,
        "repeat_policy": "do_not_repeat_conditioned_monotonic_v1_training_or_rescue_thresholds",
        "reserved_blind_state": "unavailable",
        "runnable": True,
        "schema_version": 1,
        "status": "kill",
        "structural": structural,
        "terminal": "KILL",
        "training_started": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    result = run_audit(
        repo_root=args.repo_root,
        plan_path=args.plan,
        ledger_path=args.ledger,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        path = args.output_json
        if not path.is_absolute():
            path = args.repo_root.resolve() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["terminal"] == "KILL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
