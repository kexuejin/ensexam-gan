#!/usr/bin/env python3
"""Materialize frozen train-only primary reconstruction-stage disagreement."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer.run_primary_full_page import (  # noqa: E402
    load_generator,
    pick_device,
)
from utils.page_inference import ticks  # noqa: E402


PLAN_PATH = Path("docs/reconstruction-stage-disagreement-prerequisite-v1.json")
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
CHANNELS = (
    "coarse_refine_signed_luma",
    "coarse_refine_abs_rgb",
    "coarse_ic2_abs_rgb",
    "coarse_ic4_abs_rgb",
)
FORBIDDEN_SAMPLE_COMPONENTS = {
    "label",
    "labels",
    "target",
    "targets",
    "all_labels",
}


class MaterializationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_rows(rows: list[str]) -> str:
    payload = "\n".join(sorted(rows)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MaterializationError(f"expected JSON object: {path}")
    return value


def repo_path(repo_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise MaterializationError(f"path must stay repository-relative: {value}")
    return repo_root / relative


def validate_artifact(
    repo_root: Path, artifact: dict[str, Any], label: str
) -> Path:
    if set(artifact) != {"path", "sha256"}:
        raise MaterializationError(f"{label} artifact contract changed")
    path = repo_path(repo_root, str(artifact["path"]))
    if not path.is_file():
        raise MaterializationError(f"missing {label}: {path}")
    if sha256_file(path) != artifact["sha256"]:
        raise MaterializationError(f"{label} hash changed")
    return path


def validate_authority(ledger: dict[str, Any]) -> None:
    program = ledger.get("program", {})
    if (
        program.get("product_default") != "artifacts/current-primary"
        or program.get("promotion_state") != "disabled"
        or program.get("reserved_blind_state") != "disabled"
    ):
        raise MaterializationError("product authority changed")
    active = ledger.get("active_iteration", {})
    if (
        active.get("id") != "monotonic-residual-erase-support"
        or active.get("terminal") != "PREREQUISITE_NEEDED"
        or active.get("first_gate") != "scut_inner_val15"
    ):
        raise MaterializationError("active iteration authority changed")
    statuses = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if statuses.get("materially_new_target_free_support_preregistration") != "passed":
        raise MaterializationError("stage-disagreement preregistration is not passed")
    if statuses.get("reconstruction_stage_disagreement_train_only_diagnostic") != "pending":
        raise MaterializationError("stage-disagreement diagnostic is not pending")


def validate_plan(plan: dict[str, Any]) -> None:
    if (
        plan.get("schema_version") != 1
        or plan.get("iteration_id") != "monotonic-residual-erase-support"
        or plan.get("state")
        != "preregistered_pending_stage_disagreement_materialization"
    ):
        raise MaterializationError("stage-disagreement plan state changed")
    representation = plan.get("representation", {})
    if (
        representation.get("channels") != list(CHANNELS)
        or representation.get("feature_count") != 4
        or representation.get("single_causal_change")
        != "frozen_primary_multiscale_reconstruction_stage_disagreement"
        or representation.get("no_source_rgb") is not True
        or representation.get("no_final_prediction_rgb") is not True
        or representation.get("no_masks_or_page_scalars") is not True
    ):
        raise MaterializationError("stage-disagreement representation changed")
    spec = plan.get("stage_disagreement_materialization", {})
    required = {
        "batch_size": 8,
        "derive_per_patch_before_overlap_fusion": True,
        "device": "auto",
        "dtype": "float32",
        "encoding": "one_compressed_npz_per_page_with_exact_channel_keys",
        "output_root": (
            "outputs/reconstruction-stage-disagreement-materialization-20260812"
        ),
        "page_overlap": 32,
        "provenance": (
            "networks.generator.Generator.forward via utils.page_inference "
            "full-page patch schedule"
        ),
        "source_manifest": (
            "hardcase_lists/monotonic-residual-erase-train275-v1.txt"
        ),
        "target_access": False,
    }
    for key, expected in required.items():
        if spec.get(key) != expected:
            raise MaterializationError(f"materialization field changed: {key}")
    if set(spec.get("channel_definitions", {})) != set(CHANNELS):
        raise MaterializationError("stage-disagreement channel definitions changed")
    authorization = plan.get("authorization", {})
    if (
        authorization.get("stage_disagreement_materialization") is not True
        or authorization.get("stage_disagreement_materialization_reads_targets")
        is not False
    ):
        raise MaterializationError("stage-disagreement authorization changed")


def assert_source_image_path(path: Path) -> None:
    for candidate in (path, path.resolve()):
        if {part.lower() for part in candidate.parts} & FORBIDDEN_SAMPLE_COMPONENTS:
            raise MaterializationError(f"source path appears to be a target: {path}")


def read_manifest(repo_root: Path, path: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        source = repo_path(repo_root, value)
        assert_source_image_path(source)
        name = source.name
        if name in seen or not source.is_file():
            raise MaterializationError(f"invalid or duplicate train source: {value}")
        seen.add(name)
        rows.append((name, source))
    if len(rows) != 275:
        raise MaterializationError(f"expected 275 train sources, got {len(rows)}")
    return rows


def validate_inputs(
    repo_root: Path, plan: dict[str, Any], ledger: dict[str, Any]
) -> tuple[Path, Path, Path, list[tuple[str, Path]]]:
    validate_plan(plan)
    validate_authority(ledger)
    evidence = plan["evidence"]
    config = validate_artifact(
        repo_root, evidence["current_primary_config"], "primary config"
    )
    checkpoint = validate_artifact(
        repo_root, evidence["current_primary_checkpoint"], "primary checkpoint"
    )
    validate_artifact(repo_root, evidence["generator_source"], "generator source")
    validate_artifact(
        repo_root, evidence["page_inference_source"], "page inference source"
    )
    role_plan = validate_artifact(repo_root, evidence["role_plan"], "role plan")
    role_wrapper = read_json(role_plan)
    base_role = role_wrapper.get("evidence", {}).get("base_role_contract")
    if not isinstance(base_role, dict):
        raise MaterializationError("role plan lacks base role contract")
    validate_artifact(repo_root, base_role, "base role contract")
    manifest = validate_artifact(
        repo_root, plan["data"]["manifest"], "train manifest"
    )
    rows = read_manifest(repo_root, manifest)
    if len(rows) != plan["data"]["effective_train_count"]:
        raise MaterializationError("train manifest count changed")
    output_root = repo_path(
        repo_root, plan["stage_disagreement_materialization"]["output_root"]
    )
    if output_root.exists():
        raise MaterializationError("stage-disagreement output directory already exists")
    return config, checkpoint, manifest, rows


def stage_disagreement_channels(
    ic4: torch.Tensor,
    ic2: torch.Tensor,
    ic1: torch.Tensor,
    ire: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if ic1.shape != ire.shape or ic1.ndim != 4 or ic1.shape[1] != 3:
        raise MaterializationError("invalid full-resolution stage tensors")
    ic2_full = F.interpolate(
        ic2, size=ic1.shape[-2:], mode="bilinear", align_corners=False
    )
    ic4_full = F.interpolate(
        ic4, size=ic1.shape[-2:], mode="bilinear", align_corners=False
    )
    refine_delta = (ire - ic1) * 127.5
    channels = {
        "coarse_refine_signed_luma": refine_delta.mean(dim=1),
        "coarse_refine_abs_rgb": refine_delta.abs().mean(dim=1),
        "coarse_ic2_abs_rgb": ((ic2_full - ic1) * 127.5).abs().mean(dim=1),
        "coarse_ic4_abs_rgb": ((ic4_full - ic1) * 127.5).abs().mean(dim=1),
    }
    if any(
        value.shape != ic1.shape[:1] + ic1.shape[2:]
        or not torch.isfinite(value).all()
        for value in channels.values()
    ):
        raise MaterializationError("invalid stage-disagreement tensor")
    return channels


@torch.no_grad()
def infer_stage_disagreement_full_page(
    generator: torch.nn.Module,
    rgb: np.ndarray,
    device: torch.device,
    *,
    patch_size: int = 512,
    overlap: int = 32,
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise MaterializationError("expected RGB source image")
    source = rgb.astype(np.float32) / 127.5 - 1.0
    height, width = source.shape[:2]
    stride = max(patch_size - overlap, 1)
    ys = ticks(height, patch_size, stride)
    xs = ticks(width, patch_size, stride)
    sums = {
        name: np.zeros((height, width), dtype=np.float64) for name in CHANNELS
    }
    weight = np.zeros((height, width), dtype=np.float64)
    pending_patches: list[np.ndarray] = []
    pending_meta: list[tuple[int, int, int, int]] = []

    def flush_pending() -> None:
        if not pending_patches:
            return
        patch_batch = np.stack(pending_patches, axis=0)
        patch_tensor = torch.from_numpy(patch_batch).permute(0, 3, 1, 2).to(device)
        _ms, _mb, ic4, ic2, ic1, ire, _icomp = generator(patch_tensor)
        batch_channels = stage_disagreement_channels(ic4, ic2, ic1, ire)
        cpu_channels = {
            name: value.detach().cpu().numpy().astype(np.float32, copy=False)
            for name, value in batch_channels.items()
        }
        for batch_index, (y, x, patch_h, patch_w) in enumerate(pending_meta):
            for name in CHANNELS:
                sums[name][y : y + patch_h, x : x + patch_w] += cpu_channels[
                    name
                ][batch_index, :patch_h, :patch_w].astype(np.float64)
            weight[y : y + patch_h, x : x + patch_w] += 1.0
        pending_patches.clear()
        pending_meta.clear()

    for y in ys:
        for x in xs:
            patch = source[y : y + patch_size, x : x + patch_size]
            patch_h, patch_w = patch.shape[:2]
            if patch_h != patch_size or patch_w != patch_size:
                canvas = np.ones((patch_size, patch_size, 3), dtype=np.float32)
                canvas[:patch_h, :patch_w] = patch
                patch = canvas
            pending_patches.append(patch)
            pending_meta.append((y, x, patch_h, patch_w))
            if len(pending_patches) >= max(int(batch_size), 1):
                flush_pending()
    flush_pending()
    if (weight <= 0).any():
        raise MaterializationError("full-page patch schedule left uncovered pixels")
    result = {
        name: (values / weight).astype(np.float32) for name, values in sums.items()
    }
    if any(
        value.shape != (height, width) or not np.isfinite(value).all()
        for value in result.values()
    ):
        raise MaterializationError("invalid fused stage-disagreement map")
    return result


def write_manifest(
    output_root: Path,
    *,
    repo_root: Path,
    plan: dict[str, Any],
    plan_path: Path,
    config: Path,
    checkpoint: Path,
    source_manifest: Path,
    rows: list[dict[str, Any]],
) -> Path:
    spec = plan["stage_disagreement_materialization"]
    payload = {
        "schema_version": 1,
        "terminal": "PASS",
        "provenance": spec["provenance"],
        "target_access": False,
        "train_count": len(rows),
        "plan": {
            "path": str(plan_path.relative_to(repo_root)),
            "sha256": sha256_file(plan_path),
        },
        "source_manifest": {
            "path": str(source_manifest.relative_to(repo_root)),
            "sha256": sha256_file(source_manifest),
        },
        "primary_config": {
            "path": str(config.relative_to(repo_root)),
            "sha256": sha256_file(config),
        },
        "primary_checkpoint": {
            "path": str(checkpoint.relative_to(repo_root)),
            "sha256": sha256_file(checkpoint),
        },
        "generator_source": plan["evidence"]["generator_source"],
        "page_inference_source": plan["evidence"]["page_inference_source"],
        "page_overlap": spec["page_overlap"],
        "batch_size": spec["batch_size"],
        "dtype": spec["dtype"],
        "encoding": spec["encoding"],
        "channels": list(CHANNELS),
        "channel_definitions": spec["channel_definitions"],
        "derive_per_patch_before_overlap_fusion": True,
        "pages": rows,
    }
    path = output_root / "manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def materialize(
    *, repo_root: Path, plan_path: Path = PLAN_PATH, ledger_path: Path = LEDGER_PATH
) -> dict[str, Any]:
    plan_file = repo_path(repo_root, str(plan_path))
    ledger_file = repo_path(repo_root, str(ledger_path))
    plan = read_json(plan_file)
    ledger = read_json(ledger_file)
    config, checkpoint, source_manifest, sources = validate_inputs(
        repo_root, plan, ledger
    )
    output_root = repo_path(
        repo_root, plan["stage_disagreement_materialization"]["output_root"]
    )
    page_dir = output_root / "pages"
    page_dir.mkdir(parents=True)
    spec = plan["stage_disagreement_materialization"]
    device = pick_device(spec["device"])
    generator = load_generator(str(config), str(checkpoint), device)
    rows: list[dict[str, Any]] = []
    for index, (file_name, source) in enumerate(sources, start=1):
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            raise MaterializationError(f"source decode failed: {source}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        channels = infer_stage_disagreement_full_page(
            generator,
            rgb,
            device,
            overlap=int(spec["page_overlap"]),
            batch_size=int(spec["batch_size"]),
        )
        path = page_dir / f"{Path(file_name).stem}.npz"
        np.savez_compressed(path, **channels)
        if not path.is_file():
            raise MaterializationError(f"failed to write stage NPZ: {file_name}")
        row: dict[str, Any] = {
            "file": file_name,
            "source_sha256": sha256_file(source),
            "npz_sha256": sha256_file(path),
            "height": int(rgb.shape[0]),
            "width": int(rgb.shape[1]),
            "channels": {},
        }
        for name in CHANNELS:
            values = channels[name]
            row["channels"][name] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
        rows.append(row)
        if index % 25 == 0 or index == len(sources):
            print(
                f"materialized train-only stage disagreement {index}/{len(sources)}",
                flush=True,
            )
    manifest = write_manifest(
        output_root,
        repo_root=repo_root,
        plan=plan,
        plan_path=plan_file,
        config=config,
        checkpoint=checkpoint,
        source_manifest=source_manifest,
        rows=rows,
    )
    return {
        "terminal": "PASS",
        "output_root": str(output_root),
        "manifest": str(manifest),
        "train_count": len(rows),
        "content_sha256": sha256_rows(
            [f"{row['file']} {row['npz_sha256']}" for row in rows]
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = materialize(
            repo_root=args.repo_root,
            plan_path=args.plan,
            ledger_path=args.ledger,
        )
    except (MaterializationError, OSError, ValueError) as error:
        print(f"terminal=PREREQUISITE_NEEDED reason={error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
