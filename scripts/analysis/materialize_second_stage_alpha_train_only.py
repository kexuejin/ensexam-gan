#!/usr/bin/env python3
"""Materialize frozen train-only raw second-stage alpha maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.audit_dual_input_support_separation import (  # noqa: E402
    read_json,
    repo_path,
    validate_file_artifact,
    validate_prediction_set,
)
from scripts.analysis.build_sign_separated_residual_patch_index import (  # noqa: E402
    effective_train_filenames,
)
from scripts.analysis.materialize_reconstruction_stage_disagreement_train_only import (  # noqa: E402
    sha256_file,
    sha256_rows,
)
from scripts.infer.patch_cleanup_erasemap import (  # noqa: E402
    build_model,
    resolve_device,
)

PLAN_PATH = Path("docs/second-stage-alpha-support-prerequisite-v1.json")
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
OUTPUT_ROOT = Path("outputs/second-stage-alpha-support-materialization-20260812")
RAW_ALPHA_KEY = "raw_alpha"
FORBIDDEN_SAMPLE_COMPONENTS = {
    "label",
    "labels",
    "target",
    "targets",
    "all_labels",
}


class MaterializationError(RuntimeError):
    pass


def _expected_erasemap_state_dict() -> dict[str, torch.Tensor]:
    return build_model("erasemap").state_dict()


def _validate_legacy_erasemap_state_dict(model_state: dict[str, Any]) -> None:
    expected = _expected_erasemap_state_dict()
    if set(model_state) != set(expected):
        raise MaterializationError(
            "legacy checkpoint state dict does not match erasemap keys"
        )
    for key, expected_value in expected.items():
        value = model_state[key]
        if not isinstance(value, torch.Tensor):
            raise MaterializationError(f"legacy checkpoint tensor missing: {key}")
        if tuple(value.shape) != tuple(expected_value.shape):
            raise MaterializationError(
                f"legacy checkpoint tensor shape changed: {key}"
            )


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
    if statuses.get("materially_new_support_successor_preregistration_v2") != "passed":
        raise MaterializationError("second-stage alpha preregistration is not passed")
    if statuses.get("second_stage_alpha_support_train_only_diagnostic") != "pending":
        raise MaterializationError("second-stage alpha diagnostic is not pending")
    if statuses.get("source_output_support_train_only_diagnostic") != "passed":
        raise MaterializationError("source-output KILL is not recorded")


def validate_plan(plan: dict[str, Any]) -> None:
    exact = {
        "schema_version": 1,
        "iteration_id": "monotonic-residual-erase-support",
        "state": "preregistered_pending_second_stage_alpha_materialization",
        "next_boundary_on_pass": "second_stage_alpha_data_and_training_preflight_only",
    }
    for key, expected in exact.items():
        if plan.get(key) != expected:
            raise MaterializationError(f"plan field changed: {key}")
    if plan.get("representation") != {
        "channels": [
            "second_stage_r",
            "second_stage_g",
            "second_stage_b",
            "raw_second_stage_alpha",
        ],
        "feature_count": 4,
        "no_clean_candidate_or_internal_feature": True,
        "no_masks_page_scalars_stages_source_or_primary_rgb": True,
        "no_threshold_neighborhood_or_component_transform": True,
        "single_causal_change": (
            "add_frozen_second_stage_prethreshold_raw_alpha_to_output_rgb_support_evidence"
        ),
    }:
        raise MaterializationError("registered alpha representation changed")
    spec = plan.get("second_stage_alpha_materialization", {})
    required = {
        "alpha_threshold": None,
        "batch_size": 32,
        "device": "auto",
        "dtype": "float32",
        "encoding": "one_compressed_npz_per_page_with_exact_raw_alpha_key",
        "input": "frozen_train275_primary_predictions",
        "output_root": str(OUTPUT_ROOT),
        "overlap_fusion": "arithmetic_mean_of_raw_patch_alpha_before_any_threshold",
        "provenance": "current_second_stage_erasemap_alpha_head_sigmoid",
        "stride": 160,
        "target_access": False,
        "tile_size": 160,
    }
    for key, expected in required.items():
        if spec.get(key) != expected:
            raise MaterializationError(f"materialization field changed: {key}")
    auth = plan.get("authorization", {})
    forbidden = {
        "candidate_checkpoint",
        "candidate_inference",
        "development_gate",
        "holdout40",
        "inner_val15",
        "model_training",
        "promotion",
        "reserved_blind",
        "scut115",
        "visual_review",
    }
    if any(auth.get(name) is not False for name in forbidden):
        raise MaterializationError("a candidate or quality surface was opened")
    if (
        auth.get("second_stage_alpha_materialization") is not True
        or auth.get("second_stage_alpha_materialization_reads_targets") is not False
        or auth.get("separability_diagnostic_target_decode_roles") != ["train"]
    ):
        raise MaterializationError("second-stage alpha authorization changed")
    if plan.get("planned_implementation") != {
        "audit_output": "outputs/second-stage-alpha-support-prerequisite-20260812/audit.json",
        "audit_script": "scripts/analysis/audit_second_stage_alpha_support.py",
        "materialization_output": str(OUTPUT_ROOT),
        "materializer": "scripts/analysis/materialize_second_stage_alpha_train_only.py",
        "test": "tests/test_second_stage_alpha_support_prerequisite.py",
    }:
        raise MaterializationError("planned implementation changed")


def read_manifest(repo_root: Path, manifest_path: Path) -> list[str]:
    rows = [
        row.strip()
        for row in manifest_path.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    file_names = [Path(row).name for row in rows]
    if len(file_names) != 275 or len(file_names) != len(set(file_names)):
        raise MaterializationError("train manifest count or uniqueness changed")
    for row in rows:
        source = repo_path(repo_root, row)
        lowered = {part.lower() for part in source.parts}
        if lowered & FORBIDDEN_SAMPLE_COMPONENTS:
            raise MaterializationError(f"manifest row appears to target labels: {row}")
    return rows


def load_registered_erasemap(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise MaterializationError("checkpoint payload must be a dict")
    args = state.get("args")
    if not isinstance(args, dict):
        raise MaterializationError("checkpoint args metadata is missing")
    model_state = state.get("model")
    if not isinstance(model_state, dict):
        raise MaterializationError("checkpoint model state is missing")
    model_type = args.get("model_type")
    if model_type is not None and model_type != "erasemap":
        raise MaterializationError("registered checkpoint metadata is not erasemap")
    if model_type is None:
        _validate_legacy_erasemap_state_dict(model_state)
    residual_delta_scale = float(args.get("residual_delta_scale", 0.25))
    residual_delta_bound = float(args.get("residual_delta_bound", 0.08))
    model = build_model(
        "erasemap",
        residual_delta_scale=residual_delta_scale,
        residual_delta_bound=residual_delta_bound,
    ).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model


def validate_inputs(
    repo_root: Path, plan: dict[str, Any], ledger: dict[str, Any]
) -> tuple[Path, Path, Path, Path, Path, list[str], Path]:
    validate_plan(plan)
    validate_authority(ledger)
    evidence = plan["evidence"]
    config = validate_file_artifact(
        repo_root, evidence["current_primary_config"], "primary config"
    )
    checkpoint = validate_file_artifact(
        repo_root, evidence["current_second_stage_checkpoint"], "second-stage checkpoint"
    )
    inference_source = validate_file_artifact(
        repo_root,
        evidence["second_stage_inference_source"],
        "second-stage inference source",
    )
    role_plan = validate_file_artifact(repo_root, evidence["role_plan"], "role plan")
    role_wrapper = read_json(role_plan)
    base_role_artifact = role_wrapper.get("evidence", {}).get("base_role_contract")
    if not isinstance(base_role_artifact, dict):
        raise MaterializationError("role plan lacks base role contract")
    base_role_path = validate_file_artifact(
        repo_root, base_role_artifact, "base role contract"
    )
    manifest_path = validate_file_artifact(
        repo_root, plan["data"]["manifest"], "train manifest"
    )
    manifest_rows = read_manifest(repo_root, manifest_path)
    file_names = [Path(row).name for row in manifest_rows]
    if sorted(file_names) != effective_train_filenames(repo_root, base_role_path):
        raise MaterializationError("train manifest no longer matches frozen roles")
    prediction_dir, _summary = validate_prediction_set(
        repo_root,
        evidence["primary_prediction_set"],
        file_names,
        "primary prediction set",
    )
    output_root = repo_path(
        repo_root, plan["second_stage_alpha_materialization"]["output_root"]
    )
    if output_root.exists():
        raise MaterializationError("second-stage alpha output directory already exists")
    primary_checkpoint = validate_file_artifact(
        repo_root, evidence["current_primary_checkpoint"], "primary checkpoint"
    )
    return (
        config,
        primary_checkpoint,
        checkpoint,
        inference_source,
        manifest_path,
        manifest_rows,
        prediction_dir,
    )


@torch.no_grad()
def infer_raw_alpha_full_page(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    device: torch.device,
    *,
    tile_size: int,
    stride: int,
    batch_size: int,
) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise MaterializationError("expected BGR prediction image")
    if tile_size <= 0 or stride <= 0 or batch_size <= 0:
        raise MaterializationError("tile_size, stride, and batch_size must be positive")
    height, width = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    accum = np.zeros((height, width), dtype=np.float64)
    weight = np.zeros((height, width), dtype=np.float64)

    ys = list(range(0, max(1, height - tile_size + 1), stride))
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    if not ys or ys[-1] != max(0, height - tile_size):
        ys.append(max(0, height - tile_size))
    if not xs or xs[-1] != max(0, width - tile_size):
        xs.append(max(0, width - tile_size))

    pending_tiles: list[np.ndarray] = []
    pending_meta: list[tuple[int, int, int, int]] = []

    def flush() -> None:
        if not pending_tiles:
            return
        batch = np.stack(pending_tiles, axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        _pred, alpha, _clean = model(tensor)
        alpha_np = alpha.squeeze(1).detach().cpu().numpy().astype(np.float32, copy=False)
        for batch_index, (y0, x0, tile_h, tile_w) in enumerate(pending_meta):
            accum[y0 : y0 + tile_h, x0 : x0 + tile_w] += alpha_np[
                batch_index, :tile_h, :tile_w
            ]
            weight[y0 : y0 + tile_h, x0 : x0 + tile_w] += 1.0
        pending_tiles.clear()
        pending_meta.clear()

    for y0 in ys:
        for x0 in xs:
            tile = rgb[y0 : y0 + tile_size, x0 : x0 + tile_size]
            tile_h, tile_w = tile.shape[:2]
            if tile_h != tile_size or tile_w != tile_size:
                canvas = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
                canvas[:tile_h, :tile_w] = tile
                tile = canvas
            pending_tiles.append(tile)
            pending_meta.append((y0, x0, tile_h, tile_w))
            if len(pending_tiles) >= batch_size:
                flush()
    flush()
    if (weight <= 0.0).any():
        raise MaterializationError("full-page alpha patch schedule left uncovered pixels")
    raw_alpha = (accum / weight).astype(np.float32)
    if raw_alpha.shape != (height, width) or not np.isfinite(raw_alpha).all():
        raise MaterializationError("invalid fused raw alpha map")
    if float(raw_alpha.min()) < 0.0 or float(raw_alpha.max()) > 1.0:
        raise MaterializationError("raw alpha escaped unit interval")
    return raw_alpha


def write_manifest(
    output_root: Path,
    *,
    repo_root: Path,
    plan_path: Path,
    config: Path,
    primary_checkpoint: Path,
    checkpoint: Path,
    inference_source: Path,
    source_manifest: Path,
    rows: list[dict[str, Any]],
) -> Path:
    path = output_root / "manifest.json"
    payload = {
        "schema_version": 1,
        "terminal": "PASS",
        "provenance": "current_second_stage_erasemap_alpha_head_sigmoid",
        "target_access": False,
        "train_count": len(rows),
        "tile_size": 160,
        "stride": 160,
        "batch_size": 32,
        "dtype": "float32",
        "encoding": "one_compressed_npz_per_page_with_exact_raw_alpha_key",
        "raw_alpha_key": RAW_ALPHA_KEY,
        "overlap_fusion": "arithmetic_mean_of_raw_patch_alpha_before_any_threshold",
        "output_root": str(output_root.relative_to(repo_root)),
        "pages_directory": "pages",
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
            "path": str(primary_checkpoint.relative_to(repo_root)),
            "sha256": sha256_file(primary_checkpoint),
        },
        "second_stage_checkpoint": {
            "path": str(checkpoint.relative_to(repo_root)),
            "sha256": sha256_file(checkpoint),
        },
        "second_stage_inference_source": {
            "path": str(inference_source.relative_to(repo_root)),
            "sha256": sha256_file(inference_source),
        },
        "pages": rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def materialize(
    *, repo_root: Path, plan_path: Path = PLAN_PATH, ledger_path: Path = LEDGER_PATH
) -> dict[str, Any]:
    plan_file = repo_path(repo_root, str(plan_path))
    ledger_file = repo_path(repo_root, str(ledger_path))
    plan = read_json(plan_file)
    ledger = read_json(ledger_file)
    (
        config,
        primary_checkpoint,
        checkpoint,
        inference_source,
        manifest_path,
        manifest_rows,
        prediction_dir,
    ) = validate_inputs(repo_root, plan, ledger)
    spec = plan["second_stage_alpha_materialization"]
    output_root = repo_path(repo_root, spec["output_root"])
    page_dir = output_root / "pages"
    page_dir.mkdir(parents=True)
    device = resolve_device(str(spec["device"]))
    model = load_registered_erasemap(checkpoint, device)

    rows: list[dict[str, Any]] = []
    for index, manifest_row in enumerate(manifest_rows, start=1):
        file_name = Path(manifest_row).name
        prediction_path = prediction_dir / f"{Path(file_name).stem}.png"
        image_bgr = cv2.imread(str(prediction_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise MaterializationError(f"prediction decode failed: {prediction_path}")
        raw_alpha = infer_raw_alpha_full_page(
            model,
            image_bgr,
            device,
            tile_size=int(spec["tile_size"]),
            stride=int(spec["stride"]),
            batch_size=int(spec["batch_size"]),
        )
        npz_path = page_dir / f"{Path(file_name).stem}.npz"
        np.savez_compressed(npz_path, **{RAW_ALPHA_KEY: raw_alpha})
        rows.append(
            {
                "file": file_name,
                "height": int(raw_alpha.shape[0]),
                "width": int(raw_alpha.shape[1]),
                "source_prediction_sha256": sha256_file(prediction_path),
                "npz_sha256": sha256_file(npz_path),
                "raw_alpha_min": float(raw_alpha.min()),
                "raw_alpha_max": float(raw_alpha.max()),
                "raw_alpha_mean": float(raw_alpha.mean()),
            }
        )
        if index % 25 == 0 or index == len(manifest_rows):
            print(f"materialized train-only raw alpha {index}/{len(manifest_rows)}", flush=True)

    manifest = write_manifest(
        output_root,
        repo_root=repo_root,
        plan_path=plan_file,
        config=config,
        primary_checkpoint=primary_checkpoint,
        checkpoint=checkpoint,
        inference_source=inference_source,
        source_manifest=manifest_path,
        rows=rows,
    )
    return {
        "terminal": "PASS",
        "output_root": str(output_root),
        "manifest": str(manifest),
        "train_count": len(rows),
        "content_sha256": sha256_rows([f"{row['file']} {row['npz_sha256']}" for row in rows]),
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
            repo_root=args.repo_root, plan_path=args.plan, ledger_path=args.ledger
        )
    except (MaterializationError, OSError, ValueError) as error:
        print(f"terminal=PREREQUISITE_NEEDED reason={error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
