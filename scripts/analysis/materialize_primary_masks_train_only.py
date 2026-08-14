#!/usr/bin/env python3
"""Materialize frozen primary mb/ms maps for the train-only mask prerequisite."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer.run_primary_full_page import (  # noqa: E402
    load_generator,
    pick_device,
)
from utils.page_inference import infer_full_page  # noqa: E402


PLAN_PATH = Path("docs/spatial-primary-mask-support-prerequisite-v1.json")
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
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
    return hashlib.sha256(("\n".join(sorted(rows)) + "\n").encode()).hexdigest()


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
    if statuses.get("spatial_primary_mask_support_preregistration") != "passed":
        raise MaterializationError("mask preregistration is not passed")
    if statuses.get("spatial_primary_mask_support_train_only_diagnostic") != "pending":
        raise MaterializationError("mask diagnostic is not pending")


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


def validate_plan_and_get_inputs(
    repo_root: Path, plan: dict[str, Any], ledger: dict[str, Any]
) -> tuple[Path, Path, Path, list[tuple[str, Path]]]:
    validate_authority(ledger)
    if (
        plan.get("state") != "preregistered_pending_mask_materialization"
        or plan.get("iteration_id") != "monotonic-residual-erase-support"
        or plan.get("representation", {}).get("channels")
        != ["mb", "ms", "mb_minus_ms", "mb_times_ms"]
    ):
        raise MaterializationError("mask plan representation or state changed")
    if plan.get("authorization", {}).get("mask_materialization_reads_targets"):
        raise MaterializationError("mask materializer target access was enabled")
    evidence = plan.get("evidence", {})
    checkpoint = validate_artifact(
        repo_root, evidence["current_primary_checkpoint"], "primary checkpoint"
    )
    config = validate_artifact(
        repo_root, evidence["current_primary_config"], "primary config"
    )
    role_plan = validate_artifact(repo_root, evidence["role_plan"], "role plan")
    role_wrapper = read_json(role_plan)
    base_role = role_wrapper.get("evidence", {}).get("base_role_contract")
    if not isinstance(base_role, dict):
        raise MaterializationError("role plan lacks base role contract")
    validate_artifact(repo_root, base_role, "base role contract")
    manifest_spec = plan["data"]["manifest"]
    manifest = validate_artifact(repo_root, manifest_spec, "train manifest")
    rows = read_manifest(repo_root, manifest)
    if len(rows) != plan["data"]["effective_train_count"]:
        raise MaterializationError("train manifest count changed")
    output_root = repo_path(
        repo_root, plan["mask_materialization"]["output_root"]
    )
    if output_root.exists():
        raise MaterializationError("mask output directory already exists")
    return config, checkpoint, manifest, rows


def write_manifest(
    output_root: Path,
    *,
    plan: dict[str, Any],
    config: Path,
    checkpoint: Path,
    source_manifest: Path,
    rows: list[dict[str, Any]],
) -> Path:
    payload = {
        "schema_version": 1,
        "terminal": "PASS",
        "provenance": "utils.page_inference.infer_full_page",
        "target_access": False,
        "train_count": len(rows),
        "source_manifest": {
            "path": str(source_manifest.relative_to(output_root.parents[1])),
            "sha256": sha256_file(source_manifest),
        },
        "primary_config": {
            "path": str(config.relative_to(output_root.parents[1])),
            "sha256": sha256_file(config),
        },
        "primary_checkpoint": {
            "path": str(checkpoint.relative_to(output_root.parents[1])),
            "sha256": sha256_file(checkpoint),
        },
        "page_overlap": plan["mask_materialization"]["page_overlap"],
        "batch_size": plan["mask_materialization"]["batch_size"],
        "maps": ["mb", "ms"],
        "pages": rows,
    }
    path = output_root / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def materialize(
    *, repo_root: Path, plan_path: Path = PLAN_PATH, ledger_path: Path = LEDGER_PATH
) -> dict[str, Any]:
    plan_file = repo_path(repo_root, str(plan_path))
    ledger_file = repo_path(repo_root, str(ledger_path))
    plan = read_json(plan_file)
    ledger = read_json(ledger_file)
    config, checkpoint, source_manifest, sources = validate_plan_and_get_inputs(
        repo_root, plan, ledger
    )
    output_root = repo_path(
        repo_root, plan["mask_materialization"]["output_root"]
    )
    mb_dir = output_root / "mb"
    ms_dir = output_root / "ms"
    mb_dir.mkdir(parents=True)
    ms_dir.mkdir()
    device = pick_device(plan["mask_materialization"]["device"])
    generator = load_generator(str(config), str(checkpoint), device)
    rows: list[dict[str, Any]] = []
    for index, (file_name, source) in enumerate(sources, start=1):
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            raise MaterializationError(f"source decode failed: {source}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        outputs = infer_full_page(
            generator,
            rgb,
            device,
            overlap=int(plan["mask_materialization"]["page_overlap"]),
            batch_size=int(plan["mask_materialization"]["batch_size"]),
        )
        height, width = rgb.shape[:2]
        page = {"file": file_name, "source_sha256": sha256_file(source)}
        for name, directory in (("mb", mb_dir), ("ms", ms_dir)):
            mask = outputs.get(name)
            if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8:
                raise MaterializationError(f"invalid {name} map for {file_name}")
            if mask.shape != (height, width) or not np.isfinite(mask).all():
                raise MaterializationError(f"invalid {name} shape for {file_name}")
            path = directory / f"{Path(file_name).stem}.png"
            if not cv2.imwrite(str(path), mask):
                raise MaterializationError(f"failed to write {name} map")
            page[f"{name}_sha256"] = sha256_file(path)
        page["height"] = height
        page["width"] = width
        rows.append(page)
        if index % 25 == 0 or index == len(sources):
            print(f"materialized train-only masks {index}/{len(sources)}", flush=True)
    manifest = write_manifest(
        output_root,
        plan=plan,
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
        "mb_content_sha256": sha256_rows(
            [f"{row['file']} {row['mb_sha256']}" for row in rows]
        ),
        "ms_content_sha256": sha256_rows(
            [f"{row['file']} {row['ms_sha256']}" for row in rows]
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
        print(f"terminal=PREREQUISITE_NEEDED reason={error}")
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
