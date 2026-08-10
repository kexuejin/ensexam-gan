#!/usr/bin/env python3
"""Build a train-only, direction-aware patch index for sign-separated repair."""

from __future__ import annotations

import argparse
import csv
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

from scripts.analysis.validate_sign_separated_data_roles import (  # noqa: E402
    ROLE_PLAN_PATH,
    derive_effective_roles,
    read_json,
    validate_plan_header,
    validate_reserved_blind,
    validate_role_sources,
    validate_zero_overlap,
)
from scripts.experimental.build_explicit_mask_patch_index import (  # noqa: E402
    dataset_ticks,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_rows(rows: list[str]) -> str:
    payload = "\n".join(rows) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def training_filename(identity: str) -> str:
    parts = identity.split("/", 2)
    if len(parts) != 3 or parts[1] != "train":
        raise ValueError(f"invalid train identity: {identity}")
    domain, _split, name = parts
    if domain not in {"scut", "hw5k"} or Path(name).name != name:
        raise ValueError(f"invalid train identity: {identity}")
    return f"{domain}_{name}"


def effective_train_filenames(
    repo_root: Path,
    role_plan_path: Path,
) -> list[str]:
    plan = read_json(role_plan_path)
    validate_plan_header(plan)
    roles = plan.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("role plan roles are missing")
    validate_reserved_blind(roles)
    raw, _sources = validate_role_sources(repo_root, roles)
    effective, _summary = derive_effective_roles(roles, raw)
    validate_zero_overlap(effective)
    filenames = sorted(training_filename(value) for value in effective["train"])
    if len(filenames) != len(set(filenames)):
        raise ValueError("effective train filenames are not unique")
    return filenames


def find_prediction(directory: Path, file_name: str) -> Path:
    stem = Path(file_name).stem
    for name in (f"{stem}.png", f"{stem}.clean.png", file_name):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing pipeline prediction for {file_name}")


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def direction_metrics(delta: np.ndarray, margin: float) -> dict[str, float]:
    bright = delta > margin
    dark = delta < -margin
    bright_ratio = float(bright.mean())
    dark_ratio = float(dark.mean())
    bright_mean = float(delta[bright].mean()) if bool(bright.any()) else 0.0
    dark_mean = float((-delta[dark]).mean()) if bool(dark.any()) else 0.0
    return {
        "brighten_ratio": bright_ratio,
        "darken_ratio": dark_ratio,
        "brighten_mean_delta": bright_mean,
        "darken_mean_delta": dark_mean,
        "brighten_score": bright_ratio * bright_mean,
        "darken_score": dark_ratio * dark_mean,
    }


def build_candidates(
    *,
    filenames: list[str],
    label_dir: Path,
    input_dir: Path,
    tile_size: int,
    overlap: int,
    direction_margin: float,
    min_support_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    candidates: list[dict[str, Any]] = []
    input_hashes: list[str] = []
    label_hashes: list[str] = []
    for file_name in filenames:
        input_path = find_prediction(input_dir, file_name)
        label_path = label_dir / file_name
        if not label_path.is_file():
            raise FileNotFoundError(f"missing train label: {label_path}")
        inp = read_bgr(input_path)
        target = read_bgr(label_path)
        if inp.shape != target.shape:
            raise ValueError(
                f"input/target shape mismatch for {file_name}: "
                f"{inp.shape} != {target.shape}"
            )
        input_hashes.append(f"{file_name} {sha256_file(input_path)}")
        label_hashes.append(f"{file_name} {sha256_file(label_path)}")
        delta = target.astype(np.float32).mean(axis=2) - inp.astype(
            np.float32
        ).mean(axis=2)
        height, width = delta.shape
        for y1 in dataset_ticks(height, tile_size, overlap):
            for x1 in dataset_ticks(width, tile_size, overlap):
                y2 = min(y1 + tile_size, height)
                x2 = min(x1 + tile_size, width)
                metrics = direction_metrics(
                    delta[y1:y2, x1:x2], direction_margin
                )
                if max(
                    metrics["brighten_ratio"], metrics["darken_ratio"]
                ) < min_support_ratio:
                    continue
                candidates.append(
                    {
                        "file": file_name,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        **metrics,
                    }
                )
    return candidates, {
        "input_content_sha256": sha256_rows(sorted(input_hashes)),
        "label_content_sha256": sha256_rows(sorted(label_hashes)),
    }


def select_direction_balanced(
    candidates: list[dict[str, Any]],
    top_k_per_direction: int,
) -> list[dict[str, Any]]:
    if top_k_per_direction <= 0:
        raise ValueError("top_k_per_direction must be positive")

    def ranked(direction: str) -> list[dict[str, Any]]:
        score_key = f"{direction}_score"
        ratio_key = f"{direction}_ratio"
        eligible = [row for row in candidates if float(row[ratio_key]) > 0.0]
        return sorted(
            eligible,
            key=lambda row: (
                -float(row[score_key]),
                str(row["file"]),
                int(row["y1"]),
                int(row["x1"]),
            ),
        )[:top_k_per_direction]

    selected_by_key: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
    for direction in ("brighten", "darken"):
        selected = ranked(direction)
        if not selected:
            raise RuntimeError(f"no {direction} training support found")
        for row in selected:
            key = (
                str(row["file"]),
                int(row["x1"]),
                int(row["y1"]),
                int(row["x2"]),
                int(row["y2"]),
            )
            stored = selected_by_key.setdefault(key, dict(row, selected_for=[]))
            stored["selected_for"].append(direction)

    rows = []
    for row in selected_by_key.values():
        row["selected_for"] = "+".join(row["selected_for"])
        row["rank_score"] = max(
            float(row["brighten_score"]), float(row["darken_score"])
        )
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            -float(row["rank_score"]),
            str(row["file"]),
            int(row["y1"]),
            int(row["x1"]),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--role-plan", type=Path, default=ROLE_PLAN_PATH)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--direction-margin", type=float, default=2.0)
    parser.add_argument("--min-support-ratio", type=float, default=0.001)
    parser.add_argument("--top-k-per-direction", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    role_plan = args.role_plan
    if not role_plan.is_absolute():
        role_plan = repo_root / role_plan
    data_root = args.data_root
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    input_dir = args.input_dir
    if not input_dir.is_absolute():
        input_dir = repo_root / input_dir
    output_csv = args.output_csv
    if not output_csv.is_absolute():
        output_csv = repo_root / output_csv
    output_json = args.output_json
    if not output_json.is_absolute():
        output_json = repo_root / output_json
    if output_csv.exists() or output_json.exists():
        raise FileExistsError("registered patch-index outputs must be absent")
    if args.split != "train":
        raise ValueError("sign-separated patches must use the train split")

    filenames = effective_train_filenames(repo_root, role_plan)
    label_dir = data_root / args.split / "all_labels"
    candidates, content_hashes = build_candidates(
        filenames=filenames,
        label_dir=label_dir,
        input_dir=input_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        direction_margin=args.direction_margin,
        min_support_ratio=args.min_support_ratio,
    )
    rows = select_direction_balanced(candidates, args.top_k_per_direction)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank_score",
        "selected_for",
        "file",
        "x1",
        "y1",
        "x2",
        "y2",
        "brighten_ratio",
        "darken_ratio",
        "brighten_mean_delta",
        "darken_mean_delta",
        "brighten_score",
        "darken_score",
    ]
    with output_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    selected_counts = {
        direction: sum(direction in str(row["selected_for"]) for row in rows)
        for direction in ("brighten", "darken")
    }
    summary = {
        "status": "pass",
        "terminal": "PASS",
        "train_role_count": len(filenames),
        "train_role_sha256": sha256_rows(filenames),
        "candidate_count": len(candidates),
        "patch_count": len(rows),
        "page_count": len({str(row["file"]) for row in rows}),
        "selected_counts": selected_counts,
        "patch_index": str(output_csv),
        "patch_index_sha256": sha256_file(output_csv),
        **content_hashes,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
