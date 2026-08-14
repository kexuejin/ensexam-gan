#!/usr/bin/env python3
"""Build the registered train-only brighten-support patch index."""

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

from scripts.analysis.build_sign_separated_residual_patch_index import (  # noqa: E402
    effective_train_filenames,
    find_prediction,
    sha256_rows,
)
from scripts.experimental.build_explicit_mask_patch_index import (  # noqa: E402
    dataset_ticks,
)


BASE_ROLE_PLAN_PATH = Path("docs/sign-separated-residual-data-roles.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def support_metrics(delta: np.ndarray, margin: float) -> dict[str, float]:
    positive = delta > margin
    positive_ratio = float(positive.mean())
    positive_mean = float(delta[positive].mean()) if bool(positive.any()) else 0.0
    return {
        "edit_positive_ratio": positive_ratio,
        "edit_positive_mean_delta": positive_mean,
        "edit_positive_score": positive_ratio * positive_mean,
        "preserve_negative_ratio": 1.0 - positive_ratio,
    }


def build_candidates(
    *,
    filenames: list[str],
    label_dir: Path,
    input_dir: Path,
    tile_size: int,
    overlap: int,
    luminance_margin_gray: float,
    min_positive_ratio: float,
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
                metrics = support_metrics(
                    delta[y1:y2, x1:x2], luminance_margin_gray
                )
                if metrics["edit_positive_ratio"] < min_positive_ratio:
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


def select_top_brighten(
    candidates: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    selected = sorted(
        candidates,
        key=lambda row: (
            -float(row["edit_positive_score"]),
            str(row["file"]),
            int(row["y1"]),
            int(row["x1"]),
        ),
    )[:top_k]
    if not selected:
        raise RuntimeError("no target-lighter training support found")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--role-plan", type=Path, default=BASE_ROLE_PLAN_PATH)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--luminance-margin-gray", type=float, default=2.0)
    parser.add_argument("--min-positive-ratio", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=256)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else repo_root / path

    role_plan = resolve(args.role_plan)
    data_root = resolve(args.data_root)
    input_dir = resolve(args.input_dir)
    output_csv = resolve(args.output_csv)
    output_json = resolve(args.output_json)
    if output_csv.exists() or output_json.exists():
        raise FileExistsError("registered patch-index outputs must be absent")

    filenames = effective_train_filenames(repo_root, role_plan)
    candidates, content_hashes = build_candidates(
        filenames=filenames,
        label_dir=data_root / args.split / "all_labels",
        input_dir=input_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        luminance_margin_gray=args.luminance_margin_gray,
        min_positive_ratio=args.min_positive_ratio,
    )
    rows = select_top_brighten(candidates, args.top_k)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "edit_positive_score",
        "file",
        "x1",
        "y1",
        "x2",
        "y2",
        "edit_positive_ratio",
        "edit_positive_mean_delta",
        "preserve_negative_ratio",
    ]
    with output_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "pass",
        "terminal": "PASS",
        "selection": "top_target_lighter_support_only",
        "train_role_count": len(filenames),
        "train_role_sha256": sha256_rows(filenames),
        "candidate_count": len(candidates),
        "patch_count": len(rows),
        "page_count": len({str(row["file"]) for row in rows}),
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
