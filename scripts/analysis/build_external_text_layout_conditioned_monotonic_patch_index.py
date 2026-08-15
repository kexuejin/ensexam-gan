#!/usr/bin/env python3
"""Build the registered text-layout conditioned monotonic patch index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.build_monotonic_residual_erase_patch_index import (  # noqa: E402
    BASE_ROLE_PLAN_PATH,
    read_bgr,
    select_top_brighten,
    support_metrics,
)
from scripts.analysis.build_sign_separated_residual_patch_index import (  # noqa: E402
    effective_train_filenames,
    find_prediction,
    sha256_rows,
)
from scripts.experimental.build_explicit_mask_patch_index import (  # noqa: E402
    dataset_ticks,
)
from scripts.train.train_external_text_layout_conditioned_monotonic import (  # noqa: E402
    CONDITIONED_INPUT_CHANNELS,
    find_layout_npz,
    load_layout_grids,
)


SELECTION = "top_target_lighter_support_with_frozen_external_text_layout"
CHANNEL_ORDER = [
    "second_stage_r",
    "second_stage_g",
    "second_stage_b",
    "external_text_occupancy",
    "external_text_confidence",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def layout_patch_metrics(
    occupancy: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, float]:
    if occupancy.shape != confidence.shape:
        raise ValueError("layout patch shape mismatch")
    occupied = occupancy > 0
    return {
        "text_occupancy_ratio": float(occupied.mean()),
        "text_confidence_mean": float(confidence.mean()),
        "text_confidence_occupied_mean": (
            float(confidence[occupied].mean()) if bool(occupied.any()) else 0.0
        ),
    }


def build_conditioned_candidates(
    *,
    filenames: list[str],
    label_dir: Path,
    input_dir: Path,
    layout_dir: Path,
    tile_size: int,
    overlap: int,
    luminance_margin_gray: float,
    min_positive_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    candidates: list[dict[str, Any]] = []
    input_hashes: list[str] = []
    label_hashes: list[str] = []
    layout_hashes: list[str] = []
    for file_name in filenames:
        input_path = find_prediction(input_dir, file_name)
        label_path = label_dir / file_name
        layout_path = find_layout_npz(layout_dir, file_name)
        if not label_path.is_file():
            raise FileNotFoundError(f"missing train label: {label_path}")
        inp = read_bgr(input_path)
        target = read_bgr(label_path)
        if inp.shape != target.shape:
            raise ValueError(
                f"input/target shape mismatch for {file_name}: "
                f"{inp.shape} != {target.shape}"
            )
        height, width = inp.shape[:2]
        occupancy, confidence = load_layout_grids(
            layout_path,
            expected_shape=(height, width),
        )
        input_hashes.append(f"{file_name} {sha256_file(input_path)}")
        label_hashes.append(f"{file_name} {sha256_file(label_path)}")
        layout_hashes.append(f"{file_name} {sha256_file(layout_path)}")
        delta = target.astype(np.float32).mean(axis=2) - inp.astype(
            np.float32
        ).mean(axis=2)
        for y1 in dataset_ticks(height, tile_size, overlap):
            for x1 in dataset_ticks(width, tile_size, overlap):
                y2 = min(y1 + tile_size, height)
                x2 = min(x1 + tile_size, width)
                metrics = support_metrics(
                    delta[y1:y2, x1:x2],
                    luminance_margin_gray,
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
                        **layout_patch_metrics(
                            occupancy[y1:y2, x1:x2],
                            confidence[y1:y2, x1:x2],
                        ),
                    }
                )
    return candidates, {
        "input_content_sha256": sha256_rows(sorted(input_hashes)),
        "label_content_sha256": sha256_rows(sorted(label_hashes)),
        "layout_content_sha256": sha256_rows(sorted(layout_hashes)),
    }


def summarize_rows(
    *,
    filenames: list[str],
    candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    output_csv: Path,
    content_hashes: dict[str, str],
) -> dict[str, Any]:
    positive_ratios = [float(row["edit_positive_ratio"]) for row in rows]
    preserve_ratios = [float(row["preserve_negative_ratio"]) for row in rows]
    occupancy_ratios = [float(row["text_occupancy_ratio"]) for row in rows]
    confidence_means = [float(row["text_confidence_mean"]) for row in rows]
    return {
        "candidate_count": len(candidates),
        "channel_order": CHANNEL_ORDER,
        "input_channels": CONDITIONED_INPUT_CHANNELS,
        "page_count": len({str(row["file"]) for row in rows}),
        "patch_count": len(rows),
        "patch_index": str(output_csv),
        "patch_index_sha256": sha256_file(output_csv),
        "positive_ratio_max": max(positive_ratios),
        "positive_ratio_min": min(positive_ratios),
        "preserve_ratio_min": min(preserve_ratios),
        "selection": SELECTION,
        "status": "pass",
        "target_access": "train_labels_only_for_patch_support",
        "terminal": "PASS",
        "text_confidence_mean_avg": float(np.mean(confidence_means)),
        "text_occupancy_ratio_avg": float(np.mean(occupancy_ratios)),
        "train_role_count": len(filenames),
        "train_role_sha256": sha256_rows(filenames),
        **content_hashes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--role-plan", type=Path, default=BASE_ROLE_PLAN_PATH)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--layout-dir", type=Path, required=True)
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
    layout_dir = resolve(args.layout_dir)
    output_csv = resolve(args.output_csv)
    output_json = resolve(args.output_json)
    if output_csv.exists() or output_json.exists():
        raise FileExistsError("registered conditioned patch-index outputs must be absent")

    filenames = effective_train_filenames(repo_root, role_plan)
    candidates, content_hashes = build_conditioned_candidates(
        filenames=filenames,
        label_dir=data_root / args.split / "all_labels",
        input_dir=input_dir,
        layout_dir=layout_dir,
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
        "text_occupancy_ratio",
        "text_confidence_mean",
        "text_confidence_occupied_mean",
    ]
    with output_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_rows(
        filenames=filenames,
        candidates=candidates,
        rows=rows,
        output_csv=output_csv,
        content_hashes=content_hashes,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
