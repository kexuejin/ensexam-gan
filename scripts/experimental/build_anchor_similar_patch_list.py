#!/usr/bin/env python3
"""Rank train patches by similarity to an anchor patch.

The exact129 one-step probe shows that patch identity can preserve or destroy
strict-gate behavior. This helper ranks train-split patches by local statistics
similarity to a known anchor patch, rather than by handwriting density alone.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_config  # noqa: E402
from data.dataset import EnsExamRealDataset  # noqa: E402
from data.mask_utils import generate_mask_from_pair, generate_mb_from_boxes  # noqa: E402
from train import load_optional_file_list  # noqa: E402


FEATURES = [
    "stroke_ratio",
    "printed_near_ratio",
    "bg_risk_ratio",
    "mean_diff",
    "p95_diff",
    "diff_mb_ratio",
    "box_mb_ratio",
    "box_overlap_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--anchor-patch-csv", required=True)
    parser.add_argument("--train-file-list", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--train-pages", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--min-stroke-ratio", type=float, default=0.0)
    parser.add_argument("--exclude-anchor", action="store_true")
    parser.add_argument("--stroke-weight", type=float, default=1.0)
    parser.add_argument("--printed-near-weight", type=float, default=2.0)
    parser.add_argument("--bg-risk-weight", type=float, default=2.0)
    parser.add_argument("--mean-diff-weight", type=float, default=1.0)
    parser.add_argument("--p95-diff-weight", type=float, default=1.0)
    parser.add_argument("--diff-mb-ratio-weight", type=float, default=1.0)
    parser.add_argument("--box-mb-ratio-weight", type=float, default=4.0)
    parser.add_argument("--box-overlap-ratio-weight", type=float, default=2.0)
    return parser.parse_args()


def crop_pair(
    info: dict[str, object],
    img_size: int,
    image_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    img_path = str(info["img_path"])
    gt_path = str(info["gt_path"])
    if img_path not in image_cache:
        image_cache[img_path] = cv2.imread(img_path)[:, :, ::-1]
    if gt_path not in image_cache:
        image_cache[gt_path] = cv2.imread(gt_path)[:, :, ::-1]
    iin = image_cache[img_path]
    igt = image_cache[gt_path]
    y1, y2 = int(info["y1"]), int(info["y2"])
    x1, x2 = int(info["x1"]), int(info["x2"])
    iin = np.ascontiguousarray(iin[y1:y2, x1:x2])
    igt = np.ascontiguousarray(igt[y1:y2, x1:x2])
    if bool(info["pad_h"]) or bool(info["pad_w"]):
        pad_h = img_size - iin.shape[0]
        pad_w = img_size - iin.shape[1]
        iin = cv2.copyMakeBorder(iin, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        igt = cv2.copyMakeBorder(igt, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    return iin, igt


def patch_key(info: dict[str, object]) -> tuple[str, int, int]:
    return os.path.basename(str(info["img_path"])), int(info["x1"]), int(info["y1"])


def read_anchor_key(path: Path) -> tuple[str, int, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"anchor patch CSV must contain exactly one row: {path}")
    row = rows[0]
    return row["file"], int(row["x1"]), int(row["y1"])


def compute_features(
    info: dict[str, object],
    img_size: int,
    mask_threshold: int,
    image_cache: dict[str, np.ndarray],
    kernel_near: np.ndarray,
    kernel_bg: np.ndarray,
) -> dict[str, float]:
    iin, igt = crop_pair(info, img_size, image_cache)
    ms, diff_mb = generate_mask_from_pair(iin, igt, threshold=mask_threshold)
    box_txt = str(info.get("box_txt_path") or "")
    if box_txt and Path(box_txt).exists():
        box_mb = generate_mb_from_boxes(
            box_txt,
            int(info["x1"]),
            int(info["y1"]),
            int(info["x2"]),
            int(info["y2"]),
            img_size,
        )
    else:
        box_mb = (diff_mb > 0.5).astype(np.uint8)
    stroke = ms > 0.15
    box_region = box_mb > 0
    diff = np.abs(iin.astype(np.int16) - igt.astype(np.int16)).mean(axis=-1)
    gray_gt = cv2.cvtColor(igt[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    printed = gray_gt < 185
    near_stroke = cv2.dilate(stroke.astype(np.uint8), kernel_near, iterations=1) > 0
    broad_bg = (diff > 12) & ~cv2.dilate(stroke.astype(np.uint8), kernel_bg, iterations=1).astype(bool)
    return {
        "stroke_ratio": float(stroke.mean()),
        "printed_near_ratio": float((printed & near_stroke & ~stroke).mean()),
        "bg_risk_ratio": float(broad_bg.mean()),
        "mean_diff": float(diff.mean()),
        "p95_diff": float(np.percentile(diff, 95)),
        "diff_mb_ratio": float((diff_mb > 0.5).mean()),
        "box_mb_ratio": float(box_region.mean()),
        "box_overlap_ratio": float((box_region & stroke).sum() / max(int(box_region.sum()), 1)),
    }


def normalized_abs_delta(value: float, anchor: float, floor: float = 1e-6) -> float:
    return abs(value - anchor) / max(abs(anchor), floor)


def score_row(features: dict[str, float], anchor: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights[key] * normalized_abs_delta(features[key], anchor[key]) for key in FEATURES)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_files = load_optional_file_list(args.train_file_list)
    if train_files is not None:
        train_files = train_files[: args.train_pages]

    dataset = EnsExamRealDataset(
        data_root=cfg["data"]["data_root"],
        img_size=cfg["data"]["img_size"],
        is_train=True,
        overlap=cfg["data"]["overlap"],
        mask_threshold=cfg["data"]["mask_threshold"],
        aug_cfg=None,
        file_list=train_files,
        phase="train",
    )

    anchor_key = read_anchor_key(Path(args.anchor_patch_csv))
    anchor_info = None
    for info in dataset.patch_index_map:
        if patch_key(info) == anchor_key:
            anchor_info = info
            break
    if anchor_info is None:
        raise RuntimeError(f"anchor patch not found in train split: {anchor_key}")

    image_cache: dict[str, np.ndarray] = {}
    kernel_near = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    anchor_features = compute_features(
        anchor_info,
        cfg["data"]["img_size"],
        cfg["data"]["mask_threshold"],
        image_cache,
        kernel_near,
        kernel_bg,
    )
    weights = {
        "stroke_ratio": args.stroke_weight,
        "printed_near_ratio": args.printed_near_weight,
        "bg_risk_ratio": args.bg_risk_weight,
        "mean_diff": args.mean_diff_weight,
        "p95_diff": args.p95_diff_weight,
        "diff_mb_ratio": args.diff_mb_ratio_weight,
        "box_mb_ratio": args.box_mb_ratio_weight,
        "box_overlap_ratio": args.box_overlap_ratio_weight,
    }

    rows: list[dict[str, object]] = []
    for idx, info in enumerate(dataset.patch_index_map):
        key = patch_key(info)
        if args.exclude_anchor and key == anchor_key:
            continue
        features = compute_features(
            info,
            cfg["data"]["img_size"],
            cfg["data"]["mask_threshold"],
            image_cache,
            kernel_near,
            kernel_bg,
        )
        if features["stroke_ratio"] < args.min_stroke_ratio:
            continue
        distance = score_row(features, anchor_features, weights)
        row: dict[str, object] = {
            "rank_score": distance,
            "patch_idx": idx,
            "file": key[0],
            "x1": int(info["x1"]),
            "y1": int(info["y1"]),
            "x2": int(info["x2"]),
            "y2": int(info["y2"]),
        }
        for feature in FEATURES:
            row[feature] = features[feature]
            row[f"anchor_{feature}"] = anchor_features[feature]
            row[f"delta_{feature}"] = features[feature] - anchor_features[feature]
        rows.append(row)

    rows.sort(key=lambda row: (float(row["rank_score"]), str(row["file"]), int(row["y1"]), int(row["x1"])))
    rows = rows[: args.top_k]
    if not rows:
        raise RuntimeError("no candidate patches matched the filters")

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"anchor={anchor_key}")
    print("anchor_features=" + " ".join(f"{key}={anchor_features[key]:.12f}" for key in FEATURES))
    print(f"patches={len(rows)} output={output}")
    print("top5=")
    for row in rows[:5]:
        print(row)


if __name__ == "__main__":
    main()
