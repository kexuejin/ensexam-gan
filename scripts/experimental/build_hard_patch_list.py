#!/usr/bin/env python3
"""Build a deterministic hard-patch list for SCUT finetune probes.

The score favors patches with enough handwriting signal and nearby printed
content, while penalizing broad background differences that are likely to drive
over-erasure.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_config  # noqa: E402
from data.dataset import EnsExamRealDataset  # noqa: E402
from data.mask_utils import generate_mask_from_pair  # noqa: E402
from train import load_optional_file_list  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-file-list", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--train-pages", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--min-stroke-ratio", type=float, default=0.001)
    parser.add_argument("--stroke-weight", type=float, default=100.0)
    parser.add_argument("--printed-near-weight", type=float, default=50.0)
    parser.add_argument("--bg-risk-weight", type=float, default=30.0)
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

    rows: list[dict[str, object]] = []
    image_cache: dict[str, np.ndarray] = {}
    kernel_near = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    for idx, info in enumerate(dataset.patch_index_map):
        iin, igt = crop_pair(info, cfg["data"]["img_size"], image_cache)
        ms, _ = generate_mask_from_pair(iin, igt, threshold=cfg["data"]["mask_threshold"])
        stroke = ms > 0.15
        stroke_ratio = float(stroke.mean())
        if stroke_ratio < args.min_stroke_ratio:
            continue

        gray_gt = cv2.cvtColor(igt[:, :, ::-1], cv2.COLOR_BGR2GRAY)
        printed = gray_gt < 185
        near_stroke = cv2.dilate(stroke.astype(np.uint8), kernel_near, iterations=1) > 0
        printed_near_ratio = float((printed & near_stroke & ~stroke).mean())

        diff = np.abs(iin.astype(np.int16) - igt.astype(np.int16)).mean(axis=-1)
        broad_bg = (diff > 12) & ~cv2.dilate(stroke.astype(np.uint8), kernel_bg, iterations=1).astype(bool)
        bg_risk_ratio = float(broad_bg.mean())

        score = (
            stroke_ratio * args.stroke_weight
            + printed_near_ratio * args.printed_near_weight
            - bg_risk_ratio * args.bg_risk_weight
        )
        rows.append({
            "rank_score": score,
            "patch_idx": idx,
            "file": os.path.basename(str(info["img_path"])),
            "x1": int(info["x1"]),
            "y1": int(info["y1"]),
            "x2": int(info["x2"]),
            "y2": int(info["y2"]),
            "stroke_ratio": stroke_ratio,
            "printed_near_ratio": printed_near_ratio,
            "bg_risk_ratio": bg_risk_ratio,
        })

    rows.sort(key=lambda row: (-float(row["rank_score"]), str(row["file"]), int(row["y1"]), int(row["x1"])))
    rows = rows[: args.top_k]

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"patches={len(rows)} output={output}")
    if rows:
        print("top5=")
        for row in rows[:5]:
            print(row)


if __name__ == "__main__":
    main()
