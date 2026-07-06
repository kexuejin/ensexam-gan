#!/usr/bin/env python3
"""Build a patch-index CSV from explicit handwriting masks.

The output is compatible with micro_train_region_probe.py --patch-index-file.
It is intended for datasets converted into EnsExamRealDataset layout with:

  train/all_images/*.jpg
  train/all_labels/*.jpg
  train/all_masks/*.png

Validation splits use the same layout under val/.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import cv2


def dataset_ticks(total: int, patch_size: int, overlap: int) -> list[int]:
    """Match EnsExamRealDataset._build_patch_index start coordinates."""
    if total <= patch_size:
        return [0]
    step = max(patch_size - overlap, 1)
    count = math.ceil((total - overlap) / step)
    return [index * step for index in range(count)]


def load_file_list(path: str | None) -> set[str] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return {os.path.basename(line.strip()) for line in f if line.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--train-file-list", default=None)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--mask-threshold", type=int, default=12)
    parser.add_argument("--min-mask-ratio", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    image_dir = data_root / args.split / "all_images"
    mask_dir = data_root / args.split / "all_masks"
    allowed_files = load_file_list(args.train_file_list)

    if not mask_dir.is_dir():
        raise RuntimeError(f"missing explicit mask directory: {mask_dir}")

    rows: list[dict[str, object]] = []
    for mask_path in sorted(mask_dir.iterdir()):
        if mask_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        source_name = mask_path.with_suffix(".jpg").name
        if allowed_files is not None and source_name not in allowed_files:
            continue
        image_path = image_dir / source_name
        if not image_path.exists():
            # Fall back to same suffix for converted/local probes.
            image_path = image_dir / mask_path.name
        if not image_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        h, w = mask.shape[:2]
        for y1 in dataset_ticks(h, args.img_size, args.overlap):
            for x1 in dataset_ticks(w, args.img_size, args.overlap):
                y2 = min(y1 + args.img_size, h)
                x2 = min(x1 + args.img_size, w)
                crop = mask[y1:y2, x1:x2]
                mask_ratio = float((crop > args.mask_threshold).mean())
                if mask_ratio < args.min_mask_ratio:
                    continue
                rows.append({
                    "rank_score": mask_ratio,
                    "file": source_name,
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "mask_ratio": mask_ratio,
                })

    rows.sort(key=lambda row: (-float(row["rank_score"]), str(row["file"]), int(row["y1"]), int(row["x1"])))
    if args.top_k > 0:
        rows = rows[: args.top_k]

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank_score", "file", "x1", "y1", "x2", "y2", "mask_ratio"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"patches={len(rows)} output={output}")
    for row in rows[:5]:
        print(row)


if __name__ == "__main__":
    main()
