#!/usr/bin/env python3
"""Materialize explicit target-difference masks for paired EnsExam datasets.

The output keeps the EnsExamRealDataset layout:

  output_root/train/all_images/*.jpg
  output_root/train/all_labels/*.jpg
  output_root/train/all_masks/*.png

Images and labels are symlinked by default; masks are generated from local
pixel differences between all_images and all_labels.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.path_utils import normalize_path  # noqa: E402


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_file_list(path: str | None) -> list[str] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [os.path.basename(line.strip()) for line in f if line.strip()]


def link_or_copy(source: Path, destination: Path, *, copy_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_files:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def build_mask(
    image: np.ndarray,
    target: np.ndarray,
    *,
    threshold: int,
    min_component_area: int,
    dilate: int,
    open_size: int,
) -> np.ndarray:
    diff = cv2.absdiff(image, target)
    mask = (diff.max(axis=2) > threshold).astype(np.uint8)
    if open_size > 1:
        kernel = np.ones((open_size, open_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if min_component_area > 1:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        for label in range(1, n_labels):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_component_area:
                keep[labels == label] = 1
        mask = keep
    if dilate > 0:
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return (mask * 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--file-list", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threshold", type=int, default=12)
    parser.add_argument("--min-component-area", type=int, default=3)
    parser.add_argument("--dilate", type=int, default=1)
    parser.add_argument("--open-size", type=int, default=0)
    parser.add_argument("--copy-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(normalize_path(args.source_root))
    output_root = Path(args.output_root)
    source_split = source_root / args.split
    image_dir = source_split / "all_images"
    label_dir = source_split / "all_labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise RuntimeError(f"missing source image/label directories under {source_split}")

    requested = load_file_list(args.file_list)
    if requested is None:
        files = sorted(path.name for path in image_dir.iterdir() if path.suffix.lower() in VALID_EXTENSIONS)
    else:
        files = [name for name in requested if (image_dir / name).exists()]
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise RuntimeError("no source files selected")

    out_split = output_root / args.split
    manifest_path = out_split / "target_diff_mask_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for name in files:
        image_path = image_dir / name
        label_path = label_dir / name
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        target = cv2.imread(str(label_path), cv2.IMREAD_COLOR)
        if image is None or target is None:
            continue
        if image.shape != target.shape:
            raise RuntimeError(f"shape mismatch for {name}: {image.shape} vs {target.shape}")

        mask = build_mask(
            image,
            target,
            threshold=args.threshold,
            min_component_area=args.min_component_area,
            dilate=args.dilate,
            open_size=args.open_size,
        )
        mask_name = Path(name).with_suffix(".png").name
        link_or_copy(image_path, out_split / "all_images" / name, copy_files=args.copy_files)
        link_or_copy(label_path, out_split / "all_labels" / name, copy_files=args.copy_files)
        mask_path = out_split / "all_masks" / mask_name
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"failed to write mask: {mask_path}")

        rows.append({
            "file": name,
            "mask_file": mask_name,
            "height": image.shape[0],
            "width": image.shape[1],
            "mask_pixels": int((mask > 0).sum()),
            "mask_ratio": f"{float((mask > 0).mean()):.8f}",
        })

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "mask_file", "height", "width", "mask_pixels", "mask_ratio"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"split={args.split} files={len(rows)} output={out_split}")
    print(f"manifest={manifest_path}")
    if rows:
        ratios = np.array([float(row["mask_ratio"]) for row in rows], dtype=np.float32)
        print(
            f"mask_ratio min={ratios.min():.6f} "
            f"mean={ratios.mean():.6f} max={ratios.max():.6f}"
        )


if __name__ == "__main__":
    main()
