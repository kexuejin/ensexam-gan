#!/usr/bin/env python3
"""Materialize visible-delta pages into a small EnsExam dataset root.

Visible-delta analysis can identify useful local components on pages that are
not part of the current training split. This script copies those source pages
into a separate EnsExamRealDataset-compatible root so bounded smoke/probe runs
can validate plumbing without mutating the canonical SCUT train/test folders.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--file-list", default="")
    parser.add_argument("--region-type", default="improve", choices=("improve", "regress", "all"))
    parser.add_argument("--reason-contains", default="visible_target_region")
    return parser.parse_args()


def row_matches(row: dict[str, str], region_type: str, reason_contains: str) -> bool:
    if region_type != "all" and row["region_type"] != region_type:
        return False
    if reason_contains and reason_contains not in row["reason"]:
        return False
    return True


def copy_once(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    image_dir = output_root / args.split / "all_images"
    label_dir = output_root / args.split / "all_labels"
    copied: dict[str, tuple[Path, Path]] = {}

    with Path(args.components_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row_matches(row, args.region_type, args.reason_contains):
                continue
            image_path = Path(row["image_path"])
            label_path = Path(row["label_path"])
            name = image_path.name
            if name in copied:
                continue
            image_dst = image_dir / name
            label_dst = label_dir / name
            copy_once(image_path, image_dst)
            copy_once(label_path, label_dst)
            copied[name] = (image_dst, label_dst)

    if args.file_list:
        file_list = Path(args.file_list)
        file_list.parent.mkdir(parents=True, exist_ok=True)
        file_list.write_text("".join(f"{name}\n" for name in sorted(copied)), encoding="utf-8")

    print(f"pages={len(copied)}")
    print(f"output_root={output_root}")
    if args.file_list:
        print(f"file_list={args.file_list}")
    for name in sorted(copied):
        print(name)


if __name__ == "__main__":
    main()
