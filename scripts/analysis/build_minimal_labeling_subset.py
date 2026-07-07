#!/usr/bin/env python3
"""Build a small balanced labeling subset from auto-suggested page labels."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


OUTPUT_FIELDS = [
    "subset_rank",
    "split",
    "file",
    "candidate",
    "bucket",
    "label",
    "flags",
    "reviewer",
    "review_date",
    "comment",
    "auto_suggest_label",
    "auto_confidence",
    "auto_review_priority",
    "auto_reason",
    "review_image",
]

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
SUGGESTION_ORDER = {"slight_win": 0, "slight_loss": 1, "noop": 2, "needs_review": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto-suggest-csv", required=True)
    parser.add_argument("--review-index-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-total", type=int, default=40)
    parser.add_argument("--min-per-split", type=int, default=12)
    parser.add_argument("--min-per-bucket", type=int, default=6)
    parser.add_argument("--min-per-suggest-label", type=int, default=8)
    parser.add_argument("--contact-sheet", default="")
    parser.add_argument("--chunk-dir", default="")
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--contact-width", type=int, default=1440)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return row["split"], row["file"], row["candidate"], row["bucket"]


def rank_key(row: dict[str, str]) -> tuple[int, int, int, str, str, str]:
    return (
        PRIORITY_ORDER.get(row.get("auto_review_priority", ""), 9),
        CONFIDENCE_ORDER.get(row.get("auto_confidence", ""), 9),
        SUGGESTION_ORDER.get(row.get("auto_suggest_label", ""), 9),
        row.get("bucket", ""),
        row.get("split", ""),
        row.get("file", ""),
    )


def merge_review_images(rows: list[dict[str, str]], review_index: list[dict[str, str]]) -> list[dict[str, str]]:
    image_by_key = {key(row): row.get("review_image", "") for row in review_index}
    merged = []
    for row in rows:
        out = dict(row)
        out["review_image"] = image_by_key.get(key(row), "")
        merged.append(out)
    return merged


def add_rows(
    selected: list[dict[str, str]],
    seen: set[tuple[str, str, str, str]],
    candidates: list[dict[str, str]],
    limit: int,
) -> None:
    for row in candidates:
        if len(selected) >= limit:
            return
        row_key = key(row)
        if row_key in seen:
            continue
        seen.add(row_key)
        selected.append(row)


def load_review_image(row: dict[str, str], width: int) -> np.ndarray | None:
    image_path = Path(row.get("review_image", ""))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    if image.shape[1] != width:
        scale = width / max(image.shape[1], 1)
        image = cv2.resize(
            image,
            (width, max(1, int(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    header = np.full((76, width, 3), 255, np.uint8)
    title = (
        f"rank={row.get('subset_rank')} {row.get('split')}/{row.get('file')} "
        f"{row.get('bucket')} suggest={row.get('auto_suggest_label')} "
        f"conf={row.get('auto_confidence')} priority={row.get('auto_review_priority')}"
    )
    cv2.putText(header, title[:180], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 1, cv2.LINE_AA)
    reason = row.get("auto_reason", "")
    cv2.putText(header, reason[:190], (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (70, 70, 70), 1, cv2.LINE_AA)
    return np.concatenate([header, image], axis=0)


def write_sheet(rows: list[dict[str, str]], path: Path, width: int) -> None:
    images = [image for row in rows if (image := load_review_image(row, width)) is not None]
    if not images:
        return
    separator = np.full((12, width, 3), 230, np.uint8)
    parts: list[np.ndarray] = []
    for image in images:
        parts.extend([image, separator])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(parts[:-1], axis=0))


def write_contact_outputs(rows: list[dict[str, str]], contact_sheet: str, chunk_dir: str, chunk_size: int, width: int) -> None:
    if contact_sheet:
        write_sheet(rows, Path(contact_sheet), width)
    if chunk_dir:
        output_dir = Path(chunk_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            path = output_dir / f"chunk_{start // chunk_size + 1:02d}.png"
            write_sheet(chunk, path, width)


def main() -> None:
    args = parse_args()
    rows = merge_review_images(
        read_rows(Path(args.auto_suggest_csv)),
        read_rows(Path(args.review_index_csv)),
    )
    ranked = sorted(rows, key=rank_key)
    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for split in sorted({row["split"] for row in rows}):
        add_rows(
            selected,
            seen,
            [row for row in ranked if row["split"] == split],
            min(args.max_total, len(selected) + args.min_per_split),
        )

    for bucket in sorted({row["bucket"] for row in rows}):
        current = sum(row["bucket"] == bucket for row in selected)
        add_rows(
            selected,
            seen,
            [row for row in ranked if row["bucket"] == bucket],
            min(args.max_total, len(selected) + max(0, args.min_per_bucket - current)),
        )

    for suggested in sorted({row["auto_suggest_label"] for row in rows}):
        current = sum(row["auto_suggest_label"] == suggested for row in selected)
        add_rows(
            selected,
            seen,
            [row for row in ranked if row["auto_suggest_label"] == suggested],
            min(args.max_total, len(selected) + max(0, args.min_per_suggest_label - current)),
        )

    add_rows(selected, seen, ranked, args.max_total)

    for index, row in enumerate(selected, start=1):
        row["subset_rank"] = str(index)

    write_csv(Path(args.output_csv), selected)
    write_contact_outputs(selected, args.contact_sheet, args.chunk_dir, args.chunk_size, args.contact_width)
    print(f"selected={len(selected)} output_csv={args.output_csv}")
    print(f"splits={dict(Counter(row['split'] for row in selected))}")
    print(f"buckets={dict(Counter(row['bucket'] for row in selected))}")
    print(f"suggestions={dict(Counter(row['auto_suggest_label'] for row in selected))}")
    missing_images = sum(1 for row in selected if not row.get("review_image"))
    print(f"missing_review_images={missing_images}")
    if args.contact_sheet:
        print(f"contact_sheet={args.contact_sheet}")
    if args.chunk_dir:
        print(f"chunk_dir={args.chunk_dir}")


if __name__ == "__main__":
    main()
