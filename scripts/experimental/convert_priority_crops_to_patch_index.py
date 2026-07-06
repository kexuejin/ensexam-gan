#!/usr/bin/env python3
"""Convert prioritized crop-review rows into EnsExam training patch indices."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--train-files-out", default="")
    parser.add_argument("--data-root", default="./data-links/samples/SCUT-EnsExam")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--patch-pad", type=int, default=64)
    parser.add_argument("--max-tiles-per-crop", type=int, default=3)
    parser.add_argument(
        "--triage-bucket",
        action="append",
        default=[],
        help="Optional triage_bucket value to include. May be repeated.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def image_size(path_text: str) -> tuple[int, int]:
    image = cv2.imread(str(Path(path_text)), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path_text)
    h, w = image.shape[:2]
    return w, h


def dataset_phase(split: str) -> str:
    if split == "scut115":
        return "test"
    return "train"


def image_path_for_row(row: dict[str, str], data_root: Path) -> Path:
    if row.get("source_input"):
        return Path(row["source_input"])
    return data_root / dataset_phase(row.get("split", "")) / "all_images" / row["file"]


def parse_box(value: str, pad: int, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    parts = [int(float(part)) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected 4 box values, got {value!r}")
    x0, y0, x1, y1 = parts
    return max(0, x0 - pad), max(0, y0 - pad), min(image_w, x1 + pad), min(image_h, y1 + pad)


def ticks(total: int, patch_size: int, overlap: int) -> list[int]:
    if total <= patch_size:
        return [0]
    step = max(patch_size - overlap, 1)
    count = math.ceil((total - overlap) / step)
    return [index * step for index in range(count)]


def intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    return max(0, x1 - x0) * max(0, y1 - y0)


def float_value(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0)
    except ValueError:
        return 0.0


def main() -> None:
    args = parse_args()
    emitted: dict[tuple[str, int, int], dict[str, object]] = {}
    priority_rows = read_rows(Path(args.priority_csv))
    if args.triage_bucket:
        allowed = set(args.triage_bucket)
        priority_rows = [row for row in priority_rows if row.get("triage_bucket", "") in allowed]
    data_root = Path(args.data_root)

    for source_index, row in enumerate(priority_rows, start=1):
        image_w, image_h = image_size(str(image_path_for_row(row, data_root)))
        source_box = parse_box(row["crop_box"], args.patch_pad, image_w, image_h)
        box_area = max(1, (source_box[2] - source_box[0]) * (source_box[3] - source_box[1]))
        candidates: list[tuple[float, int, int, tuple[int, int, int, int], int]] = []
        for y1 in ticks(image_h, args.img_size, args.overlap):
            for x1 in ticks(image_w, args.img_size, args.overlap):
                tile = (x1, y1, min(x1 + args.img_size, image_w), min(y1 + args.img_size, image_h))
                inter = intersection_area(source_box, tile)
                if inter <= 0:
                    continue
                coverage = inter / box_area
                score = float_value(row, "priority_score") * coverage
                candidates.append((score, x1, y1, tile, inter))
        candidates.sort(reverse=True)

        for tile_rank, (score, x1, y1, tile, inter) in enumerate(candidates[: args.max_tiles_per_crop], start=1):
            key = (row["file"], x1, y1)
            existing = emitted.get(key)
            if existing is not None and float(existing["rank_score"]) >= score:
                continue
            emitted[key] = {
                "rank_score": round(score, 6),
                "source_index": source_index,
                "source_rank": tile_rank,
                "file": row["file"],
                "x1": x1,
                "y1": y1,
                "x2": tile[2],
                "y2": tile[3],
                "source_type": row.get("source_type", ""),
                "source_bucket": row.get("bucket", ""),
                "source_area": row.get("source_area", ""),
                "source_priority_rank": row.get("priority_rank", ""),
                "source_priority_score": row.get("priority_score", ""),
                "source_handwriting_score": row.get("handwriting_likelihood_score", ""),
                "source_residual_px": row.get("residual_px", ""),
                "source_crop_index": row.get("crop_index", ""),
                "source_crop_box": row.get("crop_box", ""),
                "source_tile_intersection": inter,
            }

    rows = sorted(
        emitted.values(),
        key=lambda item: (-float(item["rank_score"]), str(item["file"]), int(item["y1"]), int(item["x1"])),
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        output_csv.write_text("", encoding="utf-8")

    if args.train_files_out:
        train_files = sorted({str(row["file"]) for row in rows})
        Path(args.train_files_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.train_files_out).write_text("\n".join(train_files) + ("\n" if train_files else ""), encoding="utf-8")

    print(f"priority_rows={len(priority_rows)}")
    print(f"patch_rows={len(rows)}")
    print(f"files={len(set(row['file'] for row in rows))}")
    print(f"output_csv={output_csv}")
    if args.train_files_out:
        print(f"train_files_out={args.train_files_out}")
    for row in rows[:5]:
        print(row)


if __name__ == "__main__":
    main()
