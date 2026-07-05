#!/usr/bin/env python3
"""Select a compact hard-negative patch list from over-erasure components."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--max-per-file", type=int, default=16)
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    return parser.parse_args()


def iou(a: dict[str, int], b: dict[str, int]) -> float:
    x0 = max(a["x0"], b["x0"])
    y0 = max(a["y0"], b["y0"])
    x1 = min(a["x1"], b["x1"])
    y1 = min(a["y1"], b["y1"])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter == 0:
        return 0.0
    area_a = (a["x1"] - a["x0"]) * (a["y1"] - a["y0"])
    area_b = (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])
    return inter / max(area_a + area_b - inter, 1)


def to_patch(row: dict[str, str]) -> dict[str, str | int | float]:
    return {
        "image_path": row["image_path"],
        "label_path": row["label_path"],
        "file": row["file"],
        "x0": int(row["patch_x0"]),
        "y0": int(row["patch_y0"]),
        "x1": int(row["patch_x1"]),
        "y1": int(row["patch_y1"]),
        "source_component_id": int(row["component_id"]),
        "source_area": int(row["area"]),
        "source_mean_over_delta": float(row["mean_over_delta"]),
        "score": int(row["area"]) * float(row["mean_over_delta"]),
    }


def main() -> None:
    args = parse_args()
    with Path(args.components_csv).open(newline="", encoding="utf-8") as f:
        candidates = [
            to_patch(row)
            for row in csv.DictReader(f)
            if int(row["area"]) >= args.min_area
        ]

    candidates.sort(key=lambda row: (float(row["score"]), int(row["source_area"])), reverse=True)
    selected: list[dict[str, str | int | float]] = []
    per_file: dict[str, int] = {}
    for candidate in candidates:
        file_name = str(candidate["file"])
        if per_file.get(file_name, 0) >= args.max_per_file:
            continue
        if any(
            file_name == str(existing["file"])
            and iou(candidate, existing) >= args.iou_threshold
            for existing in selected
        ):
            continue
        selected.append(candidate)
        per_file[file_name] = per_file.get(file_name, 0) + 1
        if len(selected) >= args.top_k:
            break

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if selected:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
            writer.writeheader()
            writer.writerows(selected)
    else:
        output_csv.write_text("", encoding="utf-8")

    print(f"selected={len(selected)}")
    print(f"output_csv={output_csv}")
    print("top_files=" + ",".join(
        f"{file}:{count}"
        for file, count in sorted(per_file.items(), key=lambda item: item[1], reverse=True)[:10]
    ))


if __name__ == "__main__":
    main()
