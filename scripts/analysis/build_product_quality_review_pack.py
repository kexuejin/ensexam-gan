#!/usr/bin/env python3
"""Build side-by-side review images from product-quality review CSV rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


PANELS = [
    ("source_input", "input"),
    ("baseline_pred", "baseline"),
    ("candidate_pred", "candidate"),
    ("target", "target"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", default="docs/product-quality-review-pages.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thumb-width", type=int, default=360)
    parser.add_argument("--thumb-height", type=int, default=260)
    parser.add_argument("--max-contact-rows", type=int, default=80)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_panel(path_text: str, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, np.uint8)
    if not path_text:
        return canvas
    path = Path(path_text)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        cv2.putText(canvas, "missing", (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
        return canvas
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    y0 = (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    bar_h = 34
    out = np.full((image.shape[0] + bar_h, image.shape[1], 3), 255, np.uint8)
    out[bar_h:] = image
    cv2.putText(out, label[:48], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1, cv2.LINE_AA)
    return out


def safe_name(row: dict[str, str], index: int) -> str:
    split = row.get("split", "unknown").replace("/", "_")
    file = row.get("file", "unknown").replace("/", "_")
    bucket = row.get("bucket", "bucket").replace("/", "_")
    candidate = row.get("candidate", "candidate").replace("/", "_")
    return f"{index:03d}_{split}_{file}_{candidate}_{bucket}.png"


def build_row_image(row: dict[str, str], width: int, height: int) -> np.ndarray:
    panels = []
    for column, title in PANELS:
        image = load_panel(row.get(column, ""), width, height)
        panels.append(add_label(image, title))
    row_image = np.concatenate(panels, axis=1)
    caption_h = 48
    caption = np.full((caption_h, row_image.shape[1], 3), 255, np.uint8)
    text = f"{row.get('split','')} | {row.get('file','')} | {row.get('bucket','')} | {row.get('candidate','')}"
    cv2.putText(caption, text[:150], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    return np.concatenate([caption, row_image], axis=0)


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.review_csv))
    output_dir = Path(args.output_dir)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    contact_rows: list[np.ndarray] = []
    index_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        row_image = build_row_image(row, args.thumb_width, args.thumb_height)
        output_path = pages_dir / safe_name(row, index)
        cv2.imwrite(str(output_path), row_image)
        index_row = dict(row)
        index_row["review_image"] = str(output_path)
        index_rows.append(index_row)
        if len(contact_rows) < args.max_contact_rows:
            contact_rows.append(row_image)

    if contact_rows:
        separator = np.full((12, contact_rows[0].shape[1], 3), 230, np.uint8)
        sheet_parts: list[np.ndarray] = []
        for row_image in contact_rows:
            sheet_parts.extend([row_image, separator])
        contact_sheet = np.concatenate(sheet_parts[:-1], axis=0)
        cv2.imwrite(str(output_dir / "contact_sheet.png"), contact_sheet)

    fieldnames = sorted({key for row in index_rows for key in row.keys()})
    with (output_dir / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"rows={len(rows)}")
    print(f"pages_dir={pages_dir}")
    print(f"contact_sheet={output_dir / 'contact_sheet.png'}")
    print(f"index_csv={output_dir / 'index.csv'}")


if __name__ == "__main__":
    main()
