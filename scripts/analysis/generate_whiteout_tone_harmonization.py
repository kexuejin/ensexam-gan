#!/usr/bin/env python3
"""Generate conservative paper-tone harmonization candidates for whiteout pages.

This is an experiment-only post-processing candidate. It starts from the
baseline prediction, detects bright low-frequency correction-fluid patches
without using the target image, and gently shifts those patches toward the
surrounding paper tone. It deliberately avoids texture synthesis.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", default="docs/product-quality-review-pages.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-name", default="whiteout_tone_harmonize_v1")
    parser.add_argument("--bucket", default="correction_fluid_white_patch")
    parser.add_argument("--min-area", type=int, default=350)
    parser.add_argument("--max-area-ratio", type=float, default=0.035)
    parser.add_argument("--bright-delta", type=float, default=10.0)
    parser.add_argument("--max-shift", type=float, default=3.0)
    parser.add_argument("--blend", type=float, default=0.20)
    parser.add_argument("--ring-dilate", type=int, default=35)
    parser.add_argument("--mask-dilate", type=int, default=5)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_image(path_text: str) -> np.ndarray:
    image = cv2.imread(str(Path(path_text)), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path_text)
    return image


def robust_paper_color(image: np.ndarray, ring: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    eligible = ring & (gray > 120) & (gray < 252) & (grad < 18)
    if int(eligible.sum()) < 50:
        eligible = ring & (gray > 120) & (grad < 28)
    if int(eligible.sum()) < 50:
        return None
    pixels = image[eligible]
    return np.median(pixels.astype(np.float32), axis=0)


def detect_whiteout_components(
    source: np.ndarray,
    baseline: np.ndarray,
    min_area: int,
    max_area_ratio: float,
    bright_delta: float,
) -> list[np.ndarray]:
    gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(baseline, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    bg = cv2.medianBlur(gray.astype(np.uint8), 81).astype(np.float32)
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    # Whiteout is usually a low-saturation patch brighter than its local paper.
    # Use this as the only seed so ordinary erased handwriting cannot create
    # page-wide false positives.
    seed = (gray > 205) & ((gray - bg) > bright_delta) & (sat < 45)

    # Dark writing in the original input may remain inside the correction-fluid
    # patch after cleanup, but it should only expand an existing bright seed.
    removed_dark = (source_gray < 145) & (gray > 190)
    seed_neighborhood = cv2.dilate(seed.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
    mask = seed | (removed_dark & seed_neighborhood & (gray > bg + 4) & (sat < 60))

    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    max_area = int(mask.shape[0] * mask.shape[1] * max_area_ratio)
    components: list[np.ndarray] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if w < 12 or h < 8:
            continue
        component = labels == label
        seed_fraction = float(seed[component].sum()) / max(area, 1)
        if seed_fraction < 0.2:
            continue
        components.append(component)
    return components


def harmonize_component(
    image: np.ndarray,
    component: np.ndarray,
    ring_dilate: int,
    mask_dilate: int,
    max_shift: float,
    blend: float,
) -> tuple[np.ndarray, dict[str, float]]:
    out = image.copy()
    comp_u8 = component.astype(np.uint8)
    mask_kernel = np.ones((mask_dilate, mask_dilate), np.uint8)
    ring_kernel = np.ones((ring_dilate, ring_dilate), np.uint8)
    soft_mask = cv2.dilate(comp_u8, mask_kernel)
    ring = cv2.dilate(comp_u8, ring_kernel).astype(bool) & ~cv2.dilate(comp_u8, mask_kernel).astype(bool)

    paper = robust_paper_color(image, ring)
    if paper is None:
        return out, {"applied": 0.0, "mean_shift": 0.0}

    patch = image[soft_mask.astype(bool)].astype(np.float32)
    patch_color = np.median(patch, axis=0)
    delta = np.clip(paper - patch_color, -max_shift, max_shift)
    if float(np.linalg.norm(delta)) < 1.0:
        return out, {"applied": 0.0, "mean_shift": float(np.linalg.norm(delta))}

    alpha = cv2.GaussianBlur(soft_mask.astype(np.float32), (0, 0), sigmaX=3.0)
    alpha = np.clip(alpha[..., None] * blend, 0.0, blend)
    shifted = np.clip(out.astype(np.float32) + delta.reshape(1, 1, 3), 0, 255)
    out = np.clip(out.astype(np.float32) * (1.0 - alpha) + shifted * alpha, 0, 255).astype(np.uint8)
    return out, {"applied": 1.0, "mean_shift": float(np.linalg.norm(delta))}


def process_row(row: dict[str, str], args: argparse.Namespace, pred_dir: Path) -> dict[str, object]:
    source = load_image(row["source_input"])
    baseline = load_image(row["baseline_pred"])
    if source.shape[:2] != baseline.shape[:2]:
        source = cv2.resize(source, (baseline.shape[1], baseline.shape[0]), interpolation=cv2.INTER_AREA)

    components = detect_whiteout_components(
        source=source,
        baseline=baseline,
        min_area=args.min_area,
        max_area_ratio=args.max_area_ratio,
        bright_delta=args.bright_delta,
    )
    candidate = baseline.copy()
    applied = 0
    shifts: list[float] = []
    for component in components:
        candidate, metrics = harmonize_component(
            candidate,
            component,
            ring_dilate=args.ring_dilate,
            mask_dilate=args.mask_dilate,
            max_shift=args.max_shift,
            blend=args.blend,
        )
        applied += int(metrics["applied"])
        shifts.append(float(metrics["mean_shift"]))

    out_path = pred_dir / f"{Path(row['file']).stem}.png"
    cv2.imwrite(str(out_path), candidate)
    return {
        "split": row["split"],
        "file": row["file"],
        "source_input": row["source_input"],
        "baseline_pred": row["baseline_pred"],
        "candidate_pred": str(out_path),
        "target": row["target"],
        "components": len(components),
        "applied_components": applied,
        "mean_shift": float(np.mean(shifts)) if shifts else 0.0,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for row in read_rows(Path(args.review_csv))
        if row.get("bucket") == args.bucket
    ]
    diagnostics = [process_row(row, args, pred_dir) for row in rows]

    review_rows = []
    for row in diagnostics:
        review_rows.append({
            "split": row["split"],
            "file": row["file"],
            "bucket": args.bucket,
            "candidate": args.candidate_name,
            "source_input": row["source_input"],
            "baseline_pred": row["baseline_pred"],
            "candidate_pred": row["candidate_pred"],
            "target": row["target"],
            "review_pack": str(output_dir),
            "notes": "conservative whiteout paper-tone harmonization candidate",
        })

    write_csv(output_dir / "diagnostics.csv", diagnostics)
    write_csv(output_dir / "review_rows.csv", review_rows)
    print(f"rows={len(rows)}")
    print(f"pred_dir={pred_dir}")
    print(f"diagnostics={output_dir / 'diagnostics.csv'}")
    print(f"review_rows={output_dir / 'review_rows.csv'}")


if __name__ == "__main__":
    main()
