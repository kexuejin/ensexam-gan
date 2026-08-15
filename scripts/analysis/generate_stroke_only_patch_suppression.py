#!/usr/bin/env python3
"""Generate target-free stroke-only repair candidates from risky patch routes.

Patch / whiteout candidates often win local metrics by removing residual marks
but fail visual review because they change the surrounding paper patch. This
experiment keeps the baseline everywhere except pixels that look like original
dark handwriting/marks and were lightened by the risky candidate.

The target image is copied into the review CSV for downstream scoring only; it
is not read while generating predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Variant:
    name: str
    source_dark: int
    source_expand_dark: int
    baseline_dark: int
    local_dark_delta: int
    lift_threshold: int
    expand_lift_threshold: int
    dilate: int
    blend: float
    min_component_area: int
    max_component_area: int


VARIANTS = (
    Variant(
        name="stroke_strict",
        source_dark=155,
        source_expand_dark=175,
        baseline_dark=225,
        local_dark_delta=6,
        lift_threshold=4,
        expand_lift_threshold=3,
        dilate=1,
        blend=1.0,
        min_component_area=2,
        max_component_area=12000,
    ),
    Variant(
        name="stroke_balanced",
        source_dark=170,
        source_expand_dark=190,
        baseline_dark=235,
        local_dark_delta=4,
        lift_threshold=3,
        expand_lift_threshold=2,
        dilate=2,
        blend=0.90,
        min_component_area=2,
        max_component_area=16000,
    ),
    Variant(
        name="stroke_soft",
        source_dark=185,
        source_expand_dark=205,
        baseline_dark=245,
        local_dark_delta=3,
        lift_threshold=2,
        expand_lift_threshold=1,
        dilate=2,
        blend=0.70,
        min_component_area=2,
        max_component_area=20000,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-prefix", default="stroke_only_patch_suppression")
    parser.add_argument("--max-rows", type=int, default=60)
    parser.add_argument(
        "--allowed-split",
        action="append",
        default=["train", "train160"],
        help=(
            "Allowed split value. Defaults to train/train160 so preregistration "
            "or preflight runs fail closed before SCUT115, holdout40, or blind data."
        ),
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=[variant.name for variant in VARIANTS],
        help="Variant(s) to run. Defaults to all built-in variants.",
    )
    return parser.parse_args()


def read_rows(path: Path, max_rows: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:max_rows] if max_rows > 0 else rows


def validate_allowed_splits(rows: list[dict[str, str]], allowed_splits: set[str]) -> None:
    disallowed = sorted(
        {
            row.get("split", "")
            for row in rows
            if row.get("split", "") not in allowed_splits
        }
    )
    if disallowed:
        raise ValueError(
            "review CSV contains split(s) outside preregistered authority: "
            + ", ".join(disallowed)
        )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_bgr(path_text: str) -> np.ndarray:
    image = cv2.imread(str(Path(path_text)), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path_text)
    return image


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)


def remove_oversized_components(mask: np.ndarray, variant: Variant) -> tuple[np.ndarray, int, int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(mask, dtype=bool)
    kept_components = 0
    rejected_components = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < variant.min_component_area or area > variant.max_component_area:
            rejected_components += 1
            continue
        kept |= labels == label
        kept_components += 1
    return kept, kept_components, rejected_components


def stroke_only_mask(source: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, variant: Variant) -> tuple[np.ndarray, dict[str, object]]:
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.int16)
    baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY).astype(np.int16)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.int16)
    local_bg = cv2.medianBlur(baseline_gray.astype(np.uint8), 81).astype(np.int16)
    lift = candidate_gray - baseline_gray

    source_stroke = source_gray < variant.source_dark
    baseline_residual = (baseline_gray < variant.baseline_dark) | ((local_bg - baseline_gray) > variant.local_dark_delta)
    seed = source_stroke & baseline_residual & (lift >= variant.lift_threshold)

    if variant.dilate > 0:
        kernel = np.ones((variant.dilate * 2 + 1, variant.dilate * 2 + 1), np.uint8)
        seed_neighborhood = cv2.dilate(seed.astype(np.uint8), kernel).astype(bool)
    else:
        seed_neighborhood = seed

    expanded = (
        seed_neighborhood
        & (source_gray < variant.source_expand_dark)
        & ((baseline_gray < variant.baseline_dark) | ((local_bg - baseline_gray) > max(1, variant.local_dark_delta - 2)))
        & (lift >= variant.expand_lift_threshold)
    )
    kept, kept_components, rejected_components = remove_oversized_components(expanded, variant)
    metrics = {
        "seed_px": int(seed.sum()),
        "mask_px": int(kept.sum()),
        "kept_components": kept_components,
        "rejected_components": rejected_components,
        "candidate_lift_px": int((lift >= variant.expand_lift_threshold).sum()),
        "mask_to_lift_ratio": float(kept.sum()) / max(int((lift >= variant.expand_lift_threshold).sum()), 1),
    }
    return kept, metrics


def materialize_candidate(baseline: np.ndarray, candidate: np.ndarray, mask: np.ndarray, blend: float) -> np.ndarray:
    out = baseline.copy().astype(np.float32)
    if mask.any():
        out[mask] = baseline[mask].astype(np.float32) * (1.0 - blend) + candidate[mask].astype(np.float32) * blend
    return np.clip(out, 0, 255).astype(np.uint8)


def process_row(row: dict[str, str], variant: Variant, variant_dir: Path, candidate_prefix: str) -> tuple[dict[str, object], dict[str, object]]:
    baseline = read_bgr(row["baseline_pred"])
    source = resize_like(read_bgr(row["source_input"]), baseline)
    candidate = resize_like(read_bgr(row["candidate_pred"]), baseline)

    mask, mask_metrics = stroke_only_mask(source, baseline, candidate, variant)
    repaired = materialize_candidate(baseline, candidate, mask, variant.blend)

    pred_dir = variant_dir / "pred" / row["split"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / f"{Path(row['file']).stem}.png"
    cv2.imwrite(str(out_path), repaired)

    sample_key = row.get("sample_key") or f"{row['split']}/{row['file']}"
    review_row = {
        "sample_key": sample_key,
        "split": row["split"],
        "file": row["file"],
        "candidate": f"{candidate_prefix}_{variant.name}",
        "variant": variant.name,
        "source_input": row["source_input"],
        "baseline_pred": row["baseline_pred"],
        "candidate_pred": str(out_path),
        "target": row["target"],
        "bucket": "stroke_only_patch_suppression",
        "source_candidate_pred": row["candidate_pred"],
        "notes": json.dumps(
            {
                "method": "target_free_stroke_only_patch_suppression",
                "source_candidate": row.get("candidate", ""),
                "source_variant": row.get("variant", ""),
                "source_notes": row.get("notes", ""),
            },
            sort_keys=True,
        ),
    }
    diagnostic = {
        **{key: review_row[key] for key in ("sample_key", "split", "file", "candidate", "variant", "candidate_pred")},
        **mask_metrics,
        "changed_ratio": float(np.any(np.abs(repaired.astype(np.int16) - baseline.astype(np.int16)) > 2, axis=2).mean()),
    }
    return review_row, diagnostic


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_names = set(args.variant or [variant.name for variant in VARIANTS])
    variants = [variant for variant in VARIANTS if variant.name in selected_names]
    rows = read_rows(Path(args.review_csv), args.max_rows)
    validate_allowed_splits(rows, set(args.allowed_split))

    combined_review_rows: list[dict[str, object]] = []
    combined_diagnostics: list[dict[str, object]] = []
    for variant in variants:
        variant_dir = output_dir / variant.name
        variant_review_rows: list[dict[str, object]] = []
        variant_diagnostics: list[dict[str, object]] = []
        for row in rows:
            review_row, diagnostic = process_row(row, variant, variant_dir, args.candidate_prefix)
            variant_review_rows.append(review_row)
            variant_diagnostics.append(diagnostic)
        write_csv(variant_dir / "review_rows.csv", variant_review_rows)
        write_csv(variant_dir / "diagnostics.csv", variant_diagnostics)
        combined_review_rows.extend(variant_review_rows)
        combined_diagnostics.extend(variant_diagnostics)

    write_csv(output_dir / "review_rows.csv", combined_review_rows)
    write_csv(output_dir / "diagnostics.csv", combined_diagnostics)
    summary = {
        "input_csv": str(Path(args.review_csv)),
        "rows": len(rows),
        "variants": [variant.name for variant in variants],
        "review_rows": str(output_dir / "review_rows.csv"),
        "diagnostics": str(output_dir / "diagnostics.csv"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
