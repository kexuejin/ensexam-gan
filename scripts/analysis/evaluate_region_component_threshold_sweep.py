#!/usr/bin/env python3
"""Evaluate region-component score thresholds without writing prediction PNGs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVAL_SCRIPT_DIR = ROOT / "scripts" / "eval"
if str(EVAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPT_DIR))

from eval_hardcase_worst_pages import (  # noqa: E402
    build_changed_mask,
    ensure_same_size,
    label_path_for,
    read_bgr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-csv", required=True)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument(
        "--threshold-summary-csv",
        default="",
        help="Optional ranker threshold_summary.csv. If omitted, score quantiles are used.",
    )
    parser.add_argument("--threshold-column", default="threshold")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--max-thresholds", type=int, default=80)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME:BASELINE_METRICS:CANDIDATE_METRICS",
        help="May be repeated.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--base-edit-threshold", type=float, default=12.0)
    parser.add_argument("--candidate-delta-threshold", type=float, default=2.0)
    parser.add_argument("--min-component-area", type=int, default=3)
    parser.add_argument("--change-threshold", type=int, default=12)
    parser.add_argument("--eval-threshold", type=int, default=12)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_split(value: str) -> tuple[str, Path, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Invalid --split {value!r}; expected NAME:BASELINE:CANDIDATE")
    return parts[0], Path(parts[1]), Path(parts[2])


def component_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["split"], row["file"], row["component_id"]


def load_scores(path: Path, score_column: str) -> dict[tuple[str, str, str], float]:
    scores: dict[tuple[str, str, str], float] = {}
    for row in read_rows(path):
        if score_column not in row:
            raise KeyError(f"Score column {score_column!r} not found in {path}")
        scores[component_key(row)] = float(row[score_column])
    return scores


def load_thresholds(args: argparse.Namespace) -> list[float]:
    if args.threshold_summary_csv:
        rows = read_rows(Path(args.threshold_summary_csv))
        thresholds = [float(row[args.threshold_column]) for row in rows if row.get(args.threshold_column, "")]
    else:
        scores = [float(row[args.score_column]) for row in read_rows(Path(args.predictions_csv))]
        quantiles = np.linspace(0, 1, max(args.max_thresholds, 2))
        thresholds = [float(value) for value in np.quantile(scores, quantiles)]
    unique = sorted(set(thresholds), reverse=True)
    if args.max_thresholds > 0:
        unique = unique[: args.max_thresholds]
    return unique


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int16)


def active_components(
    source: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    base_edit_threshold: float,
    candidate_delta_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_gray = gray(source)
    baseline_gray = gray(baseline)
    candidate_gray = gray(candidate)
    active = (
        (np.abs(baseline_gray - source_gray) >= base_edit_threshold)
        & (np.abs(candidate_gray - baseline_gray) >= candidate_delta_threshold)
    ).astype(np.uint8)
    _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(active, connectivity=8)
    return labels, stats


def component_score_by_page(
    components_csv: Path,
    scores: dict[tuple[str, str, str], float],
) -> dict[tuple[str, str], dict[int, float]]:
    result: dict[tuple[str, str], dict[int, float]] = {}
    for row in read_rows(components_csv):
        key = component_key(row)
        score = scores.get(key)
        if score is None:
            continue
        result.setdefault((row["split"], row["file"]), {})[int(row["component_id"])] = score
    return result


def split_pages(
    split_name: str,
    baseline_metrics: Path,
    candidate_metrics: Path,
    scores_by_page: dict[tuple[str, str], dict[int, float]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    candidate_by_file = {row["file"]: row for row in read_rows(candidate_metrics)}
    pages: list[dict[str, object]] = []
    for baseline_row in read_rows(baseline_metrics):
        file = baseline_row["file"]
        candidate_row = candidate_by_file[file]
        source = read_bgr(Path(baseline_row["image_path"]))
        label = ensure_same_size(read_bgr(label_path_for(Path(baseline_row["image_path"]))), source)
        baseline = ensure_same_size(read_bgr(Path(baseline_row["pred_path"])), source)
        candidate = ensure_same_size(read_bgr(Path(candidate_row["pred_path"])), source)
        labels, stats = active_components(
            source,
            baseline,
            candidate,
            args.base_edit_threshold,
            args.candidate_delta_threshold,
        )
        changed = build_changed_mask(source, label, args.change_threshold)
        outside = ~changed
        baseline_residual_mask = changed & (
            cv2.absdiff(baseline, label).mean(axis=2) >= args.eval_threshold
        )
        baseline_overerase_mask = outside & (
            cv2.absdiff(baseline, source).mean(axis=2) >= args.eval_threshold
        )
        candidate_residual_mask = changed & (
            cv2.absdiff(candidate, label).mean(axis=2) >= args.eval_threshold
        )
        candidate_overerase_mask = outside & (
            cv2.absdiff(candidate, source).mean(axis=2) >= args.eval_threshold
        )
        component_rows = []
        for component_id, score in scores_by_page.get((split_name, file), {}).items():
            if component_id >= len(stats):
                continue
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < args.min_component_area:
                continue
            component_mask = labels == component_id
            component_rows.append({
                "component_id": component_id,
                "score": score,
                "pixels": int(component_mask.sum()),
                "residual_px_delta": int(candidate_residual_mask[component_mask].sum())
                - int(baseline_residual_mask[component_mask].sum()),
                "over_px_delta": int(candidate_overerase_mask[component_mask].sum())
                - int(baseline_overerase_mask[component_mask].sum()),
            })
        pages.append({
            "split": split_name,
            "file": file,
            "changed_px": int(baseline_row.get("changed_px", int(changed.sum()))),
            "outside_px": int(baseline_row.get("outside_px", int(outside.sum()))),
            "baseline_residual_px": int(baseline_row.get("residual_px", int(baseline_residual_mask.sum()))),
            "baseline_over_px": int(baseline_row.get("over_px", int(baseline_overerase_mask.sum()))),
            "component_rows": component_rows,
        })
    return pages


def evaluate_threshold(
    pages: list[dict[str, object]],
    threshold: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    residual = 0.0
    overerase = 0.0
    baseline_residual = 0.0
    baseline_overerase = 0.0
    improved_pages = 0
    worse_pages = 0
    over_reg_pages = 0
    materialized_pages = 0
    materialized_components = 0
    selected_pixels = 0

    for page in pages:
        selected_count = 0
        page_selected_pixels = 0
        residual_px_delta = 0
        over_px_delta = 0
        for component in page["component_rows"]:
            if float(component["score"]) < threshold:
                continue
            selected_count += 1
            page_selected_pixels += int(component["pixels"])
            residual_px_delta += int(component["residual_px_delta"])
            over_px_delta += int(component["over_px_delta"])

        materialized_pages += int(selected_count > 0)
        changed_px = int(page["changed_px"])
        outside_px = int(page["outside_px"])
        baseline_residual_px = int(page["baseline_residual_px"])
        baseline_over_px = int(page["baseline_over_px"])
        residual_px = baseline_residual_px + residual_px_delta
        over_px = baseline_over_px + over_px_delta
        baseline_r = baseline_residual_px / max(changed_px, 1)
        baseline_o = baseline_over_px / max(outside_px, 1)
        residual_r = residual_px / max(changed_px, 1)
        overerase_r = over_px / max(outside_px, 1)
        residual += residual_r
        overerase += overerase_r
        baseline_residual += baseline_r
        baseline_overerase += baseline_o
        improved_pages += int(baseline_r - residual_r > 0)
        worse_pages += int(baseline_r - residual_r < 0)
        over_reg_pages += int(overerase_r - baseline_o > 0)
        materialized_components += selected_count
        selected_pixels += page_selected_pixels

    page_count = max(len(pages), 1)
    return {
        "pages": len(pages),
        "materialized_pages": materialized_pages,
        "materialized_components": materialized_components,
        "selected_pixels": selected_pixels,
        "baseline_residual": baseline_residual / page_count,
        "residual": residual / page_count,
        "residual_gain": (baseline_residual - residual) / page_count,
        "baseline_overerase": baseline_overerase / page_count,
        "overerase": overerase / page_count,
        "overerase_delta": (overerase - baseline_overerase) / page_count,
        "improved_pages": improved_pages,
        "worse_pages": worse_pages,
        "over_reg_pages": over_reg_pages,
    }


def main() -> None:
    args = parse_args()
    thresholds = load_thresholds(args)
    scores = load_scores(Path(args.predictions_csv), args.score_column)
    scores_by_page = component_score_by_page(Path(args.components_csv), scores)

    pages_by_split: dict[str, list[dict[str, object]]] = {}
    for split_name, baseline_metrics, candidate_metrics in map(parse_split, args.split):
        pages_by_split[split_name] = split_pages(
            split_name,
            baseline_metrics,
            candidate_metrics,
            scores_by_page,
            args,
        )

    rows: list[dict[str, object]] = []
    for rank, threshold in enumerate(thresholds):
        for split_name, pages in pages_by_split.items():
            rows.append({
                "rank": rank,
                "threshold": threshold,
                "split": split_name,
                **evaluate_threshold(pages, threshold, args),
            })
    write_csv(Path(args.output_csv), rows)

    print(f"thresholds={len(thresholds)} rows={len(rows)} output_csv={args.output_csv}")
    for split_name in pages_by_split:
        best = max(
            (row for row in rows if row["split"] == split_name),
            key=lambda row: (float(row["residual_gain"]), -float(row["overerase_delta"])),
        )
        print(
            f"best {split_name} threshold={float(best['threshold']):.12f} "
            f"residual_gain={float(best['residual_gain']):.9f} "
            f"overerase_delta={float(best['overerase_delta']):.9f} "
            f"materialized_pages={best['materialized_pages']} "
            f"worse_pages={best['worse_pages']} "
            f"over_reg_pages={best['over_reg_pages']}"
        )


if __name__ == "__main__":
    main()
