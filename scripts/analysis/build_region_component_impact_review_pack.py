#!/usr/bin/env python3
"""Build review crops for page-impactful region components.

This complements build_region_component_review_pack.py. The original pack is
balanced by weak component verdicts; this one ranks components by actual
page-metric pixel deltas so reviewed labels focus on components that can move
residual/overerase metrics at page level.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
EVAL_SCRIPT_DIR = ROOT / "scripts" / "eval"
if str(EVAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPT_DIR))

from build_region_component_review_pack import (  # noqa: E402
    expand_crop,
    load_image,
    path_index,
    render_crop,
    resize_like,
    safe_name,
    write_contact_sheet,
    write_csv,
)
from eval_hardcase_worst_pages import build_changed_mask  # noqa: E402


BUCKET_LIMITS = {
    "residual_help": 30,
    "residual_hurt": 30,
    "overerase_risk": 20,
    "large_noop": 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-csv", required=True)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME:BASELINE_METRICS:CANDIDATE_METRICS",
        help="May be repeated. Used to resolve page paths and component masks.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allowed-split", action="append", default=[])
    parser.add_argument("--score-csv", default="", help="Optional ranker predictions.csv with split/file/component_id/score.")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--max-total", type=int, default=80)
    parser.add_argument("--max-per-page", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=220)
    parser.add_argument("--thumb-size", type=int, default=180)
    parser.add_argument("--contact-width", type=int, default=1440)
    parser.add_argument("--base-edit-threshold", type=float, default=12.0)
    parser.add_argument("--candidate-delta-threshold", type=float, default=2.0)
    parser.add_argument("--change-threshold", type=int, default=12)
    parser.add_argument("--eval-threshold", type=int, default=12)
    parser.add_argument("--min-area", type=int, default=3)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
        scores[component_key(row)] = float(row[score_column])
    return scores


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int16)


def page_metric_masks(
    source: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    change_threshold: int,
    eval_threshold: int,
) -> dict[str, np.ndarray]:
    changed = build_changed_mask(source, target, change_threshold)
    outside = ~changed
    baseline_residual = changed & (cv2.absdiff(baseline, target).mean(axis=2) >= eval_threshold)
    candidate_residual = changed & (cv2.absdiff(candidate, target).mean(axis=2) >= eval_threshold)
    baseline_over = outside & (cv2.absdiff(baseline, source).mean(axis=2) >= eval_threshold)
    candidate_over = outside & (cv2.absdiff(candidate, source).mean(axis=2) >= eval_threshold)
    return {
        "baseline_residual": baseline_residual,
        "candidate_residual": candidate_residual,
        "baseline_over": baseline_over,
        "candidate_over": candidate_over,
    }


def active_labels(
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


def impact_bucket(row: dict[str, str]) -> str:
    residual_delta = int(row["residual_px_delta"])
    over_delta = int(row["over_px_delta"])
    area = int(float(row["area"]))
    if residual_delta < 0:
        return "residual_help"
    if residual_delta > 0:
        return "residual_hurt"
    if over_delta > 0:
        return "overerase_risk"
    if area >= 50:
        return "large_noop"
    return "other"


def impact_priority(row: dict[str, str]) -> float:
    residual_delta = abs(int(row["residual_px_delta"]))
    over_delta = max(int(row["over_px_delta"]), 0)
    area = min(int(float(row["area"])), 800)
    score = float(row.get("selector_score") or 0.0)
    return residual_delta * 10.0 + over_delta * 20.0 + area * 0.05 + score


def build_impact_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_rows(Path(args.components_csv))
    allowed_splits = set(args.allowed_split)
    if allowed_splits:
        rows = [row for row in rows if row["split"] in allowed_splits]
    scores = load_scores(Path(args.score_csv), args.score_column) if args.score_csv else {}
    path_rows = path_index(args.split)
    by_page: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        if int(float(row["area"])) < args.min_area:
            continue
        score = scores.get(component_key(row))
        if args.min_score is not None and (score is None or score < args.min_score):
            continue
        row = dict(row)
        row["selector_score"] = "" if score is None else f"{score:.12g}"
        by_page.setdefault((row["split"], row["file"]), []).append(row)

    impact_rows: list[dict[str, str]] = []
    image_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = {}
    for key, page_components in sorted(by_page.items()):
        if key not in path_rows:
            continue
        paths = path_rows[key]
        baseline = load_image(paths["baseline_pred"])
        images = {
            "source": resize_like(load_image(paths["source_input"]), baseline),
            "baseline": baseline,
            "candidate": resize_like(load_image(paths["candidate_pred"]), baseline),
            "target": resize_like(load_image(paths["target"]), baseline),
        }
        labels, stats = active_labels(
            images["source"],
            images["baseline"],
            images["candidate"],
            args.base_edit_threshold,
            args.candidate_delta_threshold,
        )
        masks = page_metric_masks(
            images["source"],
            images["target"],
            images["baseline"],
            images["candidate"],
            args.change_threshold,
            args.eval_threshold,
        )
        image_cache[key] = (labels, stats, masks)

        for row in page_components:
            component_id = int(row["component_id"])
            if component_id >= len(stats):
                continue
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < args.min_area:
                continue
            component = labels == component_id
            out = dict(row)
            out["component_pixels"] = str(int(component.sum()))
            out["baseline_residual_px"] = str(int(masks["baseline_residual"][component].sum()))
            out["candidate_residual_px"] = str(int(masks["candidate_residual"][component].sum()))
            out["residual_px_delta"] = str(
                int(masks["candidate_residual"][component].sum())
                - int(masks["baseline_residual"][component].sum())
            )
            out["baseline_over_px"] = str(int(masks["baseline_over"][component].sum()))
            out["candidate_over_px"] = str(int(masks["candidate_over"][component].sum()))
            out["over_px_delta"] = str(
                int(masks["candidate_over"][component].sum())
                - int(masks["baseline_over"][component].sum())
            )
            out["bucket"] = impact_bucket(out)
            out["impact_priority"] = f"{impact_priority(out):.6f}"
            impact_rows.append(out)
    return impact_rows


def select_rows(rows: list[dict[str, str]], max_total: int, max_per_page: int) -> list[dict[str, str]]:
    by_bucket: dict[str, list[dict[str, str]]] = {name: [] for name in BUCKET_LIMITS}
    for row in rows:
        if row["bucket"] in by_bucket:
            by_bucket[row["bucket"]].append(row)
    for bucket_rows in by_bucket.values():
        bucket_rows.sort(key=lambda row: float(row["impact_priority"]), reverse=True)

    selected: list[dict[str, str]] = []
    per_page: dict[tuple[str, str], int] = {}
    for bucket_name, limit in BUCKET_LIMITS.items():
        picked = 0
        for row in by_bucket[bucket_name]:
            page_key = (row["split"], row["file"])
            if per_page.get(page_key, 0) >= max_per_page:
                continue
            selected.append(row)
            per_page[page_key] = per_page.get(page_key, 0) + 1
            picked += 1
            if picked >= limit or len(selected) >= max_total:
                break
        if len(selected) >= max_total:
            break
    return selected[:max_total]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    paths = path_index(args.split)
    impact_rows = build_impact_rows(args)
    selected = select_rows(impact_rows, args.max_total, args.max_per_page)

    contact_images: list[np.ndarray] = []
    contact_images_by_bucket: dict[str, list[np.ndarray]] = {}
    index_rows: list[dict[str, object]] = []
    image_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for rank, row in enumerate(selected, start=1):
        key = (row["split"], row["file"])
        if key not in image_cache:
            path_row = paths[key]
            baseline = load_image(path_row["baseline_pred"])
            image_cache[key] = {
                "baseline": baseline,
                "source": resize_like(load_image(path_row["source_input"]), baseline),
                "candidate": resize_like(load_image(path_row["candidate_pred"]), baseline),
                "target": resize_like(load_image(path_row["target"]), baseline),
            }
        crop = expand_crop(row, image_cache[key]["baseline"].shape, args.crop_size)
        rendered = render_crop(image_cache[key], row, crop, args.thumb_size)
        crop_path = crops_dir / safe_name(row, rank)
        if not cv2.imwrite(str(crop_path), rendered):
            raise OSError(f"Failed to write {crop_path}")
        out = dict(row)
        out.update({
            "priority_rank": rank,
            "priority_score": row["impact_priority"],
            "crop_box": f"{crop[0]},{crop[1]},{crop[2]},{crop[3]}",
            "crop_review_image": str(crop_path),
            "label": "",
            "flags": "",
            "reviewer": "",
            "review_date": "",
            "comment": "",
        })
        index_rows.append(out)
        contact_images.append(rendered)
        contact_images_by_bucket.setdefault(row["bucket"], []).append(rendered)

    fields = [
        "priority_rank",
        "priority_score",
        "bucket",
        "split",
        "file",
        "component_id",
        "selector_score",
        "x",
        "y",
        "w",
        "h",
        "area",
        "component_pixels",
        "component_gain",
        "component_hurt_ratio",
        "residual_px_delta",
        "over_px_delta",
        "baseline_residual_px",
        "candidate_residual_px",
        "baseline_over_px",
        "candidate_over_px",
        "component_verdict",
        "crop_box",
        "crop_review_image",
        "label",
        "flags",
        "reviewer",
        "review_date",
        "comment",
    ]
    write_csv(output_dir / "component-impact-review-index.csv", index_rows, fields)
    write_csv(output_dir / "component-impact-labels-template.csv", index_rows, fields)
    write_contact_sheet(contact_images, output_dir / "contact_sheet.png", args.contact_width)
    for bucket_name, images in sorted(contact_images_by_bucket.items()):
        write_contact_sheet(images, output_dir / f"contact_sheet_{bucket_name}.png", args.contact_width)

    counts: dict[str, int] = {}
    for row in index_rows:
        counts[str(row["bucket"])] = counts.get(str(row["bucket"]), 0) + 1
    print(f"impact_rows={len(impact_rows)} selected={len(index_rows)} buckets={counts}")
    print(f"index_csv={output_dir / 'component-impact-review-index.csv'}")
    print(f"labels_template={output_dir / 'component-impact-labels-template.csv'}")
    print(f"contact_sheet={output_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
