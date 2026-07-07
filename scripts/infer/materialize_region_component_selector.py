#!/usr/bin/env python3
"""Materialize region-component selections into page prediction images.

The output starts from the baseline prediction and copies candidate pixels only
inside selected connected components. Component IDs are regenerated from the
same baseline/source and candidate/baseline thresholds used by
evaluate_region_component_selector.py, so rows from its components.csv can be
replayed without relying on target images at materialization time.
"""

from __future__ import annotations

import argparse
import csv
import operator
from pathlib import Path

import cv2
import numpy as np


OPS = {
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
    "==": operator.eq,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-csv", required=True)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME:BASELINE_METRICS:CANDIDATE_METRICS",
        help="May be repeated. Outputs are written under <output-dir>/<NAME>/pred.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selector-rule", default="", help="Optional component rule, e.g. 'area >= 10 AND fill_ratio <= 0.5'.")
    parser.add_argument("--predictions-csv", default="", help="Optional ranker predictions.csv with split/file/component_id/score.")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--score-column", default="score")
    parser.add_argument(
        "--combine-mode",
        choices=("all", "any"),
        default="all",
        help="How to combine --selector-rule and --score-threshold when both are provided.",
    )
    parser.add_argument("--base-edit-threshold", type=float, default=12.0)
    parser.add_argument("--candidate-delta-threshold", type=float, default=2.0)
    parser.add_argument("--min-component-area", type=int, default=3)
    parser.add_argument("--write-empty-pages", action="store_true", help="Also write unchanged baseline pages with no selected components.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_split(value: str) -> tuple[str, Path, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Invalid --split {value!r}; expected NAME:BASELINE:CANDIDATE")
    return parts[0], Path(parts[1]), Path(parts[2])


def read_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if image.shape[:2] == reference.shape[:2]:
        return image
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int16)


def selector_hit(row: dict[str, str], rule: str) -> bool:
    for condition in (part.strip() for part in rule.split(" AND ") if part.strip()):
        parts = condition.split()
        if len(parts) != 3:
            raise ValueError(f"Unsupported selector condition: {condition!r}")
        feature, op_text, threshold_text = parts
        if op_text not in OPS:
            raise ValueError(f"Unsupported selector operator: {op_text!r}")
        if feature not in row:
            raise KeyError(f"Selector feature {feature!r} not found")
        if not OPS[op_text](float(row[feature]), float(threshold_text)):
            return False
    return True


def component_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["split"], row["file"], row["component_id"]


def load_scores(path: Path, score_column: str) -> dict[tuple[str, str, str], float]:
    scores: dict[tuple[str, str, str], float] = {}
    for row in read_rows(path):
        if score_column not in row:
            raise KeyError(f"Score column {score_column!r} not found in {path}")
        scores[component_key(row)] = float(row[score_column])
    return scores


def component_selected(
    row: dict[str, str],
    selector_rule: str,
    scores: dict[tuple[str, str, str], float],
    score_threshold: float | None,
    combine_mode: str,
) -> tuple[bool, float | None, bool | None, bool | None]:
    checks: list[bool] = []
    rule_ok: bool | None = None
    if selector_rule:
        rule_ok = selector_hit(row, selector_rule)
        checks.append(rule_ok)

    score: float | None = None
    score_ok: bool | None = None
    if score_threshold is not None:
        key = component_key(row)
        if key not in scores:
            score_ok = False
        else:
            score = scores[key]
            score_ok = score >= score_threshold
        checks.append(score_ok)

    if not checks:
        raise ValueError("At least one selector is required: --selector-rule or --score-threshold")
    selected = all(checks) if combine_mode == "all" else any(checks)
    return selected, score, rule_ok, score_ok


def selected_components_by_file(
    components_csv: Path,
    selector_rule: str,
    predictions_csv: str,
    score_threshold: float | None,
    score_column: str,
    combine_mode: str,
) -> tuple[dict[tuple[str, str], set[int]], list[dict[str, object]]]:
    scores = load_scores(Path(predictions_csv), score_column) if predictions_csv else {}
    if score_threshold is not None and not scores:
        raise ValueError("--score-threshold requires --predictions-csv")

    selected_by_file: dict[tuple[str, str], set[int]] = {}
    component_rows: list[dict[str, object]] = []
    for row in read_rows(components_csv):
        selected, score, rule_ok, score_ok = component_selected(
            row,
            selector_rule,
            scores,
            score_threshold,
            combine_mode,
        )
        key = (row["split"], row["file"])
        component_id = int(row["component_id"])
        if selected:
            selected_by_file.setdefault(key, set()).add(component_id)
        component_rows.append({
            "split": row["split"],
            "file": row["file"],
            "component_id": component_id,
            "selected": int(selected),
            "rule_ok": "" if rule_ok is None else int(rule_ok),
            "score_ok": "" if score_ok is None else int(score_ok),
            "score": "" if score is None else score,
            "x": row.get("x", ""),
            "y": row.get("y", ""),
            "w": row.get("w", ""),
            "h": row.get("h", ""),
            "area": row.get("area", ""),
        })
    return selected_by_file, component_rows


def build_active_components(
    source: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    base_edit_threshold: float,
    candidate_delta_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_gray = gray(source)
    baseline_gray = gray(baseline)
    candidate_gray = gray(candidate)
    active = (
        (np.abs(baseline_gray - source_gray) >= base_edit_threshold)
        & (np.abs(candidate_gray - baseline_gray) >= candidate_delta_threshold)
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
    return labels, stats, np.arange(count)


def materialize_split(
    split_name: str,
    baseline_metrics: Path,
    candidate_metrics: Path,
    selected_by_file: dict[tuple[str, str], set[int]],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    baseline_rows = read_rows(baseline_metrics)
    candidate_by_file = {row["file"]: row for row in read_rows(candidate_metrics)}
    pred_dir = output_dir / split_name / "pred"
    rows: list[dict[str, object]] = []

    for baseline_row in baseline_rows:
        file = baseline_row["file"]
        candidate_row = candidate_by_file[file]
        selected_ids = selected_by_file.get((split_name, file), set())

        source = read_bgr(baseline_row["image_path"])
        baseline = resize_like(read_bgr(baseline_row["pred_path"]), source)
        candidate = resize_like(read_bgr(candidate_row["pred_path"]), source)
        labels, stats, component_ids = build_active_components(
            source,
            baseline,
            candidate,
            args.base_edit_threshold,
            args.candidate_delta_threshold,
        )

        keep_mask = np.zeros(labels.shape, dtype=bool)
        materialized_ids: list[int] = []
        for component_id in sorted(selected_ids):
            if component_id >= len(component_ids):
                continue
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < args.min_component_area:
                continue
            keep_mask |= labels == component_id
            materialized_ids.append(component_id)

        pred = baseline.copy()
        if keep_mask.any():
            pred[keep_mask] = candidate[keep_mask]
        if keep_mask.any() or args.write_empty_pages:
            pred_dir.mkdir(parents=True, exist_ok=True)
            output_path = pred_dir / f"{Path(file).stem}.png"
            if not cv2.imwrite(str(output_path), pred):
                raise OSError(f"Failed to write {output_path}")
        else:
            output_path = Path("")

        rows.append({
            "split": split_name,
            "file": file,
            "selected_components": len(selected_ids),
            "materialized_components": len(materialized_ids),
            "selected_pixels": int(keep_mask.sum()),
            "baseline_pred_path": baseline_row["pred_path"],
            "candidate_pred_path": candidate_row["pred_path"],
            "output_pred_path": str(output_path),
            "materialized_component_ids": " ".join(str(value) for value in materialized_ids),
        })
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    selected_by_file, component_rows = selected_components_by_file(
        Path(args.components_csv),
        args.selector_rule,
        args.predictions_csv,
        args.score_threshold,
        args.score_column,
        args.combine_mode,
    )
    write_csv(output_dir / "component-selection.csv", component_rows)

    page_rows: list[dict[str, object]] = []
    for split_name, baseline_metrics, candidate_metrics in map(parse_split, args.split):
        page_rows.extend(
            materialize_split(
                split_name,
                baseline_metrics,
                candidate_metrics,
                selected_by_file,
                output_dir,
                args,
            )
        )
    write_csv(output_dir / "selection.csv", page_rows)

    selected_components = sum(int(row["selected"]) for row in component_rows)
    requested_selected_components = sum(int(row["selected_components"]) for row in page_rows)
    materialized_pages = sum(int(row["materialized_components"]) > 0 for row in page_rows)
    materialized_components = sum(int(row["materialized_components"]) for row in page_rows)
    selected_pixels = sum(int(row["selected_pixels"]) for row in page_rows)
    print(
        f"pages={len(page_rows)} materialized_pages={materialized_pages} "
        f"selected_components_total={selected_components} "
        f"selected_components_in_splits={requested_selected_components} "
        f"materialized_components={materialized_components} "
        f"selected_pixels={selected_pixels}"
    )
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
