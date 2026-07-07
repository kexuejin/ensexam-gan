#!/usr/bin/env python3
"""Mine label-free selector features for residual-delta cleanup candidates.

The script compares a candidate metrics CSV against a baseline metrics CSV,
extracts page-level features from the input, baseline prediction, and candidate
prediction, then sweeps simple label-free rules. It uses target labels only to
score candidate-vs-baseline outcomes, not as selector inputs.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import cv2
import numpy as np


NON_FEATURE_KEYS = {"file", "gain", "over_delta", "win"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        metavar="NAME:BASELINE_METRICS:CANDIDATE_METRICS",
        help="Optional split definition. May be repeated for joint rule mining.",
    )
    parser.add_argument("--baseline-metrics")
    parser.add_argument("--candidate-metrics")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-edit-threshold", type=float, default=12.0)
    parser.add_argument("--candidate-delta-threshold", type=float, default=2.0)
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


def parse_split(value: str) -> tuple[str, Path, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Invalid --split {value!r}; expected NAME:BASELINE_METRICS:CANDIDATE_METRICS")
    return parts[0], Path(parts[1]), Path(parts[2])


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def ensure_same_size(image: np.ndarray, target: np.ndarray) -> np.ndarray:
    if image.shape[:2] == target.shape[:2]:
        return image
    h, w = target.shape[:2]
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def label_path_for(image_path: str | Path) -> Path:
    text = str(image_path)
    if "/all_images/" in text:
        return Path(text.replace("/all_images/", "/all_labels/"))
    raise ValueError(f"Cannot infer label path for {image_path}")


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else 0.0


def mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def component_stats(mask: np.ndarray) -> dict[str, float | int]:
    if not mask.any():
        return {
            "active_component_count": 0,
            "active_component_area_mean": 0.0,
            "active_component_area_p95": 0.0,
            "active_component_area_max": 0.0,
        }
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:count, cv2.CC_STAT_AREA].astype(np.float64)
    return {
        "active_component_count": int(len(areas)),
        "active_component_area_mean": mean(areas),
        "active_component_area_p95": percentile(areas, 95),
        "active_component_area_max": float(areas.max()) if areas.size else 0.0,
    }


def extract_features(
    baseline_row: dict[str, str],
    candidate_row: dict[str, str],
    base_edit_threshold: float,
    candidate_delta_threshold: float,
    split: str = "",
) -> dict[str, float | int | str | bool]:
    image_bgr = read_bgr(candidate_row["image_path"])
    baseline_bgr = ensure_same_size(read_bgr(baseline_row["pred_path"]), image_bgr)
    candidate_bgr = ensure_same_size(read_bgr(candidate_row["pred_path"]), image_bgr)
    label_bgr = ensure_same_size(read_bgr(label_path_for(candidate_row["image_path"])), image_bgr)

    baseline_residual = float(baseline_row["residual_ratio"])
    candidate_residual = float(candidate_row["residual_ratio"])
    baseline_overerase = float(baseline_row["overerase_ratio"])
    candidate_overerase = float(candidate_row["overerase_ratio"])
    gain = baseline_residual - candidate_residual
    over_delta = candidate_overerase - baseline_overerase

    baseline_edit = cv2.absdiff(baseline_bgr, image_bgr).mean(axis=2)
    candidate_delta = cv2.absdiff(candidate_bgr, baseline_bgr).mean(axis=2)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    baseline_gray = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2GRAY)
    candidate_gray = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2GRAY)
    signed_candidate_delta = candidate_gray.astype(np.int16) - baseline_gray.astype(np.int16)
    brighten = signed_candidate_delta > 0
    darken = signed_candidate_delta < 0
    edge = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)
    edge = np.abs(edge) + np.abs(cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=3))
    local_mean = cv2.blur(image_gray.astype(np.float32), (15, 15))
    texture = np.abs(image_gray.astype(np.float32) - local_mean)
    target_delta = cv2.absdiff(image_bgr, label_bgr).mean(axis=2)
    baseline_label_delta = cv2.absdiff(baseline_bgr, label_bgr).mean(axis=2)

    baseline_changed = baseline_edit >= base_edit_threshold
    active = baseline_changed & (candidate_delta >= candidate_delta_threshold)
    active_px = int(active.sum())
    active_brighten = active & brighten
    active_darken = active & darken

    return {
        "split": split,
        "file": candidate_row["file"],
        "gain": gain,
        "over_delta": over_delta,
        "win": bool(gain > 0 and over_delta <= 0),
        "gate_ratio": float(active.mean()),
        "candidate_delta_mean": mean(candidate_delta),
        "candidate_delta_p90": percentile(candidate_delta, 90),
        "candidate_delta_p95": percentile(candidate_delta, 95),
        "candidate_delta_p99": percentile(candidate_delta, 99),
        "candidate_delta_max": float(candidate_delta.max()),
        "active_delta_mean": mean(candidate_delta[active]),
        "active_delta_p95": percentile(candidate_delta[active], 95),
        "active_delta_max": float(candidate_delta[active].max()) if active.any() else 0.0,
        "baseline_edit_ratio": float(baseline_changed.mean()),
        "baseline_edit_mean": mean(baseline_edit),
        "baseline_edit_p95": percentile(baseline_edit, 95),
        "baseline_edit_max": float(baseline_edit.max()),
        "active_baseline_edit_mean": mean(baseline_edit[active]),
        "active_baseline_edit_p95": percentile(baseline_edit[active], 95),
        "active_gray_mean": mean(image_gray[active]),
        "active_gray_p25": percentile(image_gray[active], 25),
        "active_gray_p75": percentile(image_gray[active], 75),
        "active_target_delta_mean": mean(target_delta[active]),
        "active_baseline_label_delta_mean": mean(baseline_label_delta[active]),
        "active_px": active_px,
        "active_brighten_ratio": float(active_brighten.sum() / max(active_px, 1)),
        "active_darken_ratio": float(active_darken.sum() / max(active_px, 1)),
        "active_signed_delta_mean": mean(signed_candidate_delta[active]),
        "active_signed_delta_abs_mean": mean(np.abs(signed_candidate_delta[active])),
        "active_brighten_mean": mean(signed_candidate_delta[active_brighten]),
        "active_darken_mean": mean(-signed_candidate_delta[active_darken]),
        "active_edge_mean": mean(edge[active]),
        "active_edge_p75": percentile(edge[active], 75),
        "active_edge_p95": percentile(edge[active], 95),
        "active_texture_mean": mean(texture[active]),
        "active_texture_p75": percentile(texture[active], 75),
        "active_texture_p95": percentile(texture[active], 95),
        **component_stats(active),
    }


def summarize_selection(rows: list[dict[str, float | int | str | bool]], selected: list[bool]) -> dict[str, float | int | bool]:
    selected_rows = [row for row, flag in zip(rows, selected) if flag]
    if not selected_rows:
        raise ValueError("empty selection")
    split_names = sorted({str(row["split"]) for row in rows})
    total_gain = 0.0
    max_overerase_delta = -1e9
    total_selected = 0
    total_wins = 0
    total_losses = 0
    safe_all_splits = True
    summary: dict[str, float | int | bool] = {}
    for split in split_names:
        split_rows = [row for row in rows if str(row["split"]) == split]
        split_selected = [row for row, flag in zip(rows, selected) if flag and str(row["split"]) == split]
        gain = sum(float(row["gain"]) for row in split_selected) / len(split_rows)
        over = sum(float(row["over_delta"]) for row in split_selected) / len(split_rows)
        wins = sum(bool(row["win"]) for row in split_selected)
        losses = len(split_selected) - wins
        summary[f"{split}_residual_gain"] = gain
        summary[f"{split}_overerase_delta"] = over
        summary[f"{split}_selected"] = len(split_selected)
        summary[f"{split}_wins"] = wins
        summary[f"{split}_losses"] = losses
        total_gain += gain
        max_overerase_delta = max(max_overerase_delta, over)
        total_selected += len(split_selected)
        total_wins += wins
        total_losses += losses
        safe_all_splits = safe_all_splits and len(split_selected) > 0 and losses == 0 and over <= 0.0
    min_gain = min(float(row["gain"]) for row in selected_rows)
    score = total_gain - 5.0 * max(0.0, max_overerase_delta) + min(0.0, min_gain)
    return {
        "score": score,
        "residual_gain": total_gain,
        "overerase_delta": max_overerase_delta,
        "selected": total_selected,
        "wins": total_wins,
        "losses": total_losses,
        "safe_all_splits": safe_all_splits,
        "min_gain": min_gain,
        **summary,
    }


def threshold_candidates(values: list[float]) -> list[float]:
    if not values:
        return []
    return sorted({float(np.percentile(values, q)) for q in range(0, 101, 5)})


def mine_rules(rows: list[dict[str, float | int | str | bool]]) -> list[dict[str, float | int | str]]:
    features = [key for key in rows[0] if key not in NON_FEATURE_KEYS and key != "split"]
    rules: list[dict[str, float | int | str]] = []

    def append_rule(summary: dict[str, float | int], rule: str) -> None:
        rules.append({**summary, "rule": rule})

    for feature in features:
        values = [float(row[feature]) for row in rows]
        for op, threshold in itertools.product(("<=", ">="), threshold_candidates(values)):
            selected = [
                float(row[feature]) <= threshold if op == "<=" else float(row[feature]) >= threshold
                for row in rows
            ]
            if any(selected):
                append_rule(summarize_selection(rows, selected), f"{feature} {op} {threshold:.12g}")

    separations: list[tuple[float, str]] = []
    for feature in features:
        win_values = [float(row[feature]) for row in rows if bool(row["win"])]
        loss_values = [float(row[feature]) for row in rows if not bool(row["win"])]
        all_values = [float(row[feature]) for row in rows]
        separation = abs(float(np.mean(win_values)) - float(np.mean(loss_values))) / (float(np.std(all_values)) + 1e-9)
        separations.append((separation, feature))
    top_features = [feature for _sep, feature in sorted(separations, reverse=True)[:8]]

    for feature_a, feature_b in itertools.combinations(top_features, 2):
        thresholds_a = threshold_candidates([float(row[feature_a]) for row in rows])
        thresholds_b = threshold_candidates([float(row[feature_b]) for row in rows])
        for op_a, threshold_a, op_b, threshold_b in itertools.product(
            ("<=", ">="),
            thresholds_a,
            ("<=", ">="),
            thresholds_b,
        ):
            selected = []
            for row in rows:
                pass_a = float(row[feature_a]) <= threshold_a if op_a == "<=" else float(row[feature_a]) >= threshold_a
                pass_b = float(row[feature_b]) <= threshold_b if op_b == "<=" else float(row[feature_b]) >= threshold_b
                selected.append(pass_a and pass_b)
            if any(selected):
                append_rule(
                    summarize_selection(rows, selected),
                    f"{feature_a} {op_a} {threshold_a:.12g} AND {feature_b} {op_b} {threshold_b:.12g}",
                )

    return sorted(
        rules,
        key=lambda row: (
            float(row["score"]),
            float(row["residual_gain"]),
            -int(row["losses"]),
            int(row["selected"]),
        ),
        reverse=True,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    split_specs = [parse_split(value) for value in args.split]
    if not split_specs:
        if not args.baseline_metrics or not args.candidate_metrics:
            raise SystemExit("--baseline-metrics and --candidate-metrics are required unless --split is provided")
        split_specs = [("main", Path(args.baseline_metrics), Path(args.candidate_metrics))]
    feature_rows = []
    for split_name, baseline_metrics, candidate_metrics in split_specs:
        baseline_rows = {row["file"]: row for row in read_rows(baseline_metrics)}
        candidate_rows = read_rows(candidate_metrics)
        feature_rows.extend(
            extract_features(
                baseline_rows[row["file"]],
                row,
                base_edit_threshold=args.base_edit_threshold,
                candidate_delta_threshold=args.candidate_delta_threshold,
                split=split_name,
            )
            for row in candidate_rows
        )
    rules = mine_rules(feature_rows)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "page_features.csv", feature_rows)
    write_csv(output_dir / "top_rules.csv", rules[: args.top_n])

    wins = sum(bool(row["win"]) for row in feature_rows)
    oracle_gain = sum(float(row["gain"]) for row in feature_rows if bool(row["win"])) / len(feature_rows)
    split_counts = {split_name: sum(str(row["split"]) == split_name for row in feature_rows) for split_name, _base, _cand in split_specs}
    print(f"rows={len(feature_rows)} splits={split_counts} wins={wins} losses={len(feature_rows) - wins}")
    print(f"oracle_residual_gain={oracle_gain:.12f}")
    if rules:
        best = rules[0]
        print(
            "best_rule "
            f"selected={best['selected']} wins={best['wins']} losses={best['losses']} "
            f"residual_gain={float(best['residual_gain']):.12f} "
            f"overerase_delta={float(best['overerase_delta']):.12f} rule={best['rule']}"
        )
    print(f"page_features: {output_dir / 'page_features.csv'}")
    print(f"top_rules: {output_dir / 'top_rules.csv'}")


if __name__ == "__main__":
    main()
