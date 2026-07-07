#!/usr/bin/env python3
"""Evaluate label-free connected-component selectors for candidate edits.

This is an offline analysis tool for testing whether page-level candidate edits
can be made safer by keeping only selected local components. It searches rules
on train components using inference-time features, then materializes and scores
the selected components on both train and held-out pages. Target images are used
only for offline scoring.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    threshold: float

    def hit(self, row: dict[str, object]) -> bool:
        value = float(row[self.feature])
        if self.op == "<=":
            return value <= self.threshold
        if self.op == ">=":
            return value >= self.threshold
        raise ValueError(f"Unsupported operator: {self.op}")

    def text(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:.12g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME:BASELINE_METRICS:CANDIDATE_METRICS",
        help="May be repeated.",
    )
    parser.add_argument("--train-split", action="append", required=True)
    parser.add_argument("--test-split", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-edit-threshold", type=float, default=12.0)
    parser.add_argument("--candidate-delta-threshold", type=float, default=2.0)
    parser.add_argument("--eval-threshold", type=float, default=12.0)
    parser.add_argument("--min-component-area", type=int, default=3)
    parser.add_argument("--max-conditions", type=int, default=2)
    parser.add_argument("--max-single-candidates", type=int, default=80)
    parser.add_argument("--max-rules", type=int, default=50)
    parser.add_argument("--min-train-components", type=int, default=20)
    parser.add_argument("--min-test-pages", type=int, default=1)
    parser.add_argument("--max-train-reject-components", type=int, default=0)
    parser.add_argument("--max-train-reject-ratio", type=float, default=0.0)
    parser.add_argument(
        "--quantiles",
        default="0,0.05,0.1,0.2,0.25,0.33,0.5,0.67,0.75,0.8,0.9,0.95,1",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
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


def label_path_for(image_path: str | Path) -> Path:
    parts = list(Path(image_path).parts)
    try:
        idx = parts.index("all_images")
    except ValueError as exc:
        raise ValueError(f"Cannot derive label path from {image_path}") from exc
    parts[idx] = "all_labels"
    return Path(*parts)


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int16)


def mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else 0.0


def compute_metrics(
    source: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    change_threshold: float,
    eval_threshold: float,
) -> dict[str, float | int]:
    source_delta = cv2.absdiff(source, label).mean(axis=2)
    changed = source_delta >= change_threshold
    outside = ~changed
    residual_delta = cv2.absdiff(pred, label).mean(axis=2)
    over_delta = cv2.absdiff(pred, source).mean(axis=2)
    residual_px = int((changed & (residual_delta >= eval_threshold)).sum())
    over_px = int((outside & (over_delta >= eval_threshold)).sum())
    changed_px = int(changed.sum())
    outside_px = int(outside.sum())
    return {
        "changed_px": changed_px,
        "outside_px": outside_px,
        "residual_px": residual_px,
        "over_px": over_px,
        "residual_ratio": residual_px / max(changed_px, 1),
        "overerase_ratio": over_px / max(outside_px, 1),
    }


def component_features(
    split: str,
    file: str,
    component_id: int,
    component: np.ndarray,
    x0: int,
    y0: int,
    width: int,
    height: int,
    area: int,
    source_gray: np.ndarray,
    signed_delta: np.ndarray,
    abs_delta: np.ndarray,
    baseline_edit: np.ndarray,
    candidate_source_delta: np.ndarray,
    source_edge: np.ndarray,
    texture: np.ndarray,
    improvement: np.ndarray,
) -> dict[str, object]:
    gain = mean(improvement[component])
    hurt_ratio = float((improvement[component] < -2).mean()) if area else 0.0
    help_ratio = float((improvement[component] > 2).mean()) if area else 0.0
    over_delta = mean(candidate_source_delta[component]) - mean(baseline_edit[component])
    verdict = "accept" if gain > 0 and hurt_ratio < 0.25 else ("review" if gain > 0 else "reject")

    return {
        "split": split,
        "file": file,
        "component_id": component_id,
        "x": x0,
        "y": y0,
        "w": width,
        "h": height,
        "area": area,
        "fill_ratio": area / max(width * height, 1),
        "signed_delta_mean": mean(signed_delta[component]),
        "signed_delta_abs_mean": mean(abs_delta[component]),
        "brighten_ratio": float((signed_delta[component] > 0).mean()) if area else 0.0,
        "darken_ratio": float((signed_delta[component] < 0).mean()) if area else 0.0,
        "baseline_edit_mean": mean(baseline_edit[component]),
        "baseline_edit_p95": percentile(baseline_edit[component], 95),
        "candidate_source_delta_mean": mean(candidate_source_delta[component]),
        "source_gray_mean": mean(source_gray[component]),
        "source_gray_p25": percentile(source_gray[component], 25),
        "source_gray_p75": percentile(source_gray[component], 75),
        "source_edge_mean": mean(source_edge[component]),
        "source_edge_p95": percentile(source_edge[component], 95),
        "source_texture_mean": mean(texture[component]),
        "source_texture_p95": percentile(texture[component], 95),
        "component_gain": gain,
        "component_over_delta": over_delta,
        "component_help_ratio": help_ratio,
        "component_hurt_ratio": hurt_ratio,
        "component_verdict": verdict,
    }


def load_split_components(
    split: str,
    baseline_metrics: Path,
    candidate_metrics: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    baseline_rows = {row["file"]: row for row in read_rows(baseline_metrics)}
    candidate_rows = {row["file"]: row for row in read_rows(candidate_metrics)}
    component_rows: list[dict[str, object]] = []
    page_rows: list[dict[str, object]] = []

    for file in sorted(baseline_rows):
        baseline_row = baseline_rows[file]
        candidate_row = candidate_rows[file]
        source = read_bgr(baseline_row["image_path"])
        label = resize_like(read_bgr(label_path_for(baseline_row["image_path"])), source)
        baseline = resize_like(read_bgr(baseline_row["pred_path"]), source)
        candidate = resize_like(read_bgr(candidate_row["pred_path"]), source)

        source_gray = gray(source)
        label_gray = gray(label)
        baseline_gray = gray(baseline)
        candidate_gray = gray(candidate)
        signed_delta = candidate_gray.astype(np.int16) - baseline_gray.astype(np.int16)
        abs_delta = np.abs(signed_delta)
        baseline_edit = np.abs(baseline_gray.astype(np.int16) - source_gray.astype(np.int16))
        candidate_source_delta = np.abs(candidate_gray.astype(np.int16) - source_gray.astype(np.int16))
        source_edge_x = cv2.Sobel(source_gray.astype(np.uint8), cv2.CV_32F, 1, 0, ksize=3)
        source_edge_y = cv2.Sobel(source_gray.astype(np.uint8), cv2.CV_32F, 0, 1, ksize=3)
        source_edge = np.abs(source_edge_x) + np.abs(source_edge_y)
        local_mean = cv2.blur(source_gray.astype(np.float32), (15, 15))
        texture = np.abs(source_gray.astype(np.float32) - local_mean)
        baseline_error = np.abs(baseline_gray.astype(np.int16) - label_gray.astype(np.int16))
        candidate_error = np.abs(candidate_gray.astype(np.int16) - label_gray.astype(np.int16))
        improvement = baseline_error - candidate_error
        candidate_delta = np.abs(candidate_gray - baseline_gray)
        active = (
            (baseline_edit >= args.base_edit_threshold)
            & (candidate_delta >= args.candidate_delta_threshold)
        ).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
        component_ids: list[int] = []
        for component_id in range(1, count):
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < args.min_component_area:
                continue
            component = labels[y : y + h, x : x + w] == component_id
            component_ids.append(component_id)
            component_rows.append(
                component_features(
                    split,
                    file,
                    component_id,
                    component,
                    x,
                    y,
                    w,
                    h,
                    area,
                    source_gray[y : y + h, x : x + w],
                    signed_delta[y : y + h, x : x + w],
                    abs_delta[y : y + h, x : x + w],
                    baseline_edit[y : y + h, x : x + w],
                    candidate_source_delta[y : y + h, x : x + w],
                    source_edge[y : y + h, x : x + w],
                    texture[y : y + h, x : x + w],
                    improvement[y : y + h, x : x + w],
                )
            )

        page_rows.append({
            "split": split,
            "file": file,
            "image_path": baseline_row["image_path"],
            "baseline_pred_path": baseline_row["pred_path"],
            "candidate_pred_path": candidate_row["pred_path"],
            "components": len(component_ids),
            "baseline_residual_ratio": float(baseline_row["residual_ratio"]),
            "baseline_overerase_ratio": float(baseline_row["overerase_ratio"]),
            "candidate_residual_ratio": float(candidate_row["residual_ratio"]),
            "candidate_overerase_ratio": float(candidate_row["overerase_ratio"]),
        })
    return component_rows, page_rows


def numeric_features(rows: list[dict[str, object]]) -> list[str]:
    excluded = {
        "split",
        "file",
        "component_id",
        "component_gain",
        "component_over_delta",
        "component_help_ratio",
        "component_hurt_ratio",
        "component_verdict",
    }
    features: list[str] = []
    for key in rows[0]:
        if key in excluded:
            continue
        if all(isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key])) for row in rows):
            features.append(key)
    return sorted(features)


def quantile(sorted_values: list[float], q: float) -> float:
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def candidate_conditions(rows: list[dict[str, object]], features: list[str], quantiles: list[float]) -> list[Condition]:
    conditions: list[Condition] = []
    seen: set[tuple[str, str, float]] = set()
    for feature in features:
        values = sorted(float(row[feature]) for row in rows)
        for q in quantiles:
            threshold = quantile(values, q)
            for op in ("<=", ">="):
                key = (feature, op, threshold)
                if key in seen:
                    continue
                seen.add(key)
                conditions.append(Condition(feature, op, threshold))
    return conditions


def condition_flags(rows: list[dict[str, object]], conditions: tuple[Condition, ...]) -> list[bool]:
    return [all(condition.hit(row) for condition in conditions) for row in rows]


def component_summary(rows: list[dict[str, object]], flags: list[bool], prefix: str) -> dict[str, object]:
    kept = [row for row, flag in zip(rows, flags) if flag]
    rejects = [row for row in kept if row["component_verdict"] == "reject"]
    losses = [row for row in kept if float(row["component_gain"]) <= 0]
    return {
        f"{prefix}_components": len(kept),
        f"{prefix}_component_accept": sum(row["component_verdict"] == "accept" for row in kept),
        f"{prefix}_component_review": sum(row["component_verdict"] == "review" for row in kept),
        f"{prefix}_component_reject": len(rejects),
        f"{prefix}_component_losses": len(losses),
        f"{prefix}_component_gain": sum(float(row["component_gain"]) for row in kept) / max(len(rows), 1),
    }


def component_arrays(rows: list[dict[str, object]], features: list[str]) -> dict[str, np.ndarray]:
    arrays = {feature: np.asarray([float(row[feature]) for row in rows], dtype=np.float64) for feature in features}
    arrays["_gain"] = np.asarray([float(row["component_gain"]) for row in rows], dtype=np.float64)
    arrays["_accept"] = np.asarray([row["component_verdict"] == "accept" for row in rows], dtype=bool)
    arrays["_review"] = np.asarray([row["component_verdict"] == "review" for row in rows], dtype=bool)
    arrays["_reject"] = np.asarray([row["component_verdict"] == "reject" for row in rows], dtype=bool)
    return arrays


def condition_mask(arrays: dict[str, np.ndarray], conditions: tuple[Condition, ...], size: int) -> np.ndarray:
    mask = np.ones(size, dtype=bool)
    for condition in conditions:
        values = arrays[condition.feature]
        if condition.op == "<=":
            mask &= values <= condition.threshold
        elif condition.op == ">=":
            mask &= values >= condition.threshold
        else:
            raise ValueError(f"Unsupported operator: {condition.op}")
    return mask


def component_summary_from_mask(arrays: dict[str, np.ndarray], mask: np.ndarray, prefix: str) -> dict[str, object]:
    kept = int(mask.sum())
    gains = arrays["_gain"]
    rejects = int((mask & arrays["_reject"]).sum())
    return {
        f"{prefix}_components": kept,
        f"{prefix}_component_accept": int((mask & arrays["_accept"]).sum()),
        f"{prefix}_component_review": int((mask & arrays["_review"]).sum()),
        f"{prefix}_component_reject": rejects,
        f"{prefix}_component_reject_ratio": rejects / max(kept, 1),
        f"{prefix}_component_losses": int((mask & (gains <= 0)).sum()),
        f"{prefix}_component_gain": float(gains[mask].sum() / max(len(gains), 1)),
    }


def train_rule_allowed(summary: dict[str, object], args: argparse.Namespace) -> bool:
    selected = int(summary["train_components"])
    if selected < args.min_train_components:
        return False
    rejects = int(summary["train_component_reject"])
    if rejects > args.max_train_reject_components:
        return False
    reject_ratio = rejects / max(selected, 1)
    return reject_ratio <= args.max_train_reject_ratio


def page_summary(
    component_rows: list[dict[str, object]],
    page_rows: list[dict[str, object]],
    flags: list[bool],
    conditions: tuple[Condition, ...],
    args: argparse.Namespace,
    prefix: str,
) -> dict[str, object]:
    keep_by_file: dict[str, set[int]] = {}
    for row, flag in zip(component_rows, flags):
        if flag:
            keep_by_file.setdefault(str(row["file"]), set()).add(int(row["component_id"]))

    selected_pages = 0
    selector_residual = 0.0
    selector_overerase = 0.0
    baseline_residual = 0.0
    baseline_overerase = 0.0
    candidate_residual = 0.0
    candidate_overerase = 0.0
    for page in page_rows:
        source = read_bgr(str(page["image_path"]))
        label = resize_like(read_bgr(label_path_for(str(page["image_path"]))), source)
        baseline = resize_like(read_bgr(str(page["baseline_pred_path"])), source)
        candidate = resize_like(read_bgr(str(page["candidate_pred_path"])), source)
        source_gray = gray(source)
        baseline_gray = gray(baseline)
        candidate_gray = gray(candidate)
        active = (
            (np.abs(baseline_gray - source_gray) >= args.base_edit_threshold)
            & (np.abs(candidate_gray - baseline_gray) >= args.candidate_delta_threshold)
        ).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
        keep_mask = np.zeros(active.shape, dtype=bool)
        for component_id in keep_by_file.get(str(page["file"]), set()):
            if component_id < count and int(stats[component_id, cv2.CC_STAT_AREA]) >= args.min_component_area:
                keep_mask |= labels == component_id
        pred = baseline.copy()
        pred[keep_mask] = candidate[keep_mask]
        selected_pages += int(bool(keep_mask.any()))
        metrics = compute_metrics(source, label, pred, args.base_edit_threshold, args.eval_threshold)
        selector_residual += float(metrics["residual_ratio"])
        selector_overerase += float(metrics["overerase_ratio"])
        baseline_residual += float(page["baseline_residual_ratio"])
        baseline_overerase += float(page["baseline_overerase_ratio"])
        candidate_residual += float(page["candidate_residual_ratio"])
        candidate_overerase += float(page["candidate_overerase_ratio"])

    pages = max(len(page_rows), 1)
    return {
        f"{prefix}_pages": len(page_rows),
        f"{prefix}_selected_pages": selected_pages,
        f"{prefix}_page_coverage": selected_pages / pages,
        f"{prefix}_baseline_residual": baseline_residual / pages,
        f"{prefix}_candidate_residual": candidate_residual / pages,
        f"{prefix}_selector_residual": selector_residual / pages,
        f"{prefix}_residual_gain": (baseline_residual - selector_residual) / pages,
        f"{prefix}_baseline_overerase": baseline_overerase / pages,
        f"{prefix}_candidate_overerase": candidate_overerase / pages,
        f"{prefix}_selector_overerase": selector_overerase / pages,
        f"{prefix}_overerase_delta": (selector_overerase - baseline_overerase) / pages,
    }


def evaluate_rule(
    train_components: list[dict[str, object]],
    test_components: list[dict[str, object]],
    train_pages: list[dict[str, object]],
    test_pages: list[dict[str, object]],
    conditions: tuple[Condition, ...],
    args: argparse.Namespace,
    train_arrays: dict[str, np.ndarray] | None = None,
    test_arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    if train_arrays is None or test_arrays is None:
        train_flags = condition_flags(train_components, conditions)
        test_flags = condition_flags(test_components, conditions)
        train_component_summary = component_summary(train_components, train_flags, "train")
        test_component_summary = component_summary(test_components, test_flags, "test")
    else:
        train_mask = condition_mask(train_arrays, conditions, len(train_components))
        test_mask = condition_mask(test_arrays, conditions, len(test_components))
        train_flags = train_mask.tolist()
        test_flags = test_mask.tolist()
        train_component_summary = component_summary_from_mask(train_arrays, train_mask, "train")
        test_component_summary = component_summary_from_mask(test_arrays, test_mask, "test")
    return {
        **train_component_summary,
        **test_component_summary,
        **page_summary(train_components, train_pages, train_flags, conditions, args, "train"),
        **page_summary(test_components, test_pages, test_flags, conditions, args, "test"),
        "conditions": len(conditions),
        "rule": " AND ".join(condition.text() for condition in conditions),
    }


def main() -> None:
    args = parse_args()
    all_components: list[dict[str, object]] = []
    all_pages: list[dict[str, object]] = []
    for split_name, baseline_metrics, candidate_metrics in map(parse_split, args.split):
        components, pages = load_split_components(split_name, baseline_metrics, candidate_metrics, args)
        all_components.extend(components)
        all_pages.extend(pages)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "components.csv", all_components)
    write_csv(output_dir / "pages.csv", all_pages)
    train_splits = set(args.train_split)
    test_splits = set(args.test_split)
    train_components = [row for row in all_components if row["split"] in train_splits]
    test_components = [row for row in all_components if row["split"] in test_splits]
    train_pages = [row for row in all_pages if row["split"] in train_splits]
    test_pages = [row for row in all_pages if row["split"] in test_splits]
    if not train_components or not test_components:
        raise ValueError("Both train and test components are required")

    features = numeric_features(train_components)
    train_arrays = component_arrays(train_components, features)
    test_arrays = component_arrays(test_components, features)
    quantiles = [float(value) for value in args.quantiles.split(",") if value.strip()]
    candidates = candidate_conditions(train_components, features, quantiles)
    single_results: list[tuple[tuple[Condition, ...], dict[str, object]]] = []
    for condition in candidates:
        flags = condition_mask(train_arrays, (condition,), len(train_components))
        summary = component_summary_from_mask(train_arrays, flags, "train")
        if not train_rule_allowed(summary, args):
            continue
        single_results.append(((condition,), summary))
    single_results.sort(
        key=lambda item: (
            int(item[1]["train_component_reject"]),
            int(item[1]["train_component_losses"]),
            -int(item[1]["train_components"]),
            -float(item[1]["train_component_gain"]),
        )
    )
    condition_sets = [item[0] for item in single_results[: args.max_single_candidates]]
    if args.max_conditions >= 2:
        for left, right in itertools.combinations([item[0][0] for item in single_results[: args.max_single_candidates]], 2):
            condition_sets.append((left, right))

    rules: list[dict[str, object]] = []
    seen_rules: set[str] = set()
    for conditions in condition_sets:
        rule_text = " AND ".join(condition.text() for condition in conditions)
        if rule_text in seen_rules:
            continue
        seen_rules.add(rule_text)
        row = evaluate_rule(
            train_components,
            test_components,
            train_pages,
            test_pages,
            conditions,
            args,
            train_arrays,
            test_arrays,
        )
        if not train_rule_allowed(row, args):
            continue
        if int(row["test_selected_pages"]) < args.min_test_pages:
            continue
        rules.append(row)

    rules.sort(
        key=lambda row: (
            int(row["test_component_reject"]),
            int(row["test_component_losses"]),
            -float(row["test_residual_gain"]),
            float(row["test_overerase_delta"]),
            -int(row["test_selected_pages"]),
            int(row["conditions"]),
        )
    )
    write_csv(output_dir / "rules.csv", rules[: args.max_rules])
    print(
        f"components={len(all_components)} train={len(train_components)} test={len(test_components)} "
        f"pages={len(all_pages)} features={len(features)} rules={len(rules)}"
    )
    if rules:
        top = rules[0]
        print(
            "best "
            f"test_pages={top['test_selected_pages']}/{top['test_pages']} "
            f"test_component_reject={top['test_component_reject']} "
            f"test_residual_gain={top['test_residual_gain']} "
            f"test_overerase_delta={top['test_overerase_delta']} "
            f"rule={top['rule']}"
        )


if __name__ == "__main__":
    main()
