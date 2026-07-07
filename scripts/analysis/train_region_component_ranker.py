#!/usr/bin/env python3
"""Train a lightweight label-free ranker for local edit components.

This is an offline probe for a future learned region selector. Component
verdicts and gains are weak supervision from target comparisons; model inputs
are restricted to source/baseline/candidate component features available at
inference time.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


EXCLUDED_FEATURES = {
    "split",
    "file",
    "component_id",
    "x",
    "y",
    "component_gain",
    "component_over_delta",
    "component_help_ratio",
    "component_hurt_ratio",
    "component_verdict",
}

EXCLUDED_SUBSTRINGS = (
    "help",
    "hurt",
    "label",
    "target",
    "verdict",
    "gain",
    "loss",
    "oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-csv", required=True)
    parser.add_argument("--train-split", action="append", required=True)
    parser.add_argument("--test-split", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--positive-mode", choices=("accept", "safe"), default="accept")
    parser.add_argument("--min-train-selected", type=int, default=500)
    parser.add_argument("--min-test-selected", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=80)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def is_number(value: str) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def label_free_features(rows: list[dict[str, str]]) -> list[str]:
    features: list[str] = []
    for key in rows[0]:
        if key in EXCLUDED_FEATURES:
            continue
        if any(token in key for token in EXCLUDED_SUBSTRINGS):
            continue
        if all(is_number(row.get(key, "")) for row in rows):
            features.append(key)
    return sorted(features)


def positive_label(row: dict[str, str], mode: str) -> bool:
    gain = float(row["component_gain"])
    hurt_ratio = float(row["component_hurt_ratio"])
    if mode == "accept":
        return row["component_verdict"] == "accept" and gain > 0.0 and hurt_ratio <= 0.10
    return row["component_verdict"] != "reject" and gain > 0.0 and hurt_ratio <= 0.25


def build_matrix(rows: list[dict[str, str]], features: list[str]) -> np.ndarray:
    matrix = np.asarray([[float(row[feature]) for feature in features] for row in rows], dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(matrix, -1e6, 1e6)


def standardize(train_x: np.ndarray, all_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-9] = 1.0
    standardized = (all_x - mean) / std
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(standardized, -10.0, 10.0), mean, std


def train_logistic(x: np.ndarray, y: np.ndarray, epochs: int, lr: float, l2: float) -> tuple[np.ndarray, float]:
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    sample_weight = np.where(y > 0.5, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    for _ in range(epochs):
        logits = np.sum(x * weights.reshape(1, -1), axis=1) + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        error = (probs - y) * sample_weight
        grad_w = np.sum(x * error.reshape(-1, 1), axis=0) / len(y) + l2 * weights
        grad_b = float(error.mean())
        grad_norm = float(np.linalg.norm(grad_w))
        if grad_norm > 10.0:
            grad_w *= 10.0 / grad_norm
        weights -= lr * np.clip(grad_w, -10.0, 10.0)
        bias -= lr * max(min(grad_b, 10.0), -10.0)
        weights = np.clip(np.nan_to_num(weights, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
        bias = float(np.clip(np.nan_to_num(bias, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0))
    return weights, bias


def score_rows(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    logits = np.sum(x * weights.reshape(1, -1), axis=1) + bias
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def summarize(rows: list[dict[str, str]], selected: np.ndarray, prefix: str) -> dict[str, object]:
    picked = [row for row, flag in zip(rows, selected) if bool(flag)]
    verdicts = {"accept": 0, "review": 0, "reject": 0}
    split_counts: dict[str, int] = {}
    split_rejects: dict[str, int] = {}
    for row in picked:
        verdicts[row["component_verdict"]] = verdicts.get(row["component_verdict"], 0) + 1
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        if row["component_verdict"] == "reject":
            split_rejects[row["split"]] = split_rejects.get(row["split"], 0) + 1
    gain = sum(float(row["component_gain"]) for row in picked)
    return {
        f"{prefix}_selected": len(picked),
        f"{prefix}_coverage": len(picked) / len(rows) if rows else 0.0,
        f"{prefix}_accept": verdicts.get("accept", 0),
        f"{prefix}_review": verdicts.get("review", 0),
        f"{prefix}_reject": verdicts.get("reject", 0),
        f"{prefix}_reject_ratio": verdicts.get("reject", 0) / max(len(picked), 1),
        f"{prefix}_mean_gain_on_selected": gain / max(len(picked), 1),
        f"{prefix}_gain_per_component": gain / len(rows) if rows else 0.0,
        **{f"{prefix}_{split}_selected": count for split, count in sorted(split_counts.items())},
        **{f"{prefix}_{split}_reject": count for split, count in sorted(split_rejects.items())},
    }


def threshold_summaries(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    min_train_selected: int,
    min_test_selected: int,
    max_rows: int,
) -> list[dict[str, object]]:
    thresholds = sorted(set(float(value) for value in np.quantile(train_scores, np.linspace(0, 1, 151))))
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        train_selected = train_scores >= threshold
        test_selected = test_scores >= threshold
        if int(train_selected.sum()) < min_train_selected:
            continue
        if int(test_selected.sum()) < min_test_selected:
            continue
        rows.append({
            "threshold": threshold,
            **summarize(train_rows, train_selected, "train"),
            **summarize(test_rows, test_selected, "test"),
        })
    rows.sort(
        key=lambda row: (
            float(row["test_reject_ratio"]),
            -float(row["test_gain_per_component"]),
            -int(row["test_selected"]),
            float(row["train_reject_ratio"]),
        )
    )
    return rows[:max_rows]


def prediction_rows(rows: list[dict[str, str]], scores: np.ndarray, labels: np.ndarray, subset: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row, score, label in zip(rows, scores, labels):
        out.append({
            "subset": subset,
            "split": row["split"],
            "file": row["file"],
            "component_id": row["component_id"],
            "score": float(score),
            "label": int(label),
            "component_verdict": row["component_verdict"],
            "component_gain": float(row["component_gain"]),
            "component_hurt_ratio": float(row["component_hurt_ratio"]),
        })
    return out


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.components_csv))
    train_splits = set(args.train_split)
    test_splits = set(args.test_split)
    train_rows = [row for row in rows if row["split"] in train_splits]
    test_rows = [row for row in rows if row["split"] in test_splits]
    if not train_rows or not test_rows:
        raise ValueError("Both train and test rows are required")

    features = label_free_features(rows)
    all_rows = train_rows + test_rows
    all_x = build_matrix(all_rows, features)
    train_x_raw = all_x[: len(train_rows)]
    x, mean, std = standardize(train_x_raw, all_x)
    train_x = x[: len(train_rows)]
    test_x = x[len(train_rows) :]
    train_y = np.asarray([positive_label(row, args.positive_mode) for row in train_rows], dtype=np.float64)
    test_y = np.asarray([positive_label(row, args.positive_mode) for row in test_rows], dtype=np.float64)

    weights, bias = train_logistic(train_x, train_y, args.epochs, args.lr, args.l2)
    train_scores = score_rows(train_x, weights, bias)
    test_scores = score_rows(test_x, weights, bias)
    thresholds = threshold_summaries(
        train_rows,
        test_rows,
        train_scores,
        test_scores,
        args.min_train_selected,
        args.min_test_selected,
        args.max_rows,
    )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "threshold_summary.csv", thresholds)
    write_csv(
        output_dir / "predictions.csv",
        prediction_rows(train_rows, train_scores, train_y, "train")
        + prediction_rows(test_rows, test_scores, test_y, "test"),
    )
    coefficient_rows = [
        {
            "feature": feature,
            "weight": float(weight),
            "mean": float(feature_mean),
            "std": float(feature_std),
        }
        for feature, weight, feature_mean, feature_std in zip(features, weights, mean, std)
    ]
    coefficient_rows.sort(key=lambda row: abs(float(row["weight"])), reverse=True)
    write_csv(output_dir / "coefficients.csv", coefficient_rows, ["feature", "weight", "mean", "std"])

    best = thresholds[0] if thresholds else {}
    summary = {
        "rows": len(rows),
        "features": len(features),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_positive": int(train_y.sum()),
        "test_positive": int(test_y.sum()),
        "positive_mode": args.positive_mode,
        "epochs": args.epochs,
        "lr": args.lr,
        "l2": args.l2,
        "best_test_selected": best.get("test_selected", 0),
        "best_test_reject": best.get("test_reject", 0),
        "best_test_reject_ratio": best.get("test_reject_ratio", 0.0),
        "best_test_gain_per_component": best.get("test_gain_per_component", 0.0),
    }
    write_csv(output_dir / "summary.csv", [summary], list(summary))
    print(
        f"rows={len(rows)} features={len(features)} train={len(train_rows)} test={len(test_rows)} "
        f"train_positive={int(train_y.sum())} test_positive={int(test_y.sum())}"
    )
    if thresholds:
        print(
            "best "
            f"test_selected={best['test_selected']} "
            f"test_accept={best['test_accept']} "
            f"test_review={best['test_review']} "
            f"test_reject={best['test_reject']} "
            f"test_reject_ratio={float(best['test_reject_ratio']):.4f} "
            f"test_gain_per_component={float(best['test_gain_per_component']):.6f}"
        )
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
