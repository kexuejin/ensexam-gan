#!/usr/bin/env python3
"""Train a lightweight label-free page selector ranker.

This script is intentionally dependency-light: it uses NumPy only, so it can run
in the project Torch/MPS environment without scikit-learn. Target-derived local
proxy verdicts and metric outcomes are used only for offline supervision and
validation, never as inference-time features.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


EXCLUDED_FEATURES = {
    "file",
    "split",
    "gain",
    "over_delta",
    "win",
    "safe_win",
}

EXCLUDED_SUBSTRINGS = (
    "help",
    "hurt",
    "label",
    "metric",
    "oracle",
    "residual_gain",
    "target",
    "verdict",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-comparison-csv", required=True)
    parser.add_argument(
        "--feature-csv",
        action="append",
        required=True,
        metavar="SPLIT:CSV",
        help="May be repeated. SPLIT overrides any split column inside the CSV.",
    )
    parser.add_argument("--train-split", action="append", required=True)
    parser.add_argument("--test-split", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--max-rules", type=int, default=50)
    parser.add_argument("--min-train-selected", type=int, default=5)
    parser.add_argument("--min-test-selected", type=int, default=1)
    parser.add_argument("--positive-mode", choices=("accept", "safe"), default="safe")
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


def parse_feature_csv(value: str) -> tuple[str, Path]:
    if ":" not in value:
        return "", Path(value)
    split, path = value.split(":", 1)
    if not split or not path:
        raise ValueError(f"Invalid --feature-csv {value!r}; expected SPLIT:CSV")
    return split, Path(path)


def load_feature_rows(feature_specs: list[str]) -> tuple[list[dict[str, str]], set[str]]:
    rows: list[dict[str, str]] = []
    fieldnames: set[str] = set()
    for split, path in map(parse_feature_csv, feature_specs):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Feature CSV has no header: {path}")
            fieldnames.update(reader.fieldnames)
            for row in reader:
                row = dict(row)
                if split:
                    row["split"] = split
                elif not row.get("split"):
                    raise ValueError(f"Feature CSV requires split when no SPLIT prefix is used: {path}")
                rows.append(row)
    return rows, fieldnames


def is_number(value: str) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def label_free_features(rows: list[dict[str, str]], allowed_keys: set[str]) -> list[str]:
    features: list[str] = []
    for key in rows[0]:
        if key not in allowed_keys:
            continue
        if key in EXCLUDED_FEATURES:
            continue
        if any(token in key for token in EXCLUDED_SUBSTRINGS):
            continue
        if all(is_number(row.get(key, "")) for row in rows):
            features.append(key)
    return sorted(features)


def merge_rows(
    local_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    features_by_key = {(row["split"], row["file"]): row for row in feature_rows}
    merged: list[dict[str, str]] = []
    missing: list[tuple[str, str]] = []
    for local in local_rows:
        key = (local["split"], local["file"])
        feature = features_by_key.get(key)
        if feature is None:
            missing.append(key)
            continue
        merged.append({**local, **feature, "split": local["split"], "file": local["file"]})
    if missing:
        sample = ", ".join(f"{split}/{file}" for split, file in missing[:5])
        raise KeyError(f"Missing feature rows for {len(missing)} local rows; sample: {sample}")
    return merged


def is_safe_positive(row: dict[str, str], positive_mode: str) -> bool:
    metric_safe = float(row["gain"]) > 0.0 and float(row["over_delta"]) <= 0.0
    if positive_mode == "accept":
        return metric_safe and row["local_verdict"] == "accept"
    return metric_safe and row["local_verdict"] != "reject"


def is_negative(row: dict[str, str], positive_mode: str) -> bool:
    if is_safe_positive(row, positive_mode):
        return False
    return row["local_verdict"] == "reject" or float(row["gain"]) <= 0.0 or float(row["over_delta"]) > 0.0


def build_matrix(rows: list[dict[str, str]], features: list[str]) -> np.ndarray:
    matrix = np.asarray([[float(row[feature]) for feature in features] for row in rows], dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(matrix, -1e6, 1e6)


def standardize(train_x: np.ndarray, all_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.clip(np.nan_to_num(train_x, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    all_x = np.clip(np.nan_to_num(all_x, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
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
    for _epoch in range(epochs):
        weights = np.clip(np.nan_to_num(weights, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
        bias = float(np.clip(np.nan_to_num(bias, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0))
        logits = np.sum(x * weights.reshape(1, -1), axis=1) + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        error = (probs - y) * sample_weight
        grad_w = np.sum(x * error.reshape(-1, 1), axis=0) / len(y) + l2 * weights
        grad_w = np.clip(np.nan_to_num(grad_w, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
        grad_b = float(error.mean())
        grad_norm = float(np.linalg.norm(grad_w))
        if grad_norm > 10.0:
            grad_w = grad_w * (10.0 / grad_norm)
        grad_b = float(np.clip(grad_b, -10.0, 10.0))
        weights -= lr * grad_w
        bias -= lr * grad_b
        weights = np.clip(np.nan_to_num(weights, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
        bias = float(np.clip(np.nan_to_num(bias, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0))
    return weights, bias


def score_rows(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    weights = np.clip(np.nan_to_num(weights, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    bias = float(np.clip(np.nan_to_num(bias, nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0))
    logits = np.sum(x * weights.reshape(1, -1), axis=1) + bias
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))


def summarize(rows: list[dict[str, str]], selected: np.ndarray, prefix: str) -> dict[str, object]:
    picked = [row for row, flag in zip(rows, selected) if bool(flag)]
    metric_losses = [
        row for row in picked
        if not (float(row["gain"]) > 0.0 and float(row["over_delta"]) <= 0.0)
    ]
    verdicts = {"accept": 0, "review": 0, "reject": 0}
    split_counts: dict[str, int] = {}
    split_rejects: dict[str, int] = {}
    for row in picked:
        verdicts[row["local_verdict"]] = verdicts.get(row["local_verdict"], 0) + 1
        split = row["split"]
        split_counts[split] = split_counts.get(split, 0) + 1
        if row["local_verdict"] == "reject":
            split_rejects[split] = split_rejects.get(split, 0) + 1
    return {
        f"{prefix}_selected": len(picked),
        f"{prefix}_coverage": len(picked) / len(rows) if rows else 0.0,
        f"{prefix}_metric_losses": len(metric_losses),
        f"{prefix}_accept": verdicts.get("accept", 0),
        f"{prefix}_review": verdicts.get("review", 0),
        f"{prefix}_reject": verdicts.get("reject", 0),
        f"{prefix}_residual_gain": sum(float(row["gain"]) for row in picked) / len(rows) if rows else 0.0,
        f"{prefix}_overerase_delta": sum(float(row["over_delta"]) for row in picked) / len(rows) if rows else 0.0,
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
    thresholds = sorted(set(float(value) for value in np.quantile(train_scores, np.linspace(0, 1, 101))))
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        train_selected = train_scores >= threshold
        test_selected = test_scores >= threshold
        if int(train_selected.sum()) < min_train_selected:
            continue
        if int(test_selected.sum()) < min_test_selected:
            continue
        row = {
            "threshold": threshold,
            **summarize(train_rows, train_selected, "train"),
            **summarize(test_rows, test_selected, "test"),
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -int(row["test_reject"]),
            -int(row["test_metric_losses"]),
            float(row["test_residual_gain"]),
            int(row["test_selected"]),
            float(row["train_residual_gain"]),
        ),
        reverse=True,
    )
    return rows[:max_rows]


def prediction_rows(rows: list[dict[str, str]], scores: np.ndarray, labels: np.ndarray, subset: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row, score, label in zip(rows, scores, labels):
        out.append({
            "subset": subset,
            "split": row["split"],
            "file": row["file"],
            "score": f"{float(score):.12f}",
            "label": int(label),
            "local_verdict": row["local_verdict"],
            "gain": row["gain"],
            "over_delta": row["over_delta"],
        })
    out.sort(key=lambda item: float(item["score"]), reverse=True)
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    local_rows = read_rows(Path(args.local_comparison_csv))
    feature_rows, feature_fields = load_feature_rows(args.feature_csv)
    rows = merge_rows(local_rows, feature_rows)
    features = label_free_features(rows, feature_fields)

    train_splits = set(args.train_split)
    test_splits = set(args.test_split)
    train_rows = [row for row in rows if row["split"] in train_splits]
    test_rows = [row for row in rows if row["split"] in test_splits]
    if not train_rows or not test_rows:
        raise ValueError("Both train and test rows are required")

    train_y = np.asarray([1.0 if is_safe_positive(row, args.positive_mode) else 0.0 for row in train_rows])
    test_y = np.asarray([1.0 if is_safe_positive(row, args.positive_mode) else 0.0 for row in test_rows])
    if train_y.sum() == 0 or train_y.sum() == len(train_y):
        raise ValueError("Training labels need both positive and negative examples")

    all_x = build_matrix(train_rows + test_rows, features)
    train_x_raw = all_x[: len(train_rows)]
    all_x, mean, std = standardize(train_x_raw, all_x)
    train_x = all_x[: len(train_rows)]
    test_x = all_x[len(train_rows) :]

    weights, bias = train_logistic(train_x, train_y, args.epochs, args.lr, args.l2)
    train_scores = score_rows(train_x, weights, bias)
    test_scores = score_rows(test_x, weights, bias)

    summaries = threshold_summaries(
        train_rows,
        test_rows,
        train_scores,
        test_scores,
        args.min_train_selected,
        args.min_test_selected,
        args.max_rules,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "threshold_summary.csv", summaries)
    write_csv(
        output_dir / "predictions.csv",
        [
            *prediction_rows(train_rows, train_scores, train_y, "train"),
            *prediction_rows(test_rows, test_scores, test_y, "test"),
        ],
        fields=["subset", "split", "file", "score", "label", "local_verdict", "gain", "over_delta"],
    )
    coefficient_rows = [
        {
            "feature": feature,
            "weight": f"{weight:.12f}",
            "mean": f"{feature_mean:.12f}",
            "std": f"{feature_std:.12f}",
        }
        for feature, weight, feature_mean, feature_std in sorted(
            zip(features, weights, mean, std),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
    ]
    write_csv(output_dir / "coefficients.csv", coefficient_rows, fields=["feature", "weight", "mean", "std"])
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
        "best_rules": len(summaries),
        "best_test_selected": summaries[0]["test_selected"] if summaries else 0,
        "best_test_reject": summaries[0]["test_reject"] if summaries else 0,
        "best_test_metric_losses": summaries[0]["test_metric_losses"] if summaries else 0,
        "best_test_residual_gain": summaries[0]["test_residual_gain"] if summaries else 0.0,
    }
    write_csv(output_dir / "summary.csv", [summary], fields=list(summary))
    print(
        f"rows={len(rows)} train={len(train_rows)} test={len(test_rows)} "
        f"features={len(features)} train_pos={int(train_y.sum())} test_pos={int(test_y.sum())} "
        f"rules={len(summaries)} output_dir={output_dir}"
    )
    for row in summaries[:10]:
        print(row)


if __name__ == "__main__":
    main()
