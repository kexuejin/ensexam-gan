#!/usr/bin/env python3
"""Calibrate simple page-level selectors from visual labels and page features.

This script consumes human-confirmed page labels, joins them to label-free
features, and mines conservative rules that avoid visually bad pages. It is
intentionally small and auditable: if there are too few confirmed labels it
reports the gap instead of producing a misleading selector.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path


POSITIVE_LABELS = {"clear_win", "slight_win"}
NEGATIVE_LABELS = {"slight_loss", "clear_loss"}
NEUTRAL_LABELS = {"noop"}
DEFAULT_EXCLUDE_FEATURES = {
    "split",
    "file",
    "gain",
    "over_delta",
    "win",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--features-csv", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", default="")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--min-labeled", type=int, default=40)
    parser.add_argument("--min-selected", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalized_split(value: str) -> str:
    return "" if value == "main" else value


def read_features(paths: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    features: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        for row in read_rows(Path(path)):
            features[(normalized_split(row.get("split", "")), row["file"])] = row
    return features


def label_class(label: str) -> str:
    if label in POSITIVE_LABELS:
        return "positive"
    if label in NEGATIVE_LABELS:
        return "negative"
    if label in NEUTRAL_LABELS:
        return "neutral"
    return "unlabeled"


def join_labels(
    labels: list[dict[str, str]],
    features_by_key: dict[tuple[str, str], dict[str, str]],
    label_column: str,
    candidate: str,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    joined: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []
    for label_row in labels:
        if candidate and label_row.get("candidate") != candidate:
            continue
        label = label_row.get(label_column, "").strip()
        cls = label_class(label)
        if cls == "unlabeled":
            continue
        feature = features_by_key.get((normalized_split(label_row.get("split", "")), label_row["file"]))
        if feature is None:
            missing.append(label_row)
            continue
        joined.append({
            **feature,
            "candidate": label_row.get("candidate", ""),
            "bucket": label_row.get("bucket", ""),
            "visual_label": label,
            "visual_class": cls,
            "is_positive": cls == "positive",
            "is_negative": cls == "negative",
        })
    return joined, missing


def fnum(row: dict[str, object], key: str) -> float:
    return float(row[key])


def threshold_candidates(values: list[float]) -> list[float]:
    if not values:
        return []
    sorted_values = sorted(values)
    thresholds = set()
    for percentile in range(0, 101, 5):
        index = round((len(sorted_values) - 1) * percentile / 100)
        thresholds.add(sorted_values[index])
    return sorted(thresholds)


def summarize(rows: list[dict[str, object]], selected: list[bool]) -> dict[str, object]:
    selected_rows = [row for row, flag in zip(rows, selected) if flag]
    positives = sum(bool(row["is_positive"]) for row in selected_rows)
    negatives = sum(bool(row["is_negative"]) for row in selected_rows)
    neutral = len(selected_rows) - positives - negatives
    split_names = sorted({str(row["split"]) for row in rows})
    out: dict[str, object] = {
        "selected": len(selected_rows),
        "positive": positives,
        "negative": negatives,
        "neutral": neutral,
        "precision": positives / len(selected_rows) if selected_rows else 0.0,
        "negative_rate": negatives / len(selected_rows) if selected_rows else 0.0,
        "positive_coverage": positives / max(1, sum(bool(row["is_positive"]) for row in rows)),
    }
    for split in split_names:
        split_selected = [row for row in selected_rows if str(row["split"]) == split]
        out[f"{split}_selected"] = len(split_selected)
        out[f"{split}_positive"] = sum(bool(row["is_positive"]) for row in split_selected)
        out[f"{split}_negative"] = sum(bool(row["is_negative"]) for row in split_selected)
    return out


def feature_names(rows: list[dict[str, object]]) -> list[str]:
    names = []
    for key, value in rows[0].items():
        if key in DEFAULT_EXCLUDE_FEATURES or key.startswith("visual_") or key.startswith("is_"):
            continue
        if key in {"candidate", "bucket"}:
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            continue
        names.append(key)
    return names


def mine_rules(rows: list[dict[str, object]], min_selected: int) -> list[dict[str, object]]:
    features = feature_names(rows)
    rules: list[dict[str, object]] = []

    def add_rule(selected: list[bool], rule: str) -> None:
        if sum(selected) < min_selected:
            return
        summary = summarize(rows, selected)
        if int(summary["negative"]) > 0:
            return
        rules.append({**summary, "rule": rule})

    for feature in features:
        values = [fnum(row, feature) for row in rows]
        for threshold in threshold_candidates(values):
            add_rule([fnum(row, feature) <= threshold for row in rows], f"{feature} <= {threshold:.12g}")
            add_rule([fnum(row, feature) >= threshold for row in rows], f"{feature} >= {threshold:.12g}")

    scored_features = []
    for feature in features:
        pos_values = [fnum(row, feature) for row in rows if bool(row["is_positive"])]
        neg_values = [fnum(row, feature) for row in rows if bool(row["is_negative"])]
        if not pos_values or not neg_values:
            continue
        spread = max(fnum(row, feature) for row in rows) - min(fnum(row, feature) for row in rows)
        score = abs(sum(pos_values) / len(pos_values) - sum(neg_values) / len(neg_values)) / max(spread, 1e-12)
        scored_features.append((score, feature))
    top_features = [feature for _score, feature in sorted(scored_features, reverse=True)[:8]]

    for left, right in itertools.combinations(top_features, 2):
        left_thresholds = threshold_candidates([fnum(row, left) for row in rows])
        right_thresholds = threshold_candidates([fnum(row, right) for row in rows])
        for left_op, left_threshold, right_op, right_threshold in itertools.product(
            ("<=", ">="),
            left_thresholds,
            ("<=", ">="),
            right_thresholds,
        ):
            selected = []
            for row in rows:
                left_pass = fnum(row, left) <= left_threshold if left_op == "<=" else fnum(row, left) >= left_threshold
                right_pass = fnum(row, right) <= right_threshold if right_op == "<=" else fnum(row, right) >= right_threshold
                selected.append(left_pass and right_pass)
            add_rule(selected, f"{left} {left_op} {left_threshold:.12g} AND {right} {right_op} {right_threshold:.12g}")

    return sorted(
        rules,
        key=lambda row: (
            float(row["positive_coverage"]),
            float(row["precision"]),
            int(row["positive"]),
            int(row["selected"]),
        ),
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    features = read_features(args.features_csv)
    joined, missing = join_labels(
        read_rows(Path(args.labels_csv)),
        features,
        label_column=args.label_column,
        candidate=args.candidate,
    )

    summary = {
        "labels_csv": args.labels_csv,
        "label_column": args.label_column,
        "candidate": args.candidate,
        "joined_labeled": len(joined),
        "missing_features": len(missing),
        "min_labeled": args.min_labeled,
        "status": "ready" if len(joined) >= args.min_labeled else "insufficient_labeled",
        "positive": sum(bool(row["is_positive"]) for row in joined),
        "negative": sum(bool(row["is_negative"]) for row in joined),
        "neutral": sum(row["visual_class"] == "neutral" for row in joined),
    }
    write_csv(output_dir / "summary.csv", [summary])
    write_csv(output_dir / "joined_labels_features.csv", joined)
    write_csv(output_dir / "missing_features.csv", missing)

    if len(joined) < args.min_labeled:
        print(
            "status=insufficient_labeled "
            f"joined={len(joined)} min_labeled={args.min_labeled} missing_features={len(missing)}"
        )
        print(f"summary={output_dir / 'summary.csv'}")
        return

    rules = mine_rules(joined, args.min_selected)
    write_csv(output_dir / "top_rules.csv", rules[: args.top_n])
    print(f"status=ready joined={len(joined)} rules={len(rules)}")
    if rules:
        best = rules[0]
        print(
            "best_rule "
            f"selected={best['selected']} positive={best['positive']} negative={best['negative']} "
            f"precision={float(best['precision']):.3f} rule={best['rule']}"
        )
    print(f"top_rules={output_dir / 'top_rules.csv'}")


if __name__ == "__main__":
    main()
