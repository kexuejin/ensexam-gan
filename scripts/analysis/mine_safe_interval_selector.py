#!/usr/bin/env python3
"""Mine offline interval-box selector headroom from replay page choices.

This is an oracle-style analysis tool: it uses target-derived residual/overerase
metrics to estimate whether a candidate family has enough separable safe pages
to justify a real label-free selector. Do not promote mined boxes directly as a
product rule without independent validation.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_FEATURES = [
    "copy_mask_cov8",
    "primary_edit_px",
    "primary_p95_edit_delta",
    "second_stage_gate_ratio",
]


def fnum(row: dict[str, str], key: str) -> float:
    return float(row[key])


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def feature_ranges(rows: list[dict[str, str]], features: list[str]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for feature in features:
        values = [fnum(row, feature) for row in rows]
        ranges[feature] = (min(values), max(values))
    return ranges


def normalized_distance(
    a: dict[str, str],
    b: dict[str, str],
    features: list[str],
    ranges: dict[str, tuple[float, float]],
) -> float:
    total = 0.0
    for feature in features:
        lo, hi = ranges[feature]
        scale = max(hi - lo, 1e-12)
        delta = (fnum(a, feature) - fnum(b, feature)) / scale
        total += delta * delta
    return math.sqrt(total)


def make_box(rows: list[dict[str, str]], features: list[str]) -> dict[str, tuple[float, float]]:
    box: dict[str, tuple[float, float]] = {}
    for feature in features:
        values = [fnum(row, feature) for row in rows]
        box[feature] = (min(values), max(values))
    return box


def in_box(row: dict[str, str], box: dict[str, tuple[float, float]]) -> bool:
    for feature, (lo, hi) in box.items():
        value = fnum(row, feature)
        if value < lo or value > hi:
            return False
    return True


def selected_by_boxes(row: dict[str, str], boxes: list[dict[str, tuple[float, float]]]) -> bool:
    return any(in_box(row, box) for box in boxes)


def summarize_selection(
    rows: list[dict[str, str]],
    selected: list[bool],
    max_overerase_regret: float,
) -> dict[str, object]:
    split_names = sorted({row["split"] for row in rows})
    out: dict[str, object] = {}
    total_selected = 0
    total_residual_gain = 0.0
    safe_all = True
    max_split_overerase_regret = -1e9
    safe_positive_selected = 0
    unsafe_selected = 0
    for split in split_names:
        split_pairs = [(row, flag) for row, flag in zip(rows, selected) if row["split"] == split]
        baseline_residual = sum(fnum(row, "baseline_residual_ratio") for row, _ in split_pairs) / len(split_pairs)
        baseline_overerase = sum(fnum(row, "baseline_overerase_ratio") for row, _ in split_pairs) / len(split_pairs)
        residual = sum(
            fnum(row, "candidate_residual_ratio") if flag else fnum(row, "baseline_residual_ratio")
            for row, flag in split_pairs
        ) / len(split_pairs)
        overerase = sum(
            fnum(row, "candidate_overerase_ratio") if flag else fnum(row, "baseline_overerase_ratio")
            for row, flag in split_pairs
        ) / len(split_pairs)
        selected_count = sum(1 for _row, flag in split_pairs if flag)
        residual_gain = baseline_residual - residual
        overerase_regret = overerase - baseline_overerase
        out[f"{split}_selected"] = selected_count
        out[f"{split}_residual_gain"] = residual_gain
        out[f"{split}_overerase_regret"] = overerase_regret
        total_selected += selected_count
        total_residual_gain += residual_gain
        max_split_overerase_regret = max(max_split_overerase_regret, overerase_regret)
        safe_all = safe_all and overerase_regret <= max_overerase_regret

    for row, flag in zip(rows, selected):
        if not flag:
            continue
        is_positive = fnum(row, "residual_gain") > 0
        is_non_overerase = fnum(row, "overerase_regret") <= 0
        if is_positive and is_non_overerase:
            safe_positive_selected += 1
        if not is_positive or not is_non_overerase:
            unsafe_selected += 1

    out["total_selected"] = total_selected
    out["safe_positive_selected"] = safe_positive_selected
    out["unsafe_selected"] = unsafe_selected
    out["total_residual_gain"] = total_residual_gain
    out["max_split_overerase_regret"] = max_split_overerase_regret
    out["safe_all_splits"] = int(safe_all)
    return out


def candidate_boxes(
    rows: list[dict[str, str]],
    features: list[str],
    max_box_size: int,
    min_box_selected: int,
) -> list[dict[str, object]]:
    ranges = feature_ranges(rows, features)
    safe_positive_rows = [
        row for row in rows
        if fnum(row, "residual_gain") > 0 and fnum(row, "overerase_regret") <= 0
    ]
    boxes: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, float, float], ...]] = set()
    for seed in safe_positive_rows:
        nearest = sorted(rows, key=lambda row: normalized_distance(seed, row, features, ranges))
        for size in range(min_box_selected, max_box_size + 1):
            group = nearest[:size]
            box = make_box(group, features)
            key = tuple((feature, box[feature][0], box[feature][1]) for feature in features)
            if key in seen:
                continue
            seen.add(key)
            selected = [in_box(row, box) for row in rows]
            if sum(selected) < min_box_selected:
                continue
            summary = summarize_selection(rows, selected, max_overerase_regret=0.0)
            boxes.append({
                "box": box,
                "seed_split": seed["split"],
                "seed_file": seed["file"],
                **summary,
            })
    boxes.sort(
        key=lambda item: (
            int(item["safe_all_splits"]),
            float(item["total_residual_gain"]),
            int(item["safe_positive_selected"]),
            -int(item["unsafe_selected"]),
        ),
        reverse=True,
    )
    return boxes


def greedy_select_boxes(
    rows: list[dict[str, str]],
    candidate_box_rows: list[dict[str, object]],
    max_boxes: int,
    max_overerase_regret: float,
) -> tuple[list[dict[str, object]], dict[str, object], list[bool]]:
    chosen: list[dict[str, object]] = []
    selected = [False] * len(rows)
    current_summary = summarize_selection(rows, selected, max_overerase_regret)
    remaining = list(candidate_box_rows)
    for _ in range(max_boxes):
        best: tuple[float, int, dict[str, object], dict[str, object], list[bool]] | None = None
        for item in remaining:
            box = item["box"]
            assert isinstance(box, dict)
            trial_selected = [
                flag or in_box(row, box)  # type: ignore[arg-type]
                for row, flag in zip(rows, selected)
            ]
            summary = summarize_selection(rows, trial_selected, max_overerase_regret)
            if not int(summary["safe_all_splits"]):
                continue
            gain_delta = float(summary["total_residual_gain"]) - float(current_summary["total_residual_gain"])
            selected_delta = int(summary["safe_positive_selected"]) - int(current_summary["safe_positive_selected"])
            if gain_delta <= 0 and selected_delta <= 0:
                continue
            rank = (gain_delta, selected_delta, item)
            if best is None or rank[:2] > best[:2]:
                best = (gain_delta, selected_delta, item, summary, trial_selected)
        if best is None:
            break
        _gain_delta, _selected_delta, item, current_summary, selected = best
        chosen.append(item)
        remaining = [candidate for candidate in remaining if candidate is not item]
    return chosen, current_summary, selected


def box_to_row(box: dict[str, tuple[float, float]], features: list[str]) -> dict[str, object]:
    row: dict[str, object] = {}
    for feature in features:
        lo, hi = box[feature]
        row[f"min_{feature}"] = lo
        row[f"max_{feature}"] = hi
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-choices", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--max-box-size", type=int, default=8)
    parser.add_argument("--min-box-selected", type=int, default=2)
    parser.add_argument("--max-boxes", type=int, default=8)
    parser.add_argument("--max-overerase-regret", type=float, default=0.0)
    parser.add_argument("--top-candidate-boxes", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.page_choices))
    features = [feature.strip() for feature in args.features.split(",") if feature.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    boxes = candidate_boxes(
        rows,
        features,
        max_box_size=args.max_box_size,
        min_box_selected=args.min_box_selected,
    )
    chosen, summary, selected = greedy_select_boxes(
        rows,
        boxes,
        max_boxes=args.max_boxes,
        max_overerase_regret=args.max_overerase_regret,
    )

    candidate_path = output_dir / "candidate_boxes.csv"
    candidate_fields = [
        "rank",
        "seed_split",
        "seed_file",
        *[field for feature in features for field in (f"min_{feature}", f"max_{feature}")],
        "total_selected",
        "safe_positive_selected",
        "unsafe_selected",
        "total_residual_gain",
        "max_split_overerase_regret",
        "safe_all_splits",
    ]
    with candidate_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields)
        writer.writeheader()
        for rank, item in enumerate(boxes[: args.top_candidate_boxes], start=1):
            box = item["box"]
            assert isinstance(box, dict)
            writer.writerow({
                "rank": rank,
                "seed_split": item["seed_split"],
                "seed_file": item["seed_file"],
                **box_to_row(box, features),  # type: ignore[arg-type]
                "total_selected": item["total_selected"],
                "safe_positive_selected": item["safe_positive_selected"],
                "unsafe_selected": item["unsafe_selected"],
                "total_residual_gain": item["total_residual_gain"],
                "max_split_overerase_regret": item["max_split_overerase_regret"],
                "safe_all_splits": item["safe_all_splits"],
            })

    selected_path = output_dir / "selected_pages.csv"
    selected_fields = [
        "split",
        "file",
        "selected",
        "residual_gain",
        "overerase_regret",
        *features,
    ]
    with selected_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=selected_fields)
        writer.writeheader()
        for row, flag in zip(rows, selected):
            if not flag:
                continue
            writer.writerow({
                "split": row["split"],
                "file": row["file"],
                "selected": int(flag),
                "residual_gain": row["residual_gain"],
                "overerase_regret": row["overerase_regret"],
                **{feature: row[feature] for feature in features},
            })

    chosen_path = output_dir / "chosen_boxes.csv"
    chosen_fields = [
        "rank",
        "seed_split",
        "seed_file",
        *[field for feature in features for field in (f"min_{feature}", f"max_{feature}")],
    ]
    with chosen_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=chosen_fields)
        writer.writeheader()
        for rank, item in enumerate(chosen, start=1):
            box = item["box"]
            assert isinstance(box, dict)
            writer.writerow({
                "rank": rank,
                "seed_split": item["seed_split"],
                "seed_file": item["seed_file"],
                **box_to_row(box, features),  # type: ignore[arg-type]
            })

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"rows={len(rows)} candidate_boxes={len(boxes)} chosen_boxes={len(chosen)}")
    print(
        f"selected={summary['total_selected']} "
        f"safe_positive={summary['safe_positive_selected']} "
        f"unsafe={summary['unsafe_selected']} "
        f"total_residual_gain={float(summary['total_residual_gain']):.12f} "
        f"max_overerase_regret={float(summary['max_split_overerase_regret']):.12f}"
    )
    print(f"summary: {summary_path}")
    print(f"chosen_boxes: {chosen_path}")
    print(f"selected_pages: {selected_path}")


if __name__ == "__main__":
    main()
