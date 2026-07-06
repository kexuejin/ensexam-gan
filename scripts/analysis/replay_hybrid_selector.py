#!/usr/bin/env python3
"""Replay label-free hybrid selector rules across one or more evaluation splits.

The hybrid gate runner writes final metrics for the selected output, but
threshold tuning needs both choices for every page. This script recomputes
candidate-page metrics from saved ``candidate/`` images, joins them with the
baseline metrics, then sweeps label-free feature thresholds.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVAL_SCRIPT_DIR = ROOT / "scripts" / "eval"
if str(EVAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPT_DIR))

from eval_hardcase_worst_pages import (  # noqa: E402
    compute_residual_metrics,
    ensure_same_size,
    label_path_for,
    read_bgr,
)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    baseline_metrics: Path
    gate_metrics: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME:BASELINE_METRICS:GATE_METRICS",
        help="Evaluation split definition. May be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-subdir", default="candidate")
    parser.add_argument("--change-threshold", type=int, default=12)
    parser.add_argument("--eval-threshold", type=int, default=12)
    parser.add_argument(
        "--max-overerase-regret",
        type=float,
        default=0.0,
        help="Allowed per-split overerase increase versus baseline.",
    )
    parser.add_argument(
        "--min-selected-total",
        type=int,
        default=1,
        help="Discard rules selecting fewer pages across all splits.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of ranked rules to write to top_rules.csv.",
    )
    parser.add_argument(
        "--max-thresholds-per-feature",
        type=int,
        default=28,
        help="Cap unique threshold candidates per feature to keep sweeps fast.",
    )
    parser.add_argument(
        "--pin-min-copy-mask-cov8",
        action="append",
        type=float,
        default=[],
        help="Additional exact min copy-mask coverage threshold to include.",
    )
    parser.add_argument(
        "--pin-max-primary-edit-px",
        action="append",
        type=float,
        default=[],
        help="Additional exact max primary edit pixel threshold to include.",
    )
    parser.add_argument(
        "--pin-max-primary-p95-edit-delta",
        action="append",
        type=float,
        default=[],
        help="Additional exact max primary p95 edit delta threshold to include.",
    )
    parser.add_argument(
        "--pin-max-second-stage-gate-ratio",
        action="append",
        type=float,
        default=[],
        help="Additional exact max second-stage gate ratio threshold to include.",
    )
    parser.add_argument(
        "--named-rule",
        action="append",
        default=[],
        metavar="NAME:MIN_COV8:MAX_EDIT_PX:MAX_P95:MAX_GATE",
        help=(
            "Optional fixed selector rule to summarize separately. Use large "
            "MAX_P95/MAX_GATE values to disable those caps."
        ),
    )
    parser.add_argument(
        "--named-interval-rule",
        action="append",
        default=[],
        metavar="NAME:MIN_COV8:MAX_COV8:MIN_EDIT_PX:MAX_EDIT_PX:MIN_P95:MAX_P95:MIN_GATE:MAX_GATE",
        help=(
            "Optional fixed interval selector rule to summarize separately. "
            "Use 0/larger-than-observed bounds to disable a lower/upper bound."
        ),
    )
    parser.add_argument(
        "--named-interval-union-rule",
        action="append",
        default=[],
        metavar="NAME:BOX;BOX",
        help=(
            "Optional fixed OR-of-intervals selector to summarize separately. "
            "Each BOX is MIN_COV8,MAX_COV8,MIN_EDIT_PX,MAX_EDIT_PX,MIN_P95,MAX_P95,MIN_GATE,MAX_GATE."
        ),
    )
    return parser.parse_args()


def parse_split(value: str) -> SplitSpec:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Invalid --split {value!r}; expected NAME:BASELINE_METRICS:GATE_METRICS")
    return SplitSpec(parts[0], Path(parts[1]), Path(parts[2]))


def parse_named_rule(value: str) -> tuple[str, float, float, float, float]:
    parts = value.split(":", 4)
    if len(parts) != 5 or not all(parts):
        raise ValueError(
            f"Invalid --named-rule {value!r}; expected NAME:MIN_COV8:MAX_EDIT_PX:MAX_P95:MAX_GATE"
        )
    name, min_cov8, max_edit_px, max_p95, max_gate = parts
    return name, float(min_cov8), float(max_edit_px), float(max_p95), float(max_gate)


def parse_named_interval_rule(
    value: str,
) -> tuple[str, float, float, float, float, float, float, float, float]:
    parts = value.split(":", 8)
    if len(parts) != 9 or not all(parts):
        raise ValueError(
            f"Invalid --named-interval-rule {value!r}; expected "
            "NAME:MIN_COV8:MAX_COV8:MIN_EDIT_PX:MAX_EDIT_PX:MIN_P95:MAX_P95:MIN_GATE:MAX_GATE"
        )
    name, min_cov8, max_cov8, min_edit_px, max_edit_px, min_p95, max_p95, min_gate, max_gate = parts
    return (
        name,
        float(min_cov8),
        float(max_cov8),
        float(min_edit_px),
        float(max_edit_px),
        float(min_p95),
        float(max_p95),
        float(min_gate),
        float(max_gate),
    )


IntervalBox = tuple[float, float, float, float, float, float, float, float]


def parse_interval_box(value: str) -> IntervalBox:
    parts = value.split(",")
    if len(parts) != 8 or not all(parts):
        raise ValueError(
            f"Invalid interval box {value!r}; expected "
            "MIN_COV8,MAX_COV8,MIN_EDIT_PX,MAX_EDIT_PX,MIN_P95,MAX_P95,MIN_GATE,MAX_GATE"
        )
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def parse_named_interval_union_rule(value: str) -> tuple[str, list[IntervalBox]]:
    name, separator, boxes_text = value.partition(":")
    if not name or separator != ":" or not boxes_text:
        raise ValueError(f"Invalid --named-interval-union-rule {value!r}; expected NAME:BOX;BOX")
    boxes = [parse_interval_box(box) for box in boxes_text.split(";") if box]
    if not boxes:
        raise ValueError(f"Invalid --named-interval-union-rule {value!r}; no interval boxes found")
    return name, boxes


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def candidate_path_for(gate_metrics: Path, row: dict[str, str], candidate_subdir: str) -> Path:
    return gate_metrics.parent / candidate_subdir / f"{Path(row['file']).stem}.png"


def fnum(row: dict[str, object], key: str) -> float:
    value = row[key]
    if value == "":
        return math.nan
    return float(value)


def recompute_candidate_rows(
    split: SplitSpec,
    candidate_subdir: str,
    change_threshold: int,
    eval_threshold: int,
) -> list[dict[str, object]]:
    baseline_rows = read_rows(split.baseline_metrics)
    gate_rows = read_rows(split.gate_metrics)
    baseline_by_file = {row["file"]: row for row in baseline_rows}
    rows: list[dict[str, object]] = []

    for gate_row in gate_rows:
        file_name = gate_row["file"]
        baseline_row = baseline_by_file[file_name]
        image_bgr = read_bgr(Path(gate_row["image_path"]))
        label_bgr = ensure_same_size(read_bgr(label_path_for(Path(gate_row["image_path"]))), image_bgr)
        candidate_path = candidate_path_for(split.gate_metrics, gate_row, candidate_subdir)
        candidate_bgr = ensure_same_size(read_bgr(candidate_path), image_bgr)
        candidate_metrics = compute_residual_metrics(
            image_bgr,
            label_bgr,
            candidate_bgr,
            change_threshold=change_threshold,
            eval_threshold=eval_threshold,
        )

        row: dict[str, object] = {
            "split": split.name,
            "file": file_name,
            "image_path": gate_row["image_path"],
            "candidate_pred_path": str(candidate_path),
            "baseline_pred_path": baseline_row["pred_path"],
            "baseline_residual_ratio": float(baseline_row["residual_ratio"]),
            "baseline_overerase_ratio": float(baseline_row["overerase_ratio"]),
            "candidate_residual_ratio": float(candidate_metrics["residual_ratio"]),
            "candidate_overerase_ratio": float(candidate_metrics["overerase_ratio"]),
            "copy_mask_cov8": float(gate_row["copy_mask_cov8"]),
            "primary_edit_px": int(float(gate_row["primary_edit_px"])),
            "primary_p95_edit_delta": float(gate_row["primary_p95_edit_delta"]),
            "primary_mean_edit_delta": float(gate_row["primary_mean_edit_delta"]),
            "second_stage_gate_ratio": float(gate_row["second_stage_gate_ratio"]),
        }
        row["residual_gain"] = row["baseline_residual_ratio"] - row["candidate_residual_ratio"]
        row["overerase_regret"] = row["candidate_overerase_ratio"] - row["baseline_overerase_ratio"]
        rows.append(row)
    return rows


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / max(len(vals), 1)


def thresholds(
    rows: list[dict[str, object]],
    key: str,
    direction: str,
    max_count: int,
    pinned: list[float],
) -> list[float]:
    values = sorted({float(row[key]) for row in rows})
    if max_count <= 0 or len(values) <= max_count:
        selected = values
    else:
        selected_set = {values[0], values[-1]}
        steps = max_count - 1
        for index in range(max_count):
            selected_set.add(values[round(index * (len(values) - 1) / steps)])
        selected = sorted(selected_set)
    selected = sorted(set(selected).union(pinned))
    if direction in {"min", "max"}:
        return selected
    raise ValueError(direction)


def selected_by_rule(
    row: dict[str, object],
    min_cov8: float,
    max_edit_px: float,
    max_p95: float,
    max_gate: float,
) -> bool:
    return (
        float(row["copy_mask_cov8"]) >= min_cov8
        and float(row["primary_edit_px"]) <= max_edit_px
        and float(row["primary_p95_edit_delta"]) <= max_p95
        and float(row["second_stage_gate_ratio"]) <= max_gate
    )


def selected_by_interval_rule(
    row: dict[str, object],
    min_cov8: float,
    max_cov8: float,
    min_edit_px: float,
    max_edit_px: float,
    min_p95: float,
    max_p95: float,
    min_gate: float,
    max_gate: float,
) -> bool:
    return (
        min_cov8 <= float(row["copy_mask_cov8"]) <= max_cov8
        and min_edit_px <= float(row["primary_edit_px"]) <= max_edit_px
        and min_p95 <= float(row["primary_p95_edit_delta"]) <= max_p95
        and min_gate <= float(row["second_stage_gate_ratio"]) <= max_gate
    )


def selected_by_interval_box(row: dict[str, object], box: IntervalBox) -> bool:
    min_cov8, max_cov8, min_edit_px, max_edit_px, min_p95, max_p95, min_gate, max_gate = box
    return selected_by_interval_rule(
        row,
        min_cov8=min_cov8,
        max_cov8=max_cov8,
        min_edit_px=min_edit_px,
        max_edit_px=max_edit_px,
        min_p95=min_p95,
        max_p95=max_p95,
        min_gate=min_gate,
        max_gate=max_gate,
    )


def selected_by_interval_union_rule(row: dict[str, object], boxes: list[IntervalBox]) -> bool:
    return any(selected_by_interval_box(row, box) for box in boxes)


def summarize_rule(
    rows: list[dict[str, object]],
    split_names: list[str],
    min_cov8: float,
    max_edit_px: float,
    max_p95: float,
    max_gate: float,
    max_overerase_regret: float,
) -> dict[str, object]:
    out: dict[str, object] = {
        "min_copy_mask_cov8": min_cov8,
        "max_primary_edit_px": int(max_edit_px),
        "max_primary_p95_edit_delta": max_p95,
        "max_second_stage_gate_ratio": max_gate,
    }
    total_selected = 0
    total_residual_gain = 0.0
    max_split_overerase_regret = -1e9
    safe_all = True

    for split in split_names:
        split_rows = [row for row in rows if row["split"] == split]
        selected = [
            selected_by_rule(row, min_cov8, max_edit_px, max_p95, max_gate)
            for row in split_rows
        ]
        residual = mean(
            fnum(row, "candidate_residual_ratio") if is_selected else fnum(row, "baseline_residual_ratio")
            for row, is_selected in zip(split_rows, selected)
        )
        overerase = mean(
            fnum(row, "candidate_overerase_ratio") if is_selected else fnum(row, "baseline_overerase_ratio")
            for row, is_selected in zip(split_rows, selected)
        )
        baseline_residual = mean(fnum(row, "baseline_residual_ratio") for row in split_rows)
        baseline_overerase = mean(fnum(row, "baseline_overerase_ratio") for row in split_rows)
        residual_gain = baseline_residual - residual
        overerase_regret = overerase - baseline_overerase
        selected_count = sum(selected)

        out[f"{split}_selected"] = selected_count
        out[f"{split}_residual"] = residual
        out[f"{split}_overerase"] = overerase
        out[f"{split}_residual_gain"] = residual_gain
        out[f"{split}_overerase_regret"] = overerase_regret
        total_selected += selected_count
        total_residual_gain += residual_gain
        max_split_overerase_regret = max(max_split_overerase_regret, overerase_regret)
        safe_all = safe_all and overerase_regret <= max_overerase_regret

    out["total_selected"] = total_selected
    out["total_residual_gain"] = total_residual_gain
    out["max_split_overerase_regret"] = max_split_overerase_regret
    out["safe_all_splits"] = int(safe_all)
    return out


def summarize_interval_rule(
    rows: list[dict[str, object]],
    split_names: list[str],
    min_cov8: float,
    max_cov8: float,
    min_edit_px: float,
    max_edit_px: float,
    min_p95: float,
    max_p95: float,
    min_gate: float,
    max_gate: float,
    max_overerase_regret: float,
) -> dict[str, object]:
    out: dict[str, object] = {
        "min_copy_mask_cov8": min_cov8,
        "max_copy_mask_cov8": max_cov8,
        "min_primary_edit_px": int(min_edit_px),
        "max_primary_edit_px": int(max_edit_px),
        "min_primary_p95_edit_delta": min_p95,
        "max_primary_p95_edit_delta": max_p95,
        "min_second_stage_gate_ratio": min_gate,
        "max_second_stage_gate_ratio": max_gate,
    }
    total_selected = 0
    total_residual_gain = 0.0
    max_split_overerase_regret = -1e9
    safe_all = True

    for split in split_names:
        split_rows = [row for row in rows if row["split"] == split]
        selected = [
            selected_by_interval_rule(
                row,
                min_cov8,
                max_cov8,
                min_edit_px,
                max_edit_px,
                min_p95,
                max_p95,
                min_gate,
                max_gate,
            )
            for row in split_rows
        ]
        residual = mean(
            fnum(row, "candidate_residual_ratio") if is_selected else fnum(row, "baseline_residual_ratio")
            for row, is_selected in zip(split_rows, selected)
        )
        overerase = mean(
            fnum(row, "candidate_overerase_ratio") if is_selected else fnum(row, "baseline_overerase_ratio")
            for row, is_selected in zip(split_rows, selected)
        )
        baseline_residual = mean(fnum(row, "baseline_residual_ratio") for row in split_rows)
        baseline_overerase = mean(fnum(row, "baseline_overerase_ratio") for row in split_rows)
        residual_gain = baseline_residual - residual
        overerase_regret = overerase - baseline_overerase
        selected_count = sum(selected)

        out[f"{split}_selected"] = selected_count
        out[f"{split}_residual"] = residual
        out[f"{split}_overerase"] = overerase
        out[f"{split}_residual_gain"] = residual_gain
        out[f"{split}_overerase_regret"] = overerase_regret
        total_selected += selected_count
        total_residual_gain += residual_gain
        max_split_overerase_regret = max(max_split_overerase_regret, overerase_regret)
        safe_all = safe_all and overerase_regret <= max_overerase_regret

    out["total_selected"] = total_selected
    out["total_residual_gain"] = total_residual_gain
    out["max_split_overerase_regret"] = max_split_overerase_regret
    out["safe_all_splits"] = int(safe_all)
    return out


def summarize_interval_union_rule(
    rows: list[dict[str, object]],
    split_names: list[str],
    boxes: list[IntervalBox],
    max_overerase_regret: float,
) -> dict[str, object]:
    out: dict[str, object] = {
        "interval_boxes": ";".join(",".join(f"{value:.12g}" for value in box) for box in boxes),
        "interval_box_count": len(boxes),
    }
    total_selected = 0
    total_residual_gain = 0.0
    max_split_overerase_regret = -1e9
    safe_all = True

    for split in split_names:
        split_rows = [row for row in rows if row["split"] == split]
        selected = [selected_by_interval_union_rule(row, boxes) for row in split_rows]
        residual = mean(
            fnum(row, "candidate_residual_ratio") if is_selected else fnum(row, "baseline_residual_ratio")
            for row, is_selected in zip(split_rows, selected)
        )
        overerase = mean(
            fnum(row, "candidate_overerase_ratio") if is_selected else fnum(row, "baseline_overerase_ratio")
            for row, is_selected in zip(split_rows, selected)
        )
        baseline_residual = mean(fnum(row, "baseline_residual_ratio") for row in split_rows)
        baseline_overerase = mean(fnum(row, "baseline_overerase_ratio") for row in split_rows)
        residual_gain = baseline_residual - residual
        overerase_regret = overerase - baseline_overerase
        selected_count = sum(selected)

        out[f"{split}_selected"] = selected_count
        out[f"{split}_residual"] = residual
        out[f"{split}_overerase"] = overerase
        out[f"{split}_residual_gain"] = residual_gain
        out[f"{split}_overerase_regret"] = overerase_regret
        total_selected += selected_count
        total_residual_gain += residual_gain
        max_split_overerase_regret = max(max_split_overerase_regret, overerase_regret)
        safe_all = safe_all and overerase_regret <= max_overerase_regret

    out["total_selected"] = total_selected
    out["total_residual_gain"] = total_residual_gain
    out["max_split_overerase_regret"] = max_split_overerase_regret
    out["safe_all_splits"] = int(safe_all)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    split_specs = [parse_split(value) for value in args.split]
    split_names = [spec.name for spec in split_specs]
    output_dir = Path(args.output_dir)

    rows: list[dict[str, object]] = []
    for split in split_specs:
        rows.extend(
            recompute_candidate_rows(
                split,
                candidate_subdir=args.candidate_subdir,
                change_threshold=args.change_threshold,
                eval_threshold=args.eval_threshold,
            )
        )
    write_csv(output_dir / "page_choices.csv", rows)

    cov_thresholds = thresholds(
        rows,
        "copy_mask_cov8",
        "min",
        args.max_thresholds_per_feature,
        args.pin_min_copy_mask_cov8,
    )
    edit_thresholds = thresholds(
        rows,
        "primary_edit_px",
        "max",
        args.max_thresholds_per_feature,
        args.pin_max_primary_edit_px,
    )
    p95_thresholds = thresholds(
        rows,
        "primary_p95_edit_delta",
        "max",
        args.max_thresholds_per_feature,
        args.pin_max_primary_p95_edit_delta,
    )
    gate_thresholds = thresholds(
        rows,
        "second_stage_gate_ratio",
        "max",
        args.max_thresholds_per_feature,
        args.pin_max_second_stage_gate_ratio,
    )
    print(
        "threshold_grid "
        f"cov={len(cov_thresholds)} edit={len(edit_thresholds)} "
        f"p95={len(p95_thresholds)} gate={len(gate_thresholds)} "
        f"rules_to_score={len(cov_thresholds) * len(edit_thresholds) * len(p95_thresholds) * len(gate_thresholds)}",
        flush=True,
    )

    summaries: list[dict[str, object]] = []
    for min_cov8 in cov_thresholds:
        for max_edit_px in edit_thresholds:
            for max_p95 in p95_thresholds:
                for max_gate in gate_thresholds:
                    summary = summarize_rule(
                        rows,
                        split_names,
                        min_cov8=min_cov8,
                        max_edit_px=max_edit_px,
                        max_p95=max_p95,
                        max_gate=max_gate,
                        max_overerase_regret=args.max_overerase_regret,
                    )
                    if int(summary["total_selected"]) >= args.min_selected_total:
                        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            int(row["safe_all_splits"]),
            float(row["total_residual_gain"]),
            -float(row["max_split_overerase_regret"]),
            int(row["total_selected"]),
        ),
        reverse=True,
    )
    write_csv(output_dir / "sweep_summary.csv", summaries)
    write_csv(output_dir / "top_rules.csv", summaries[: args.top_n])

    fixed_rule_rows: list[dict[str, object]] = []
    for name, min_cov8, max_edit_px, max_p95, max_gate in [
        parse_named_rule(value) for value in args.named_rule
    ]:
        row = summarize_rule(
            rows,
            split_names,
            min_cov8=min_cov8,
            max_edit_px=max_edit_px,
            max_p95=max_p95,
            max_gate=max_gate,
            max_overerase_regret=args.max_overerase_regret,
        )
        row["rule_name"] = name
        fixed_rule_rows.append(row)
    if fixed_rule_rows:
        write_csv(output_dir / "named_rules.csv", fixed_rule_rows)

    fixed_interval_rule_rows: list[dict[str, object]] = []
    for (
        name,
        min_cov8,
        max_cov8,
        min_edit_px,
        max_edit_px,
        min_p95,
        max_p95,
        min_gate,
        max_gate,
    ) in [parse_named_interval_rule(value) for value in args.named_interval_rule]:
        row = summarize_interval_rule(
            rows,
            split_names,
            min_cov8=min_cov8,
            max_cov8=max_cov8,
            min_edit_px=min_edit_px,
            max_edit_px=max_edit_px,
            min_p95=min_p95,
            max_p95=max_p95,
            min_gate=min_gate,
            max_gate=max_gate,
            max_overerase_regret=args.max_overerase_regret,
        )
        row["rule_name"] = name
        fixed_interval_rule_rows.append(row)
    if fixed_interval_rule_rows:
        write_csv(output_dir / "named_interval_rules.csv", fixed_interval_rule_rows)

    fixed_interval_union_rule_rows: list[dict[str, object]] = []
    for name, boxes in [parse_named_interval_union_rule(value) for value in args.named_interval_union_rule]:
        row = summarize_interval_union_rule(
            rows,
            split_names,
            boxes=boxes,
            max_overerase_regret=args.max_overerase_regret,
        )
        row["rule_name"] = name
        fixed_interval_union_rule_rows.append(row)
    if fixed_interval_union_rule_rows:
        write_csv(output_dir / "named_interval_union_rules.csv", fixed_interval_union_rule_rows)

    safe = [row for row in summaries if int(row["safe_all_splits"])]
    best = safe[0] if safe else (summaries[0] if summaries else {})
    print(f"page_choices: {output_dir / 'page_choices.csv'}")
    print(f"sweep_summary: {output_dir / 'sweep_summary.csv'}")
    print(f"top_rules: {output_dir / 'top_rules.csv'}")
    if fixed_rule_rows:
        print(f"named_rules: {output_dir / 'named_rules.csv'}")
    if fixed_interval_rule_rows:
        print(f"named_interval_rules: {output_dir / 'named_interval_rules.csv'}")
    if fixed_interval_union_rule_rows:
        print(f"named_interval_union_rules: {output_dir / 'named_interval_union_rules.csv'}")
    print(f"rules={len(summaries)} safe={len(safe)}")
    if best:
        print(
            "best "
            f"safe={best['safe_all_splits']} "
            f"selected={best['total_selected']} "
            f"residual_gain={float(best['total_residual_gain']):.12f} "
            f"max_overerase_regret={float(best['max_split_overerase_regret']):.12f} "
            f"cov8>={float(best['min_copy_mask_cov8']):.6f} "
            f"edit<={int(best['max_primary_edit_px'])} "
            f"p95<={float(best['max_primary_p95_edit_delta']):.6f} "
            f"gate<={float(best['max_second_stage_gate_ratio']):.9f}"
        )


if __name__ == "__main__":
    main()
