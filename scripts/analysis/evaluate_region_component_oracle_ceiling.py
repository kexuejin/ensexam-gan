#!/usr/bin/env python3
"""Estimate the oracle ceiling for a region-component candidate family.

This intentionally uses target-derived residual/overerase deltas, so it is not
an inference-time selector. It answers a different question: if a perfect
component selector existed for this candidate family, how much page-level metric
headroom would remain?
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_region_component_threshold_sweep import (  # noqa: E402
    component_score_by_page,
    read_rows,
    split_pages,
)


ComponentPredicate = Callable[[dict[str, object]], bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-csv", required=True)
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


def parse_split(value: str) -> tuple[str, Path, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Invalid --split {value!r}; expected NAME:BASELINE:CANDIDATE")
    return parts[0], Path(parts[1]), Path(parts[2])


def component_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["split"], row["file"], row["component_id"]


def include_all_component_scores(components_csv: Path, split_names: set[str]) -> dict[tuple[str, str, str], float]:
    scores: dict[tuple[str, str, str], float] = {}
    for row in read_rows(components_csv):
        if row["split"] in split_names:
            scores[component_key(row)] = 1.0
    return scores


def policy_residual_only(component: dict[str, object]) -> bool:
    return int(component["residual_px_delta"]) < 0


def policy_strict_no_over(component: dict[str, object]) -> bool:
    return int(component["residual_px_delta"]) < 0 and int(component["over_px_delta"]) <= 0


def policy_net_penalty3(component: dict[str, object]) -> bool:
    residual_delta = int(component["residual_px_delta"])
    over_delta = max(int(component["over_px_delta"]), 0)
    return residual_delta + 3 * over_delta < 0


def policy_no_metric_hurt(component: dict[str, object]) -> bool:
    residual_delta = int(component["residual_px_delta"])
    over_delta = int(component["over_px_delta"])
    return residual_delta <= 0 and over_delta <= 0 and (residual_delta < 0 or over_delta < 0)


POLICIES: dict[str, ComponentPredicate] = {
    "oracle_residual_only": policy_residual_only,
    "oracle_strict_no_over": policy_strict_no_over,
    "oracle_net_penalty3": policy_net_penalty3,
    "oracle_no_metric_hurt": policy_no_metric_hurt,
}


def evaluate_policy(pages: list[dict[str, object]], predicate: ComponentPredicate) -> dict[str, object]:
    totals: dict[str, float] = {
        "pages": len(pages),
        "materialized_pages": 0,
        "materialized_components": 0,
        "selected_pixels": 0,
        "baseline_residual": 0.0,
        "residual": 0.0,
        "baseline_overerase": 0.0,
        "overerase": 0.0,
        "improved_pages": 0,
        "worse_pages": 0,
        "over_reg_pages": 0,
    }
    for page in pages:
        residual_delta = 0
        over_delta = 0
        selected = 0
        pixels = 0
        for component in page["component_rows"]:
            if not predicate(component):
                continue
            selected += 1
            pixels += int(component["pixels"])
            residual_delta += int(component["residual_px_delta"])
            over_delta += int(component["over_px_delta"])

        changed_px = max(int(page["changed_px"]), 1)
        outside_px = max(int(page["outside_px"]), 1)
        baseline_residual_px = int(page["baseline_residual_px"])
        baseline_over_px = int(page["baseline_over_px"])
        baseline_residual = baseline_residual_px / changed_px
        baseline_overerase = baseline_over_px / outside_px
        residual = (baseline_residual_px + residual_delta) / changed_px
        overerase = (baseline_over_px + over_delta) / outside_px

        totals["materialized_pages"] += int(selected > 0)
        totals["materialized_components"] += selected
        totals["selected_pixels"] += pixels
        totals["baseline_residual"] += baseline_residual
        totals["residual"] += residual
        totals["baseline_overerase"] += baseline_overerase
        totals["overerase"] += overerase
        totals["improved_pages"] += int(baseline_residual - residual > 0)
        totals["worse_pages"] += int(baseline_residual - residual < 0)
        totals["over_reg_pages"] += int(overerase - baseline_overerase > 0)

    page_count = max(len(pages), 1)
    for key in ("baseline_residual", "residual", "baseline_overerase", "overerase"):
        totals[key] /= page_count
    totals["residual_gain"] = totals["baseline_residual"] - totals["residual"]
    totals["overerase_delta"] = totals["overerase"] - totals["baseline_overerase"]
    return totals


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    split_specs = [parse_split(value) for value in args.split]
    scores = include_all_component_scores(Path(args.components_csv), {name for name, _base, _candidate in split_specs})
    scores_by_page = component_score_by_page(Path(args.components_csv), scores)

    rows: list[dict[str, object]] = []
    for split_name, baseline_metrics, candidate_metrics in split_specs:
        pages = split_pages(split_name, baseline_metrics, candidate_metrics, scores_by_page, args)
        for policy_name, predicate in POLICIES.items():
            rows.append({
                "policy": policy_name,
                "split": split_name,
                **evaluate_policy(pages, predicate),
            })

    write_csv(Path(args.output_csv), rows)
    print(f"rows={len(rows)} output_csv={args.output_csv}")
    for row in rows:
        print(
            f"{row['policy']} {row['split']} "
            f"residual_gain={float(row['residual_gain']):.9f} "
            f"overerase_delta={float(row['overerase_delta']):.9f} "
            f"materialized_pages={int(row['materialized_pages'])} "
            f"worse_pages={int(row['worse_pages'])} "
            f"over_reg_pages={int(row['over_reg_pages'])}"
        )


if __name__ == "__main__":
    main()
