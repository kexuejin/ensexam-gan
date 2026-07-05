#!/usr/bin/env python3
"""Select per-file copy-mask thresholds from local hardcase metric sweeps."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-root",
        default="outputs",
        help="Directory containing eval output subdirectories.",
    )
    parser.add_argument(
        "--glob",
        default="current_best_copyout_mb_t*_d0_*",
        help="Glob for threshold sweep output directories under sweep-root.",
    )
    parser.add_argument(
        "--exclude-substring",
        action="append",
        default=["minedit"],
        help="Skip sweep directories containing this substring. Can be repeated.",
    )
    parser.add_argument("--max-overerase", type=float, default=0.0015)
    parser.add_argument("--overerase-penalty", type=float, default=8.0)
    parser.add_argument("--output-map", required=True)
    parser.add_argument("--output-summary", required=True)
    return parser.parse_args()


def threshold_from_dir(path: Path) -> int | None:
    match = re.search(r"_mb_t(\d+)_d0_", path.name)
    if not match:
        return None
    return int(match.group(1))


def score_row(row: dict[str, str], overerase_penalty: float) -> float:
    baseline_residual = float(row["baseline_residual_ratio"])
    baseline_overerase = float(row["baseline_overerase_ratio"])
    residual = float(row["residual_ratio"])
    overerase = float(row["overerase_ratio"])
    return (baseline_residual - residual) - overerase_penalty * max(overerase - baseline_overerase, 0.0)


def main() -> None:
    args = parse_args()
    sweep_root = Path(args.sweep_root)
    excluded = tuple(args.exclude_substring or [])

    candidates: dict[str, list[dict[str, str | int | float]]] = {}
    for sweep_dir in sorted(sweep_root.glob(args.glob)):
        if any(token and token in sweep_dir.name for token in excluded):
            continue
        threshold = threshold_from_dir(sweep_dir)
        metrics_path = sweep_dir / "hardcase_worst_metrics.csv"
        if threshold is None or not metrics_path.exists():
            continue

        with metrics_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scored = dict(row)
                scored["threshold"] = threshold
                scored["score"] = score_row(row, args.overerase_penalty)
                scored["source_dir"] = str(sweep_dir)
                candidates.setdefault(row["file"], []).append(scored)

    if not candidates:
        raise FileNotFoundError(f"No threshold sweep metrics matched {sweep_root / args.glob}")

    selected = []
    for file_name, rows in sorted(candidates.items()):
        selected.append(max(rows, key=lambda row: float(row["score"])))

    avg_residual = sum(float(row["residual_ratio"]) for row in selected) / len(selected)
    avg_overerase = sum(float(row["overerase_ratio"]) for row in selected) / len(selected)
    avg_baseline_residual = sum(float(row["baseline_residual_ratio"]) for row in selected) / len(selected)
    avg_baseline_overerase = sum(float(row["baseline_overerase_ratio"]) for row in selected) / len(selected)
    aggregate_score = (
        avg_baseline_residual - avg_residual
        - args.overerase_penalty * max(avg_overerase - avg_baseline_overerase, 0.0)
    )

    output_map = Path(args.output_map)
    output_map.parent.mkdir(parents=True, exist_ok=True)
    threshold_map = ",".join(f"{row['file']}:{row['threshold']}" for row in selected)
    output_map.write_text(threshold_map + "\n", encoding="utf-8")

    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "threshold",
        "residual_ratio",
        "overerase_ratio",
        "baseline_residual_ratio",
        "baseline_overerase_ratio",
        "score",
        "source_dir",
    ]
    with output_summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row[key] for key in fieldnames})

    print(f"threshold_map={threshold_map}")
    print(f"summary_csv={output_summary}")
    print(f"residual={avg_residual:.6f}")
    print(f"overerase={avg_overerase:.6f}")
    print(f"score={aggregate_score:+.6f}")
    print(f"passes_overerase_gate={avg_overerase <= args.max_overerase}")


if __name__ == "__main__":
    main()
