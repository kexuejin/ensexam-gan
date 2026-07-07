#!/usr/bin/env python3
"""Summarize candidate metric folders against split baselines.

This is a read-only discovery tool for finding candidate families worth deeper
local-proxy or visual validation. It compares output ``metrics.csv`` files only
when their file set exactly matches the configured baseline split.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="append",
        required=True,
        metavar="SPLIT:METRICS_CSV",
        help="Baseline split metrics. May be repeated.",
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--name-contains", action="append", default=[])
    parser.add_argument("--top-n-per-split", type=int, default=25)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_baseline(value: str) -> tuple[str, Path]:
    split, separator, path = value.partition(":")
    if separator != ":" or not split or not path:
        raise ValueError(f"Invalid --baseline {value!r}; expected SPLIT:METRICS_CSV")
    return split, Path(path)


def metric_summary(path: Path) -> dict[str, object] | None:
    rows = read_rows(path)
    if not rows:
        return None
    if "residual_ratio" not in rows[0] or "overerase_ratio" not in rows[0] or "file" not in rows[0]:
        return None
    return {
        "rows": len(rows),
        "files": {row["file"] for row in rows},
        "residual": sum(float(row["residual_ratio"]) for row in rows) / len(rows),
        "overerase": sum(float(row["overerase_ratio"]) for row in rows) / len(rows),
    }


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
    baselines = []
    for split, path in map(parse_baseline, args.baseline):
        summary = metric_summary(path)
        if summary is None:
            raise ValueError(f"Baseline metrics missing required fields: {path}")
        baselines.append((split, path, summary))

    rows: list[dict[str, object]] = []
    for output_dir in sorted(Path(args.outputs_root).iterdir()):
        if not output_dir.is_dir():
            continue
        if args.name_contains and not any(token in output_dir.name for token in args.name_contains):
            continue
        metrics_path = output_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        candidate = metric_summary(metrics_path)
        if candidate is None:
            continue
        for split, baseline_path, baseline in baselines:
            if candidate["files"] != baseline["files"]:
                continue
            residual = float(candidate["residual"])
            overerase = float(candidate["overerase"])
            baseline_residual = float(baseline["residual"])
            baseline_overerase = float(baseline["overerase"])
            rows.append({
                "split": split,
                "candidate": output_dir.name,
                "metrics_csv": str(metrics_path),
                "baseline_metrics_csv": str(baseline_path),
                "pages": int(candidate["rows"]),
                "residual": residual,
                "overerase": overerase,
                "baseline_residual": baseline_residual,
                "baseline_overerase": baseline_overerase,
                "residual_delta_vs_baseline": residual - baseline_residual,
                "overerase_delta_vs_baseline": overerase - baseline_overerase,
                "has_pred": (output_dir / "pred").is_dir(),
            })

    rows.sort(key=lambda row: (str(row["split"]), float(row["residual_delta_vs_baseline"]), float(row["overerase_delta_vs_baseline"])))
    write_csv(Path(args.output_csv), rows)

    print(f"rows={len(rows)} output_csv={args.output_csv}")
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        print(f"\n== {split} candidates={len(split_rows)}")
        for row in split_rows[: args.top_n_per_split]:
            print(
                f"{row['candidate']}\tres_delta={float(row['residual_delta_vs_baseline']):+.9f}"
                f"\tover_delta={float(row['overerase_delta_vs_baseline']):+.9f}"
                f"\tres={float(row['residual']):.9f}\tover={float(row['overerase']):.9f}"
            )


if __name__ == "__main__":
    main()
