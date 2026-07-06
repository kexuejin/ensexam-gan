#!/usr/bin/env python3
"""Compare hybrid-gate feature drift across candidate metric CSVs.

This is intended for bounded continuation probes where the same strict gate
pages stop passing after training. It reports whether pages were rejected due to
copy-mask coverage, edit-size, or both compared with a reference metrics file.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-metrics", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME:METRICS_CSV",
        help="Candidate metrics CSV to compare. May be repeated.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--min-copy-mask-cov8", type=float, required=True)
    parser.add_argument("--max-primary-edit-px", type=float, required=True)
    parser.add_argument(
        "--reference-selected-only",
        action="store_true",
        help="Only compare rows where the reference used the candidate output.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["file"]: row for row in csv.DictReader(handle)}


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def fail_reason(row: dict[str, str], min_cov8: float, max_edit_px: float) -> str:
    reasons: list[str] = []
    if as_float(row, "copy_mask_cov8") < min_cov8:
        reasons.append("copy_mask_cov8")
    if as_float(row, "primary_edit_px") > max_edit_px:
        reasons.append("primary_edit_px")
    return "+".join(reasons) or "passes"


def main() -> None:
    args = parse_args()
    reference = read_rows(Path(args.reference_metrics))
    candidates: list[tuple[str, dict[str, dict[str, str]]]] = []
    for raw in args.candidate:
        if ":" not in raw:
            raise ValueError(f"--candidate must be NAME:METRICS_CSV: {raw}")
        name, path = raw.split(":", 1)
        candidates.append((name, read_rows(Path(path))))

    files = sorted(reference)
    if args.reference_selected_only:
        files = [
            file
            for file in files
            if reference[file].get("use_candidate") == "1" or reference[file].get("source") == "candidate"
        ]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "candidate",
        "reference_source",
        "candidate_source",
        "reference_fail_reason",
        "candidate_fail_reason",
        "reference_cov8",
        "candidate_cov8",
        "delta_cov8",
        "reference_edit_px",
        "candidate_edit_px",
        "delta_edit_px",
        "reference_p95_edit_delta",
        "candidate_p95_edit_delta",
        "delta_p95_edit_delta",
        "reference_gate_ratio",
        "candidate_gate_ratio",
        "delta_gate_ratio",
    ]
    summary: dict[str, dict[str, int]] = {}
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for file in files:
            ref = reference[file]
            ref_reason = fail_reason(ref, args.min_copy_mask_cov8, args.max_primary_edit_px)
            for name, rows in candidates:
                row = rows[file]
                cand_reason = fail_reason(row, args.min_copy_mask_cov8, args.max_primary_edit_px)
                summary.setdefault(name, {})
                summary[name][cand_reason] = summary[name].get(cand_reason, 0) + 1
                writer.writerow(
                    {
                        "file": file,
                        "candidate": name,
                        "reference_source": ref.get("source", ""),
                        "candidate_source": row.get("source", ""),
                        "reference_fail_reason": ref_reason,
                        "candidate_fail_reason": cand_reason,
                        "reference_cov8": f"{as_float(ref, 'copy_mask_cov8'):.12f}",
                        "candidate_cov8": f"{as_float(row, 'copy_mask_cov8'):.12f}",
                        "delta_cov8": f"{as_float(row, 'copy_mask_cov8') - as_float(ref, 'copy_mask_cov8'):.12f}",
                        "reference_edit_px": f"{as_float(ref, 'primary_edit_px'):.0f}",
                        "candidate_edit_px": f"{as_float(row, 'primary_edit_px'):.0f}",
                        "delta_edit_px": f"{as_float(row, 'primary_edit_px') - as_float(ref, 'primary_edit_px'):.0f}",
                        "reference_p95_edit_delta": f"{as_float(ref, 'primary_p95_edit_delta'):.6f}",
                        "candidate_p95_edit_delta": f"{as_float(row, 'primary_p95_edit_delta'):.6f}",
                        "delta_p95_edit_delta": (
                            f"{as_float(row, 'primary_p95_edit_delta') - as_float(ref, 'primary_p95_edit_delta'):.6f}"
                        ),
                        "reference_gate_ratio": f"{as_float(ref, 'second_stage_gate_ratio'):.12f}",
                        "candidate_gate_ratio": f"{as_float(row, 'second_stage_gate_ratio'):.12f}",
                        "delta_gate_ratio": (
                            f"{as_float(row, 'second_stage_gate_ratio') - as_float(ref, 'second_stage_gate_ratio'):.12f}"
                        ),
                    }
                )

    print(f"rows={len(files) * len(candidates)}")
    print(f"output_csv={output_path}")
    for name, counts in summary.items():
        parts = " ".join(f"{key}={counts[key]}" for key in sorted(counts))
        print(f"{name}: {parts}")


if __name__ == "__main__":
    main()
