#!/usr/bin/env python3
"""Apply the predeclared whole-dev-set gate to a candidate prediction CSV.

This is deliberately a development-set tool.  It accepts or rejects one
predeclared candidate checkpoint as a whole; it never selects page-specific
predictions, thresholds, or repairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = (
    "file",
    "image_path",
    "label_path",
    "pred_path",
    "residual_ratio",
    "overerase_ratio",
)

P95_CONVENTIONS = {
    "linear": "numpy.quantile(q=0.95, method='linear')",
    "frozen-15page-lower-index": "sorted ascending index 13 (second-worst of 15)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-12,
        help="Numerical comparison tolerance; not a permitted quality regression.",
    )
    parser.add_argument(
        "--p95-convention",
        choices=tuple(P95_CONVENTIONS),
        default="linear",
        help="P95 calculation convention. The frozen rule requires exactly 15 pages.",
    )
    parser.add_argument(
        "--six-metric-only",
        action="store_true",
        help=(
            "Apply only the six aggregate/tail metrics. This is the registered "
            "frozen inner-val15 screen; page deltas remain diagnostic, not promotion gates."
        ),
    )
    parser.add_argument(
        "--min-mean-residual-improvement-pct",
        "--min-mean-residual-improvement-percent",
        dest="min_mean_residual_improvement_pct",
        type=float,
        default=0.0,
        help=(
            "Required mean residual improvement percentage in six-metric mode. "
            "Must be finite, non-negative, and less than 100; defaults to 0 for "
            "backward-compatible behavior."
        ),
    )
    args = parser.parse_args()
    try:
        validate_min_mean_residual_improvement_pct(args.min_mean_residual_improvement_pct)
    except ValueError as exc:
        parser.error(str(exc))
    if args.min_mean_residual_improvement_pct > 0 and not args.six_metric_only:
        parser.error(
            "a nonzero min_mean_residual_improvement_pct requires --six-metric-only"
        )
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_identity_path(value: object, csv_path: Path, line_number: int, field_name: str) -> Path:
    path = Path(str(value or ""))
    if not value or not path.is_file():
        raise ValueError(f"{csv_path}:{line_number} {field_name} does not exist: {value}")
    return path.resolve()


def read_metric_rows(path: Path) -> dict[str, dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = sorted(set(REQUIRED_COLUMNS) - set(fieldnames))
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

        rows: dict[str, dict[str, object]] = {}
        for line_number, raw in enumerate(reader, start=2):
            file_name = (raw.get("file") or "").strip()
            if not file_name:
                raise ValueError(f"{path}:{line_number} has an empty file value")
            if file_name in rows:
                raise ValueError(f"{path} contains a duplicate file: {file_name}")

            pred_path = Path((raw.get("pred_path") or "").strip())
            if not pred_path.is_file():
                raise ValueError(f"{path}:{line_number} prediction does not exist: {pred_path}")
            image_path = resolve_identity_path(raw.get("image_path"), path, line_number, "image_path")
            label_path = resolve_identity_path(raw.get("label_path"), path, line_number, "label_path")
            try:
                residual = float(raw["residual_ratio"])
                overerase = float(raw["overerase_ratio"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number} has non-numeric metrics") from exc
            if not math.isfinite(residual) or not math.isfinite(overerase):
                raise ValueError(f"{path}:{line_number} has non-finite metrics")
            rows[file_name] = {
                "residual_ratio": residual,
                "overerase_ratio": overerase,
                "pred_path": str(pred_path),
                "image_path": str(image_path),
                "label_path": str(label_path),
                "image_sha256": sha256_file(image_path),
                "label_sha256": sha256_file(label_path),
            }
    if not rows:
        raise ValueError(f"{path} has no metric rows")
    return rows


def p95(values: np.ndarray, convention: str) -> float:
    if convention == "linear":
        return float(np.quantile(values, 0.95, method="linear"))
    if convention == "frozen-15page-lower-index":
        if len(values) != 15:
            raise ValueError(
                "p95 convention frozen-15page-lower-index requires exactly 15 metric rows"
            )
        return float(np.sort(values)[13])
    raise ValueError(f"unsupported p95 convention: {convention}")


def summarize(rows: dict[str, dict[str, object]], p95_convention: str) -> dict[str, float]:
    residuals = np.asarray([float(row["residual_ratio"]) for row in rows.values()])
    overerases = np.asarray([float(row["overerase_ratio"]) for row in rows.values()])
    return {
        "mean_residual_ratio": float(residuals.mean()),
        "mean_overerase_ratio": float(overerases.mean()),
        "p95_residual_ratio": p95(residuals, p95_convention),
        "p95_overerase_ratio": p95(overerases, p95_convention),
        "max_residual_ratio": float(residuals.max()),
        "max_overerase_ratio": float(overerases.max()),
    }


def validate_min_mean_residual_improvement_pct(value: float) -> float:
    if not math.isfinite(value) or value < 0 or value >= 100:
        raise ValueError(
            "min_mean_residual_improvement_pct must be finite, non-negative, and less than 100"
        )
    return value


def mean_residual_bound(
    baseline_mean_residual: float,
    min_mean_residual_improvement_pct: float,
    *,
    six_metric_only: bool,
) -> float:
    pct = validate_min_mean_residual_improvement_pct(min_mean_residual_improvement_pct)
    if pct > 0 and not six_metric_only:
        raise ValueError(
            "a nonzero min_mean_residual_improvement_pct requires --six-metric-only"
        )
    if not six_metric_only:
        return baseline_mean_residual
    return baseline_mean_residual * (1.0 - pct / 100.0)


def check_gate(
    baseline: dict[str, dict[str, object]],
    candidate: dict[str, dict[str, object]],
    tolerance: float,
    p95_convention: str,
    six_metric_only: bool,
    min_mean_residual_improvement_pct: float = 0.0,
) -> tuple[list[dict[str, object]], dict[str, float], dict[str, float]]:
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be a finite non-negative number")
    min_mean_residual_improvement_pct = validate_min_mean_residual_improvement_pct(
        min_mean_residual_improvement_pct
    )
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(
            "baseline and candidate file sets differ: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    mismatched_identity = [
        file_name
        for file_name in baseline
        if (
            baseline[file_name]["image_path"] != candidate[file_name]["image_path"]
            or baseline[file_name]["label_path"] != candidate[file_name]["label_path"]
            or baseline[file_name]["image_sha256"] != candidate[file_name]["image_sha256"]
            or baseline[file_name]["label_sha256"] != candidate[file_name]["label_sha256"]
        )
    ]
    if mismatched_identity:
        raise ValueError(
            "baseline and candidate source/label identity differs: "
            f"{mismatched_identity[:5]}"
        )

    baseline_summary = summarize(baseline, p95_convention)
    candidate_summary = summarize(candidate, p95_convention)
    checks: list[dict[str, object]] = []
    residual_bound = mean_residual_bound(
        baseline_summary["mean_residual_ratio"],
        min_mean_residual_improvement_pct,
        six_metric_only=six_metric_only,
    )

    def add(name: str, passed: bool, actual: float, bound: float) -> None:
        checks.append({"name": name, "passed": passed, "actual": actual, "bound": bound})

    add(
        "mean_residual_improves",
        (
            candidate_summary["mean_residual_ratio"] <= residual_bound + tolerance
            if min_mean_residual_improvement_pct > 0 and six_metric_only
            else candidate_summary["mean_residual_ratio"] < residual_bound - tolerance
        ),
        candidate_summary["mean_residual_ratio"],
        residual_bound,
    )
    for metric in ("mean_overerase_ratio", "p95_residual_ratio", "p95_overerase_ratio", "max_residual_ratio", "max_overerase_ratio"):
        add(
            f"{metric}_does_not_regress",
            candidate_summary[metric] <= baseline_summary[metric] + tolerance,
            candidate_summary[metric],
            baseline_summary[metric],
        )

    if not six_metric_only:
        residual_regressions = sorted(
            file_name
            for file_name in baseline
            if float(candidate[file_name]["residual_ratio"]) > float(baseline[file_name]["residual_ratio"]) + tolerance
        )
        overerase_regressions = sorted(
            file_name
            for file_name in baseline
            if float(candidate[file_name]["overerase_ratio"]) > float(baseline[file_name]["overerase_ratio"]) + tolerance
        )
        checks.append({
            "name": "no_page_residual_regressions",
            "passed": not residual_regressions,
            "count": len(residual_regressions),
            "files": residual_regressions,
        })
        checks.append({
            "name": "no_page_overerase_regressions",
            "passed": not overerase_regressions,
            "count": len(overerase_regressions),
            "files": overerase_regressions,
        })
    return checks, baseline_summary, candidate_summary


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline_csv)
    candidate_path = Path(args.candidate_csv)
    baseline = read_metric_rows(baseline_path)
    candidate = read_metric_rows(candidate_path)
    checks, baseline_summary, candidate_summary = check_gate(
        baseline,
        candidate,
        args.tolerance,
        args.p95_convention,
        args.six_metric_only,
        args.min_mean_residual_improvement_pct,
    )
    requested_improvement_pct = validate_min_mean_residual_improvement_pct(
        args.min_mean_residual_improvement_pct
    )
    residual_bound = mean_residual_bound(
        baseline_summary["mean_residual_ratio"],
        requested_improvement_pct,
        six_metric_only=args.six_metric_only,
    )
    passed = all(bool(check["passed"]) for check in checks)
    result = {
        "protocol": "scut_dev_candidate_gate",
        "decision": "accept" if passed else "reject",
        "candidate_is_whole_checkpoint": True,
        "page_specific_selection_forbidden": True,
        "paired_source_and_label_identity_required": True,
        "sample_count": len(candidate),
        "baseline_csv": str(baseline_path),
        "baseline_csv_sha256": sha256_file(baseline_path),
        "candidate_csv": str(candidate_path),
        "candidate_csv_sha256": sha256_file(candidate_path),
        "tolerance": args.tolerance,
        "p95_convention": args.p95_convention,
        "p95_method": P95_CONVENTIONS[args.p95_convention],
        "min_mean_residual_improvement_pct": requested_improvement_pct,
        "mean_residual_bound": residual_bound,
        "gate_scope": "six_metrics_only" if args.six_metric_only else "six_metrics_plus_per_page_nonregression",
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "checks": checks,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "output_json": str(output_path)}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
