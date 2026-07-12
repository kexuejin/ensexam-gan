#!/usr/bin/env python3
"""Report the sustainable quality-goal status for the selector pipeline.

This is a lightweight gatekeeper for the ongoing EnsExam-GAN quality loop. It
summarizes the current release candidate, target-aware quality state, and
post-125 label-queue backlog so each continuation can start from the same
measurable definition of done.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("outputs/balanced007_ranker_expansion_source_eval_20260708")
DEFAULT_SELECTOR_QUALITY_CSV = ROOT / "target_quality_scoring_20260708/selector_target_quality_summary.csv"
DEFAULT_SELECTOR_SUMMARY_DIR = ROOT / "final_candidate_selectors"
DEFAULT_LABEL_QUEUE_CSV = ROOT / "labeling_queue_post125_coverage_20260708/auto_triage/auto_triage_labels.csv"
DEFAULT_TARGET_QUALITY_SUMMARY_JSON = ROOT / "target_quality_scoring_20260708/full435_target_quality_v2_summary.json"
DEFAULT_RELEASE_SELECTOR = "zero_reject_veto_112_zero_target_loss"
DEFAULT_TRACKED_SELECTORS = (
    "zero_reject_veto_112_zero_target_loss",
    "zero_reject_veto_125_accept_clean",
    "zero_reject_veto_134_manual_accept_whitelist",
    "zero_reject_veto_auto_triage_promote_candidate",
)
POSITIVE_DECISIONS = {"promote", "accept", "approve", "approved", "yes", "y", "true", "1"}
NEGATIVE_DECISIONS = {"reject", "rejected", "keep", "keep_review", "skip", "defer", "no", "n", "false", "0"}
REVIEW_LABELS = {"promote_candidate", "borderline_review"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-quality-csv", default=str(DEFAULT_SELECTOR_QUALITY_CSV))
    parser.add_argument("--selector-summary-dir", default=str(DEFAULT_SELECTOR_SUMMARY_DIR))
    parser.add_argument("--label-queue-csv", default=str(DEFAULT_LABEL_QUEUE_CSV))
    parser.add_argument("--target-quality-summary-json", default=str(DEFAULT_TARGET_QUALITY_SUMMARY_JSON))
    parser.add_argument("--release-selector", default=DEFAULT_RELEASE_SELECTOR)
    parser.add_argument("--track-selector", action="append", default=[])
    parser.add_argument("--min-selected", type=int, default=112)
    parser.add_argument("--max-metric-losses", type=int, default=0)
    parser.add_argument("--max-local-reject", type=int, default=0)
    parser.add_argument("--max-target-losses", type=int, default=0)
    parser.add_argument(
        "--max-target-borderline",
        type=int,
        default=0,
        help="Maximum unresolved target-borderline selected pages for release-ready status.",
    )
    parser.add_argument("--max-missing-quality-rows", type=int, default=0)
    parser.add_argument(
        "--allow-unresolved-label-queue",
        action="store_true",
        help="Do not fail release-ready status on blank manual labels in promote/borderline queues.",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def normalize(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def num(row: dict[str, Any], field: str, default: float = 0.0) -> float:
    value = row.get(field, "")
    if value in (None, ""):
        return default
    return float(value)


def int_num(row: dict[str, Any], field: str, default: int = 0) -> int:
    return int(round(num(row, field, default)))


def selector_summary_path(summary_dir: Path, selector: str) -> Path:
    return summary_dir / f"{selector}_summary.json"


def load_selector_status(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    quality_rows = {row["selector"]: row for row in read_rows(Path(args.selector_quality_csv))}
    summary_dir = Path(args.selector_summary_dir)
    selectors = list(dict.fromkeys([args.release_selector, *DEFAULT_TRACKED_SELECTORS, *args.track_selector]))
    statuses: dict[str, dict[str, Any]] = {}
    for selector in selectors:
        quality = quality_rows.get(selector, {})
        summary = read_json(selector_summary_path(summary_dir, selector))
        statuses[selector] = {
            "selector": selector,
            "selected": int_num(quality or summary, "selected"),
            "selected_coverage_pct": num(quality, "selected_coverage_pct", num(summary, "selected_coverage_pct", num(summary, "coverage_pct", 0.0))),
            "target_confirmed_wins": int_num(quality or summary, "target_confirmed_wins"),
            "target_borderline": int_num(quality or summary, "target_borderline"),
            "target_losses": int_num(quality or summary, "target_losses"),
            "local_accept": int_num(quality or summary, "local_accept", int_num(summary, "accept")),
            "local_review": int_num(quality or summary, "local_review", int_num(summary, "review")),
            "local_reject": int_num(quality or summary, "local_reject", int_num(summary, "reject")),
            "metric_losses": int_num(summary, "metric_losses"),
            "missing_quality_rows": int_num(quality, "missing_quality_rows"),
            "summary_path": str(selector_summary_path(summary_dir, selector)),
            "summary_exists": selector_summary_path(summary_dir, selector).exists(),
        }
    return statuses


def label_queue_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "rows": 0, "unresolved_review_rows": 0, "next_review_files": []}
    rows = read_rows(path)
    decision_counts = Counter()
    by_triage = Counter()
    by_triage_decision: dict[str, Counter[str]] = defaultdict(Counter)
    unresolved: list[str] = []
    positive: list[str] = []
    negative: list[str] = []
    for row in rows:
        triage = normalize(row.get("auto_triage_label", "")) or "missing"
        decision = normalize(row.get("manual_label", ""))
        page = f"{row.get('split', '').strip()}/{row.get('file', '').strip()}"
        if not decision:
            bucket = "blank"
            if triage in REVIEW_LABELS:
                unresolved.append(page)
        elif decision in POSITIVE_DECISIONS:
            bucket = "positive"
            positive.append(page)
        elif decision in NEGATIVE_DECISIONS:
            bucket = "negative"
            negative.append(page)
        else:
            bucket = "other"
        decision_counts[bucket] += 1
        by_triage[triage] += 1
        by_triage_decision[triage][bucket] += 1
    return {
        "exists": True,
        "path": str(path),
        "rows": len(rows),
        "decision_counts": dict(decision_counts),
        "by_triage": dict(by_triage),
        "by_triage_decision": {key: dict(value) for key, value in sorted(by_triage_decision.items())},
        "unresolved_review_rows": len(unresolved),
        "next_review_files": unresolved[:20],
        "positive_files": positive,
        "negative_files": negative,
    }


def build_checks(args: argparse.Namespace, release: dict[str, Any], queue: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [
        {
            "name": "release_selector_summary_exists",
            "passed": bool(release.get("summary_exists")),
            "actual": release.get("summary_path"),
            "required": "existing summary JSON",
        },
        {
            "name": "selected_floor",
            "passed": int(release["selected"]) >= args.min_selected,
            "actual": release["selected"],
            "required": f">= {args.min_selected}",
        },
        {
            "name": "metric_losses",
            "passed": int(release["metric_losses"]) <= args.max_metric_losses,
            "actual": release["metric_losses"],
            "required": f"<= {args.max_metric_losses}",
        },
        {
            "name": "local_reject",
            "passed": int(release["local_reject"]) <= args.max_local_reject,
            "actual": release["local_reject"],
            "required": f"<= {args.max_local_reject}",
        },
        {
            "name": "target_losses",
            "passed": int(release["target_losses"]) <= args.max_target_losses,
            "actual": release["target_losses"],
            "required": f"<= {args.max_target_losses}",
        },
        {
            "name": "target_borderline",
            "passed": int(release["target_borderline"]) <= args.max_target_borderline,
            "actual": release["target_borderline"],
            "required": f"<= {args.max_target_borderline}",
        },
        {
            "name": "missing_quality_rows",
            "passed": int(release["missing_quality_rows"]) <= args.max_missing_quality_rows,
            "actual": release["missing_quality_rows"],
            "required": f"<= {args.max_missing_quality_rows}",
        },
    ]
    if not args.allow_unresolved_label_queue:
        checks.append({
            "name": "post125_review_queue_resolved",
            "passed": int(queue.get("unresolved_review_rows", 0)) == 0,
            "actual": queue.get("unresolved_review_rows", 0),
            "required": "0 blank manual_label rows in promote_candidate/borderline_review",
        })
    return checks


def next_actions(checks: list[dict[str, Any]], release_selector: str, queue: dict[str, Any]) -> list[str]:
    failed = {check["name"] for check in checks if not check["passed"]}
    actions: list[str] = []
    if "target_losses" in failed:
        actions.append(f"Remove or repair target-loss pages from {release_selector}; prefer target-aware candidate generation before widening coverage.")
    if "target_borderline" in failed:
        actions.append(f"Convert selected target-borderline pages in {release_selector} into visual labels or confirmed wins.")
    if "post125_review_queue_resolved" in failed:
        examples = ", ".join(queue.get("next_review_files", [])[:8])
        actions.append(f"Continue post-125 visual labeling for promote/borderline rows; next examples: {examples}.")
    if "selected_floor" in failed:
        actions.append("Do not shrink below the selected-page floor without documenting the quality/coverage tradeoff.")
    if "metric_losses" in failed or "local_reject" in failed:
        actions.append("Reject the candidate selector or add vetoes before any product-default promotion.")
    if not actions:
        actions.append("All configured gates pass; materialize predictions and run final page/crop review plus readiness smoke before declaring release quality.")
    return actions


def markdown_report(report: dict[str, Any]) -> str:
    release = report["release_selector"]
    lines = [
        "# Quality Goal Status",
        "",
        f"Release selector: `{release['selector']}`",
        f"Release ready: **{report['release_ready']}**",
        "",
        "## Gates",
    ]
    for check in report["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- {mark}: `{check['name']}` actual={check['actual']} required={check['required']}")
    lines.extend(["", "## Selector Snapshot"])
    for selector, status in report["selectors"].items():
        lines.append(
            f"- `{selector}`: selected={status['selected']} target_losses={status['target_losses']} "
            f"target_borderline={status['target_borderline']} metric_losses={status['metric_losses']} "
            f"local_reject={status['local_reject']}"
        )
    queue = report["label_queue"]
    lines.extend([
        "",
        "## Label Queue",
        f"- rows={queue.get('rows', 0)} unresolved_review_rows={queue.get('unresolved_review_rows', 0)}",
        f"- decision_counts={queue.get('decision_counts', {})}",
        "",
        "## Next Actions",
    ])
    for action in report["next_actions"]:
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    selectors = load_selector_status(args)
    if args.release_selector not in selectors:
        raise ValueError(f"Release selector not loaded: {args.release_selector}")
    queue = label_queue_status(Path(args.label_queue_csv))
    target_quality_summary = read_json(Path(args.target_quality_summary_json))
    checks = build_checks(args, selectors[args.release_selector], queue)
    report = {
        "release_ready": all(check["passed"] for check in checks),
        "release_selector": selectors[args.release_selector],
        "checks": checks,
        "selectors": selectors,
        "label_queue": queue,
        "target_quality_summary": target_quality_summary,
        "next_actions": next_actions(checks, args.release_selector, queue),
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report))


if __name__ == "__main__":
    main()
