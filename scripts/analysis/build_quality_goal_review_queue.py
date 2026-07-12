#!/usr/bin/env python3
"""Build prioritized review queues for the sustainable quality goal.

The quality goal has two active review surfaces:
1. target-borderline pages already selected by the quality-first release selector;
2. unresolved post-125 auto-triage rows that need explicit manual labels before
   selector promotion.

This helper turns those surfaces into stable CSV queues and first-batch review
inputs for the existing page/crop review pack builders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path("outputs/balanced007_ranker_expansion_source_eval_20260708")
DEFAULT_SELECTED_CSV = ROOT / "final_candidate_selectors/zero_reject_veto_112_zero_target_loss_selected.csv"
DEFAULT_QUALITY_CSV = ROOT / "target_quality_scoring_20260708/full435_target_quality_v2.csv"
DEFAULT_POST125_CSV = ROOT / "labeling_queue_post125_coverage_20260708/auto_triage/auto_triage_labels.csv"
REVIEW_TRIAGE_LABELS = {"promote_candidate", "borderline_review"}
POSITIVE_DECISIONS = {"promote", "accept", "approve", "approved", "yes", "y", "true", "1"}
NEGATIVE_DECISIONS = {"reject", "rejected", "keep", "keep_review", "skip", "defer", "no", "n", "false", "0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", default=str(DEFAULT_SELECTED_CSV))
    parser.add_argument("--quality-csv", default=str(DEFAULT_QUALITY_CSV))
    parser.add_argument("--post125-csv", default=str(DEFAULT_POST125_CSV))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--post125-batch-size", type=int, default=16)
    parser.add_argument(
        "--include-post125-label",
        action="append",
        default=[],
        help="auto_triage_label to include from post-125 queue. Defaults to promote_candidate and borderline_review.",
    )
    return parser.parse_args()


def normalize(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def page_key(row: dict[str, str]) -> tuple[str, str]:
    split = row.get("split", "").strip()
    file = row.get("file", "").strip()
    page = row.get("page", "").strip()
    if split and file:
        return split, file
    if page.count("/") == 1:
        split, file = page.split("/", 1)
        return split, file
    if file.count("/") == 1:
        split, file = file.split("/", 1)
        return split, file
    return "", ""


def page_text(row: dict[str, str]) -> str:
    split, file = page_key(row)
    return f"{split}/{file}" if split and file else ""


def fnum(row: dict[str, str], field: str, default: float = 0.0) -> float:
    value = row.get(field, "")
    if value == "":
        return default
    try:
        number = float(value)
    except ValueError:
        return default
    return number if math.isfinite(number) else default


def selected_pages(path: Path) -> set[tuple[str, str]]:
    pages = {page_key(row) for row in read_rows(path)}
    if ("", "") in pages:
        raise ValueError(f"Selected CSV contains a row without split/file: {path}")
    return pages


def target_borderline_score(row: dict[str, str]) -> tuple[float, str]:
    reason = row.get("target_quality_reason", "")
    local_verdict = normalize(row.get("local_verdict", ""))
    dark = fnum(row, "target_dark_damage_changed_ratio")
    over = fnum(row, "overerase_changed_ratio")
    changed = fnum(row, "changed_ratio")
    help_hurt = fnum(row, "help_hurt_ratio")
    residual_help_hurt = fnum(row, "residual_help_hurt_ratio")
    score = 0.0
    parts: list[str] = []
    if local_verdict == "review":
        score += 100.0
        parts.append("local_review")
    if "target_dark_damage" in reason:
        score += 70.0 + dark * 100.0
        parts.append(f"target_dark_damage={dark:.4f}")
    if "overerase" in reason:
        score += 55.0 + over * 100.0
        parts.append(f"overerase={over:.4f}")
    score += min(changed * 10000.0, 30.0)
    if help_hurt and help_hurt < 1.75:
        score += (1.75 - help_hurt) * 20.0
        parts.append(f"low_help_hurt={help_hurt:.3f}")
    if residual_help_hurt > 1000.0:
        score += min(residual_help_hurt / 1000.0, 10.0)
        parts.append(f"large_residual_signal={residual_help_hurt:.1f}")
    if not parts:
        parts.append(reason or "target_borderline")
    return score, "; ".join(parts)


def post125_score(row: dict[str, str]) -> tuple[float, str]:
    triage = normalize(row.get("auto_triage_label", ""))
    priority = normalize(row.get("auto_review_priority", ""))
    group = normalize(row.get("label_queue_group", ""))
    gain = fnum(row, "gain")
    help_hurt = fnum(row, "help_hurt_ratio")
    residual_help_hurt = fnum(row, "residual_help_hurt_ratio")
    changed = fnum(row, "changed_ratio")
    score = 0.0
    parts: list[str] = []
    if triage == "promote_candidate":
        score += 100.0
        parts.append("promote_candidate")
    elif triage == "borderline_review":
        score += 60.0
        parts.append("borderline_review")
    if priority == "low":
        score += 20.0
        parts.append("low_review_risk")
    elif priority == "medium":
        score += 10.0
        parts.append("medium_review_risk")
    if group == "post125_metric_safe_accept":
        score += 20.0
        parts.append("new_metric_safe_accept")
    elif group == "post125_metric_safe_review":
        score += 12.0
        parts.append("new_metric_safe_review")
    elif group == "current125_review":
        score += 5.0
        parts.append("current125_quality_confirmation")
    score += min(gain * 1000.0, 25.0)
    score += min(help_hurt, 10.0)
    score += min(residual_help_hurt / 1000.0, 8.0)
    score -= min(changed * 1000.0, 8.0)
    return score, "; ".join(parts)


def frontmatter(row: dict[str, str], stream: str, rank: int, score: float, reason: str, guidance: str) -> dict[str, str]:
    split, file = page_key(row)
    return {
        "review_rank": str(rank),
        "review_stream": stream,
        "page": f"{split}/{file}",
        "manual_label": "",
        "manual_notes": "",
        "review_guidance": guidance,
        "priority_score": f"{score:.6f}",
        "priority_reason": reason,
    }


def with_frontmatter(rows: Iterable[tuple[dict[str, str], float, str]], stream: str, guidance: str) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for rank, (row, score, reason) in enumerate(rows, start=1):
        output.append({**frontmatter(row, stream, rank, score, reason, guidance), **row})
    return output


def unresolved_decision(row: dict[str, str]) -> bool:
    decision = normalize(row.get("manual_label", ""))
    if not decision:
        return True
    return decision not in POSITIVE_DECISIONS and decision not in NEGATIVE_DECISIONS


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    selected = selected_pages(Path(args.selected_csv))
    quality_rows = read_rows(Path(args.quality_csv))
    post125_rows = read_rows(Path(args.post125_csv))
    include_post125 = {normalize(value) for value in args.include_post125_label} or set(REVIEW_TRIAGE_LABELS)

    target_scored = []
    for row in quality_rows:
        if page_key(row) not in selected:
            continue
        if normalize(row.get("target_quality_label", "")) != "borderline":
            continue
        score, reason = target_borderline_score(row)
        target_scored.append((row, score, reason))
    target_scored.sort(key=lambda item: (item[1], page_text(item[0])), reverse=True)
    target_queue = with_frontmatter(
        target_scored,
        "release_target_borderline",
        "confirm_win / keep_borderline / reject_selector_page / repair_needed",
    )

    post125_scored = []
    for row in post125_rows:
        triage = normalize(row.get("auto_triage_label", ""))
        if triage not in include_post125:
            continue
        if not unresolved_decision(row):
            continue
        score, reason = post125_score(row)
        post125_scored.append((row, score, reason))
    post125_scored.sort(key=lambda item: (item[1], page_text(item[0])), reverse=True)
    post125_queue = with_frontmatter(
        post125_scored,
        "post125_unresolved",
        "promote / keep_review / reject",
    )

    target_batch = target_queue[: args.target_batch_size]
    post125_batch = post125_queue[: args.post125_batch_size]
    combined_batch = target_batch + post125_batch

    paths = {
        "target_queue_csv": output_dir / "target_borderline_queue.csv",
        "target_batch_csv": output_dir / "target_borderline_batch01.csv",
        "post125_queue_csv": output_dir / "post125_unresolved_review_queue.csv",
        "post125_batch_csv": output_dir / "post125_unresolved_batch01.csv",
        "combined_batch_csv": output_dir / "combined_next_review_batch01.csv",
        "summary_json": output_dir / "quality_goal_review_queue_summary.json",
    }
    write_csv(paths["target_queue_csv"], target_queue)
    write_csv(paths["target_batch_csv"], target_batch)
    write_csv(paths["post125_queue_csv"], post125_queue)
    write_csv(paths["post125_batch_csv"], post125_batch)
    write_csv(paths["combined_batch_csv"], combined_batch)

    summary = {
        "selected_csv": args.selected_csv,
        "quality_csv": args.quality_csv,
        "post125_csv": args.post125_csv,
        "target_borderline_rows": len(target_queue),
        "target_batch_rows": len(target_batch),
        "post125_unresolved_rows": len(post125_queue),
        "post125_batch_rows": len(post125_batch),
        "combined_batch_rows": len(combined_batch),
        "target_borderline_by_split": dict(Counter(row.get("split", "") for row in target_queue)),
        "post125_by_triage": dict(Counter(row.get("auto_triage_label", "") for row in post125_queue)),
        "post125_by_group": dict(Counter(row.get("label_queue_group", "") for row in post125_queue)),
        "top_target_pages": [row["page"] for row in target_batch[:10]],
        "top_post125_pages": [row["page"] for row in post125_batch[:10]],
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
