# Sustainable Quality Goal

This is the active long-running goal for the EnsExam-GAN handwriting-removal pipeline: keep improving selector and repair quality until the candidate is safe enough to promote, and record rejected paths when they fail the gates.

## Baseline

Use `zero_reject_veto_125_accept_clean` as the high-coverage safety reference, but do not treat it as fully quality-solved: it still has target-aware loss and borderline pages. Use `zero_reject_veto_112_zero_target_loss` as the quality-first release candidate until reviewed labels or improved generation safely recover coverage.

Current quality facts to preserve:

```text
zero_reject_veto_125_accept_clean: selected=125 metric_losses=0 local_reject=0 target_losses=13 target_borderline=58
zero_reject_veto_112_zero_target_loss: selected=112 metric_losses=0 local_reject=0 target_losses=0 target_borderline=58
post-125 auto-triage queue: 53 rows, with promote_candidate and borderline_review rows requiring explicit visual labels
```

## Definition Of Done

A selector is release-ready only when all configured gates pass:

```text
metric_losses <= 0
local_reject <= 0
target_losses <= 0
missing_quality_rows <= 0
unresolved selected target_borderline pages <= 0, unless explicitly accepted by durable visual labels
post-125 promote_candidate/borderline_review rows have explicit manual_label decisions before promotion
materialized predictions pass page/crop review on changed pages across holdout40, SCUT115, next120, and train160
readiness smoke does not regress residual or overerase against the documented baseline
```

The default gate is intentionally strict on `target_borderline`: unresolved borderlines keep the long-running goal active. If a product release accepts a smaller zero-loss selector while coverage work continues, record that as a scoped release decision rather than closing the quality goal.

## Status Command

Run this at the start and end of each continuation pass:

```bash
$ENSEXAM_PYTHON scripts/analysis/report_quality_goal_status.py \
  --release-selector zero_reject_veto_112_zero_target_loss \
  --output-json outputs/balanced007_ranker_expansion_source_eval_20260708/quality_goal_status_latest.json \
  --output-md outputs/balanced007_ranker_expansion_source_eval_20260708/quality_goal_status_latest.md
```

Generated `outputs/` status files are local evidence and should not be committed. Commit only reusable scripts, docs, or consolidated decision records.

## Sustainable Loop

1. Run `report_quality_goal_status.py` and identify the first failing gate.
2. If post-125 labels are unresolved, review `promote_candidate` before `borderline_review`; write explicit `manual_label` values before applying any selector overlay.
3. Apply reviewed labels with `apply_selector_label_queue_promotions.py`; blank labels must promote zero pages.
4. Materialize the candidate with `materialize_fixed_page_selector.py` and build page/crop review packs for every changed page.
5. Re-run target-quality summaries and readiness smoke; reject candidates with target losses, local rejects, metric losses, or visible paper/text regressions.
6. Promote the selector only when the status command and visual/materialized evidence pass; otherwise document the rejected family and move to the next failure bucket.

## Stop Rules

Do not run another broad retrain, threshold-only micro-probe, or selector-only widening pass unless it names the failure bucket, expected page-level lift, and regression guard before it runs. Prefer target-aware loss repair, formal visual labels, and bounded hardcase fine-tunes over repeated threshold search.
