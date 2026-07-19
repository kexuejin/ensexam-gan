# Zero-Loss 112 Review-First Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce and process bounded evidence shards that remove unresolved
quality status from the 112-page zero-target-loss selector before expanding it.

**Architecture:** Existing quality-goal scripts remain the source of truth.
The queue builder ranks selected target-borderline and post-125 rows, review
packs provide bounded evidence, and the label applier updates overlays without
altering inference rules. Status and materialization commands are the promotion
gate.

**Tech Stack:** Python 3.10 environment from `.env`, CSV/JSON evidence, existing
`scripts/analysis` and `scripts/infer` tools.

---

### Task 1: Establish the immutable baseline

**Files:**
- Read: `docs/quality-goal.md`
- Read: `outputs/balanced007_ranker_expansion_source_eval_20260708/final_candidate_selectors/zero_reject_veto_112_zero_target_loss_selected.csv`
- Create: a fresh `outputs/.../quality_goal_review_YYYYMMDD/quality_goal_status_before.json`

**Step 1:** Load the project environment.

Run: `source .env && $ENSEXAM_PYTHON --version`

**Step 2:** Generate the baseline status with the fixed 112-page release selector.

Run: `source .env && $ENSEXAM_PYTHON scripts/analysis/report_quality_goal_status.py --release-selector zero_reject_veto_112_zero_target_loss --output-json outputs/balanced007_ranker_expansion_source_eval_20260708/quality_goal_review_YYYYMMDD/quality_goal_status_before.json --output-md outputs/balanced007_ranker_expansion_source_eval_20260708/quality_goal_review_YYYYMMDD/quality_goal_status_before.md`

**Step 3:** Confirm the only selected-page quality blocker is unresolved
target-borderline evidence, not a target loss, metric loss, or local reject.

### Task 2: Materialize one bounded review shard

**Files:**
- Read: `scripts/analysis/build_quality_goal_review_queue.py`
- Read: `scripts/analysis/build_target_quality_bucket_review_packs.py`
- Create: `outputs/.../quality_goal_review_YYYYMMDD/target_borderline_batch01.csv`
- Create: `outputs/.../quality_goal_review_YYYYMMDD/review_pack/`

**Step 1:** Build a fresh queue; never overwrite an earlier evidence directory.

Run: `source .env && $ENSEXAM_PYTHON scripts/analysis/build_quality_goal_review_queue.py --output-dir outputs/balanced007_ranker_expansion_source_eval_20260708/quality_goal_review_YYYYMMDD`

**Step 2:** Validate the batch contains unique page keys, permitted decision
columns, and only bounded review items.

**Step 3:** Generate exactly one compliant review pack from the highest-priority
batch. Do not inspect original-resolution pages.

**Step 4:** Persist one row per reviewed sample with `sample_key`, `decision`,
`reason_code`, `metric_summary`, `evidence_paths`, and `reviewer`.

### Task 3: Apply only explicit labels and test the promotion gate

**Files:**
- Read: `scripts/analysis/apply_quality_goal_review_labels.py`
- Read: `scripts/analysis/apply_selector_label_queue_promotions.py`
- Create: fresh target-quality and post-125 overlay CSVs under the shard output
- Create: materialized selector output only if labels authorize a change

**Step 1:** Run the applicable label applier on the completed shard. Blank or
ambiguous labels must make no selected-page or coverage change.

**Step 2:** Re-run `summarize_selector_target_quality.py` and
`report_quality_goal_status.py` against the new overlays.

**Step 3:** If an explicit post-125 acceptance expands coverage, materialize the
selector with `scripts/infer/materialize_fixed_page_selector.py` and compare all
changed pages using existing local metric and review-pack tools.

**Step 4:** Run the documented readiness smoke. Reject the change if target
losses, metric losses, local rejects, source/output mismatches, residual
regression, or overerase regression appear.

### Task 4: Record the result

**Files:**
- Modify: `docs/quality-goal.md` only for a consolidated result or a new
  promotion/rejection decision
- Modify: `docs/rejected-directions.md` only if a reusable route is rejected

**Step 1:** Write only the aggregate result and evidence paths; do not commit
generated predictions, datasets, or review images.

**Step 2:** Run the relevant unit tests plus strict CSV/JSON artifact validation.

**Step 3:** Commit reusable scripts and decision documentation with Lore trailers
only after the result is consolidated.

