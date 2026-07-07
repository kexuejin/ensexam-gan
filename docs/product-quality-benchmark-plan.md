# Product Quality Benchmark Plan

## Goal

Move the project from threshold-only micro-tuning to page-level product-quality evaluation.

The current union gate is safe but too narrow:

```text
validated pages: 435
candidate-selected pages: 6
next120 selected pages: 0
current status: safety hypothesis, not product-quality solution
```

The next milestone is not another hand-tuned selector. The next milestone is a benchmark that can
tell whether a candidate improves enough pages without visible regressions.

## Phase 1: Build The Manual Review Set

Create a compact review set from existing local outputs before training more models.

Initial page sources:

```text
union-selected wins:
  scut115: 17.jpg, 156.jpg, 303.jpg, 370.jpg, 371.jpg
  holdout40: 466.jpg

known selector rejects / unsafe pages:
  train160 relaxed-interval bad pages: 166.jpg, 190.jpg, 192.jpg
  old strict selected regression: 254.jpg

coverage negatives:
  next120 pages where union selected 0/120

special failure cases:
  correction-fluid / whiteout pages from existing whiteout repair analysis
  gray paper-tone residue pages
  low-contrast handwriting pages
  printed-text preservation risk pages
```

Deliverable:

```text
docs/product-quality-review-pages.csv
```

Required columns:

```text
split,file,bucket,source_input,baseline_pred,candidate_pred,target,review_pack,notes
```

## Phase 2: Add Page-Level Labels

Every candidate comparison should receive one page-level visual label:

```text
clear_win      user would clearly prefer candidate
slight_win     candidate is better but subtle
noop           no meaningful visible difference
slight_loss    candidate is slightly worse or dirtier
clear_loss     candidate visibly damages content/background
```

Add issue flags independently of the win/loss label:

```text
residual_handwriting
correction_fluid_white_patch
gray_paper_tone
low_contrast_handwriting
printed_text_damage
paper_texture_dirty
halo_or_edge_artifact
overerase
```

Deliverable:

```text
docs/product-quality-labels.csv
```

Required columns:

```text
split,file,candidate,label,flags,reviewer,review_date,comment
```

## Phase 3: Report Candidate Quality By Bucket

Add a local analysis script that summarizes page labels and metric deltas together.

Input:

```text
docs/product-quality-labels.csv
candidate metrics.csv files
baseline metrics.csv files
```

Output:

```text
outputs/product_quality_reports/<candidate>/summary.csv
outputs/product_quality_reports/<candidate>/bucket_summary.csv
outputs/product_quality_reports/<candidate>/selected_pages.csv
```

Minimum report fields:

```text
pages
selected_pages
coverage_rate
clear_win
slight_win
noop
slight_loss
clear_loss
net_win_rate
clear_loss_rate
mean_residual_delta
mean_overerase_delta
bucket
```

Promotion thresholds for a serious candidate:

```text
coverage_rate >= 20%
clear_loss = 0 on core benchmark
clear_win + slight_win materially exceeds slight_loss + clear_loss
no printed-text damage clear_loss
no paper-tone / whiteout regression on correction-fluid bucket
aggregate overerase delta <= 0 on validation splits
```

## Phase 4: Replace Hand-Tuned Selector Work

Only after labels exist, compare three higher-leverage routes:

```text
learned selector:
  train/calibrate a page-level selector using local metric features and visual labels

candidate objective:
  change generator training objective to improve named failure buckets while preserving paper tone

specialized repair:
  separate correction-fluid / paper-tone repair branch if whiteout pages remain visually worse
```

Do not run another threshold-only selector sweep unless it is used as a baseline for one of those
routes.

## Phase 5: Expand Validation

Use remaining local and external data only after the benchmark can score candidates.

Validation order:

```text
1. remaining unused SCUT train pages
2. held-out correction-fluid pages
3. ExamInk-Seg-like external pages where target semantics are clear
4. broader document cleanup pages with manual labels
```

Target before product default:

```text
coverage_rate: 20-30% on relevant failure buckets
clear_loss: 0 on core benchmark
manual visual review: pass
aggregate residual: improves or unchanged
aggregate overerase: unchanged or lower
```

## Immediate Next Task

Generate `docs/product-quality-review-pages.csv` from existing outputs and include:

```text
union selected pages
visible-delta regress pages
train160 relaxed bad pages
whiteout repair examples
representative next120 no-op pages
```

Then create contact-sheet review packs for those pages and label them before any more training.

## Manual Labeling Workflow

The first review pack has already been generated locally:

```text
contact sheet:
  outputs/product_quality_review_pack_initial_20260706/contact_sheet.png

chunked sheets:
  outputs/product_quality_review_pack_initial_20260706/chunks/chunk_01.png
  outputs/product_quality_review_pack_initial_20260706/chunks/chunk_02.png
  outputs/product_quality_review_pack_initial_20260706/chunks/chunk_03.png
  outputs/product_quality_review_pack_initial_20260706/chunks/chunk_04.png
  outputs/product_quality_review_pack_initial_20260706/chunks/chunk_05.png

single-page review images:
  outputs/product_quality_review_pack_initial_20260706/pages/
```

Each review image is ordered left to right:

```text
input | baseline | candidate | target
```

Fill this file after local visual review:

```text
docs/product-quality-labels.csv
```

Allowed `label` values:

```text
clear_win      candidate is clearly better than baseline
slight_win     candidate is better, but subtle
noop           no meaningful visible difference
slight_loss    candidate is slightly worse / dirtier
clear_loss     candidate clearly damages content, paper tone, or readability
```

Allowed `flags` values are comma-separated:

```text
residual_handwriting
correction_fluid_white_patch
gray_paper_tone
low_contrast_handwriting
printed_text_damage
paper_texture_dirty
halo_or_edge_artifact
overerase
missed_coverage
```

Label conservatively:

```text
- If candidate and baseline look the same at page scale, use noop.
- If the candidate removes handwriting but also dirties paper tone, use slight_loss or clear_loss
  depending on whether the damage is user-visible.
- If the candidate is selected only by metrics but not visibly better, use noop.
- If a next120 page should have been improved but the selector stayed baseline, use noop plus
  missed_coverage in flags.
- Leave label blank only when the review image is insufficient or the page needs a zoomed crop.
```

After labels are filled, run:

```bash
$ENSEXAM_PYTHON scripts/analysis/summarize_product_quality_labels.py \
  --labels-csv docs/product-quality-labels.csv \
  --review-csv docs/product-quality-review-pages.csv \
  --output-dir outputs/product_quality_label_summary_manual_YYYYMMDD
```

Promotion discussion should use:

```text
outputs/product_quality_label_summary_manual_YYYYMMDD/candidate_summary.csv
outputs/product_quality_label_summary_manual_YYYYMMDD/bucket_summary.csv
outputs/product_quality_label_summary_manual_YYYYMMDD/pending_labels.csv
```

Do not promote a candidate if `pending_labels.csv` still contains rows from a core bucket.

## Residual-Delta Review Queue

The residual-delta branch now has a deterministic page-level review queue generator. It expands the
manual review surface from the 30-row seed set into metric-win, metric-loss, selector-hit, and
high-activity risk buckets without inventing labels.

```text
script:
  scripts/analysis/build_residual_delta_review_set.py

verified output:
  outputs/product_quality_review_residual_delta_20260707/review-pages.csv
  outputs/product_quality_review_residual_delta_20260707/labels-template.csv
  outputs/product_quality_review_residual_delta_20260707/review_pack/contact_sheet.png

rows:
  total = 142
  scut115 = 78
  holdout40 = 64

buckets:
  residual_delta_metric_win = 39
  residual_delta_metric_loss = 48
  residual_delta_joint_selector = 7
  residual_delta_high_activity_risk = 48
```

Regenerate the queue and contact sheet with:

```bash
$ENSEXAM_PYTHON scripts/analysis/build_residual_delta_review_set.py \
  --split scut115:outputs/scut_test115_second_stage_baseline_20260705/metrics.csv:outputs/eval_scut115_residual_delta_bias3_scale008_best_base12_delta2_20260707/metrics.csv:outputs/selector_features_residual_delta_joint_20260707_nodummy/page_features.csv \
  --split holdout40:outputs/holdout40_second_stage_nearworst_safe_step1_t98_20260705/metrics.csv:outputs/eval_holdout40_residual_delta_bias3_scale008_best_base12_delta2_20260707/metrics.csv:outputs/selector_features_residual_delta_joint_20260707_nodummy/page_features.csv \
  --candidate-name residual_delta_bias3_scale008 \
  --output-review-csv outputs/product_quality_review_residual_delta_YYYYMMDD/review-pages.csv \
  --output-labels-template outputs/product_quality_review_residual_delta_YYYYMMDD/labels-template.csv \
  --review-pack outputs/product_quality_review_residual_delta_YYYYMMDD

$ENSEXAM_PYTHON scripts/analysis/build_product_quality_review_pack.py \
  --review-csv outputs/product_quality_review_residual_delta_YYYYMMDD/review-pages.csv \
  --output-dir outputs/product_quality_review_residual_delta_YYYYMMDD/review_pack \
  --thumb-width 360 \
  --thumb-height 260 \
  --max-contact-rows 80
```

Do not merge the generated labels template into `docs/product-quality-labels.csv` until the pages
have been reviewed visually. For selector training, the next required artifact is the filled
`labels-template.csv`; metric wins alone are not acceptable labels.

To reduce manual review cost, generate auto-suggested labels as triage hints:

```bash
$ENSEXAM_PYTHON scripts/analysis/suggest_product_quality_labels.py \
  --labels-template outputs/product_quality_review_residual_delta_YYYYMMDD/labels-template.csv \
  --review-csv outputs/product_quality_review_residual_delta_YYYYMMDD/review-pages.csv \
  --features-csv outputs/selector_features_residual_delta_joint_20260707_nodummy/page_features.csv \
  --output-csv outputs/product_quality_review_residual_delta_YYYYMMDD/labels-auto-suggest.csv
```

Verified residual-delta suggestion output:

```text
rows = 142
auto_suggest_label:
  slight_loss = 74
  slight_win = 48
  noop = 20
auto_confidence:
  medium = 114
  low = 28
auto_review_priority:
  high = 102
  medium = 26
  low = 14
```

Treat `auto_suggest_label` as a review aid only. Human review should focus first on
`auto_review_priority=high`, especially low-confidence metric wins with high gate activity and any
suggested losses. Low-priority noops can be sampled instead of exhaustively reviewed.

After visual labels are filled, calibrate a page-level selector from confirmed labels and page
features:

```bash
$ENSEXAM_PYTHON scripts/analysis/calibrate_page_quality_selector.py \
  --labels-csv outputs/product_quality_review_residual_delta_YYYYMMDD/labels-template.csv \
  --features-csv outputs/selector_features_residual_delta_joint_20260707_nodummy/page_features.csv \
  --candidate residual_delta_bias3_scale008 \
  --output-dir outputs/page_quality_selector_residual_delta_YYYYMMDD \
  --min-labeled 40
```

The calibrator refuses to mine rules when confirmed labels are insufficient:

```text
empty labels-template smoke:
  status = insufficient_labeled
  joined_labeled = 0
  min_labeled = 40
```

An auto-suggest smoke run proves the rule-mining path, but must not be treated as product evidence:

```text
labels source: labels-auto-suggest.csv
joined_labeled = 142
positive = 48
negative = 74
neutral = 20
best smoke rule:
  active_gray_p25 >= 114 AND candidate_delta_p99 >= 2
  selected = 20
  positive = 19
  negative = 0
  precision = 0.950
```

Only use rules mined from visually confirmed `label` values for promotion decisions. Rules mined
from `auto_suggest_label` are smoke tests for plumbing and prioritization only.

For the first calibration pass, build a balanced 40-row subset instead of reviewing all 142 pages:

```bash
$ENSEXAM_PYTHON scripts/analysis/build_minimal_labeling_subset.py \
  --auto-suggest-csv outputs/product_quality_review_residual_delta_YYYYMMDD/labels-auto-suggest.csv \
  --review-index-csv outputs/product_quality_review_residual_delta_YYYYMMDD/review_pack/index.csv \
  --output-csv outputs/product_quality_review_residual_delta_YYYYMMDD/minimal-labeling-subset-40.csv \
  --contact-sheet outputs/product_quality_review_residual_delta_YYYYMMDD/minimal-labeling-subset-40.png \
  --chunk-dir outputs/product_quality_review_residual_delta_YYYYMMDD/minimal_subset_chunks \
  --max-total 40 \
  --min-per-split 12 \
  --min-per-bucket 6 \
  --min-per-suggest-label 8
```

Verified residual-delta minimal subset:

```text
rows = 40
splits:
  holdout40 = 22
  scut115 = 18
buckets:
  residual_delta_high_activity_risk = 16
  residual_delta_metric_win = 10
  residual_delta_metric_loss = 8
  residual_delta_joint_selector = 6
auto_suggest_label:
  slight_win = 25
  noop = 8
  slight_loss = 7
auto_review_priority:
  high = 32
  medium = 6
  low = 2
missing_review_images = 0
contact_sheet:
  outputs/product_quality_review_residual_delta_20260707/minimal-labeling-subset-40.png
chunked sheets:
  outputs/product_quality_review_residual_delta_20260707/minimal_subset_chunks/chunk_01.png
  outputs/product_quality_review_residual_delta_20260707/minimal_subset_chunks/chunk_02.png
  outputs/product_quality_review_residual_delta_20260707/minimal_subset_chunks/chunk_03.png
  outputs/product_quality_review_residual_delta_20260707/minimal_subset_chunks/chunk_04.png
```

Fill the `label`, `flags`, `reviewer`, `review_date`, and `comment` columns in this subset first.
Once at least 40 rows are confirmed, use it as the first input to `calibrate_page_quality_selector.py`.

## AI-Provisional Visual Labels

An AI visual pass over the 40-row minimal subset produced a provisional label file:

```text
outputs/product_quality_review_residual_delta_20260707/minimal-labeling-subset-40-ai-provisional.csv
```

Provisional summary:

```text
slight_win = 22
noop = 11
slight_loss = 7
clear_win = 0
clear_loss = 0
net_wins = 15
```

Bucket-level provisional signal:

```text
residual_delta_metric_win:
  wins = 10 / 10
  losses = 0 / 10
residual_delta_joint_selector:
  wins = 5 / 6
  noop = 1 / 6
  losses = 0 / 6
residual_delta_metric_loss:
  wins = 0 / 8
  noop = 2 / 8
  losses = 6 / 8
residual_delta_high_activity_risk:
  wins = 7 / 16
  noop = 8 / 16
  losses = 1 / 16
```

This is useful directionally but remains provisional. Do not merge it into the manual benchmark as a
confirmed label set without review.

## AI-Provisional Fixed Rule Replay

The provisional 40-row selector calibration surfaced a simple candidate rule:

```text
active_gray_p25 >= 123
```

Full SCUT115 + holdout40 replay for this fixed rule:

```text
script:
  scripts/analysis/evaluate_fixed_page_selector.py
output:
  outputs/page_quality_selector_ai_provisional_40_20260707/fixed_rule_active_gray_p25_ge_123_eval.csv

SCUT115:
  pages = 115
  selected = 19
  coverage = 16.5%
  residual: 0.114225 -> 0.113408
  residual_gain = 0.000817
  overerase_delta = -0.00000782
  selected metric wins/losses = 13 / 6

holdout40:
  pages = 40
  selected = 5
  coverage = 12.5%
  residual: 0.130543 -> 0.129855
  residual_gain = 0.000688
  overerase_delta = -0.00000235
  selected metric wins/losses = 3 / 2

combined:
  pages = 155
  selected = 24
  coverage = 15.5%
  residual: 0.118436 -> 0.117653
  residual_gain = 0.000783
  overerase_delta = -0.00000641
  selected metric wins/losses = 16 / 8
```

This is the first selector candidate in this branch that improves aggregate residual and overerase
while selecting more than a tiny handful of pages. It is still below the 20-30% product-coverage
target and includes 8 selected metric-loss pages, so it should be treated as a candidate for visual
review and refinement, not a product default.

A stricter zero-metric-loss refinement is also reproducible:

```text
active_gray_p25 >= 123
AND active_baseline_edit_p95 <= 149
AND candidate_delta_mean >= 0.0182428157494
```

Replay output:

```text
outputs/page_quality_selector_ai_provisional_40_20260707/fixed_rule_active_gray_p25_ge_123_refined_zero_loss_eval.csv
```

Result:

```text
SCUT115:
  selected = 12 / 115
  coverage = 10.4%
  residual: 0.114225 -> 0.113242
  residual_gain = 0.000983
  overerase_delta = -0.00000668
  selected metric wins/losses = 12 / 0

holdout40:
  selected = 1 / 40
  coverage = 2.5%
  residual: 0.130543 -> 0.129716
  residual_gain = 0.000826
  overerase_delta = -0.00000050
  selected metric wins/losses = 1 / 0

combined:
  selected = 13 / 155
  coverage = 8.4%
  residual: 0.118436 -> 0.117494
  residual_gain = 0.000942
  overerase_delta = -0.00000509
  selected metric wins/losses = 13 / 0
```

This refined rule is safer but too narrow, especially on holdout40. Use it as the conservative
anchor when comparing future selectors: any broader rule should preserve the zero-loss behavior or
provide visual evidence explaining why selected metric-loss pages are still acceptable.

Pareto refinement around the broad `active_gray_p25 >= 123` rule is reproducible with:

```bash
python3 scripts/analysis/refine_fixed_page_selector.py \
  --base-rule 'active_gray_p25 >= 123' \
  --split scut115:outputs/selector_features_residual_delta_scut115_20260707_retry/page_features.csv:outputs/scut_test115_second_stage_baseline_20260705/metrics.csv:outputs/eval_scut115_residual_delta_bias3_scale008_best_base12_delta2_20260707/metrics.csv \
  --split holdout40:outputs/selector_features_residual_delta_holdout40_20260707/page_features.csv:outputs/holdout40_second_stage_nearworst_safe_step1_t98_20260705/metrics.csv:outputs/eval_holdout40_residual_delta_bias3_scale008_best_base12_delta2_20260707/metrics.csv \
  --output-csv outputs/page_quality_selector_ai_provisional_40_20260707/refine_active_gray_p25_ge_123_pareto_loss_le3_script.csv \
  --max-losses 3 \
  --min-selected 8 \
  --top-features 10 \
  --top-n 200
```

Search result:

```text
base rule:
  selected = 24
  wins/losses = 16 / 8

best rule for loss_limit 0:
  active_baseline_edit_p95 <= 149
  AND candidate_delta_mean >= 0.0182428157494
  selected = 13
  wins/losses = 13 / 0
  residual_gain = 0.000942
  overerase_delta = -0.00000509

best rule for loss_limit 1, 2, or 3:
  same as loss_limit 0
```

Within the current two-condition feature search, allowing 1-3 metric-loss pages does not recover
extra coverage. The next coverage gain likely needs either visual acceptance of selected metric-loss
pages, a learned selector using confirmed labels, or a better candidate model; further local
threshold searching around these features is low-leverage.

The eight metric-loss pages selected by the broad `active_gray_p25 >= 123` rule were diagnosed with
local target-distance deltas:

```bash
python3 scripts/analysis/diagnose_selected_metric_losses.py \
  --review-csv outputs/page_quality_selector_ai_provisional_40_20260707/active_gray_p25_ge_123_metric_losses_review.csv \
  --output-csv outputs/page_quality_selector_ai_provisional_40_20260707/active_gray_p25_ge_123_metric_losses_diagnostics_script.csv
```

Result:

```text
rows = 8
diag_heuristic_verdict:
  likely_true_loss = 8

hurt ratios:
  holdout40/333.jpg = 0.720
  holdout40/337.jpg = 0.460
  scut115/174.jpg = 0.450
  scut115/179.jpg = 0.749
  scut115/189.jpg = 0.739
  scut115/314.jpg = 0.406
  scut115/315.jpg = 0.467
  scut115/316.jpg = 0.412
```

This supports treating the broad-rule metric-loss pages as real risk pages rather than metric-only
false negatives. The zero-loss refined selector should remain the safety anchor until a learned
selector or improved candidate can recover coverage without selecting these hurt-heavy cases.

The refined zero-loss selector can be materialized into prediction folders with:

```bash
python3 scripts/infer/materialize_fixed_page_selector.py \
  --selector-rule 'active_gray_p25 >= 123 AND active_baseline_edit_p95 <= 149 AND candidate_delta_mean >= 0.0182428157494' \
  --split scut115:outputs/selector_features_residual_delta_scut115_20260707_retry/page_features.csv:outputs/scut_test115_second_stage_baseline_20260705/metrics.csv:outputs/eval_scut115_residual_delta_bias3_scale008_best_base12_delta2_20260707/metrics.csv \
  --split holdout40:outputs/selector_features_residual_delta_holdout40_20260707/page_features.csv:outputs/holdout40_second_stage_nearworst_safe_step1_t98_20260705/metrics.csv:outputs/eval_holdout40_residual_delta_bias3_scale008_best_base12_delta2_20260707/metrics.csv \
  --output-dir outputs/materialized_refined_zero_loss_selector_YYYYMMDD
```

Verified materialized output:

```text
outputs/materialized_refined_zero_loss_selector_20260707/
  selection.csv
  scut115/pred/*.png
  holdout40/pred/*.png

rows = 155
selected = 13
selected metric wins/losses = 13 / 0
scut115 pred files = 115
holdout40 pred files = 40
```

Use this materialized folder for side-by-side visual review of the current safety anchor. It should
not replace the current default pipeline until the selected pages are visually confirmed and the
coverage limitation is accepted or improved.
