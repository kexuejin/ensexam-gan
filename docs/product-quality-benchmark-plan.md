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
```

Fill the `label`, `flags`, `reviewer`, `review_date`, and `comment` columns in this subset first.
Once at least 40 rows are confirmed, use it as the first input to `calibrate_page_quality_selector.py`.
