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
