# Optimization Roadmap

## Current Evidence

The current benchmark says the project is not blocked by a lack of selector thresholds. It is blocked
by candidate quality and page-level acceptance:

```text
exact129_lam16_union_gate:
  labeled pages: 21
  wins: 4
  losses: 2
  noops: 15
  win_rate: 19.0%
  status: safe but too narrow

whiteout_inpaint_d10_g215_a120:
  labeled pages: 6
  wins: 0
  losses: 4
  noops: 2
  status: metric-positive but visually worse
```

This makes threshold-only tuning a low-leverage path. Future work should improve the candidate
families that the selector can choose from, then score them with the product-quality label set.

## Most Promising Directions

### 1. Paper-Tone Harmonization For Correction Fluid

Use a conservative whiteout-specific candidate that adjusts low-frequency tone only.

The current inpaint route tries to synthesize paper texture inside correction-fluid regions. Visual
labels show that this often makes the area look dirtier than the baseline smooth white patch. The
better product target is not "reconstruct texture"; it is "make the remaining white patch less
distracting while preserving clean readability."

Experiment shape:

```text
detect candidate whiteout patch:
  bright low-saturation region
  local high residual was already removed
  region is not printed text

estimate surrounding paper tone:
  use a ring outside the patch
  ignore dark ink and high-gradient pixels
  estimate low-frequency luminance/chroma only

blend:
  apply gentle tone shift inside the patch
  preserve smoothness
  avoid adding texture or noise
  cap delta so a clean white patch is never made dirty
```

Promotion gate:

```text
correction-fluid bucket:
  wins > losses
  clear_loss = 0
  paper_texture_dirty does not increase
  no printed_text_damage
```

### 2. Better Candidate Generation Before Better Selection

The union selector is narrow because most candidates either do not visibly improve enough pages or
create paper/background risk. A learned selector cannot fix bad candidate supply. The next generator
experiments should explicitly preserve paper and printed text while removing residual handwriting.

Experiment shape:

```text
train objective:
  residual handwriting reduction inside erase mask
  outside-mask edit penalty
  low-frequency background preservation
  printed-text preservation guard

sampling:
  named failure buckets, not generic stroke-density patches
  include hard residual pages and background-risk pages together
  keep validation splits fixed before running

output:
  candidate predictions
  local metric report
  product-quality review pack
```

Promotion gate:

```text
coverage >= 20% on targeted failure bucket
clear_loss = 0
wins materially exceed losses
overerase unchanged or lower on validation splits
```

### 3. Calibrated Selector, Not A Full Learned Model Yet

The current label count is too small for a reliable learned selector. Use labels first to calibrate
features and reject unsafe candidate families.

Useful features:

```text
copy_mask_cov8
primary_edit_px
primary_p95_edit_delta
second_stage_gate_ratio
visible residual delta
visible overerase delta
background low-frequency delta
whiteout patch area / tone delta
printed-text overlap risk
```

Near-term selector work should be a small, auditable calibration report:

```text
for each candidate family:
  summarize feature ranges for wins, noops, losses
  identify features that separate losses from wins
  propose one candidate-family-specific gate
  verify it against fixed labels and non-overlap pages
```

Do not train a black-box selector until the review set has enough labeled pages per bucket.

### 4. Zoom Review Packs For Ambiguous Failures

Page-scale contact sheets miss small edge artifacts and low-contrast residuals. Add crop review packs
for each failure bucket before changing model behavior again.

Crop sources:

```text
visible-delta improve/regress components
whiteout patch bounding boxes
gray paper-tone residue regions
printed-text overlap risk boxes
low-contrast handwriting residual components
```

This should reduce false conclusions from page-scale review and make labels more consistent.

### 5. External Data Only After Scoring Is Stable

ExamInk-Seg and other external data are useful for coverage, but only after the local scoring loop can
separate real visual wins from metric-only wins.

Use external data for:

```text
more correction-fluid examples
more gray / colored paper backgrounds
more handwriting styles
more low-contrast erasure cases
```

Avoid using it for:

```text
blind full retraining
mask-only adaptation without page-level validation
claiming product lift from aggregate residual alone
```

## Recommended Next Experiment

Start with paper-tone harmonization because it targets the clearest current bucket failure and does
not require expensive retraining.

Implementation order:

```text
1. Build a whiteout patch detector and crop review pack.
2. Generate a conservative tone-harmonized candidate on the six seeded whiteout pages.
3. Add candidate rows to docs/product-quality-review-pages.csv.
4. Build a review pack and label page/crop outcomes.
5. Promote only if whiteout labels improve from 0 wins / 4 losses to positive net wins.
```

If this fails, the evidence is still useful: it means correction-fluid handling needs model-level
training data or should be left as baseline rather than patched with post-processing.

## Whiteout Tone-Harmonization Probe

An initial local probe generated six correction-fluid candidates with a conservative post-processing
script:

```text
script:
  scripts/analysis/generate_whiteout_tone_harmonization.py

default candidate:
  max_shift = 3
  blend = 0.20
```

The probe shows this direction is only marginally useful unless the detector becomes more precise:

```text
best conservative grid point:
  mean target MAE delta: -0.009769
  improved pages by MAE: 3 / 6
  mean overdark delta: +0.007446
  max overdark delta: +0.015796
  mean changed ratio: 0.000462
  max changed ratio: 0.001086
```

More aggressive tone shifts improve target MAE slightly more, but consistently increase the
overdark / dirty-paper proxy and modify too much page area. Treat this as a tentative candidate
generator, not a product path. It needs crop-level review and a tighter whiteout detector before any
promotion.

## Crop-Level Review Packs

Page-level contact sheets are not enough for deciding small local changes. Use crop-level packs for
whiteout, gray-background residue, low-contrast residual strokes, and printed-text protection before
promoting a candidate.

Reusable tooling:

```text
script:
  scripts/analysis/build_product_quality_crop_review_pack.py

example:
  $ENSEXAM_PYTHON scripts/analysis/build_product_quality_crop_review_pack.py \
    --review-csv outputs/whiteout_tone_harmonize_v1_default_20260706/review_rows.csv \
    --output-dir outputs/product_quality_crop_review_pack_whiteout_tone_v1_20260706 \
    --bucket correction_fluid_white_patch \
    --candidate whiteout_tone_harmonize_v1 \
    --max-crops-per-row 6 \
    --crop-size 320 \
    --thumb-size 240

missed-coverage example:
  $ENSEXAM_PYTHON scripts/analysis/build_product_quality_crop_review_pack.py \
    --review-csv docs/product-quality-review-pages.csv \
    --output-dir outputs/product_quality_crop_review_pack_union_gate_residual_20260706 \
    --candidate exact129_lam16_union_gate \
    --include-target-residual \
    --max-crops-per-row 4
```

Initial whiteout tone-harmonization crop pack:

```text
rows: 6
crops: 33
contact sheet:
  outputs/product_quality_crop_review_pack_whiteout_tone_v1_20260706/contact_sheet.png
index:
  outputs/product_quality_crop_review_pack_whiteout_tone_v1_20260706/index.csv
label template:
  outputs/product_quality_crop_review_pack_whiteout_tone_v1_20260706/crop-labels-template.csv
```

Do not treat the crop pack itself as product evidence until labels are added. Its purpose is to make
local failure review cheaper and more consistent than page-scale inspection.

Crop labels can be summarized with:

```text
$ENSEXAM_PYTHON scripts/analysis/summarize_product_quality_crop_labels.py \
  --labels-csv outputs/product_quality_crop_review_pack_whiteout_tone_v1_20260706/crop-labels-template.csv \
  --output-dir outputs/product_quality_crop_label_summary_whiteout_tone_v1_20260706
```

When a crop pack is large, generate a prioritized manual-label subset first:

```text
$ENSEXAM_PYTHON scripts/analysis/prioritize_product_quality_crops.py \
  --index-csv outputs/product_quality_crop_review_pack_union_gate_residual_typed_20260706/index.csv \
  --output-csv outputs/product_quality_crop_review_pack_union_gate_residual_typed_20260706/priority_top24.csv \
  --max-total 24 \
  --max-per-page 3
```

Initial union residual priority output:

```text
input crops: 84
selected: 24
source_type: target_residual = 24
bucket: coverage_negative_noop = 24
pages covered: 9
```

This makes the next manual review question concrete: are the largest no-op residual crops true
missed handwriting, or mostly target/background mismatch? The answer determines whether the next
model work should expand coverage or avoid chasing noisy target differences.
