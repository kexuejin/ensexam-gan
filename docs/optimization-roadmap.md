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

The residual-delta family comparison reinforces this direction. The current t4
sweep is the best existing residual-removal candidate on SCUT115, but its
SCUT115 residual gain is only 0.000193 while holdout40 pays +0.000245
overerase. Variants that reduce overerase more strongly, especially
`mixed_td_w2_t4`, do so by worsening SCUT115 residual. Treat t4 as the
regression baseline for future candidate objectives, not as the product path.

The overguard scale0.06 follow-up proved the safer objective direction, but the
balanced scale0.07 follow-up is now the stronger candidate. Balanced007 improves
aggregate residual and aggregate overerase on all four checked splits:
SCUT115, holdout40, train160, and next120. It beats t4's SCUT115 residual gain,
removes t4's holdout40 overerase regression, and recovers more holdout residual
coverage than overguard. It is still a research candidate until page/crop review
confirms that the aggregate gains do not hide visible regressions.

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

The current page-label count is too small for a reliable black-box page selector. Region/component
review packs are now available and should be used to build a small reviewed selector dataset before
more model tuning. Weak target-derived component labels were tested and rejected as a selector
training signal because held-out reject ratios remained too high.

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

Region selector workflow now available:

```text
1. Generate component features with evaluate_region_component_selector.py.
2. Build impact-ranked component crops with build_region_component_impact_review_pack.py.
3. Use the balanced build_region_component_review_pack.py output only as a sanity/backfill set.
4. Label components as keep / drop / review using docs/region-component-labeling.md.
5. Validate labels with validate_region_component_labels.py.
6. Train with train_region_component_ranker.py --label-csv.
7. Promote only after separate reviewed held-out labels show near-zero reject rate.
```

Current region evidence:

```text
simple component rules: no <=5% train-reject-ratio rule found at useful scale
best simple pair: held-out reject ratio 36.1%
weak-label ranker: held-out reject ratio 37.0%
weak-label materialized page result:
  SCUT115 residual_gain=0.000001815, overerase_delta=0.000000000
  holdout40 residual_gain=0.000007298, overerase_delta=0.000000000
  status: bridge works, quality lift is negligible
weak-label threshold sweep:
  best sampled shared threshold=0.5130248089880388
  SCUT115 residual_gain=0.000231559, worse_pages=32/115
  holdout40 residual_gain=0.000227216, worse_pages=13/40
  status: broader threshold is still too weak and too noisy for promotion
review pack seed: 20 accept / 20 review / 20 reject held-out component crops
impact review pack seed:
  output: outputs/region_component_impact_review_pack_weak_t0513_20260707
  selected: 76 components from 10041 impact-scored components
  buckets: residual_help=30, residual_hurt=30, overerase_risk=14, large_noop=2
  status: preferred next label source because rows are ranked by page-level metric impact
t4 oracle ceiling:
  output: outputs/region_component_oracle_ceiling_t4_20260707/oracle_ceiling.csv
  SCUT115 best residual_gain=0.000941709 with zero worse pages
  holdout40 best residual_gain=0.000452334 with zero worse pages
  status: selector improvements alone cannot make this candidate family product-quality
```

This means the next selector milestone is not a better weak-label threshold; it
is a reviewed component label set large enough to train and validate a region
selector only after a stronger candidate family exists. For the current t4
residual-delta candidate, the oracle ceiling is too low to justify more selector
micro-tuning as a product-quality path.

Next candidate objective should be judged against t4 on four splits:

```text
SCUT115: must beat residual_delta=-0.000193191 without overerase regression
holdout40: must keep the residual gain while reducing +0.000244709 overerase penalty
train160/next120: must stay residual-positive, not just overerase-positive
mixed_td-style tradeoff: reject if residual worsens to buy overerase reduction
```

Updated overguard benchmark:

```text
SCUT115: residual_delta=-0.000786008, overerase_delta=-0.000009437
holdout40: residual_delta=-0.000831641, overerase_delta=-0.000009240
gate: alpha=0.3, second_delta=2 only; stricter gates are no-op
next objective: recover holdout40 residual coverage without losing overerase safety
```

Updated balanced residual-delta benchmark:

```text
candidate: outputs/train_scut_residual_delta_balanced_scale007_step80_20260707/cleanup_best.pt
gate: alpha=0.3, base_edit=12, second_delta=2
SCUT115: residual_delta=-0.001349386843, overerase_delta=-0.000020558496
holdout40: residual_delta=-0.001418918619, overerase_delta=-0.000018467665
train160: residual_delta=-0.004913059464, overerase_delta=-0.000024327035
next120: residual_delta=-0.004139072483, overerase_delta=-0.000023137928
status: current best residual-delta candidate; pending page/crop visual review
```

The next experiment should not be another blind scalar tweak. Build a focused
review pack for balanced007 versus the current second-stage baseline, t4, and
overguard. Prioritize high-gate pages, pages where balanced007 loses residual,
and buckets that previously caused visible failures: correction-fluid white
patches, gray backgrounds, low-contrast handwriting, printed-text protection,
and paper-texture dirtying.

Balanced007 local review triage is now available:

```text
output: outputs/product_quality_review_balanced007_four_split_20260707/
review rows: 96 pressure-test pages, 24 per split
crop rows: 384
local target proxy: accept=38, review=3, reject=55
metric-win bucket: accept=28, review=2, reject=2
metric-loss/overrisk bucket: reject=32
high-activity bucket: accept=10, review=1, reject=21
```

This does not overturn the aggregate result because the queue is deliberately
risk-heavy. It does show that balanced007 should be promoted only behind a
page/region selector or veto. The next higher-leverage step is to mine features
that separate the metric-win accepts from high-activity rejects, then validate a
candidate-family-specific selector on the same four splits before another
training tweak.

Initial selector mining found a conservative balanced007 gate:

```text
rule: active_baseline_edit_mean >= 96.6185399391 AND active_gray_p75 <= 192
full four-split result:
  selected=26/435
  wins/losses=26/0
  residual_gain=0.001247317
  overerase_delta=-0.000002939
pressure-review local proxy:
  selected=20/96
  accept=20
  reject=0
```

This is directionally useful because it converts balanced007 from a broad risky
candidate into a small zero-loss selector on current metrics. It is still below
the desired 20%-30% reliable coverage, so the next work should either widen this
selector with additional features/review labels or improve the candidate
objective. Do not treat the 6% safe gate as productized quality.

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

Before manual review, compute deterministic residual features:

```text
$ENSEXAM_PYTHON scripts/analysis/analyze_crop_residual_features.py \
  --index-csv outputs/product_quality_crop_review_pack_union_gate_residual_typed_20260706/index.csv \
  --output-csv outputs/product_quality_crop_review_pack_union_gate_residual_typed_20260706/target_residual_features.csv \
  --source-type target_residual
```

Use the resulting `handwriting_likelihood_score` as a triage signal only. It helps find crops where
the source input and baseline still contain dark, edge-like residuals that the target removed; it is
not a substitute for crop labels.

The priority queue can consume the feature report:

```text
$ENSEXAM_PYTHON scripts/analysis/prioritize_product_quality_crops.py \
  --index-csv outputs/product_quality_crop_review_pack_union_gate_residual_typed_20260706/index.csv \
  --feature-csv outputs/product_quality_crop_review_pack_union_gate_residual_typed_20260706/target_residual_features.csv \
  --output-csv outputs/product_quality_crop_review_pack_union_gate_residual_typed_20260706/priority_feature_top15.csv \
  --contact-sheet outputs/product_quality_crop_review_pack_union_gate_residual_typed_20260706/priority_feature_top15.png \
  --source-type target_residual \
  --max-total 15 \
  --max-per-page 2
```

Initial target-residual feature run:

```text
input rows: 40
output rows: 40
top scoring pages:
  next120/281.jpg
  next120/374.jpg
  next120/273.jpg
  next120/274.jpg
```

These pages should be reviewed before lower-scoring, large-area crops because they have stronger
source-dark overlap and edge density signals. If they are true residual handwriting, the next useful
work is candidate generation for coverage expansion. If they are mostly target/background mismatch,
the coverage-negative bucket should be down-weighted.

## Coverage-Residual Training Probe

The feature-ranked residual queue can be converted into a bounded micro-training patch index:

```text
$ENSEXAM_PYTHON scripts/experimental/convert_priority_crops_to_patch_index.py \
  --priority-csv outputs/product_quality_crop_review_pack_union_gate_residual_typed_20260706/priority_feature_top15_with_sheet.csv \
  --output-csv outputs/coverage_residual_feature_top15_patch_index_20260706/patch_index.csv \
  --train-files-out outputs/coverage_residual_feature_top15_patch_index_20260706/train_files.txt \
  --max-tiles-per-crop 3
```

This is the next training candidate source. It targets the named `coverage_negative_noop` bucket and
uses residual-feature evidence instead of generic stroke density.

Initial conversion output:

```text
priority rows: 15
patch rows: 45
files: 9
train files:
  273.jpg 274.jpg 275.jpg 281.jpg 282.jpg 362.jpg 374.jpg 389.jpg 392.jpg
```

MPS training smoke:

```text
command:
  scripts/train/micro_train_region_probe.py
output:
  outputs/smoke_coverage_residual_feature_top15_step1_20260706
matched patches:
  45 / 1232
step:
  1 / 1
sampled patch:
  281.jpg x1=1440 y1=1120 x2=1696 y2=1376
loss:
  G=31.760653 D=3.581648
checkpoint:
  outputs/smoke_coverage_residual_feature_top15_step1_20260706/micro_region_probe.pth
```

This verifies the training entrypoint only. It does not prove quality. The next meaningful run should
train a bounded multi-step candidate from this patch index and evaluate it through the existing
SCUT115 / holdout40 / next120 product-quality reports.

Bounded step5 candidates were evaluated on the same nine targeted train pages against the existing
next120 second-stage baseline:

```text
full-generator step5:
  output: outputs/exp_coverage_residual_feature_top15_step5_20260706
  eval: outputs/eval_coverage_residual_feature_top15_step5_target9_20260706
  baseline residual: 0.338039701
  candidate residual: 0.428630279
  delta residual: +0.090590578
  baseline overerase: 0.006130962
  candidate overerase: 0.014270546
  delta overerase: +0.008139585

mask-only decoder step5:
  output: outputs/exp_coverage_residual_feature_top15_maskonly_decoder_step5_20260706
  eval: outputs/eval_coverage_residual_feature_top15_maskonly_decoder_step5_target9_20260706
  baseline residual: 0.338039701
  candidate residual: 0.421282252
  delta residual: +0.083242551
  baseline overerase: 0.006130962
  candidate overerase: 0.016514974
  delta overerase: +0.010384012
```

Do not continue this exact patch-index route by adding more steps. The likely issue is that
target-residual crops include target/background mismatch or regions where direct patch training
damages page tone more than it removes useful residual handwriting. The next training route needs
crop labels or a stronger preserve objective before more coverage expansion.

A stronger preservation-aware probe did not fix this route:

```text
outside-edit step2:
  output: outputs/exp_coverage_residual_feature_top15_outsideedit_step2_20260706
  eval: outputs/eval_coverage_residual_feature_top15_outsideedit_step2_target9_20260706
  lr: 2e-6
  lambda_outside_edit_size: 8.0
  outside_edit_threshold_px: 4.0
  baseline residual: 0.338039701
  candidate residual: 0.508559486
  delta residual: +0.170519785
  baseline overerase: 0.006130962
  candidate overerase: 0.043658455
  delta overerase: +0.037527493
```

This makes the failure mode clearer: direct training on these target-residual crops is unsafe even
with an outside-edit penalty. The next useful step is to label or filter the crop queue before
creating another training candidate.

The crop queue can be triaged before another training attempt:

```text
$ENSEXAM_PYTHON scripts/analysis/bucket_residual_crop_candidates.py \
  --priority-csv outputs/product_quality_crop_review_pack_union_gate_residual_typed_20260706/priority_feature_top15_with_sheet.csv \
  --output-csv outputs/product_quality_crop_review_pack_union_gate_residual_typed_20260706/priority_feature_top15_triage.csv
```

This creates review/training buckets such as `high_confidence_residual_handwriting`,
`possible_residual_handwriting`, and `probable_background_or_target_mismatch`. These buckets are
triage signals only; product promotion still requires visual crop labels and split validation.

Initial top15 triage:

```text
high_confidence_residual_handwriting: 5
possible_residual_handwriting: 4
probable_background_or_target_mismatch: 6
```

This explains the failed direct training probes: the queue is not clean enough to use as supervision
without filtering or manual labels.

High-confidence-only residual training still failed:

```text
triage source:
  high_confidence_residual_handwriting only
patch rows:
  15
files:
  273.jpg 274.jpg 281.jpg 374.jpg
smoke:
  outputs/smoke_coverage_residual_highconf_step1_20260706
  matched patches: 15 / 560
step2:
  outputs/exp_coverage_residual_highconf_step2_20260706
eval:
  outputs/eval_coverage_residual_highconf_step2_target4_20260706
baseline residual: 0.334932443
candidate residual: 0.510497900
delta residual: +0.175565458
baseline overerase: 0.007438648
candidate overerase: 0.030668833
delta overerase: +0.023230184
```

The filtered queue still fails, so the next training route should not be another residual-crop
micro-tune. The model needs a different objective/data representation for coverage expansion, such
as explicit residual masks with a stronger paper/printed-text preservation target, or a separate
candidate generator that does not update the full image synthesis path.

Target-difference explicit-mask smoke was added as a safer representation probe:

```text
mask materialization:
  scripts/experimental/materialize_target_diff_masks.py
  output: data-links/samples/SCUT-EnsExam-targetdiff-explicit
patch queue:
  scripts/experimental/build_explicit_mask_patch_index.py
  output: outputs/scut_targetdiff_explicit_patch_index_20260706/patch_index_exact.csv
  matched patches after coordinate fix: 128 / 3848
heads-only step1:
  train: outputs/smoke_scut_targetdiff_explicit_mask_heads_exact_step1_20260706
  train4 eval: residual 0.089141230 -> 0.086848954, overerase 0.001060701 -> 0.001049271
  holdout4 eval: residual 0.156151160 -> 0.158561260, overerase 0.002021731 -> 0.002092774
decoder step3:
  train: outputs/exp_scut_targetdiff_explicit_mask_decoder_step3_20260706
  train4 eval: residual 0.089141230 -> 0.090902478, overerase 0.001060701 -> 0.000975507
```

Conclusion: the explicit-mask path is now runnable and useful for future data experiments, but the
current SCUT target-diff mask-only candidates should not be promoted. Heads-only improvement on the
training probe did not transfer to holdout, and decoder-scope training worsened residual on the
training probe.

ExamInk-Seg external-mask intake is now validated at smoke scale:

```text
dataset:
  Hugging Face: ynyg/ExamInk-Seg
  actual layout: data/{train,val}/{source,target,mask}
  train triplets: 1511
  val triplets: 115
download helper:
  scripts/experimental/prepare_examink_seg_dataset.py
  supports HF tree pagination, source/target jpg + mask png stem matching,
  per-triplet progress, train-only probes via --limit-val 0, and curl fallback
smoke data:
  data-links/samples/ExamInk-Seg-smoke
  train files: 0.jpg 1.jpg 2.jpg 3.jpg
  mask ratios: 0.014298 0.011411 0.017351 0.019216
patch queue:
  outputs/examink_seg_smoke_patch_index_20260706/patch_index.csv
  matched patches: 64 / 507
heads-only step1:
  train: outputs/smoke_examink_seg_mask_heads_step1_20260706
  eval: outputs/eval_examink_seg_smoke_mask_heads_step1_train4_20260706
  baseline:  residual 0.193198, overerase 0.004355
  candidate: residual 0.195089, overerase 0.003790
```

Conclusion: external explicit masks are now usable locally, but the first heads-only mask probe
became more conservative rather than better: overerase improved, residual worsened on 3/4 pages.
The next useful route is not to promote this checkpoint, but to use ExamInk-Seg for a larger
selector/mask calibration experiment that explicitly balances residual coverage against overerase.

Current exact129/outside-edit union candidate selector ceiling:

```text
four-split replay:
  outputs/selector_replay_exact129_outside_edit_lam16_union_four_split_20260706
  splits: scut115 + holdout40 + train160 + next120 = 435 pages
current named union:
  selected=6/435
  total_residual_gain=0.000323590327
  max_split_overerase_regret=0
best monotonic threshold sweep:
  selected=1/435
  total_residual_gain=0.000073397331
oracle safe-positive pages:
  selected=26/435 (5.98%)
  total_residual_gain=0.000891888940
offline interval-box mining:
  selected=23/435
  safe_positive_selected=13
  unsafe_selected=10
  total_residual_gain=0.000960965951
```

This closes the threshold-search route for the current candidate family. Even target-aware oracle
selection cannot approach 20%-30% safe coverage, so the next real quality step needs a different
candidate representation/objective rather than another selector sweep.
