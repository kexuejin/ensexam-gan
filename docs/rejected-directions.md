# Rejected Directions

## Threshold-Only Micro-Tuning Loop

Rejected as the primary path forward. The recent exact129 / outside-edit / union-gate work found a
safe selector window, but the useful coverage is too small for productization:

```text
SCUT115: selected=5/115
holdout40: selected=1/40
train160: selected=0/160
next120: selected=0/120
combined selected=6/435
```

The next120 non-overlap check is useful safety evidence because it selected no pages and caused no
metric regression, but it also proves that the current selector family does not broaden coverage.
More hand-tuned intervals or one-step patch probes are unlikely to close the product-quality gap
unless they are anchored to explicit failure buckets and page-level visual labels.

Do not continue the default loop of:

```text
one-step probe -> replay selector thresholds -> tiny safe window -> document narrow result
```

Future experiments should first define the target failure bucket, expected page coverage lift, and
manual visual acceptance criteria. Without that, improve the benchmark and labeling workflow instead
of launching another probe.

## ExamInk-Seg Direct / Mask-Only Adaptation

Rejected for current product path. Direct ExamInk partial and mixed updates made the model overly conservative. Mask-only updates worsened hardcase residual.

## Identity-Safe Erasemap Cleanup Train31 Promotion

Rejected for product promotion in its current checkpoint forms. The identity-safe cleanup training
and validation scripts are useful infrastructure, but the ExamInk-only cleanup checkpoints do not
survive SCUT115 validation.

Small-set checks looked positive:

```text
ExamInk train31 same-sample:
  residual 0.198286 -> 0.193751
  overerase 0.004114 -> 0.004060
SCUT holdout4:
  residual 0.156151 -> 0.154797
  overerase 0.002022 -> 0.001983
```

Full SCUT115 checks failed:

```text
primary input:
  residual=0.118313 overerase=0.003048
current second-stage baseline:
  residual=0.114225 overerase=0.003048
train31 cleanup as replacement:
  residual=0.136329 overerase=0.003003
train31 cleanup after current second-stage:
  residual=0.132060 overerase=0.003009
```

Do not promote `outputs/train_examink_cleanup_erasemap_identity_train31_step500_20260707/cleanup_probe.pt`
or repeat replacement/third-stage evaluation for this checkpoint. Future cleanup work should train
on a larger frozen ExamInk-Seg train/val split and require SCUT115 plus holdout40 validation before
selector tuning.

A later train24/val7 split added held-out validation and selected `cleanup_best.pt` by validation
loss:

```text
validation loss:
  step100=1.727791
  step500=1.573239
ExamInk val7:
  residual 0.182208 -> 0.159401
  overerase 0.004473 -> 0.004433
SCUT115 replacement:
  current second-stage baseline residual=0.114225 overerase=0.003048
  train24/val7 cleanup residual=0.162796 overerase=0.003009
```

This means an ExamInk held-out validation win is not enough evidence for SCUT product quality. Do not
spend more cycles promoting ExamInk-only cleanup checkpoints through threshold tuning; the next
cleanup attempt needs SCUT-style validation or a mixed-domain objective before SCUT115 evaluation.

A follow-up mixed-domain run added SCUT target-diff explicit patches to the cleanup training and
validation set. It improved over the ExamInk-only cleanup checkpoints but still failed product
promotion:

```text
mixed cleanup best checkpoint:
  outputs/train_mixed_cleanup_erasemap_step500_20260707/cleanup_best.pt
validation:
  best step100 loss=1.229255
SCUT115 replacement:
  current second-stage baseline residual=0.114225 overerase=0.003048
  mixed cleanup residual=0.150084 overerase=0.003008
SCUT115 conservative gate sweep:
  base_edit=12 second_delta=12 gate=0.002447
  residual=0.116928 overerase=0.003021
  residual regret=+0.002703 versus current baseline
```

Do not continue selector or threshold tuning for this mixed cleanup checkpoint. Even highly
conservative gates only approach the primary-input result and remain worse than the current
second-stage baseline. Future cleanup work needs a stronger objective or architecture change, not
more gate narrowing around this candidate family.

An alpha-threshold 0.5 rerun confirmed this is not a hidden gate issue:

```text
SCUT115 alpha=0.5, base_edit=12, second_delta=2:
  gate=0.000000
  residual=0.118313
  overerase=0.003048
```

This exactly falls back to the primary input and remains worse than the current second-stage
baseline residual `0.114225`. Treat the current cleanup checkpoint family as closed unless the
training objective or architecture changes.

Metric-aligned SCUT cleanup training also failed to produce a promotable checkpoint. A conservative
identity-safe run with `alpha_init_bias=-6` stayed locked near identity:

```text
SCUT115, metric-aligned cleanup best:
  residual=0.118313
  overerase=0.003048
  gate=0.000000
diagnostic:
  alpha ~= 0.002476 across sampled tiles
  max second_delta ~= 0.335/255
```

Mask-normalized alpha BCE fixed the gradient dilution issue, but stronger alpha settings caused
over-broad edits rather than useful residual cleanup:

```text
alpha_init_bias=-2, lr=5e-5:
  residual=0.278154
  overerase=0.003030
  gate=0.023974
alpha_init_bias=-3 with stronger outside/overerase constraints:
  residual=0.209663
  overerase=0.003024
  gate=0.018118
current second-stage baseline:
  residual=0.114225
  overerase=0.003048
```

Do not continue scalar alpha-bias / alpha-threshold tuning for this cleanup architecture. The useful
lesson is that alpha supervision must be mask-normalized, but this branch still lacks a mechanism to
produce residual-specific corrections without damaging the page. Future work should change the
candidate representation or use page-level acceptance labels instead of attempting to rescue this
checkpoint family with gate tweaks.

A residual-delta cleanup candidate reduced the over-broad edit problem but still failed promotion:

```text
model_type=residual_delta, residual_delta_scale=0.08:
  sampled max second_delta ~= 5.48/255
  sampled px_delta_ge_12=0.000000
  SCUT115 residual=0.118880
  overerase=0.003014
  gate=0.016595
```

This is much safer than the full clean-head variants, but it remains worse than the current
second-stage baseline residual `0.114225`. Do not promote it directly. Treat it as a better
candidate representation for future selector/page-label experiments, not as a solved cleanup model.

## Patent-Style Standalone Mask Calibration

Rejected as a standalone branch. SCUT pseudo-mask heads-only training worsened hardcase residual, indicating isolated mask branch tuning breaks erase gating.

## Whiteout Inpaint Repair

Metric-positive but visually rejected. The current local inpaint approach can make correction-fluid areas look dirtier / less natural than the unmodified second-stage output.

## Current-Primary Step4 Continuation

Rejected for promotion. The 2026-07-05 four-step continuation from the current primary checkpoint
worsened holdout40 residual on every candidate checkpoint. The least-bad candidate was `step0001`
with residual `0.138113` and overerase `0.002797`, versus the current primary baseline residual
`0.136111` and overerase `0.002482`.

Keep `artifacts/current-primary` unchanged and treat this run as evidence that naive continuation
from the current primary is not enough; future attempts should use safer lower-LR or narrower
hardcase variants with explicit preserve/overerase gating.

## Visible-Delta Patch-Only Step10 Probe

Rejected for promotion. The 2026-07-05 10-step probe trained only on visible-delta improve patches
from the strict SCUT test115 gate, using an isolated two-page derived dataset and the current primary
checkpoint as the resume source.

SCUT test115 strict-gate evaluation selected `0/115` candidate pages, so the output exactly fell back
to the baseline second-stage metrics: residual `0.114225`, overerase `0.003048`. The original strict
gate selected `6/115` pages with residual `0.113956`, overerase `0.003047`.

Feature drift on the original six strict-gate pages shows why it failed: `copy_mask_cov8` collapsed and
`primary_edit_px` rose sharply, so all previously selected pages failed the safety gate. Example drift:
`17.jpg` cov8 `0.806133 -> 0.255107`, edit_px `67884 -> 208413`; `303.jpg` cov8
`0.877718 -> 0.272895`, edit_px `91129 -> 381554`.

Do not continue this exact patch-only full-generator training path. Future visible-delta probes should
try narrower trainable scopes, mask-only/head-only updates, lower LR, or an auxiliary objective that
preserves gate features instead of updating the whole generator against a tiny two-page patch set.

## High Stroke-Ratio Train Patch Sensitivity

Rejected as a direct patch-selection strategy. A deterministic top4 sweep used ranked train-split
hard patches from `outputs/train_hard_patch_index_for_sensitivity_20260706/patch_index.csv` and
one-step MPS micro-probes from the current primary checkpoint.

All four top-ranked patches came from `362.jpg` and had very high local stroke density, but strict
SCUT115 and holdout40 evaluation selected no candidate pages:

```text
001_362_x1280_y320: scut selected=0/115 holdout selected=0/40
002_362_x1280_y640: scut selected=0/115 holdout selected=0/40
003_362_x1280_y480: scut selected=0/115 holdout selected=0/40
004_362_x1440_y480: scut selected=0/115 holdout selected=0/40
```

The metrics therefore exactly matched baseline second-stage outputs: SCUT115 residual
`0.114224963938`, overerase `0.003048296717`; holdout40 residual `0.134026304621`,
overerase `0.002481606117`.

Do not expand this naive high-stroke sweep. Future patch sensitivity work should rank patches by
their ability to preserve strict-gate features such as `copy_mask_cov8` and `primary_edit_px`, not
by stroke density alone.

## Current Hybrid Gate Promotion

Rejected for default product promotion. A joint SCUT test115 + holdout40 selector replay over saved
strict-candidate outputs found that the only rules with non-worse overerase on both splits select too
few pages to matter.

Best jointly safe label-free rule:

```text
copy_mask_cov8 >= 0.458438
primary_edit_px <= 101340
primary_p95_edit_delta <= 5
second_stage_gate_ratio <= 0.000248943
selected=2/155 pages
total residual gain=0.000074976101
max split overerase regret=-0.000000187820
```

The strict and SCUT7 rules produce larger residual gains but still raise holdout40 overerase:

```text
strict_cov806_edit98868 selected=9 total residual gain=0.000699581706 max overerase regret=+0.000010712336
scut7_cov65_edit98868_p95_5_gate0015 selected=10 total residual gain=0.000702181859 max overerase regret=+0.000010712336
```

Keep `scripts/analysis/replay_hybrid_selector.py` for future selector analysis, but do not promote
the current hybrid selector as a product default without a new candidate that improves residual
without cross-split overerase regression.

Four-split selector replay with SCUT115, holdout40, train160, and next120 makes the selector ceiling
lower, not higher:

```text
replay:
  outputs/selector_replay_exact129_outside_edit_lam16_union_four_split_20260706
splits:
  scut115=115, holdout40=40, train160=160, next120=120
named union rule:
  selected=6/435
  total_residual_gain=0.000323590327
  max_split_overerase_regret=0
best monotonic threshold rule:
  selected=1/435
  total_residual_gain=0.000073397331
  max_split_overerase_regret=0
safe positive oracle:
  selected=26/435 (5.98%)
  total_residual_gain=0.000891888940
  max_split_overerase_regret=-0.000000728701
min-2 interval-box oracle:
  selected=23/435
  safe_positive_selected=13
  unsafe_selected=10
  total_residual_gain=0.000960965951
  max_split_overerase_regret=-0.000000055094
```

The current candidate family cannot reach the desired 20%-30% safe coverage even with target-metric
oracle selection. Stop threshold-only selector searches on this family; future work needs a new
candidate generator or objective that separates residual cleanup from overerase before selector
calibration can matter.

## Low-Diff Anchor Selector-Only Promotion

Rejected as a product default. Low-diff anchor-like patch probes can restore some strict-gate
eligibility, but replaying label-free selector thresholds across SCUT115 and holdout40 shows that
safe rules have too little stable residual gain.

Best safe selector replay results:

```text
001_161_x1120_y0: selected=1/155 total residual gain=0.000276124066 max overerase regret=0
002_130_x2240_y960: selected=4/155 total residual gain=0.000119013592 max overerase regret=-0.000000740458
003_84_x800_y480: selected=1/155 total residual gain=-0.000000895287 max overerase regret=0
004_378_x0_y960: no safe rules; best unsafe max overerase regret=0.000317846011
```

The best-looking raw candidate, `003_84_x800_y480`, improves SCUT/holdout residual under the strict
rule but raises overerase. Once the replay requires no cross-split overerase regression, it keeps
only one holdout page with slightly negative residual gain. `001` and `002` expose useful diagnostic
signals, but their product-level gain is negligible.

Keep the low-diff anchor ranking and replay tooling for future probes, but do not spend more cycles
promoting the current low-diff candidates through threshold search alone. Future work should change
the training objective or candidate generation so useful pages keep high `copy_mask_cov8` while
reducing `primary_edit_px`, p95 edit delta, and background overerase.

## Low-Diff Outside-Edit Step1 Promotion

Rejected for promotion, but keep the training objective. The new `outside_edit_size` loss directly
penalizes large edits outside `Mb_gt`, and a one-step lowdiff002 probe confirmed that the loss runs
on MPS and logs a non-zero `outside_edit` term. However, the resulting candidate is not good enough
for product promotion.

Strict-gate result:

```text
scut115: selected=3/115 residual=0.114371078175 overerase=0.003050526697
holdout40: selected=2/40 residual=0.133809982834 overerase=0.002500990117
```

Compared with the original lowdiff002 one-step probe, SCUT improves slightly, but holdout40 residual
and overerase both move in the wrong direction:

```text
scut115 residual 0.114388655784 -> 0.114371078175
scut115 overerase 0.003050635477 -> 0.003050526697
holdout40 residual 0.133796433250 -> 0.133809982834
holdout40 overerase 0.002500568850 -> 0.002500990117
```

Joint selector replay with `max-overerase-regret=0` found only a negligible safe window:

```text
selected=4/155
total residual gain=0.000027968598
max split overerase regret=0
selected split coverage: scut115=4, holdout40=0
```

Do not promote this checkpoint or repeat the exact one-step lowdiff002 setting as a product path.
Future probes can still use `lambda_outside_edit_size` with lower learning rate, different anchors,
or mixed patch sampling because the objective is more targeted than global preservation weight
increases.

## Low-Diff Outside-Edit Mix25 Step2 Promotion

Rejected for promotion. A more conservative mixed-sampling probe used the lowdiff002 anchor with
`patch_index_mix_ratio=0.25` for two steps. It successfully exercised the outside-edit objective,
but it traded residual quality for minor overerase improvement and still failed the product gate.

Strict-gate result:

```text
scut115: selected=4/115 residual=0.114540661017 overerase=0.003039349003
holdout40: selected=3/40 residual=0.134347379815 overerase=0.002487066735
```

Compared with baseline, SCUT overerase improved but residual worsened, and holdout40 worsened on
both residual and overerase:

```text
scut115 baseline residual=0.114224963938 overerase=0.003048296717
scut115 mix25 residual=0.114540661017 overerase=0.003039349003
holdout40 baseline residual=0.134026304621 overerase=0.002481606117
holdout40 mix25 residual=0.134347379815 overerase=0.002487066735
```

Joint selector replay with `max-overerase-regret=0` found only one safe SCUT page:

```text
selected=1/155
total residual gain=0.000105093877
max split overerase regret=0
selected split coverage: scut115=1, holdout40=0
```

Do not continue this exact lowdiff002 mix25 step2 path. If outside-edit training continues, vary the
anchor or objective schedule rather than spending more cycles on selector replay for this candidate.

## Low-Diff Outside-Edit 001 Step1 Promotion

Rejected for promotion. Switching the outside-edit one-step probe from lowdiff002 to lowdiff001 did
not solve the cross-split tradeoff. The probe slightly improved residual versus the original
lowdiff001 candidate, but it increased overerase and stayed unsafe on holdout40.

Strict-gate result:

```text
scut115: selected=1/115 residual=0.114160171559 overerase=0.003060470866
holdout40: selected=2/40 residual=0.134636204411 overerase=0.002526421284
```

Compared with the original lowdiff001 one-step probe:

```text
scut115 residual 0.114160943919 -> 0.114160171559
scut115 overerase 0.003060213443 -> 0.003060470866
holdout40 residual 0.134691305786 -> 0.134636204411
holdout40 overerase 0.002524365006 -> 0.002526421284
```

Joint selector replay with `max-overerase-regret=0` again found only a one-page SCUT-only safe
window:

```text
selected=1/155
total residual gain=0.000275221700
max split overerase regret=0
selected split coverage: scut115=1, holdout40=0
```

Do not continue the exact lowdiff001 outside-edit step1 path. This reinforces that the current
low-diff anchors expose diagnostic signal but are not enough for product quality without a better
candidate-generation objective.

## Low-Diff Outside-Edit 003 Step1 Promotion

## Coverage-Residual Feature Top15 Patch Training

Rejected for continuation. A feature-ranked `target_residual` queue converted the top 15 missed
coverage crops into 45 training patches across nine next120 train pages:

```text
patch index: outputs/coverage_residual_feature_top15_patch_index_20260706/patch_index.csv
train files: 273.jpg 274.jpg 275.jpg 281.jpg 282.jpg 362.jpg 374.jpg 389.jpg 392.jpg
```

The training entrypoint is valid: a one-step MPS smoke matched `45/1232` patches, sampled
`281.jpg x1=1440 y1=1120`, and saved a finite-loss checkpoint.

However, both bounded step5 candidates worsened the same nine target pages compared with the
existing next120 second-stage baseline:

```text
baseline:
  residual=0.338039701
  overerase=0.006130962

full-generator step5:
  residual=0.428630279  delta=+0.090590578
  overerase=0.014270546 delta=+0.008139585

mask-only decoder step5:
  residual=0.421282252  delta=+0.083242551
  overerase=0.016514974 delta=+0.010384012

outside-edit step2:
  lr=2e-6
  lambda_outside_edit_size=8.0
  outside_edit_threshold_px=4.0
  residual=0.508559486  delta=+0.170519785
  overerase=0.043658455 delta=+0.037527493
```

Do not continue this exact patch-index route by adding more steps. It is useful evidence that
feature-ranked target-residual crops alone are not enough to expand coverage safely. Future coverage
training should first add crop labels to separate true residual handwriting from target/background
mismatch, or add a stronger paper/printed-text preservation objective before revisiting these pages.

The filtered high-confidence subset was also rejected. After triaging the feature top15 queue, only
`high_confidence_residual_handwriting` crops were converted into 15 patch rows across four pages:

```text
files: 273.jpg 274.jpg 281.jpg 374.jpg
smoke: matched 15/560 patches
step2 run: outputs/exp_coverage_residual_highconf_step2_20260706
eval: outputs/eval_coverage_residual_highconf_step2_target4_20260706
baseline residual=0.334932443 overerase=0.007438648
candidate residual=0.510497900 overerase=0.030668833
delta residual=+0.175565458 overerase=+0.023230184
```

Do not continue residual-crop micro-tuning by simply filtering the same queue harder. Even the
high-confidence subset teaches the full generator in a way that damages page tone and overerase.
Future coverage work needs a different training target or architecture path, not more steps on these
patches.

Rejected for promotion. The lowdiff003 outside-edit one-step probe produced the strongest strict
residual improvement among the tested low-diff outside-edit anchors, but it increased overerase on
both SCUT115 and holdout40. The selector replay safe window collapses to one holdout page with
negligible total gain.

Strict-gate result:

```text
scut115: selected=5/115 residual=0.113975166648 overerase=0.003064461676
holdout40: selected=3/40 residual=0.133709426395 overerase=0.002514275694
```

Compared with the original lowdiff003 one-step probe, outside-edit improved residual but worsened
overerase on both splits:

```text
scut115 residual 0.114007786589 -> 0.113975166648
scut115 overerase 0.003061889507 -> 0.003064461676
holdout40 residual 0.133760842647 -> 0.133709426395
holdout40 overerase 0.002508892113 -> 0.002514275694
```

Joint selector replay with `max-overerase-regret=0` found only a negligible holdout-only safe
window:

```text
selected=1/155
total residual gain=0.000015518312
max split overerase regret=0
selected split coverage: scut115=0, holdout40=1
```

Do not promote this checkpoint. This confirms that outside-edit alone can move residual in the
right direction, but it is not enough to solve the product-level residual/overerase tradeoff.

## Mask-Confidence Feature Selector

Rejected as a product inference selector for the current lowdiff003 outside-edit candidate family.
The diagnostic script `scripts/analysis/analyze_mask_confidence_features.py` re-runs the primary
model to recover page-level `ms`/`mb` masks, loads saved candidate images, and writes residual /
overerase deltas plus mask-confidence features.

The analysis confirmed the core tradeoff: many pages improve residual, but almost none are safe on
overerase.

```text
scut115:
  pages=115 positive_gain=71 safe_positive=1
  avg_residual_gain=0.003083072386
  avg_overerase_regret=0.000349649965
  only safe positive: 397.jpg gain=0.002036560758 overerase_regret=-0.000174437785

holdout40:
  pages=40 positive_gain=26 safe_positive=1
  avg_residual_gain=0.003024969740
  avg_overerase_regret=0.000350554299
  only safe positive: 477.jpg gain=0.000620732464 overerase_regret=-0.000002347434
```

A local threshold sweep over `mb_cov8`, `mb_cov98`, `primary_edit_px`,
`primary_p95_edit_delta`, and `ms_cov98` found no safe rules. Keep the script as diagnostic
tooling, but do not continue product work by tuning mask-confidence thresholds on the same saved
candidates. The next useful route remains better candidate generation or a training objective that
reduces background edits while preserving residual improvements.

## Simple Candidate-Only Background Protection

Rejected as a product inference switch. Local analysis of pages where the candidate improves residual
but increases overerase found 649 candidate-only overerase components. The dominant failures are many
small background edits and page-edge artifacts rather than a few large regions:

```text
small_background_edit: 431 components, area=21354
page_edge_artifact: 127 components, area=7363
near_changed_region_halo: 57 components, area=2492
near_target_boundary_or_low_contrast_label: 28 components, area=1286
```

A top24 high-risk page probe reverted candidate-only connected components to the baseline prediction.
This reduced overerase modestly, but it did not make any high-risk page safe and it also reduced the
residual gain:

```text
none:           residual_gain=0.176509 overerase_regret=0.015874 safe_pages=0/24
edge_only:      residual_gain=0.176170 overerase_regret=0.015132 safe_pages=0/24
small_nonlarge: residual_gain=0.154762 overerase_regret=0.012318 safe_pages=0/24
small_or_edge:  residual_gain=0.154762 overerase_regret=0.012291 safe_pages=0/24
all_components: residual_gain=0.154762 overerase_regret=0.012248 safe_pages=0/24
```

Keep `scripts/analysis/analyze_candidate_overerase_delta.py` as a diagnostic tool. Future work should
focus on model-side background preservation or stronger confidence masks rather than simple
post-inference connected-component rollback.

## Global Preservation Weight Increase

Rejected as a direct continuation path from the current checkpoint. Two 2-step MPS probes increased
the existing `input_preserve` and `mb_leak` loss weights:

```text
preserve16/leak0.75: lambda_input_preserve=16.0, lambda_mb_leak=0.75
preserve24/leak1:    lambda_input_preserve=24.0, lambda_mb_leak=1.0
```

Both probes were numerically stable and exposed useful loss telemetry, but strict-gate evaluation
became over-conservative:

```text
preserve16/leak0.75:
  scut115 residual=0.114210374926 overerase=0.003048663423 selected=1/115
  holdout40 residual=0.134427251448 overerase=0.002498697889 selected=3/40

preserve24/leak1:
  scut115 residual=0.114210460744 overerase=0.003048568711 selected=1/115
  holdout40 residual=0.134429857005 overerase=0.002498202272 selected=3/40

original strict:
  scut115 residual=0.113955546480 overerase=0.003046624105 selected=6/115
  holdout40 residual=0.133596140373 overerase=0.002492318453 selected=3/40
```

Do not keep scaling global preservation weights as the next product-quality route. It removes the
few useful strict-gate pages without solving cross-split overerase. Future attempts should use more
selective preservation targets or confidence masks.

## SCUT Target-Diff Explicit-Mask Mask-Only Promotion

Rejected as a model promotion path from the current checkpoint. The explicit-mask data path itself is
valid and should be kept, but bounded mask-only fine-tunes did not produce holdout-safe quality.

Reusable setup:

```text
materialize masks:
  scripts/experimental/materialize_target_diff_masks.py
patch index:
  scripts/experimental/build_explicit_mask_patch_index.py
  fixed to match EnsExamRealDataset edge-patch coordinates exactly
smoke config:
  configs/local/config.local-scut-targetdiff-maskonly-smoke-mps.yaml
```

Observed probes:

```text
heads-only step1:
  train: outputs/smoke_scut_targetdiff_explicit_mask_heads_exact_step1_20260706
  train4:   residual 0.089141230 -> 0.086848954, overerase 0.001060701 -> 0.001049271
  holdout4: residual 0.156151160 -> 0.158561260, overerase 0.002021731 -> 0.002092774

decoder step3:
  train: outputs/exp_scut_targetdiff_explicit_mask_decoder_step3_20260706
  train4: residual 0.089141230 -> 0.090902478, overerase 0.001060701 -> 0.000975507
```

The heads-only result was a training-set-only improvement dominated by one page and did not transfer
to holdout. Decoder-scope training worsened train residual. Do not promote either checkpoint; future
explicit-mask work should use higher-quality external masks or a selector/candidate branch evaluated
on larger split coverage before any model registry update.

## ExamInk-Seg Heads-Only Step1 Promotion

Rejected as a model promotion path. ExamInk-Seg is valuable as a real external explicit-mask source,
and the intake script now downloads its current Hugging Face layout, but the first heads-only
one-step probe made the model more conservative instead of improving erasure quality.

Tested setup:

```text
download:
  scripts/experimental/prepare_examink_seg_dataset.py
  output: data-links/samples/ExamInk-Seg-smoke
  train files: 0.jpg 1.jpg 2.jpg 3.jpg
patch index:
  outputs/examink_seg_smoke_patch_index_20260706/patch_index.csv
  matched: 64 / 507
train:
  outputs/smoke_examink_seg_mask_heads_step1_20260706
eval:
  outputs/eval_examink_seg_smoke_baseline_train4_20260706
  outputs/eval_examink_seg_smoke_mask_heads_step1_train4_20260706
```

Result:

```text
baseline:  residual 0.193198, overerase 0.004355
candidate: residual 0.195089, overerase 0.003790

0.jpg residual -0.000474, overerase -0.000862
1.jpg residual +0.004418, overerase -0.000200
2.jpg residual +0.002942, overerase -0.000425
3.jpg residual +0.000675, overerase -0.000772
```

Do not promote this checkpoint. The result is useful evidence that explicit masks can reduce
overerase pressure, but coverage worsened on 3/4 pages. Future ExamInk-Seg use should train or
calibrate a selector/mask candidate with an explicit residual-coverage objective and larger
SCUT/holdout validation.

## Selective Box-Preserve Class2 Training

Rejected as a direct continuation path from the current checkpoint. The `target_diff_non_erase`
box-preserve target correctly identifies class `2` as non-erase/preserve on the hard proxy subset,
but bounded probes made the candidate too conservative for the existing strict hybrid gate.

Tested probes:

```text
box_preserve4 class2-only:
  run: outputs/exp_box_preserve4_class2patch_step2_20260705
  scut eval: outputs/scut_test115_hybrid_gate_box_preserve4_class2patch_step2_strict_20260705
  holdout eval: outputs/holdout40_hybrid_gate_box_preserve4_class2patch_step2_strict_20260705
  result: scut115 selected=0/115 residual=0.114224963938 overerase=0.003048296717
          holdout40 selected=0/40 residual=0.134026304621 overerase=0.002481606117

box_preserve1 class2-only:
  run: outputs/exp_box_preserve1_class2patch_step2_20260706
  scut eval: outputs/scut_test115_hybrid_gate_box_preserve1_class2patch_step2_strict_20260706
  holdout eval: outputs/holdout40_hybrid_gate_box_preserve1_class2patch_step2_strict_20260706
  result: scut115 selected=0/115 residual=0.114224963938 overerase=0.003048296717
          holdout40 selected=1/40 residual=0.134092140149 overerase=0.002443962885

box_preserve1 class2-mix50 step4:
  run: outputs/exp_box_preserve1_class2mix50_step4_20260706
  scut eval: outputs/scut_test115_hybrid_gate_box_preserve1_class2mix50_step4_strict_20260706
  holdout eval: outputs/holdout40_hybrid_gate_box_preserve1_class2mix50_step4_strict_20260706
  result: scut115 selected=0/115 residual=0.114224963938 overerase=0.003048296717
          holdout40 selected=1/40 residual=0.134166309288 overerase=0.002446837699

original strict:
  scut115 selected=6/115 residual=0.113955546480 overerase=0.003046624105
  holdout40 selected=3/40 residual=0.133596140373 overerase=0.002492318453
```

The failure mode is gate-feature drift, especially lower `copy_mask_cov8`, not a single threshold
miss. The original SCUT strict pages all lost eligibility:

```text
17.jpg:  cov8 0.806133 -> 0.485925 on mix50
156.jpg: cov8 0.806202 -> 0.535520 on mix50
254.jpg: cov8 0.923547 -> 0.775880 on mix50
303.jpg: cov8 0.877718 -> 0.648027 on mix50
370.jpg: cov8 0.848411 -> 0.713620 on mix50
371.jpg: cov8 0.835678 -> 0.673334 on mix50
```

Keep `micro_train_region_probe.py`'s patch-index mixed sampling and trace marker as reusable
diagnostic tooling, but do not continue class2-only or high-ratio class2 box-preserve training as
the next product path. Future attempts should first preserve the original strict gate features, then
try to enlarge selected coverage.

## Residual-Delta Direct Promotion And Hand-Mined Selectors

Rejected as a product default. The bounded residual-delta cleanup branch is safer than the full
clean-image head, but direct promotion worsened residual on both SCUT115 and holdout40. Page-level
oracle selection shows some real wins, but simple label-free threshold selectors only transfer at
very narrow coverage.

Tested evidence:

```text
SCUT115:
  baseline:  residual=0.114225, overerase=0.003048
  candidate: residual=0.118880, overerase=0.003014
  oracle wins=40/115, losses=75/115, oracle_gain=0.002361903162
  best local safe rule selected=12, wins=12, losses=0, gain=0.000982513297

holdout40:
  baseline:  residual=0.130543, overerase=0.002732
  candidate: residual=0.134065, overerase=0.002694
  oracle wins=15/40, losses=25/40, oracle_gain=0.001651170237

joint safe selector:
  active_gray_p25 >= 111.6 AND candidate_delta_max <= 200.133333333
  selected=7 total, wins=7, losses=0
  SCUT115 selected=6, holdout40 selected=1
  total residual_gain=0.001496633750
```

Do not spend more cycles on scalar thresholds or global edit-strength increases for this candidate.
The reusable part is `scripts/analysis/analyze_residual_delta_selector_features.py`; future work
should train or label a page-level selector and require materially higher coverage before promotion.

## Residual-Delta Dark-Preservation W2 Cleanup

Rejected as the next product path. Adding `--dark-preserve-weight 2.0` to the residual-delta
cleanup training reduced edit magnitude and slightly improved the average local proxy gain on
original reject pages, but it did not materially change page-level local proxy verdicts and it
gave up a large share of the original t4 residual metric improvement.

Training/eval:

```text
checkpoint:
  outputs/train_scut_residual_delta_darkpreserve_w2_step150_20260707/cleanup_best.pt

four-split metric result versus baseline:
  scut115:   residual_delta=-0.000189791, overerase_delta=-0.000003162
  holdout40: residual_delta=-0.000322906, overerase_delta=-0.000004172
  train160:  residual_delta=-0.000967412, overerase_delta=-0.000006834
  next120:   residual_delta=-0.000710610, overerase_delta=-0.000002712
  combined:  residual_delta=-0.000631727, overerase_delta=-0.000004482

original residual-delta t4 combined:
  residual_delta=-0.000935733, overerase_delta=-0.000005804
```

Local proxy verdicts changed only marginally:

```text
original t4:       accept=76, review=41, reject=318
dark-preserve w2:  accept=76, review=44, reject=315

verdict transitions:
  reject -> reject: 310
  accept -> accept: 74
  review -> review: 34
  reject -> review: 8
  review -> reject: 5
  review -> accept: 2
  accept -> review: 2
```

Aggregate local-proxy effects:

```text
accept pages: delta_changed=-0.00095431, delta_gain=-0.01079021
review pages: delta_changed=-0.00024989, delta_gain=-0.00045024
reject pages: delta_changed=-0.00032377, delta_gain=+0.00624071
```

Interpretation: the loss does activate and makes reject pages less bad on average, but the verdict
distribution barely moves and accepted pages lose useful cleanup strength. Do not repeat
`dark-preserve-weight=2.0` step-150 training as-is. If revisiting this direction, use it as one term
inside a broader signed-delta or failure-bucket training set that explicitly preserves accepted
high-gain brightening while targeting target-darker regressions.

## Mixed Target-Darker Hard-Negative Residual-Delta Cleanup

Rejected as the next product path. Mixing target-darker brightening hard negatives back into the
metric patch training set caused the second-stage model to edit far more area while preserving less
useful cleanup. It reduced aggregate overerase, but residual error regressed badly on every split,
including the held-out SCUT115 test pages.

Training/eval:

```text
checkpoint:
  outputs/train_mixed_target_darker_residual_delta_w2_step150_20260707/cleanup_best.pt

training patch index:
  outputs/mixed_metric_target_darker_patch_index_20260707/train_patch_index.csv
  rows=359
  target_darker_brightening=256
  metric=103

four-split metric result versus baseline:
  scut115:   residual_delta=+0.003735417, overerase_delta=-0.000178287
  holdout40: residual_delta=+0.002953104, overerase_delta=-0.000157442
  train160:  residual_delta=+0.011339396, overerase_delta=-0.000210210
  next120:   residual_delta=+0.008812737, overerase_delta=-0.000147597
  combined:  residual_delta=+0.007860986, overerase_delta=-0.000179646

reference residual-delta t4 combined:
  residual_delta=-0.000935733, overerase_delta=-0.000005804

dark-preserve w2 combined:
  residual_delta=-0.000631727, overerase_delta=-0.000004482
```

The mixed candidate's average gate ratio also jumped from the t4 reference `0.001316005` to
`0.011182734`, which is consistent with over-broad second-stage edits rather than selective
repair.

Interpretation: the target-darker brightening failure bucket is real, but training directly on
those patches with the current residual-delta objective turns into a conservative/over-broad edit
model rather than a product-safe repair model. Do not repeat this mixed hard-negative setup as-is.
If revisiting, separate it into an auxiliary signed-delta constraint or a selector/candidate
ranking problem instead of merging it into the main cleanup patch objective.

## Signed-Delta W1 Inside-Only Residual-Delta Cleanup

Rejected as the next product path. A narrower training-only signed-delta direction loss avoids the
mixed hard-negative data expansion problem, but the first `--signed-delta-weight 1.0
--signed-delta-inside-only` run still did not beat the original residual-delta t4 or dark-preserve
w2 candidates. It improved train160 and slightly improved next120 versus baseline, but regressed
the SCUT115 test split and holdout40.

Training/eval:

```text
checkpoint:
  outputs/train_scut_residual_delta_signed_delta_w1_step150_20260707/cleanup_best.pt

four-split metric result versus baseline:
  scut115:   residual_delta=+0.000598372, overerase_delta=-0.000007248
  holdout40: residual_delta=+0.000769251, overerase_delta=-0.000007612
  train160:  residual_delta=-0.002021265, overerase_delta=-0.000012558
  next120:   residual_delta=-0.000116853, overerase_delta=-0.000006055
  combined:  residual_delta=-0.000546763, overerase_delta=-0.000008905

reference residual-delta t4 combined:
  residual_delta=-0.000935733, overerase_delta=-0.000005804

dark-preserve w2 combined:
  residual_delta=-0.000631727, overerase_delta=-0.000004482
```

The signed-delta w1 run lowered overerase more than t4, but residual improvement was weaker and
less stable across splits. The average gate ratio also increased from the t4 reference
`0.001316005` to `0.003895908`, so it is not yet a safer drop-in replacement.

Interpretation: keep the default-disabled signed-delta loss as reusable training infrastructure,
but do not promote `signed-delta-weight=1.0` inside-only step-150 as the main cleanup checkpoint.
If revisiting, sweep lower weights/margins or combine with a selector/ranking stage, and require no
SCUT115/holdout40 residual regression before local-proxy review.

## Label-Free Hand-Ruled Selector From Train Splits

Rejected as a product selector path in its current hand-ruled form. A split-validated selector
check trained rules on train160+next120 and evaluated them on scut115+holdout40. The best rules
that were clean on train splits did not generalize: both selected three test/holdout pages and all
three were local-proxy rejects.

Validation:

```text
script:
  scripts/analysis/validate_label_free_selector_rules.py

input:
  outputs/residual_delta_t4_local_proxy_20260707/local_target_comparison_with_diagnostics.csv
  outputs/selector_features_residual_delta_t4_20260707/page_features.csv

split:
  train=train160,next120
  test=scut115,holdout40

best observed rules:
  active_delta_mean <= 4.11656099509 AND candidate_delta_mean >= 0.00781938404494
  active_delta_mean <= 4.11656099509 AND gate_ratio >= 0.00187798218501

train result:
  selected=10/280
  metric_losses=0
  accept=9
  review=1
  reject=0
  residual_gain=+0.000340916470

test/holdout result:
  selected=3/155
  metric_losses=3
  accept=0
  review=0
  reject=3
  residual_gain=-0.000039258120
```

Interpretation: current scalar, hand-mined label-free rules are not stable enough to be product
selectors, even at very low coverage. Keep the split-validation script for guardrail testing, but
do not spend more cycles widening hand-written rule searches unless a learned selector/ranking
signal is introduced and validated on held-out pages.

## Lightweight Learned Page Selector Ranker

Rejected as a product selector path in its current feature/label form. A dependency-light
logistic ranker trained on train160+next120 local-proxy supervision and validated on
scut115+holdout40 improves over fully hand-mined rules in some settings, but it still does not
reach the zero-reject / zero-metric-loss bar required for promotion.

Validation:

```text
script:
  scripts/analysis/train_page_selector_ranker.py

input:
  outputs/residual_delta_t4_local_proxy_20260707/local_target_comparison_with_diagnostics.csv
  outputs/selector_features_residual_delta_t4_20260707/page_features.csv

split:
  train=train160,next120
  test=scut115,holdout40

sweep:
  positive_mode=safe,accept
  lr=0.005,0.01,0.02
  l2=0.01,0.05,0.1,0.2

best sweep result:
  mode=safe
  lr=0.01
  l2=0.01
  threshold=0.833743115485
  test_selected=2/155
  test_accept=0
  test_review=2
  test_reject=0
  test_metric_losses=1
  test_residual_gain=+0.000000515789

stable smoke after numeric fixes:
  test_selected=4/155
  test_accept=1
  test_review=2
  test_reject=1
  test_metric_losses=2
  test_residual_gain=-0.000211910547
```

Interpretation: a learned selector is directionally more appropriate than hand-written scalar
rules, but the current page-level features and local-proxy labels are insufficient. Keep the
ranker as reusable infrastructure for future feature/label experiments, but do not use this model
or sweep as product gating. The next selector attempt should add stronger non-target inference-time
features or a larger manually reviewed page set, then require held-out zero reject, zero metric
loss, and meaningful coverage before promotion.

## Enhanced Non-Target Page Selector Features

Not promoted yet. Adding inference-time-only direction, connected-component, edge, and texture
features to the residual-delta selector feature extractor improved the learned ranker versus the
initial page-level feature set, but it still did not clear the product safety gate.

Validation:

```text
script:
  scripts/analysis/analyze_residual_delta_selector_features.py
  scripts/analysis/train_page_selector_ranker.py

added non-target features:
  active_brighten_ratio / active_darken_ratio
  active_signed_delta_mean / active_signed_delta_abs_mean
  active_brighten_mean / active_darken_mean
  active_edge_mean / p75 / p95
  active_texture_mean / p75 / p95
  active_component_count / area_mean / area_p95 / area_max

split:
  train=train160,next120
  test=scut115,holdout40

enhanced ranker best held-out row:
  test_selected=9/155
  test_accept=6
  test_review=2
  test_reject=1
  test_metric_losses=0
  test_residual_gain=+0.000297278664
  test_overerase_delta=-0.000000077781

initial ranker best sweep:
  test_selected=2/155
  test_reject=0
  test_metric_losses=1
  test_residual_gain=+0.000000515789
```

Interpretation: richer inference-time features are a real improvement because they eliminate
held-out metric losses while increasing useful coverage. The remaining reject means this is still
not product-safe. Keep these features, but require a second-stage risk veto or stronger reviewed
labels before promoting a selector.

## Enhanced Ranker Veto Search

Not promoted yet. A label-free veto search on top of the enhanced residual-delta ranker found
non-target feature rules that remove the remaining held-out reject, but the resulting coverage is
still too low for product gating and the rule is based on a small selected set.

Validation:

```text
script:
  scripts/analysis/evaluate_page_selector_veto.py

input:
  outputs/page_selector_ranker_t4_enhanced_train160_next120_to_scut115_holdout40_20260707/predictions.csv
  outputs/selector_features_residual_delta_t4_enhanced_20260707/page_features.csv
  outputs/residual_delta_t4_local_proxy_20260707/local_target_comparison_with_diagnostics.csv

base selector:
  enhanced ranker threshold=0.780156652583
  train_selected=40
  test_selected=9
  test_accept=6
  test_review=2
  test_reject=1
  test_metric_losses=0

best train-mined veto/keep rule:
  active_component_area_mean <= 3.36374043737
  AND active_texture_p95 <= 138.780294495

held-out after veto:
  test_selected=7/9 selected pages
  test_accept=5
  test_review=2
  test_reject=0
  test_metric_losses=0
  test_residual_gain=+0.004322267301 per selected-set denominator
```

Interpretation: the remaining reject is separable by inference-time component/texture features, so
the direction is better than pure score thresholding. This is still not enough to promote because
coverage over the full held-out set remains only 7/155 pages and the selected train set is small.
Keep the veto evaluator as reusable infrastructure; the next quality lift should use more reviewed
pages or a richer patch/region selector, not another hand-tuned page-level rule.

## Region Component Selector Probe

Not promoted. A connected-component selector was added to test whether the residual-delta candidate
can be made safer by keeping only local edit components selected by inference-time features. This
is a more appropriate granularity than whole-page thresholding, but simple component-level rules
still do not separate safe edits well enough.

Validation:

```text
script:
  scripts/analysis/evaluate_region_component_selector.py

input:
  baseline/candidate metrics for train160,next120,scut115,holdout40
  candidate=residual_delta_t4

components:
  total=106489
  train=79442
  test=27047
  features=20

strict rule search:
  max_train_reject_components=0
  rules=0

ratio rule search:
  max_train_reject_ratio=0.05
  min_train_components=500
  rules=0

best observed simple rules:
  single: candidate_source_delta_mean >= 140.548663102
    train_n=7945
    train_reject_ratio=0.259
    test_reject_ratio=0.474
  pair: baseline_edit_mean >= 136 AND fill_ratio <= 0.25
    train_n=792
    train_reject_ratio=0.182
    test_reject_ratio=0.361
```

Interpretation: region-level selection is still the right direction, but the current hand-written
component features are not enough. The train/test reject-ratio gap is large, and even the best
simple pair keeps too many harmful components. Do not continue hand-tuning component thresholds.
The next viable step should be a learned patch/region selector with reviewed labels, or richer
component features that include local context without target/label leakage.

## Region Component Ranker Probe

Not promoted. A lightweight NumPy logistic ranker was trained on the region-component features to
test whether learned weighting improves over simple component thresholds. It improves coverage
versus the strict threshold search, but still cannot separate harmful held-out components safely.

Validation:

```text
script:
  scripts/analysis/train_region_component_ranker.py

input:
  outputs/region_component_selector_t4_ratio05_train160_next120_to_scut115_holdout40_20260707/components.csv

split:
  train=train160,next120
  test=scut115,holdout40

training:
  positive_mode=accept
  epochs=2500
  lr=0.01
  l2=0.05

best held-out threshold:
  test_selected=230/27047 components
  test_accept=129
  test_review=16
  test_reject=85
  test_reject_ratio=0.369565
  test_gain_per_component=+0.010094

comparison:
  best simple pair reject ratio: train=0.182, test=0.361
  learned ranker reject ratio: test=0.370
```

Interpretation: the current component features contain some useful signal but not enough for a
safe region selector. The ranker does not materially beat the best simple pair on held-out reject
ratio, so weak target-derived component labels alone are insufficient. Keep the script for future
experiments with reviewed region labels, but do not promote this ranker or use it for product
gating.

## Sign-Separated Residual Route/Magnitude v2

Killed before inner-val15. The representation passed deterministic identity, signed-output,
branch-gradient, role-isolation, train275 materialization, and application-reachability
prerequisites. Its only authorized real run nevertheless collapsed every audited train-patch
route argmax to `darken`:

```text
checkpoint: artifacts/archive/sign-separated-residual-repair-20260810/training-output/sign_separated_probe.pt
steps / lr: 80 / 0.0001
audited patches / pixels: 512 / 33,554,432
route argmax identity / brighten / darken: 0 / 0 / 33,554,432
application-eligible brighten / darken: 0 / 8,514,478
candidate maximum delta: 13.231361 gray
inner-val15 started: false
decision: KILL
```

Do not repeat this route/magnitude family or attempt learning-rate, step-count, loss-weight, or
application-threshold rescue. The v1 `0.00002` learning rate was separately proven synthetically
unreachable at 80 steps; v2 fixed reachability but exposed a real one-route collapse. Any future
work on residual/overerase direction separation requires a materially different representation
and a new preregistration.
