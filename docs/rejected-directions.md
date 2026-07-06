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
