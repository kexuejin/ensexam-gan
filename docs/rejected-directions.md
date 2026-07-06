# Rejected Directions

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
