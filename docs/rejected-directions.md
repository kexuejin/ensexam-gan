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
