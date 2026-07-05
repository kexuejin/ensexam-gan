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
