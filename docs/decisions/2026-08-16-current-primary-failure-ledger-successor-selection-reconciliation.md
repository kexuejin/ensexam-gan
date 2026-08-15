# Current-Primary Failure Ledger Successor Selection Reconciliation

## Decision

`PREREQUISITE_NEEDED`. The durable failure-bucket ledger is reconciled with the
newer current-primary quality-loop ledger: the old active
`cross_domain_residual_headroom_vs_source_solved_pixel_regression` bucket is no
longer an executable bucket. Its one named next path, the D3 train-only
baseline-tail non-regression sidecar, consumed its prerequisite, preflight, and
single inner-val15 run, then KILLed on the same source residual regression class
as D2. The follow-up D4 primary-edit direction and D5 folded-direction attempts
also reached terminal no-lift/noop KILL states.

The program remains active, but the active work item is now successor selection:
choose a named failure bucket outside the closed cross-domain sidecar,
stroke-only source-candidate, and target-dark/overerase-risk buckets with
available leakage-safe train-only evidence; preregister one materially different
candidate family; or record broader durable exhaustion if no such path remains.

## Evidence

```text
current quality-loop ledger last terminal:
  id = target-dark-overerase-bucket-exhaustion
  terminal = KILL
  next pending = post_target_dark_overerase_bucket_exhaustion_successor_selection

cross-domain sidecar terminal evidence:
  universal-sidecar-d3-baseline-tail-step80 = KILL
  universal-sidecar-d4-primary-edit-direction-inner-val15-step80 = KILL
  universal-sidecar-d5-folded-direction-inner-val15-step80 = KILL

closed successor buckets already recorded:
  stroke-only-source-candidate-bucket-exhaustion = KILL
  target-dark-overerase-bucket-exhaustion = KILL
```

## Boundaries

No training, candidate inference, SCUT115, holdout40, visual review, reserved
blind, promotion, or `artifacts/current-primary` replacement is opened by this
reconciliation. It only removes a stale active-bucket pointer and makes the
successor-selection prerequisite explicit in the fail-closed program reporter.

Intent: Prevent stale active-bucket status from reopening a killed sidecar path.
Constraint: The quality-loop ledger is the more complete current-state record and keeps promotion, validation, reserved blind, and replacement closed.
Rejected: Keep cross-domain sidecar as the active bucket | its registered baseline-tail and direction-repair routes already reached terminal KILL states.
Rejected: Mark the full goal complete | successor selection or broader durable exhaustion still needs an explicit current-state decision.
Confidence: high
Scope-risk: narrow
Directive: Do not execute D3/D4/D5 sidecar rescues from the old failure ledger; use the successor-selection bucket until a new leakage-safe bucket is preregistered or broader exhaustion is recorded.
Tested: Fail-closed current-primary program reporter passed with active_bucket=successor_selection_outside_closed_buckets and all 7 bucket evidence checks true; quality-loop status reporter remained active_not_promotion_eligible with candidate_admission_ready=false, promotion_eligible=false, and reserved_blind_authorized=false.
Not-tested: Training, candidate inference, validation surfaces, visual review, reserved blind, promotion, or current-primary replacement.
