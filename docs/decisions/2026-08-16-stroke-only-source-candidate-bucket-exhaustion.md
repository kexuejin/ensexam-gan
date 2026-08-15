# Stroke-Only Source Candidate Bucket Exhaustion

## Decision

`KILL`. The stroke-only source-candidate successor bucket is exhausted for the
current `monotonic-residual-erase-support` iteration.

This closes the source-only postprocess route after every materially distinct
available selector family either required missing historical selector-replay
candidate PNGs, failed the train-only preserve-first gate, failed
`inner_val15`, or failed SCUT115. Continuing inside the same bucket would be
threshold, alpha, component, kernel, dilation, stroke-only variant,
page-specific, visual-review, or held-out rescue, which the ledger already
forbids.

## Evidence

- Exhaustion record:
  `docs/stroke-only-source-candidate-bucket-exhaustion-v1.json`
- Covered terminal families:
  - `exact_selector_replay_candidate_pngs`: `PREREQUISITE_NEEDED` because the
    exact historical candidate PNGs are still locally absent.
  - `source_dark_local_paper_lift_v1`: `KILL` at train160 preserve-first
    screening because residual gain came with severe overerase regression.
  - `source_dark_thin_component_lift_v1`: `KILL` at train160 preserve-first
    screening because usable residual gain was negligible and broader variants
    regressed overerase.
  - `source_chroma_primary_edit_lift_v1`: `KILL` at `inner_val15`.
  - `source_achroma_primary_edit_lift_v1`: `KILL` at `inner_val15` on
    page-level residual regression.
  - `source_edge_primary_edit_lift_v1`: `KILL` at SCUT115 because `179.jpg`
    regressed residual despite aggregate and tail residual improvement.

## Boundary

This is not a promotion event. No model training, checkpoint generation,
candidate inference, `inner_val15`, development gate, SCUT115, holdout40,
visual review, reserved blind, promotion, or `artifacts/current-primary`
replacement is opened by this record.

The next allowed move is outside the closed stroke-only source-candidate
successor bucket:

- preregister a materially different non-source-only candidate family with
  train-only evidence,
- select a new named failure bucket with an available train-only path, or
- record durable exhaustion if no remaining failure bucket has an available
  leakage-safe path.

Intent: Stop repeating exhausted stroke-only source-candidate successors and move the quality loop to a materially different failure bucket or candidate family.
Constraint: Source-only candidate rescue after held-out or preserve-first failure would violate the evidence-gated loop.
Rejected: Tune thresholds, alpha, component limits, kernels, dilation, or stroke-only blends | those are same-family rescues after terminal gate failures.
Rejected: Use visual review, held-out repeats, or page-specific replacement to save `source_edge_primary_edit_lift_v1` | SCUT115 no-page-regression failure is terminal.
Rejected: Promote or replace `artifacts/current-primary` from any source-only bucket result | no candidate passed all required gates and reserved blind remains closed.
Confidence: high
Scope-risk: narrow
Directive: Do not open another stroke-only source-candidate successor unless exact missing selector-replay PNGs are restored with hash custody or a genuinely new source of train-only evidence is preregistered.
Tested: ledger source families reviewed against terminal PASS/KILL/PREREQUISITE_NEEDED records.
Not-tested: new non-source-only family, reserved blind, promotion.
Related: docs/decisions/2026-08-16-source-edge-primary-edit-lift-scut115-kill.md
