# Candidate 5 Promotion-Readiness Evidence Admission

## Decision

Close the current Candidate 5 specialist-promotion evidence line as
`specialist_not_promotable_line_closed` with reason
`fresh_blind_unavailable_current_workspace`.

This is an evidence-admission result, not a finding that Candidate 5 fails on a
fresh HW5K-domain blind set. No such admissible set is currently registered.
Candidate 5 remains `research_only/gate_qualified_nonpromotion`,
`artifacts/current-primary` remains unchanged, and automatic routing remains
unauthorized.

## Evidence

- Candidate 5 passed the 232-page HW5K development gate but failed the SCUT
  source-domain gate as a universal replacement. The stopped gate chain did not
  run a new final blind set
  (`2026-08-02-hw5k-mixed-candidate5-gate-b-rejection.md`, lines 3-15 and
  118-126).
- The 525-page official HW5K test set was already registered, inferred, scored,
  and audited for current-primary on 2026-07-26. That decision explicitly
  forbids reuse for subsequent model selection and requires a separately
  reserved unseen final test set for a later candidate
  (`2026-07-26-hw5k-final-blind-current-primary.md`, lines 56-75).
- The explicit-domain research-harness contract independently requires a fresh
  unseen HW5K-domain blind set and explicitly says HW5K-test cannot be reused
  (`2026-08-03-explicit-domain-dual-checkpoint-research-harness.md`, lines
  99-109).
- The local HW5K train and development splits have already participated in
  training and candidate selection. They remain development evidence and cannot
  be relabeled as final-blind evidence.

## Actions Not Authorized

- Do not run Candidate 5 on the consumed official HW5K test set for a new
  promotion claim.
- Do not reinterpret the absence of a fresh blind set as a measured Candidate 5
  quality failure.
- Do not tune thresholds, selectors, routing, or preprocessing on the consumed
  HW5K test outputs.
- Do not enter automatic-router work as a fallback.

## Handoff

The next milestone belongs to the sustainable multi-domain product program: run
a bounded source-and-custody search for a new, task-compatible HW5K-like blind
set, then either register one before any Candidate 5 inference or record an
external-data prerequisite. Candidate evaluation begins only after admission.

## Verification

- No training or inference was run for this admission decision.
- No checkpoint, prediction, selector, or current-primary artifact was changed.
- The stale local HW5K provenance summary was corrected to reflect the completed
  2026-07-26 final-blind use.
