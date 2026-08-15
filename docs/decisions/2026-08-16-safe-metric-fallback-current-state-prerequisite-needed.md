# Safe-Metric Fallback Current-State Prerequisite Needed

## Decision

`PREREQUISITE_NEEDED`. The safe-metric fallback route is not currently usable as
train-only evidence for candidate admission.

The quality notes reference a fail-closed fallback queue and oracle-preflight
summary, but the referenced output root and both target files are absent from
the current worktree. This is an evidence custody problem, not a candidate PASS.
The route may only resume after rebuilding the queue and oracle-preflight
evidence in a fresh output directory, or it should be bypassed for another named
failure bucket with available leakage-safe train-only evidence.

## Evidence

- Current-state prerequisite record:
  `docs/safe-metric-fallback-current-state-prerequisite-v1.json`
- Current-state prerequisite record SHA256:
  `12df9684140228aec20195399dd6a8d43b56b4f73806030e5fefc4803146e78a`

Missing current-state paths:

```text
outputs/balanced007_ranker_expansion_source_eval_20260708
outputs/balanced007_ranker_expansion_source_eval_20260708/quality_goal_review_20260719/safe_metric_fallback_queue_failclosed_reverify2/safe_metric_fallback_queue.csv
outputs/balanced007_ranker_expansion_source_eval_20260708/quality_goal_review_20260719/safe_metric_fallback_oracle_preflight_failclosed_v3_20260719/summary.json
```

A narrow filesystem scan for `safe_metric_fallback`, `fallback_oracle`, and
`quality_goal_review` under `outputs/` found no substitute evidence files.

## Boundary

No selector expansion, candidate inference, generator training, `inner_val15`,
development gate, SCUT115, holdout40, visual review, reserved blind, promotion,
or `artifacts/current-primary` replacement is authorized.

The next move must either rebuild the missing safe-metric fallback evidence in a
new output directory, choose another named failure bucket with available
train-only evidence, or record broader durable exhaustion if no such path
remains.

Intent: Prevent a missing safe-metric fallback evidence path from being treated as candidate-admission proof.
Constraint: The quality loop requires current-state evidence files or a new fail-closed rebuild before route selection.
Rejected: Use the referenced but absent queue and oracle summary as evidence | the files are missing in the current worktree.
Rejected: Substitute unrelated output paths or quality-note prose | successor admission must be grounded in verifiable current-state artifacts.
Rejected: Open selector expansion, training, validation, reserved blind, or promotion | no candidate or train-only PASS exists for this route.
Confidence: high
Scope-risk: narrow
Directive: Rebuild safe-metric fallback evidence in a fresh output directory before using this route, or move to another evidence-backed failure bucket.
Tested: current-state path checks for the referenced output root, queue CSV, and oracle-preflight summary.
Tested: narrow outputs scan found no safe-metric fallback substitute files.
Not-tested: regenerated safe-metric fallback queue, regenerated oracle preflight, reserved blind, promotion.
Related: docs/decisions/2026-08-16-balanced007-delta-trust-oracle-ceiling-recheck-kill.md
