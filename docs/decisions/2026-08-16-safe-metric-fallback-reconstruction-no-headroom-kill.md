# Safe-Metric Fallback Reconstruction No-Headroom KILL

## Decision

`KILL`. A fresh current-state reconstruction of the safe-metric fallback route
does not provide candidate-admission evidence.

The originally referenced `zero_reject_veto_112` / 163-page fallback queue is
still absent, so the recheck used the only available aligned selector snapshot:
`outputs/materialized_zero_reject_veto_156_alt_borderline_metric_safe_confirmed_safe_20260720/selection.csv`
joined with the smoke target-quality overlay. That narrower reconstruction
produced 125 safe-metric baseline fallback rows, all of which were either
`retain_baseline` or `review_required`; there were zero auto-expand rows.

The strict oracle preflight then passed provenance checks for the train/next
development rows, but both selected development pages had negative
candidate-relative residual headroom:

```text
next120/308.jpg: candidate_minus_oracle=-0.0031394351782701446
train160/132.jpg: candidate_minus_oracle=-0.001267511674449634
mean_candidate_minus_oracle=-0.0022034734263598893
headroom_pages=0/2
passes_candidate_relative_headroom_gate=false
```

## Evidence

- Current-state reconstruction record:
  `docs/safe-metric-fallback-reconstruction-no-headroom-v1.json`
- Current-state reconstruction record SHA256:
  `f8534e4966d7fc205115886a6f4baffe31b116c2fde2b0e9d8a5f1c2d55fc17b`
- Queue CSV:
  `outputs/safe_metric_fallback_reconstruction_probe_20260816_v2/queue_from_materialized156_smoke_quality/safe_metric_fallback_queue.csv`
- Queue CSV SHA256:
  `a6ce03319e6f51023645281baea0b981cc96d13e03993c49345aabffcefaef0a`
- Queue summary:
  `outputs/safe_metric_fallback_reconstruction_probe_20260816_v2/queue_from_materialized156_smoke_quality/safe_metric_fallback_queue_summary.json`
- Queue summary SHA256:
  `4b918a56b31da014ac47d7bf9aa699cc9f693da1ddc5ed4e0028b4a4f49fbac6`
- Oracle preflight summary:
  `outputs/safe_metric_fallback_reconstruction_probe_20260816_v2/oracle_preflight_from_materialized156_smoke_quality/summary.json`
- Oracle preflight summary SHA256:
  `d348ec24ba6054a559db592676d390a48279085df76dfed2424c1e664797a31a`
- Oracle preflight per-page CSV:
  `outputs/safe_metric_fallback_reconstruction_probe_20260816_v2/oracle_preflight_from_materialized156_smoke_quality/per_page.csv`
- Oracle preflight per-page CSV SHA256:
  `c2b530b9866f81a4b0ee9fea47e081fa7bc0d9696b3277096b9147b33e9436f4`

## Boundary

No selector expansion, candidate inference, generator training, `inner_val15`,
development gate, SCUT115, holdout40, visual review, reserved blind, promotion,
or `artifacts/current-primary` replacement is authorized.

The next move must choose another named failure bucket with leakage-safe
train-only evidence, preregister a materially different non-safe-metric-fallback
family, or record broader durable exhaustion if no such path remains.

Intent: Close a reconstructed safe-metric fallback route that lacks candidate-relative headroom.
Constraint: The original 112-selector queue is absent; the only available current-state reconstruction is the materialized156 selector plus smoke target-quality overlay.
Rejected: Treat the reconstructed review-required pages as candidate evidence | both development pages have negative candidate-minus-oracle residual headroom.
Rejected: Expand selector coverage from the queue | the queue produced zero auto-expand rows and the oracle gate failed.
Rejected: Rescue with thresholds, page-specific review, validation, or reserved blind | no train-only PASS exists for this family.
Confidence: high
Scope-risk: narrow
Directive: Do not repeat safe_metric_fallback_oracle_reconstruction_v1 or route around its failed headroom gate with selector expansion, threshold tuning, page-specific overrides, validation, visual review, reserved blind, or promotion.
Tested: Rebuilt available current-state queue from materialized156 selector and smoke target-quality overlay; ran strict train160/next120 oracle preflight against reverify oracle CSVs.
Not-tested: Original absent 112-selector 163-page queue, candidate inference, generator training, validation gates, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-16-safe-metric-fallback-current-state-prerequisite-needed.md
Related: docs/decisions/2026-08-16-balanced007-delta-trust-oracle-ceiling-recheck-kill.md
