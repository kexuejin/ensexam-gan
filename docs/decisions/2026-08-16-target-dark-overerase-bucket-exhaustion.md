# Target-Dark / Overerase Bucket Exhaustion

## Decision

`KILL`. The currently available `target_dark_or_overerase_risk` successor bucket
is exhausted for this iteration.

This does not close the long-lived quality goal and does not promote a
candidate. It only closes the current bucket after its available leakage-safe
routes all failed before candidate admission:

- `target_dark_component_context_feature_v1`: the train160 component-ranker
  recheck failed for both 18 scalar features and the 26-feature context /
  printed-line extension.
- `balanced007_delta_trust_oracle_ceiling_v1`: the development-only oracle was
  worse than the current candidate on both train160 and next120.
- `safe_metric_fallback_oracle_reconstruction_v1`: the only available
  current-state reconstruction produced zero auto-expand rows and negative
  candidate-relative oracle headroom on both development pages.

## Evidence

- Bucket exhaustion record:
  `docs/target-dark-overerase-bucket-exhaustion-v1.json`
- Bucket exhaustion record SHA256:
  `0226ac8e0daa00cf8ebab15f8574fe934f49b08ee7cf71fdbfd937d45e69fae5`
- Component-context recheck record:
  `docs/target-dark-component-context-feature-recheck-v1.json`
- Component-context recheck record SHA256:
  `b94882379a22e9c152333a14621700d455e136ad9056cb197983720653fd9957`
- Delta-Trust oracle recheck record:
  `docs/balanced007-delta-trust-oracle-ceiling-recheck-v1.json`
- Delta-Trust oracle recheck record SHA256:
  `1024407a756c50029eac2e63887c9ec442353032109ba5ffe63209917c74c635`
- Safe-metric fallback reconstruction record:
  `docs/safe-metric-fallback-reconstruction-no-headroom-v1.json`
- Safe-metric fallback reconstruction record SHA256:
  `f8534e4966d7fc205115886a6f4baffe31b116c2fde2b0e9d8a5f1c2d55fc17b`

## Boundary

No candidate inference, generator training, `inner_val15`, development gate,
SCUT115, holdout40, visual review, reserved blind, promotion, selector
promotion, or `artifacts/current-primary` replacement is authorized.

The next move must leave this closed bucket: select another named failure bucket
with an available leakage-safe train-only path, preregister a materially
different non-component / non-Delta-Trust / non-safe-metric-fallback family, or
record broader durable exhaustion if no such path remains.

Intent: Stop returning to target-dark/overerase-risk routes that have no admissible train-only or development headroom.
Constraint: All current-state routes in this bucket failed before candidate admission and before any validation or blind surface opened.
Rejected: Add more labels, thresholds, feature subsets, page-specific overrides, selector expansion, validation, or reserved blind to rescue the bucket | those are prohibited after the failed train-only/development gates.
Rejected: Treat this as full goal completion | other named buckets or broader exhaustion still need explicit selection or evidence.
Confidence: high
Scope-risk: narrow
Directive: Do not repeat target_dark_or_overerase_risk successor routes through component-ranker, Delta-Trust, or safe-metric fallback variants unless a materially new train-only evidence source is preregistered.
Tested: Current ledger status after safe-metric fallback KILL remained active_not_promotion_eligible with reserved blind and promotion disabled.
Not-tested: New failure bucket selection, candidate inference, generator training, validation gates, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-16-target-dark-component-context-feature-recheck-kill.md
Related: docs/decisions/2026-08-16-balanced007-delta-trust-oracle-ceiling-recheck-kill.md
Related: docs/decisions/2026-08-16-safe-metric-fallback-reconstruction-no-headroom-kill.md
