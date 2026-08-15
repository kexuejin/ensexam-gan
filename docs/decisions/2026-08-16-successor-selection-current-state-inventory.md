# Successor Selection Current-State Inventory

## Decision

`PREREQUISITE_NEEDED`. Successor selection after the target-dark/overerase
bucket exhaustion has been reduced to an explicit current-state inventory. The
inventory does not admit a candidate and does not close the long-lived goal. It
records that the currently ledgered successor buckets are either terminal or
blocked:

- `cross_domain_residual_headroom_vs_source_solved_pixel_regression` is
  exhausted by D3/D4/D5 terminal sidecar outcomes.
- `external_text_layout_support_successors` is exhausted: the support
  diagnostic passed, but every currently registered edit-use route failed
  before candidate or quality surfaces.
- `stroke_only_source_candidate_successors` is exhausted unless exact missing
  selector-replay candidate PNGs are restored with hash custody.
- `target_dark_or_overerase_risk_successors` is exhausted under current-state
  evidence.
- `universal_mechanism_admission` remains blocked on an explicit product-owner
  entry decision.

## Boundary

This is an inventory and gating record only. It opens no training, candidate
inference, `inner_val15`, development gate, SCUT115, holdout40, visual review,
reserved blind, promotion, or `artifacts/current-primary` replacement.

The active next step is now narrower: select a materially different named
failure bucket with new leakage-safe train-only evidence, restore exact
selector-replay candidate PNGs with hash custody before reopening the source
bucket, obtain the product-owner entry decision for universal mechanism
admission, or record broader durable exhaustion if no remaining bucket has an
available path.

## Evidence

```text
inventory = docs/successor-selection-current-state-inventory-v1.json
current quality loop = docs/current-primary-quality-loop-ledger.json
program ledger = docs/current-primary-failure-ledger.md
candidate_admission_ready = false
promotion_eligible = false
reserved_blind_authorized = false
```

Intent: Make successor selection fail-closed instead of letting old external-layout, source-only, target-dark, or sidecar routes reopen.
Constraint: Current evidence contains no promotion-safe candidate and no authorized validation or blind surface.
Rejected: Select external text layout alone as the next bucket | full score, incremental score, and binary mask residual routes all failed preserve separation or application reachability.
Rejected: Reopen source-only candidates without exact selector-replay PNG custody | the bucket is terminal unless those exact missing assets are restored and hash-audited.
Rejected: Mark the full goal complete | this inventory does not prove every possible future bucket is exhausted.
Confidence: high
Scope-risk: narrow
Directive: Continue only with a materially different leakage-safe bucket, exact asset restoration, product-owner unblock, or a separate broader-exhaustion record.
Tested: `report_current_primary_program_status.py` passed with active_bucket=successor_selection_outside_closed_buckets, 10 buckets, and 10/10 evidence checks true; `report_current_primary_quality_loop_status.py` remained active_not_promotion_eligible with pending post_successor_inventory_broader_exhaustion_or_new_bucket_selection, candidate_admission_ready=false, promotion_eligible=false, reserved_blind_authorized=false, and successor_readiness=not_blocked_by_evidence_audit.
Not-tested: Training, candidate inference, validation gates, SCUT115, holdout40, visual review, reserved blind, promotion, or current-primary replacement.
