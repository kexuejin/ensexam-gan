# Spatial Continuous Reconstruction Mixture Gate A

## Decision

`PASS`. The spatial continuous reconstruction mixture implementation is
complete for the code and synthetic Gate A preflight stage. The preflight
verified current-primary artifact custody, disabled-default legacy behavior,
base freeze and BatchNorm immutability, optimizer ownership, parameter counts,
initialization equivalence, gate simplex/range invariants, edit-range
reachability, synthetic gradient liveness, metadata-free public inputs, fold
custody, prohibited-surface isolation, and absence of sealed Phase 0 outputs.

This record is an implementation-stage pass only. It does not authorize Phase 0
training, `inner_val15`, Dev40, SCUT115, holdout40, visual review, reserved
blind, promotion, or `artifacts/current-primary` replacement.

## Boundary

The old current-primary successor ledger remains closed by durable exhaustion.
This record belongs to the new user-authorized spatial mixture program outside
that exhausted ledger. `artifacts/current-primary` remains the product default
and must stay independently runnable.

The next allowed action is a separate sealed Phase 0 command-matrix record. Real
Phase 0 training remains disabled until that handoff exists.

## Evidence

```text
record = docs/spatial-continuous-reconstruction-mixture-gate-a-v1.json
preflight audit = outputs/spatial_mixture_gate_a_preflight_20260816/audit.json
preflight audit sha256 = b38e3b7a3d6f097144ffd3d1084cd19a7b2e9f2b395fd606910fe75338cac615
preflight terminal = PASS
MPS = built=true available=true alloc_ok=true
active reconstruction params = 312198
spatial gate params = 28723
mixture trainable params = 340921
fold union count = 383
sealed matrix present = false
```

Intent: Close the spatial mixture implementation/preflight stage before any training surface opens.
Constraint: The prior quality loop closed by exhaustion, so this must remain a separate new-program stage and must not imply product quality lift.
Rejected: Start Phase 0 training immediately | the implementation plan requires a separate implementation-complete record and sealed command matrix before real training.
Rejected: Treat fallback behavior as quality evidence | fallback preserves runtime safety but is not a PASS condition for model quality.
Confidence: high
Scope-risk: moderate
Directive: Do not run Phase 0, open quality splits, or replace `artifacts/current-primary` from this record alone. Seal the Phase 0 matrix in a separate record first.
Tested: `/Volumes/Tool/source/clean-doc/.venv-torch310-mps-stable/bin/python scripts/analysis/validate_spatial_mixture_preflight.py --output-dir outputs/spatial_mixture_gate_a_preflight_20260816 --require-mps --device cpu` passed with terminal=PASS; `source .env && $ENSEXAM_PYTHON -m pytest tests/test_spatial_reconstruction_mixture.py tests/test_spatial_mixture_parameter_budget.py tests/test_spatial_mixture_losses.py tests/test_materialize_spatial_mixture_phase0_folds.py tests/test_validate_spatial_mixture_preflight.py tests/test_train_spatial_mixture_probe.py tests/test_evaluate_spatial_mixture_phase0.py` passed 143/143.
Not-tested: Phase 0 training, candidate inference, `inner_val15`, Dev40, SCUT115, holdout40, visual review, reserved blind, promotion, or current-primary replacement.
