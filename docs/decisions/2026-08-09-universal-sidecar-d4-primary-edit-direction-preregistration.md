# Universal Sidecar D4 Primary-Edit-Direction Preregistration

## Status

`PREREQUISITE_NEEDED`. This record freezes the only admissible D4 causal
change before any real preflight, training, inference, or later gate. It does
not replace `artifacts/current-primary`.

## Exact Causal Diff

D4 is an exact D2 clone except for these two normalized semantic changes:

~~~text
model.universal_residual_adapter_sidecar.residual_parameterization = primary_edit_direction
train.save_dir = ./artifacts/trials/universal-sidecar-d4-d1-mixed-scut130-hw5k260-step80-primary-edit-direction-20260809
~~~

Everything else remains frozen from D2: losses, mixed SCUT130/HW5K260 train
manifest, optimizer, learning-rate schedule, seed, strict reproducibility,
80-step budget, sidecar-only trainable scope, frozen BatchNorm statistics, and
the matched-copy inference protocol.

## Explicit Exclusions

No D3 cache family is admitted. The following fields must be absent:

~~~text
loss.lambda_cached_baseline_tail_nonregress
loss.cached_baseline_tail_residual_alpha
loss.cached_baseline_tail_overerase_alpha
loss.cached_baseline_tail_fraction
loss.cached_baseline_tail_residual_threshold_px
loss.cached_baseline_tail_edit_threshold_px
loss.cached_baseline_tail_event_temperature_px
data.cached_baseline_tail_dir
~~~

Validation, final-test, SCUT115, holdout40, reserved-blind, promotion, and any
later gate remain disabled until D4 first passes its preregistered preflight
and then the first kill gate.

## Preflight Boundary

The required fail-closed preflight output path is:

~~~text
outputs/universal-sidecar-d4-preflight-20260809/preflight.json
~~~

The validator must return deterministic JSON and fail closed unless:

- the flattened D4/D2 diff is exactly the two frozen keys above;
- `validate_universal_sidecar_config` passes;
- the D4 save_dir does not exist;
- current-primary config and checkpoint hashes match the ledger;
- the D2 train manifest is unchanged and has zero filename overlap with
  inner-val15;
- default/current-primary checkpoint compatibility and D4 sidecar structure are
  audited;
- only sidecar tensors are trainable, BatchNorm freeze is inherited, and the
  exact D2 optimizer/seed/80-step controls remain inherited.

## First Gate And Kill Criteria

The first authorized real gate is leakage-safe SCUT inner-val15 only.

D4 is killed immediately if inner-val15 fails any of these:

- aggregate residual gain of at least the ledger calibration floor `0.0005`;
- zero positive page residual delta;
- zero positive page overerase delta;
- no measurable movement kill avoidance.

No SCUT115, holdout40, reserved-blind, threshold rescue, parameter-family
sweep, or current-primary replacement is authorized before that pass.
Current-primary must remain unchanged throughout.

Intent: Isolate the direction-restricted sidecar parameterization as the only D4 causal change over D2.
Constraint: One normalized semantic diff plus a unique save_dir; every optimizer, data, evaluation, and training control stays fixed.
Rejected: Carry forward any D3 cache or baseline-tail field | that would mix experiment families and break exact-D2 attribution.
Rejected: Allow later gates before inner-val15 pass | later splits are unauthorized until the first zero-regression gate succeeds.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Run only the real D4 fail-closed preflight next; do not create the preflight output directory, train save_dir, or any later-gate outputs before that pass.
Tested: Preregistration package only.
Not-tested: Real D4 preflight output, D4 training, D4 inference, inner-val15 metrics, SCUT115, holdout40, reserved blind, or promotion evidence.
Related: docs/current-primary-quality-loop-ledger.json
Related: docs/decisions/2026-08-08-universal-sidecar-primary-edit-direction-prerequisite-pass.md
