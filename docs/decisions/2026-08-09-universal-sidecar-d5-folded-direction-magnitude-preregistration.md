# Universal Sidecar D5 Folded-Direction-Magnitude Preregistration

## Status

`PREREQUISITE_NEEDED`. This record freezes the only admissible D5 successor
after the D4 root-cause closure. It does not authorize implementation,
preflight output creation, training, inference, later gates, or any change to
`artifacts/current-primary`.

## Successor Identity

~~~text
id = universal-sidecar-d5-folded-direction-magnitude
residual_parameterization = primary_edit_direction_folded
~~~

## Exact Material Causal Change

Relative to the frozen D4 branch, the future runnable configuration may differ
only in:

~~~text
model.universal_residual_adapter_sidecar.residual_parameterization = primary_edit_direction_folded
train.save_dir = ./artifacts/trials/universal-sidecar-d5-d1-mixed-scut130-hw5k260-step80-primary-edit-direction-folded-20260809
~~~

For `primary_edit_direction_folded`, the residual path is preregistered as:

~~~text
folded_magnitude = torch.where(mixed_residual >= 0, mixed_residual, -mixed_residual)
bounded_magnitude = residual_bound * tanh(folded_magnitude)
scaled_residual = nonnegative_global_scale * bounded_magnitude * normalized_primary_edit_direction
candidate = clamp(internal_baseline + scaled_residual)
~~~

D4 behavior under `primary_edit_direction` remains untouched.

Everything else stays fixed to frozen D4 controls: the same current-primary
baseline, data roles, loss weights, optimizer, learning rate, seed, sidecar
architecture, 80-step budget, first gate (`scut_inner_val15`), matched-copy
protocol, later-gate closure, and product-default policy.

## Fail-Closed Synthetic Prerequisite

D5 remains blocked until a synthetic prerequisite proves all of the following:

- strict baseline/current-primary compatibility;
- exact zero output at initialization;
- all base tensors frozen;
- zero-branch final projection gradients are nonzero;
- forced positive and forced negative raw magnitudes both yield nonnegative
  primary-direction-aligned residual with zero opposed channels;
- a two-step synthetic update from zero keeps magnitude gradients alive after
  either raw sign, creates nonzero folded support, gives the global scale a
  nonzero gradient, and moves it away from `0.001` by the second step;
- the residual bound is still honored;
- zero-primary-edit remains a no-op;
- no public interface regression.

This turn does not implement that prerequisite. The next admissible action is
to implement the synthetic tests/tooling and run that fail-closed preflight
only.

## Future Real Preflight Boundary

Only after the synthetic prerequisite passes may a real D5 preflight be
admitted. That future preflight must prove:

- exact D4 semantic diff except for the preregistered
  `residual_parameterization` and unique `train.save_dir`;
- clean train/inner-val split with zero overlap;
- unique absent save_dir;
- strict checkpoint audit against current-primary;
- exactly `17` sidecar-only trainable tensors;
- frozen BatchNorm statistics and frozen base generator parameters;
- later gates disabled.

Only a passing real preflight may authorize exactly one `80`-step MPS run and
inner-val15 only.

## First Real Acceptance Gate

If implementation and both prerequisites eventually pass, the first real
acceptance gate remains unchanged:

- aggregate residual gain at least `0.0005`;
- no aggregate residual regression;
- no aggregate overerase regression;
- no page-level residual regression;
- no page-level overerase regression;
- measurable movement.

Later gates remain closed until that pass.

## Rejected Alternatives

Record these as rejected non-default paths:

- increase scale, learning rate, or steps as scalar rescue;
- threshold rescue;
- softplus floor, because it breaks exact zero initialization;
- naive leaky signed magnitude, because it can violate nonnegative direction
  semantics or reintroduce a clamp dead zone;
- changing D4 in place.

Intent: Gate the only admissible post-D4 successor behind a folded-magnitude synthetic prerequisite instead of opening another unbounded training branch.
Constraint: Relative to frozen D4, only residual_parameterization and a unique save_dir may change in the future runnable config.
Constraint: This preregistration does not authorize implementation, training, inference, or later-gate evaluation.
Rejected: Direct real-run retry | the folded mechanism must first prove fail-closed synthetic behavior before any training is reconsidered.
Rejected: Softplus or leaky signed rescue | those alternatives either break exact zero initialization or weaken the nonnegative direction contract.
Rejected: Mutate D4 in place | D4 is frozen causal evidence and must remain unchanged.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Implement only the synthetic folded-magnitude prerequisite next; do not create D5 train outputs or run a real D5 step budget until the synthetic and exact-diff preflights both pass.
Tested: Preregistration package only.
Not-tested: D5 implementation, synthetic prerequisite execution, real preflight execution, 80-step training, inner-val15 metrics, SCUT115, holdout40, reserved blind, or promotion evidence.
Related: docs/decisions/2026-08-09-universal-sidecar-d4-subthreshold-noop-root-cause.md
Related: docs/current-primary-quality-loop-ledger.json
