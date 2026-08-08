# Universal Sidecar D3 Baseline-Tail Preregistration

## Status

`PENDING`. This record freezes the only admissible D3 causal change before
training. It is not a promotion claim and does not replace
`artifacts/current-primary`.

## Failure Bucket

`sidecar_measurable_movement_source_residual_regression`

D2 step80 produced measurable movement but regressed SCUT inner-val15 page
`301.jpg`. D2D removed that regression by halving the learning rate but became
a safe no-lift result. Nearby learning-rate and step sweeps are exhausted.

## Frozen Hypothesis

Start from D2 step80 and add only a train-only frozen current-primary
baseline-tail non-regression constraint. The two cached support channels mark
changed pixels that current-primary solved and unchanged pixels it did not
edit. Penalizing candidate threshold events on those supports should preserve
D2's measurable movement while preventing its source residual regression.

The architecture, mixed SCUT130/HW5K260 train manifest, seed, 80-step budget,
optimizer, learning-rate schedule, augmentation, sidecar-only trainable scope,
frozen BatchNorm statistics, and matched-copy inference protocol remain fixed.

~~~text
lambda_cached_baseline_tail_nonregress = 0.20
cached_baseline_tail_residual_alpha = 0.25
cached_baseline_tail_overerase_alpha = 1.0
cached_baseline_tail_fraction = 0.10
cached_baseline_tail_residual_threshold_px = 12.0
cached_baseline_tail_edit_threshold_px = 12.0
cached_baseline_tail_event_temperature_px = 0.25
~~~

Cache:

~~~text
artifacts/caches/baseline-tail-universal-sidecar-d3-mixed-scut130-hw5k260-20260807
sample_count = 383
manifest_sha256 = 92c78488cbc59e5b380fa0496f395dcfd69624b8aff58186e1559bcc66bfa21b
rows_csv_sha256 = 592f6383164af92ec10008881a8b160cee6828132831ac66c4d3316d2742545a
inner_val15_name_overlap = 0
~~~

## Gate And Falsification

The first and only currently authorized evaluation gate is leakage-safe SCUT
inner-val15. D3 is killed immediately by any positive page-level residual or
overerase delta, an aggregate overerase increase, a worse p95/max tail, a
material printed-text or paper-tone regression, or no measurable movement.
Measurable aggregate residual lift must meet the calibrated `0.0005` floor.

No SCUT115, holdout40, reserved-blind, threshold rescue, development split,
learning-rate/step sweep, or product-default replacement is authorized by this
record. A passing preflight authorizes bounded training only; a passing
inner-val15 gate is required before any development scoring.

Intent: Test whether train-only baseline-safe support preserves measurable universal-sidecar movement without source regression.
Constraint: One causal change over D2; every architecture, optimizer, manifest, seed, schedule, step, and inference control stays fixed.
Rejected: Use D2D half learning rate as the base | it already passed as safe no-lift and would confound the registered causal change.
Rejected: Add relative-teacher, virtual-tail, selector, threshold, or routing logic | those are separate experiment families and would violate one-change attribution.
Confidence: medium
Scope-risk: moderate
Reversibility: clean
Directive: Do not run later splits or replace current-primary unless D3 first passes every zero-page-regression inner-val15 requirement.
Tested: Preregistration and exact-cache prerequisite only.
Not-tested: D3 training, inner-val15 predictions, development gates, SCUT115, holdout40, visual review, or reserved-blind verification.
Related: docs/current-primary-quality-loop-ledger.json
Related: docs/decisions/2026-08-08-universal-sidecar-baseline-tail-cache-prerequisite.md
