# Universal Sidecar D5 Real Preflight PASS

## Decision

`PASS`. The exact preregistered D5 folded-direction configuration is runnable
for one bounded `80`-step MPS training run. This authorizes only that run,
checkpoint audit, and leakage-safe inner-val15 evaluation. It does not
authorize SCUT115, holdout40, reserved blind, parameter sweeps, or replacement
of `artifacts/current-primary`.

## Exact Causal Delta

Relative to the frozen D4 configuration, the D5 configuration differs only in:

~~~text
model.universal_residual_adapter_sidecar.residual_parameterization = primary_edit_direction_folded
train.save_dir = ./artifacts/trials/universal-sidecar-d5-d1-mixed-scut130-hw5k260-step80-primary-edit-direction-folded-20260809
~~~

No optimizer, learning-rate, loss, dataset, seed, augmentation, step-budget,
gate, or baseline change was admitted.

## Preflight Evidence

The fail-closed real preflight proved:

- the synthetic prerequisite PASS record and all four evidence hashes are
  intact, including audit SHA-256
  `afe1eb4a500fd144f346ff6752285c574f49118318fab85e25668e2336371348`;
- the current-primary generator and discriminator checkpoint strict-load
  against the ledger-frozen config;
- D5 initialization has exactly `17` sidecar-only missing keys and zero
  unexpected keys;
- exactly `17` sidecar tensors (`7,335` parameters) are trainable while `226`
  base parameter tensors are frozen;
- all `40` generator BatchNorm layers are frozen in evaluation mode;
- the train manifest contains `383` unique samples, SHA-256
  `0385fb96aa7aee1812b95b90acd4198e2af39e96c895a7cd8cfb2681258470ca`,
  with zero overlap against inner-val15;
- validation, final test, standalone test, and all later gates remain disabled;
- the unique D5 save directory did not exist before admission.

## Registered Evidence

~~~text
configs/local/config.local-universal-sidecar-d5-d1-mixed-scut130-hw5k260-step80-primary-edit-direction-folded-mps.yaml
sha256 = 602c91ccfa48bd71a6dee78c280a754f4f2342034ced7f554579537b2fc77f65

scripts/analysis/validate_universal_sidecar_d5_preflight.py
sha256 = bc26a358b5335766d4225656f7801af2786430a6613307397449fc4f2c3cf1fe

tests/test_universal_sidecar_d5_real_preflight.py
sha256 = e7048366df3316f1f6093bb22bda2cf7e9f67a3cee5d1b701f4d90344fbc292b

outputs/universal-sidecar-d5-preflight-20260809/preflight.json
sha256 = 7d8518e57b290a3b9a2de09d4953e83958894d65c1165113778b73a635c330c5
~~~

Focused verification passed with `11` real-preflight tests, including
float32-exact scale-movement evidence and post-PASS idempotent revalidation.

## Next Boundary

Run the exact registered config once for at most `80` steps on MPS. After the
run, audit the produced checkpoint before generating matched-copy predictions
for inner-val15 only. The first quality gate remains:

- aggregate residual gain at least `0.0005`;
- no aggregate residual regression;
- no aggregate overerase regression;
- no page-level residual regression;
- no page-level overerase regression;
- measurable movement.

Any gate failure is a `KILL`; do not rescue it with learning-rate, step,
threshold, or selector sweeps.

Intent: Permit one bounded D5 run only after exact-delta, split, checkpoint, and training-scope authority is proven fail-closed.
Constraint: The admitted run is exactly 80 steps with the registered config and current-primary initialization.
Constraint: SCUT115, holdout40, reserved blind, sweeps, and current-primary replacement remain prohibited.
Rejected: Treat preflight success as quality evidence | preflight proves run integrity, not residual or overerase lift.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Execute the registered D5 run once, then audit its checkpoint and evaluate inner-val15 only.
Tested: Eleven fail-closed D5 preflight tests and post-PASS real preflight revalidation.
Not-tested: D5 training, checkpoint movement, inner-val15, SCUT115, holdout40, reserved blind, or promotion.
Related: docs/decisions/2026-08-09-universal-sidecar-d5-folded-direction-magnitude-synthetic-prerequisite-pass.md
Related: docs/decisions/2026-08-09-universal-sidecar-d5-folded-direction-magnitude-preregistration.md
