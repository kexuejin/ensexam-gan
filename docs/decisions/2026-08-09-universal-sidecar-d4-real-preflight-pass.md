# Universal Sidecar D4 Real Preflight Pass

## Result

`PASS` for bounded D4 training readiness. The first authorized downstream
action is one fixed D4 80-step MPS training run, followed by leakage-safe SCUT
inner-val15 only. This preflight is not evidence of quality lift, later-gate
authorization, or current-primary replacement.

## Evidence

~~~text
preflight_summary = outputs/universal-sidecar-d4-preflight-20260809/preflight.json
preflight_summary_sha256 = 46ed97cb263a48189a5af10a480cce43637353efa6e3b610aea5ff3b2f94ed94
terminal = PASS
runnable = true
first_gate = scut_inner_val15
train_manifest_sample_count = 383
train_manifest_sha256 = 0385fb96aa7aee1812b95b90acd4198e2af39e96c895a7cd8cfb2681258470ca
inner_val15_overlap = 0
baseline_strict_load = true
sidecar_missing_keys = 17
sidecar_unexpected_keys = 0
structure_status = pass
trainable_tensors = 17
trainable_params = 7335
frozen_tensors = 226
frozen_batchnorm_layers = 40
gate_weight_shape = [1, 3]
~~~

The admitted evidence is the corrected isolated-worktree PASS JSON above. The
earlier wrong-root operational attempt is not a D4 model result and is not
counted as experiment evidence.

## Frozen Authorization

D4 remains an exact D2 clone except for the already-preregistered
`residual_parameterization = primary_edit_direction` change and its unique
`train.save_dir`. The real preflight confirms the frozen current-primary
hashes, unchanged 383-sample train manifest, zero inner-val15 overlap, strict
baseline checkpoint compatibility, passing sidecar structure audit, sidecar-only
trainability, frozen BatchNorm inheritance, disabled later gates, and a runnable
fail-closed configuration.

## First Gate And Kill Criteria

The first and only authorized next gate is SCUT inner-val15.

D4 is killed immediately if inner-val15 shows any of the following:

- aggregate residual gain below `0.0005`;
- any positive page residual delta;
- any positive page overerase delta;
- no measurable movement.

No later gate is authorized before that pass. SCUT115, holdout40,
reserved-blind, threshold rescue, learning-rate or step sweep, and
current-primary replacement remain prohibited until inner-val15 passes.

Intent: Admit the corrected real D4 preflight PASS as the only authorization for one fixed D4 training run and the first inner-val15 gate.
Constraint: Only the isolated-worktree PASS JSON is admissible evidence; the wrong-root operational attempt is excluded from model-result accounting.
Rejected: Treat the wrong-root attempt as a failed or partial D4 run | it did not establish model evidence in the bounded workspace and would corrupt the record.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Run exactly one fixed D4 80-step MPS training and then inner-val15 only; do not open later gates before an inner-val15 PASS.
Tested: Corrected real preflight JSON inspection; ledger-style evidence reconciliation against preregistration and prerequisite records.
Not-tested: D4 optimization trajectory, D4 inner-val15 metrics, SCUT115, holdout40, reserved blind, threshold rescue, or current-primary replacement.
Related: docs/decisions/2026-08-09-universal-sidecar-d4-primary-edit-direction-preregistration.md
Related: docs/current-primary-quality-loop-ledger.json
