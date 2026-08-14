# Universal Sidecar Primary-Edit-Direction Prerequisite Pass

## Result

`PASS` for the synthetic mechanism prerequisite. This result authorizes a
separate D4 preregistration and exact-diff preflight only. It does not authorize
training by itself, consume inner-val15, or change `artifacts/current-primary`.

## Causal Basis

D3's only new bad event occurred on `301.jpg` at pixel `(822, 551)`:

~~~text
source = [29, 26, 21]
current_primary = [243, 241, 236]
D3 = [243, 241, 235]
target = [253, 254, 248]
current_primary_edit = [214, 215, 215]
D3_delta = [0, 0, -1]
primary_edit_dot_D3_delta = -215
target_direction_dot_D3_delta = -12
baseline_target_mean_delta = 11.666666666666666
D3_target_mean_delta = 12.0
~~~

The free RGB sidecar moved one channel opposite both the same-call
current-primary edit and the target direction, creating the exact threshold
crossing that killed D2 and D3. The new parameterization removes that degree of
freedom: each adapter predicts one nonnegative magnitude map, and the applied
RGB residual is that magnitude times the normalized same-call
`current_primary_output - input_image` vector.

## Contract And Evidence

The default `free_rgb` mode remains unchanged and a historical D2 checkpoint
loads with `strict=True`. The explicit `primary_edit_direction` mode preserves
the single public `Generator.forward(Iin, ...)` interface and uses no domain,
path, page, label, or caller signal.

~~~text
audit = outputs/primary-edit-direction-sidecar-preflight-20260808/audit-final.json
audit_sha256 = a5e1f0c2f878052cc313bc4f3ce1cae0d05c3640af6824fefb78b5042a3e7937
audit_script_sha256 = 0718021ebd5c96929294c9942b432e49d3576b31504087a8188ec96f1a990693
terminal = PASS
exact_zero_init = true
gradient_live_final_bias_count = 3
opposed_channel_count = 0
residual_abs_max = 0.047058820724487305
residual_bound = 0.047058823529411764
zero_primary_edit_noop = true
sidecar_missing_keys = 17
trainable_tensors = 17
frozen_tensors = 226
~~~

Negative global scale is a no-op rather than a direction reversal. Missing or
shape-mismatched same-call input fails closed. Every enabled direction-mode
delta channel has a nonnegative product with the corresponding current-primary
edit channel, and absolute residual remains bounded by `12/255`.

Verification:

~~~text
py_compile = pass
focused_pytest = 31 passed, 4 subtests passed
D2_checkpoint_strict_load = all keys matched successfully
synthetic_audit = PASS
git_diff_check = pass
~~~

## Next Boundary

D4 may change only `model.universal_residual_adapter_sidecar.residual_parameterization`
from implicit `free_rgb` to `primary_edit_direction`, plus a unique output
directory. It must return to D2's losses, data manifest, seed, optimizer,
learning-rate schedule, 80-step budget, sidecar-only trainability, frozen BN,
and matched-copy protocol. D3's cache and baseline-tail loss must be absent.

Before training, a fail-closed validator must prove the exact D4/D2 semantic
diff, default checkpoint compatibility, structure audit, disabled later gates,
and a new/empty output directory. Inner-val15 remains the first kill gate.

Intent: Remove the free RGB degree of freedom that directly caused D3's only thresholded residual regression.
Constraint: Direction comes only from the same-call current-primary edit; no target, source identity, selector, or external route enters inference.
Rejected: Add another train-side scalar guard | D3 and historical baseline-relative families exhausted that representation.
Rejected: Force a target direction at inference | target pixels are unavailable and would violate the product interface.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Preserve free_rgb checkpoint compatibility and require an exact D4/D2 preflight before any training.
Tested: Synthetic equivalence, gradient, bound, direction, zero-edit, negative-scale, interface, trainable-scope, and historical checkpoint compatibility checks.
Not-tested: D4 config, D4 training, inner-val15 behavior, development gates, SCUT115, holdout40, visual review, or reserved blind.
Related: docs/decisions/2026-08-08-universal-sidecar-d3-baseline-tail-inner-val15-kill.md
Related: docs/current-primary-quality-loop-ledger.json
