# Universal Sidecar D5 Folded-Direction-Magnitude Synthetic Prerequisite PASS

## Decision

`PASS`. The preregistered `primary_edit_direction_folded` mechanism satisfies
the fail-closed synthetic prerequisite. This result authorizes only the next
real D5 preflight. It does not authorize training, prediction generation,
inner-val15, any later quality gate, or replacement of `artifacts/current-primary`.

## Frozen Compatibility

The audit verified the ledger-registered current-primary artifacts before
loading them:

~~~text
config sha256     = 8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e
checkpoint sha256 = e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae
~~~

The current-primary generator state loaded strictly into its frozen model
shape. Loading the same state into the folded candidate produced exactly `17`
sidecar-only missing keys and no unexpected or base-model missing keys. The
candidate exposed `226/226` frozen base parameter tensors and exactly `17`
trainable sidecar tensors. Its zero-initialized output matched current-primary
exactly and did not trigger fallback.

## Synthetic Mechanism Evidence

Both preregistered two-step probes passed from exact zero initialization:

| Raw sign after step one | Folded support | Min projection gradient, step one | Scale gradient, step two | Scale after step two |
| --- | ---: | ---: | ---: | ---: |
| positive | 16 | 2.257253072457388e-06 | 1.794235870988814e-08 | 0.0010000045876950026 |
| negative | 16 | 2.2572528450837126e-06 | 1.794235515717446e-08 | 0.0009999955072999 |

The first-step scale gradient was exactly zero in both probes, as required
while the magnitude branch was still zero. After the first projection update,
both raw signs had nonzero support; at step two, projection and scale gradients
were nonzero and the scale moved away from its captured float32 initialization
`0.0010000000474974513`. The audit compares the final parameter against that
captured tensor value, not against a Python decimal approximation.

Forced raw magnitudes `+2` and `-2` produced equal nonnegative
primary-direction residuals. The maximum absolute residual was
`0.019280552864074707` under a `0.02` bound, with zero opposed channels.
Zero-primary-edit remained an exact no-op, and `Generator.forward` retained its
existing public parameters. A focused regression test also proves the frozen
D4 `primary_edit_direction` mode still truncates forced negative raw values to
zero rather than folding them.

## Evidence

~~~text
scripts/analysis/audit_primary_edit_direction_folded_sidecar.py
sha256 = e72511b8487c2bc8c0c04c95d5dbd6ef85c2d3174323ebef563df481ff95a666

tests/test_universal_sidecar_d5_folded_direction.py
sha256 = 2ee2ae0a9bb310cef4232095f0113b38f219aad438833e37d68ba6bee2db9ca1

outputs/primary-edit-direction-folded-sidecar-preflight-20260809/audit-final.json
sha256 = afe1eb4a500fd144f346ff6752285c574f49118318fab85e25668e2336371348
~~~

Focused verification passed with `22` tests and `6` subtests across the D5
contract and the existing universal sidecar suite.

## Next Boundary

The next admissible action is to implement and execute a fail-closed real D5
preflight proving the exact D4 semantic diff, clean train/inner-val split,
unique absent save directory, current-primary checkpoint compatibility,
exactly `17` sidecar-only trainable tensors, frozen BatchNorm/base parameters,
and disabled later gates. No real D5 training is authorized until that
preflight passes.

Intent: Admit real D5 preflight only after the folded mechanism proves both-sign gradient liveness without weakening current-primary or D4 guarantees.
Constraint: This PASS is synthetic mechanism evidence, not model-quality or generalization evidence.
Constraint: Training, predictions, inner-val15, later gates, and current-primary replacement remain prohibited.
Rejected: Treat synthetic movement as quality lift | synthetic tensors cannot establish residual, overerase, paper-tone, or printed-text performance.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Run the exact D5 real preflight next; do not train or generate predictions until it passes.
Tested: D5 synthetic audit; strict current-primary state compatibility; focused D5 and universal-sidecar tests.
Not-tested: Real D5 preflight, 80-step training, inner-val15, SCUT115, holdout40, reserved blind, or promotion.
Related: docs/decisions/2026-08-09-universal-sidecar-d5-folded-direction-magnitude-preregistration.md
Related: docs/decisions/2026-08-09-universal-sidecar-d4-subthreshold-noop-root-cause.md
