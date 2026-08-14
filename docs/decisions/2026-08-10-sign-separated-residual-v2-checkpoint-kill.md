# Sign-Separated Residual v2 Checkpoint KILL

## Decision

`KILL`. The single authorized v2 training run completed structurally, but its
checkpoint failed the pre-quality-gate application audit. The route collapsed
completely to `darken` on all `33,554,432` audited train-patch pixels;
zero pixels were application-eligible for `brighten`. The candidate is not
allowed into inner-val15 and is not eligible for any later gate or promotion.

The full run is closed. Do not rescue it by changing thresholds, adding steps,
changing the learning rate, changing loss weights, or retraining the same
sign-separated family on the same materialization.

## Evidence

~~~text
training steps                  80
history rows                    80
parameter movement              total L1 903.729711
output-head movement            L1 48.156827
audited patches                 512
audited pixels                  33,554,432
application-eligible pixels    8,514,478 (25.3751%)
eligible brighten pixels        0
eligible darken pixels          8,514,478
route argmax identity           0
route argmax brighten           0
route argmax darken             33,554,432
candidate maximum delta         13.231361 gray
target-direction correct        7,942,478 pixels
target-direction opposed        202,702 pixels
target-identity applied         369,298 pixels
inner-val15 started             false
current-primary replaced       false
~~~

The structural audit returned `KILL` before any quality metric or visual gate
was opened. This is a representation/training failure, not evidence that the
current product baseline regressed. `artifacts/current-primary` remains the
default.

## Failure Bucket Closure

The attempted bucket
`single_head_residual_brighten_darken_entanglement` is now closed for this
sign-separated route/magnitude family. The synthetic contract and data-role
isolation passed, but real train-patch optimization selected only the darken
route and applied edits to identity pixels. The failure cannot be repaired by
repeating the same family or tuning its application gate.

The active quality loop remains open only to preregister the next materially
different named failure bucket. SCUT115, holdout40, reserved blind, visual
review, and promotion remain closed.

## Evidence Hashes

~~~text
outputs/archive/sign-separated-residual-repair-20260810/checkpoint-audit/audit.json
sha256 = 3889612efe6ded767e1ecbb1fb44cad563dfbd6f04b0a76107b098e7060b6b3d

artifacts/archive/sign-separated-residual-repair-20260810/training-output/sign_separated_probe.pt
sha256 = e9d75a525173a7ddf913f01765d7b5bdbc2bdb228deebfc742a24607292d05fc

artifacts/archive/sign-separated-residual-repair-20260810/training-output/sign_separated_loss_history.csv
sha256 = cefc72f340f7d8dbc8a8eb427c70d8816a41ebaaf1307df9e99303f067443538

scripts/analysis/audit_sign_separated_residual_checkpoint.py
sha256 = 245f3ea65b71b05b2c93cf7b1e56ce33997cfbaab44430dd06302a09986d6212

tests/test_sign_separated_residual_checkpoint_audit.py
sha256 = 513f494f4fbd1f590b40fab89374b80430c10e2c3ae3717a88ed12d820c6e36b
~~~

Intent: Close the sign-separated v2 attempt before any quality gate because its real training output collapsed to one signed route.
Constraint: Inner-val15 is a promotion-safety gate, not a debugging surface for a structurally invalid checkpoint.
Rejected: Lower the 12-gray application threshold | would expose the same one-route/identity-pixel failure and turn audit into threshold rescue.
Rejected: Train longer or change loss weights | would repeat the killed family after a decisive structural failure.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not reopen this sign-separated route/magnitude family without a materially different representation and a new preregistration.
Tested: One exact v2 MPS run, 80 finite history rows, checkpoint serialization, full 512-patch structural audit, bound check, route distribution, signed application counts, and pre-gate closure.
Not-tested: inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion; these are intentionally not admissible after KILL.
Related: docs/decisions/2026-08-10-sign-separated-residual-candidate-application-preflight-pass.md
Related: docs/current-primary-quality-loop-ledger.json
