# External Text Layout Conditioned Monotonic Checkpoint KILL

## Decision

`KILL`. The exact registered 80-step CPU conditioned monotonic training run
completed on the audited 256-patch train index, but the resulting checkpoint is
structurally subthreshold before candidate inference. The audit loaded the
checkpoint with the default PyTorch loader, verified five-channel conditioned
metadata, replayed all 256 training patches on CPU, and found zero pixels or
patches reaching the frozen 12-gray application gate.

This closes the conditioned monotonic v1 training path. Do not rescue it by
changing the detector/layout transforms, patch index, learning rate, step count,
loss weights, or application threshold after seeing this result.

## Checkpoint Evidence

~~~text
checkpoint / final step              external_text_layout_conditioned_monotonic.pt / 80
parameter count / bound              385,154 / 0.08 (20.4 gray)
training device                      cpu
history rows                         80
loss first / final / delta           1.45080352 / 1.41070092 / -0.04010260
audited train patches                256
positive / preserve pixels           10,932,547 / 5,844,669
positive / preserve alpha mean       0.499263 / 0.474475
positive / preserve support >= 0.5   0.560795 / 0.073762
positive / preserve delta mean       1.006779 / 1.005637 gray
maximum / minimum delta              1.373539 / 0.186997 gray
positive / preserve gate ratio       0.0 / 0.0
patches with reachable gate          0 / 256
negative-delta pixels                0
candidate inference / quality gate   false / false
~~~

The trainer now serializes `Path` arguments as strings before writing the
checkpoint. This keeps the checkpoint compatible with PyTorch 2.6's default
weights-only loader path and avoids repeating the old non-portable metadata
failure mode.

## Evidence Hashes

~~~text
artifacts/trials/external-text-layout-conditioned-monotonic-v1/external_text_layout_conditioned_monotonic.pt
sha256 = c204d53405db11f4333bfa308c2cf9fc832aa8134fd089d29ff8531b28b94cad

artifacts/trials/external-text-layout-conditioned-monotonic-v1/conditioned_monotonic_loss_history.csv
sha256 = a40e49d0214d8f9b963388ce66b5d3f8af84601c2d91382960932ce1097c191f

outputs/external-text-layout-conditioned-monotonic-checkpoint-audit-20260815/audit.json
sha256 = 140ceb8c24711e7a527367761aef57c6d892c24509788f2b846121b4518b0341

scripts/analysis/audit_external_text_layout_conditioned_monotonic_checkpoint.py
sha256 = c599bf4d51857cad31756be7eb51422ab893ecc42ec62cd1224f126213714cd7

scripts/train/train_external_text_layout_conditioned_monotonic.py
sha256 = 703945f8331c4ac8604fe753db99719d9b61d3044ac1ea560e9069dd019063e8

tests/test_external_text_layout_conditioned_checkpoint_audit.py
sha256 = e772bdd27e75c9712fa2e79af1886c9d814cdc7a5f25cf535cce50f97c54748d

tests/test_external_text_layout_conditioned_monotonic_surface.py
sha256 = 57688cf741ce7744a444380089726ccf463c0c5707bddf9868acc58aaa4e2bbc
~~~

Intent: Block a trained conditioned monotonic checkpoint that cannot reach its own registered application gate.
Constraint: The 12-gray application gate, 80-step CPU schedule, 256-patch index, and no-threshold-rescue rule were frozen before training.
Rejected: Open inner-val15 to measure the candidate | the checkpoint has zero reachable train-patch gate coverage, so candidate inference would evaluate a structural no-op.
Rejected: Lower the gate or continue training | that would be post-result threshold or schedule rescue after a registered KILL.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Close this conditioned monotonic path; the next quality attempt needs a materially different, preregistered mechanism rather than tuning this one.
Tested: Registered 80-step CPU training; py313 checkpoint audit KILL; 256-patch CPU structural replay; default PyTorch checkpoint load; no-darkening and bound checks; candidate and quality outputs absent.
Not-tested: Candidate page inference, inner-val15, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-15-external-text-layout-conditioned-monotonic-patch-materialization-pass.md
