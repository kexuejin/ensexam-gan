# Monotonic Residual Erase V2 Checkpoint KILL

## Decision

`KILL`. The exact registered monotonic v2 training run completed on the
audited train275 top-256 target-lighter patches, but its checkpoint cannot
produce one meaningful candidate edit on those same patches. The structural
audit found no negative deltas, so the monotonic bound held, yet all 256 patches
remain below the frozen 12-gray application gate. Candidate inference,
inner-val15, and every later quality surface therefore remain closed.

This result rejects the v2 family, including its `1e-4` learning rate, 80-step
schedule, current patch index, losses, and 12-gray application protocol. The
synthetic 8x8 reachability result did not transfer to real train patches:
the checkpoint's maximum real delta is only `2.667739` gray and its support
does not separate target-lighter from preserve pixels.

## Checkpoint Evidence

~~~text
checkpoint / final step              monotonic_residual_erase_probe.pt / 80
parameter count / bound              384,578 / 0.08 (20.4 gray)
audited train patches                256
positive / preserve pixels           10,932,547 / 5,844,669
positive / preserve alpha mean       0.479730 / 0.462166
positive / preserve support >= 0.5   0.024918 / 0.001735
positive / preserve delta mean       1.482286 / 2.030916 gray
maximum delta                        2.667739 gray
positive / preserve gate ratio       0.0 / 0.0
patches with reachable gate          0 / 256
negative-delta pixels                0
candidate inference / quality gate   false / false
~~~

The checkpoint contains `Path` objects in its training arguments, so PyTorch
2.6's default weights-only loader cannot reload it without an explicit trusted
`weights_only=False` context. This compatibility flaw is documented but is not
the KILL criterion: even a trusted local structural load proves the checkpoint
cannot reach its registered application gate.

## Next Boundary

Do not repeat this v2 training run or rescue it through a learning-rate,
step-count, loss-weight, patch-selection, or application-threshold sweep. The
next iteration must preregister a materially different support-separation
mechanism and first prove it against real train-only patch diagnostics before
training. It must also serialize portable checkpoint metadata before any future
candidate inference is admitted.

## Evidence Hashes

~~~text
artifacts/trials/monotonic-residual-erase-v2/monotonic_residual_erase_probe.pt
sha256 = b12f358bc8457e7a90f7f7220e44867c531a810ad7f17fea2f52c71c18358c0e

artifacts/trials/monotonic-residual-erase-v2/monotonic_loss_history.csv
sha256 = 93392e60a82a008305aa3e6af13ebb79587f81ef5ec8d6070d2025dd9cf1bc13

outputs/monotonic-residual-erase-v2-checkpoint-audit-20260810/audit.json
sha256 = 73782017723a75279f0d0fe1b69481835ac41570f60c655025c15a01edf6184a

scripts/analysis/audit_monotonic_residual_erase_v2_checkpoint.py
sha256 = e9d9dfbe81f30a07e0e0b862ae448f10a493360cac6a158c3b67547a094a28a3

tests/test_audit_monotonic_residual_erase_v2_checkpoint.py
sha256 = f6184e1ea64ea83f84c3497013344338d581a2195f413c78e059b2746e44f0be
~~~

Intent: Block a structurally subthreshold checkpoint before it can contaminate quality evaluation.
Constraint: The 12-gray candidate gate was frozen before real training and the model may brighten by at most 20.4 gray.
Rejected: Open inner-val15 to measure a zero-edit candidate | all audited train patches already fail the application gate.
Rejected: Lower the gate or continue v2 optimization | that would be a post-result threshold or parameter rescue after a registered KILL.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not repeat monotonic v2. Require real train-only support-separation evidence and portable checkpoint serialization in any successor.
Tested: Exact 80-step MPS training; train275 256-patch structural audit; bound/no-darkening checks; all candidate and quality outputs absent.
Not-tested: Candidate page inference, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-10-monotonic-residual-erase-candidate-application-preflight-pass.md
