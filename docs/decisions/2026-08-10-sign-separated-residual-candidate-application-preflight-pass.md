# Sign-Separated Residual Candidate Application Preflight PASS

## Decision

`PASS`. A data-independent third-stage application protocol is now frozen for
the sign-separated candidate, and the original v1 learning rate is superseded
before real training because it is synthetically incapable of producing a
measurable candidate edit in the registered 80 steps.

This preflight did not decode a real image or target, train on a real patch,
generate a checkpoint, open inner-val15 or another quality gate, perform
visual review, access reserved blind data, promote a candidate, or change
`artifacts/current-primary`.

## Root Cause And Correction

The model output bound is `0.08`, or approximately `20.4` gray levels. The
legacy current-second-stage application threshold of `32` is therefore
unreachable. Separately, the v1 learning rate `0.00002` produced only the
following output after the full registered 80-step synthetic schedule:

~~~text
bright target max delta  0.013740 gray
dark target max delta    0.015366 gray
~~~

That is a deterministic scale mismatch, not a poor real-data result. No v1
training run is authorized or needed.

The v2 plan changes only the optimization scale needed to make the already
registered representation observable: learning rate `0.0001`, still 80 steps,
batch size 1, seed 42, bound 0.08, and the same four unit-weight loss terms.
The data, patch index, model, direction margin, and split roles are unchanged.

At the same 80-step synthetic schedule, v2 produced:

~~~text
bright target mean/max delta  +19.850992 / 20.399920 gray
dark target mean/max delta    -20.231514 / 20.399996 gray
opposed pixels                 0 / 0
edit probability mean          0.999098 / 0.999148
identity target                exact no-op
finite final loss              yes
~~~

## Frozen Application

The candidate is a third stage over the frozen current-primary plus
current-second-stage prediction. It does not replace those baselines.

~~~text
input                        frozen current pipeline prediction
tile / stride                256 / 160
edit probability threshold   0.5
minimum mean absolute delta   12 gray
device                       mps
labels available to inference no
legacy base-edit gate         absent
legacy second-delta gate      absent
~~~

`0.5` is the natural majority boundary between the explicit identity route
and the combined brighten/darken routes. `12` is fixed before real training
from the existing meaningful-change/evaluation threshold and is below the
model's `20.4` bound. It is not selected from candidate gate results and must
not be swept after training.

## Next Boundary

Exactly one v2 training run is now admissible at
`artifacts/trials/sign-separated-residual-repair-v2`. Before inner-val15 can
open, a checkpoint audit must verify exact arguments, step 80, finite loss
history, nonzero parameter movement, valid signed outputs on registered train
patches, unchanged baseline hashes, and absent quality-gate outputs.

## Evidence Hashes

~~~text
docs/sign-separated-residual-candidate-plan-v2.json
sha256 = 7f2c35a86efe05fb50c91008bba055c1ea5dd6d3578eee1256813721d707e205

scripts/infer/run_sign_separated_residual_candidate.py
sha256 = 2993e1dbfb739e9e883d0bac75c099cc9a79986c293e8a27215c8bae811f5550

scripts/analysis/validate_sign_separated_candidate_application_preflight.py
sha256 = 05c6d6e4c2ad4b8a7df718893bef5922526c34479a679583b7c2e0fcd896d7db

tests/test_sign_separated_candidate_application_preflight.py
sha256 = f7cda0f97c3103524065319d43e2c19adad8c52f1a497b2f05563430935ca2a7

outputs/sign-separated-residual-candidate-application-preflight-20260810/preflight.json
sha256 = e7d35ec3a3bb5090a2bb8335e2b8fb2086eb5d310cbb9e33f70951c890de933a
~~~

Intent: Ensure the new bounded representation can produce and apply measurable edits before spending the only registered real training run.
Constraint: The 0.08 model bound makes the legacy 32-gray application threshold analytically unreachable.
Rejected: Keep v1 learning rate 0.00002 | the full 80-step synthetic schedule remains below 0.016 gray in both directions.
Rejected: Tune gates after inner-val15 | the 0.5 route majority and 12-gray meaningful-delta rule are frozen before real training.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Run v2 exactly once, then require checkpoint audit before any quality gate; do not train v1 or sweep learning rate, steps, probability, or delta thresholds.
Tested: Eleven focused tests, deterministic 80-step CPU reachability for legacy and v2 learning rates in both directions, identity-target no-op, application gate semantics, frozen plan/artifact hashes, absent outputs, and MPS availability.
Not-tested: Real v2 training, checkpoint behavior on real patches, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-10-sign-separated-residual-train275-materialization-pass.md
Related: docs/current-primary-quality-loop-ledger.json
