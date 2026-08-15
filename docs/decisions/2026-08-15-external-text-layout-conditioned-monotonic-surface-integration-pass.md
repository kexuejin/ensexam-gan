# External Text Layout Conditioned Monotonic Surface Integration Pass

## Decision

`PASS`. The exact five-channel implementation surface is now available through
dedicated trainer and candidate runner entry points. The existing three-channel
monotonic trainer and candidate runner remain unchanged; the conditioned path
uses recovered second-stage RGB plus frozen external text occupancy and
confidence, and still emits RGB-only preserve-or-brighten candidates.

## Evidence

- Trainer: `scripts/train/train_external_text_layout_conditioned_monotonic.py`
- Trainer SHA256: `f44666ee64b61c930f45448bb2a507ce203980397a6adb0b4e5b406b2c018d0c`
- Candidate runner: `scripts/infer/run_external_text_layout_conditioned_monotonic_candidate.py`
- Candidate runner SHA256: `5d4b81c9332430cd798e564d2eeb00dbc8519252eb6b2d7c2ff46a998068c97a`
- Test: `tests/test_external_text_layout_conditioned_monotonic_surface.py`
- Test SHA256: `46b87a43eeecd573086600f9dc03f02070c3df893ff6941feab858546396b8f5`

## Results

| Check | Result |
| --- | --- |
| Trainer input | five channels: RGB, occupancy, confidence |
| Trainer output target | RGB only |
| Candidate checkpoint requirement | `input_channels == 5` |
| Candidate layout requirement | NPZ occupancy/confidence shape matches baseline page |
| Legacy target/label options | absent |
| Resume/model-type alternatives | absent |
| Synthetic identity candidate | exact no-op |
| No-darkening gate | preserved through shared monotonic application guard |

## Boundary

This integration does not run real training, write a checkpoint, run candidate
inference on repository samples, open inner-val15, SCUT115, holdout40, visual
review, reserved blind, promotion, or replace current-primary. The next step is
to freeze or materialize the exact conditioned train patch index before any real
training command is eligible.

Intent: Add the frozen five-channel trainer/application surface without disturbing the existing three-channel monotonic lineage.
Constraint: Prior monotonic trainer and runner are hash-bound by earlier records, so the conditioned route must use dedicated entry points.
Rejected: Patch the existing three-channel trainer/runner in place | it would invalidate older preflight evidence and widen review scope.
Rejected: Run candidate inference immediately | implementation integration is not a checkpoint or quality-gate authorization.
Confidence: high
Scope-risk: narrow
Directive: Use only the dedicated conditioned entry points for this family; do not retrofit layout channels into the legacy monotonic scripts.
Tested: py313 conditioned surface tests 7/7; py310 conditioned surface tests 7/7; py313 conditioned preflight validator PASS; py313 conditioned preflight tests 7/7; py313 py_compile; py310 py_compile; git diff --check.
Not-tested: real conditioned training, checkpoint audit, candidate inference, inner-val15, SCUT115, holdout40, visual review, reserved blind, promotion.
