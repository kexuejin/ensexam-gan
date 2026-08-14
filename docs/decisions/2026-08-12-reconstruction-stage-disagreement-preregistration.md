# Reconstruction-Stage Disagreement Preregistration

## Decision

`PREREQUISITE_NEEDED`. The current-primary support program has now rejected
both previously authorized target-free sources before training:

- the 13-channel final-pipeline/context representation reached mean fold AUC
  `0.648757` but added only `0.004251` over second-stage RGB;
- the four-channel frozen `mb`/`ms` representation reached mean fold AUC
  `0.580727` and lost to second-stage RGB by `0.063779`.

Neither family may be repeated, expanded, or rescued. The next bounded
uncertainty is materially different: does disagreement between the frozen
primary generator's own coarse and refine reconstruction stages identify
residual handwriting support before final RGB composition, masks, page gates,
or targets are available?

Only one label-free train275 stage-disagreement materialization and one fixed
train-only separability diagnostic are authorized. No model, optimizer,
checkpoint, candidate inference, inner-val15, development gate, SCUT115,
holdout40, visual review, reserved blind, promotion, or current-primary
replacement is authorized.

## Causal Distinction

The frozen generator already emits `Ic4`, `Ic2`, `Ic1`, and `Ire` during one
forward pass. `Ic4` and `Ic2` are lower-resolution coarse reconstructions,
`Ic1` is the full-resolution coarse reconstruction, and `Ire` is the refine
reconstruction before `Mb` composition. Their disagreement measures whether
the frozen restoration hierarchy converges on a pixel; it does not reuse final
primary RGB, second-stage RGB as a full feature, `mb`, `ms`, page-broadcast
scalars, thresholded components, neighborhoods, OCR, or target-derived input.

The exact four feature channels, in model gray units, are:

1. signed arithmetic-mean RGB difference `Ire - Ic1`;
2. mean absolute RGB difference `Ire - Ic1`;
3. mean absolute RGB difference between bilinear-upsampled `Ic2` and `Ic1`;
4. mean absolute RGB difference between bilinear-upsampled `Ic4` and `Ic1`.

`Ic2` and `Ic4` are resized to the `Ic1` patch shape with bilinear
interpolation and `align_corners=False`. Features are derived from model
floating-point tensors per patch before overlap fusion, then overlap-averaged
with the frozen full-page patch schedule. The materializer stores one
compressed float32 NPZ per page with exactly these four keys. It reads no
labels or targets.

## Frozen Data And Materialization

Reuse the exact train275 source role, current-primary config/checkpoint,
`page_overlap=32`, `batch_size=8`, and `device=auto`. The materializer must pin
source, config, checkpoint, generator source, page-inference source, channel
definitions, page shape, NPZ content, and manifest hashes. It must fail closed
on target/label paths, output reuse, source-role drift, non-finite values,
channel or interpolation drift, unexpected files, or any candidate/quality
output.

The materialization stage may execute the frozen generator only on train-role
source images. It may not open train labels. Target decode becomes legal only
after all 275 materialized pages pass provenance and hash validation, and then
only inside the registered separability diagnostic.

## Frozen Diagnostic

Use the same train-only protocol as both closed support diagnostics:

- target-lighter is exactly `target_luma - second_stage_luma > 2` gray;
- all other pixels are preserve;
- assign pages with `int(sha256(utf8_basename), 16) % 5`;
- select the same deterministic SplitMix64-ranked
  `min(1024, positive_count, preserve_count)` pixels per class and page;
- divide the four gray-unit disagreement channels by `255`;
- fit float64 closed-form ridge with `lambda=1.0`, fitting-fold-only
  standardization, and an unpenalized intercept;
- compare against the unchanged three-channel second-stage-RGB ablation on the
  same coordinates and folds.

No raw RGB may be added to the full representation. No stage subset,
alternative resize, ratio, product, threshold, neighborhood, component,
nonlinear probe, fold, sampling, lambda, or acceptance search is authorized.
Per-channel quartile strata are descriptive only.

## Acceptance Contract

`PASS` requires every condition:

- 275 finite, pixel-aligned, hash-stable stage-disagreement NPZ files with
  exact frozen provenance and no target access;
- exact train roles, source identities, label content, fold membership, and
  deterministic sample coordinates;
- mean held-out fold AUC at least `0.65`;
- every held-out fold AUC at least `0.55`;
- macro median per-page AUC at least `0.60`;
- mean AUC at least `0.03` above the fixed second-stage-RGB ablation;
- positive mean score above preserve in at least four of five folds.

Any failed condition is `KILL`. Missing or drifting materialization evidence is
`PREREQUISITE_NEEDED`. `PASS` authorizes only a separate stage-disagreement
data/training preflight with portable string path metadata and a default
`weights_only=True` checkpoint load test. It does not authorize training or
candidate inference directly.

## Terminal Successors

- `PASS`: freeze a separate data/training/application preflight before any
  optimizer starts.
- `KILL`: close the exact reconstruction-stage-disagreement family without
  stage, feature, probe, or threshold rescue.
- `PREREQUISITE_NEEDED`: repair only missing provenance or implementation
  evidence without changing the registered representation or gates.

## Registered Surface

~~~text
plan:
  docs/reconstruction-stage-disagreement-prerequisite-v1.json
future materializer:
  scripts/analysis/materialize_reconstruction_stage_disagreement_train_only.py
future audit:
  scripts/analysis/audit_reconstruction_stage_disagreement.py
future test:
  tests/test_reconstruction_stage_disagreement_prerequisite.py
future materialization:
  outputs/reconstruction-stage-disagreement-materialization-20260812/
future audit output:
  outputs/reconstruction-stage-disagreement-prerequisite-20260812/audit.json
~~~

## Evidence Hashes

~~~text
docs/reconstruction-stage-disagreement-prerequisite-v1.json
sha256 = 90310fbc9c435bf714fc427bf1a1c1f1fe7440183896d01713230c09374166e0

networks/generator.py
sha256 = a2791f8765f54d7e6568f1fb3905deb392abc04c4762e6f9ebee1e70de9d576e

utils/page_inference.py
sha256 = 5c699ca4920c468e1473912a5ca97af2321537ad3c84e79bc108b1a971afd49a

docs/decisions/2026-08-11-dual-input-support-separation-diagnostic-kill.md
sha256 = 64c52693cc60bab60d33586e73a46a3789d0dcbd24a3b1f4492a88fc654f94d9

docs/decisions/2026-08-11-spatial-primary-mask-support-diagnostic-kill.md
sha256 = 5da07daa99b6efb8783f4a5c0ef1f37edd2a6b50f36c9e5d21e8a200de42820a
~~~

Intent: Test frozen reconstruction-stage convergence as a new target-free support source after final-pipeline context and masks failed.
Constraint: Only label-free train275 materialization and a fixed train-label separability diagnostic are authorized; every model and quality surface remains closed.
Rejected: Add final RGB, masks, page scalars, local neighborhoods, or thresholded components | those reuse closed evidence families or introduce an unfrozen selector.
Rejected: Train directly on reconstruction features | support must first beat the independent RGB ablation before another optimizer can open.
Confidence: medium
Scope-risk: moderate
Reversibility: clean
Directive: Do not alter stage channels, interpolation, folds, sampling, probe, lambda, or acceptance after materialization starts. A failed ablation margin closes the exact family.
Tested: Generator and full-page inference interfaces inspected; existing train275 roles, current-primary hashes, second-stage prediction hashes, and prior KILL records reused without data execution.
Not-tested: Stage-disagreement materialization, channel ranges or hashes, train-only separability, model training, checkpoint portability, candidate inference, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-11-spatial-primary-mask-support-diagnostic-kill.md
