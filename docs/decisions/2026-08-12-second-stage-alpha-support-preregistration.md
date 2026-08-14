# Second-Stage Alpha Support Preregistration

## Decision

`PREREQUISITE_NEEDED`. Four exact target-free support families are closed:
final-pipeline RGB/context, primary `mb`/`ms`, primary reconstruction-stage
disagreement, and raw source-plus-output RGB. Each failed to improve mean
held-out AUC over the fixed second-stage-RGB ablation by the registered `0.03`.

The next bounded uncertainty is materially different: does the current frozen
second-stage erasemap model's raw, prethreshold alpha head identify residual
handwriting support that final RGB hides after alpha thresholding and the
product gate? The alpha head was directly trained as a spatial edit mask and
is emitted before `cleanup_alpha_threshold=0.3`, base-edit gating, and
second-delta gating. It is not a threshold selector derived from diagnostic
labels.

Only one label-free train275 alpha materialization and one fixed train-label
separability diagnostic are authorized. No model training, checkpoint,
candidate inference, inner-val15, development gate, SCUT115, holdout40,
visual review, reserved blind, promotion, or current-primary replacement is
authorized.

## Causal Distinction

The frozen `EraseMapCleanupNet` maps each primary-prediction tile to three
outputs: blended prediction, raw sigmoid alpha, and clean candidate. Product
inference thresholds alpha at `0.3` before later pipeline gates. On a frozen
train tile inspected before registration, raw alpha was finite and nonconstant
(`0.001065` to `0.045282`) while threshold coverage was exactly zero. This
proves that the raw map can contain spatial evidence erased from the final RGB
surface by the fixed product threshold.

The version-1 full representation contains exactly:

1. frozen final second-stage red;
2. frozen final second-stage green;
3. frozen final second-stage blue;
4. frozen erasemap raw alpha before all thresholds.

The ablation remains the first three RGB channels. Do not add clean-candidate
RGB, alpha-threshold masks, primary/source RGB, primary masks, reconstruction
stages, page scalars, output differences, neighborhoods, components, ratios,
products, gradients, alternative model layers, or nonlinear probes.

## Frozen Materialization

Use exact train275 primary predictions as erasemap inputs. Validate the frozen
roles, prediction content, current second-stage checkpoint, and
`patch_cleanup_erasemap.py` source before inference. Load only the registered
`erasemap` checkpoint with `weights_only=True`-compatible metadata.

Use `tile_size=160`, `stride=160`, `batch_size=32`, and `device=auto`. For each
tile, persist only raw sigmoid alpha. Fuse any overlap by arithmetic mean before
thresholding; version 1 has no threshold. Write one compressed float32 NPZ per
page with exactly `raw_alpha`, plus page shape, min/max/mean, source prediction
hash, NPZ hash, checkpoint/source provenance, and one terminal manifest. The
materializer must not open labels or targets.

## Frozen Diagnostic

After all 275 alpha maps pass provenance, shape, range `[0,1]`, finite-value,
and hash validation, target decode is allowed only for train-role labels.
Target-lighter remains exactly
`target_luma - final_second_stage_luma > 2` gray; all other pixels are
preserve. Reuse the five basename-hash folds and deterministic SplitMix64
sampling with at most 1024 pixels per class per page.

Fit float64 closed-form ridge with `lambda=1.0`, fitting-fold-only
standardization, and an unpenalized intercept. Compare the exact four-channel
representation against second-stage RGB on identical coordinates and folds.
No threshold or hyperparameter is learned.

## Acceptance Contract

`PASS` requires every condition:

- exactly 275 finite, aligned, hash-stable raw-alpha maps with no target access;
- exact train roles, prediction identities, labels, folds, and sample coordinates;
- mean held-out fold AUC at least `0.65`;
- every held-out fold AUC at least `0.55`;
- macro median per-page AUC at least `0.60`;
- mean AUC at least `0.03` above the fixed second-stage-RGB ablation;
- positive mean score above preserve in at least four of five folds.

Any metric failure is `KILL`. Missing or drifting provenance is
`PREREQUISITE_NEEDED`. `PASS` authorizes only a separate alpha-conditioned
data/training/application preflight with portable checkpoint metadata; it does
not authorize training or candidate inference directly.

## Terminal Successors

- `PASS`: freeze a separate data/training/application preflight.
- `KILL`: close raw second-stage alpha without threshold, transform, layer,
  neighborhood, probe, or training rescue.
- `PREREQUISITE_NEEDED`: repair provenance or implementation only, without
  changing channels, inference, folds, sampling, or gates.

## Registered Surface

~~~text
plan:
  docs/second-stage-alpha-support-prerequisite-v1.json
future materializer:
  scripts/analysis/materialize_second_stage_alpha_train_only.py
future audit:
  scripts/analysis/audit_second_stage_alpha_support.py
future test:
  tests/test_second_stage_alpha_support_prerequisite.py
future materialization:
  outputs/second-stage-alpha-support-materialization-20260812/
future audit:
  outputs/second-stage-alpha-support-prerequisite-20260812/audit.json
~~~

Intent: Test frozen prethreshold second-stage edit confidence after visible RGB and primary internal evidence failed.
Constraint: Only label-free train275 alpha materialization and a fixed train-label audit are authorized; all model and quality surfaces remain closed.
Rejected: Use page-level second-stage gate ratio | it is a closed coarse context feature derived after thresholds.
Rejected: Use clean candidate or alpha-threshold masks | that adds output RGB or a post-result selector instead of one raw causal signal.
Rejected: Tune alpha thresholds, layers, transforms, neighborhoods, folds, sampling, lambda, or probe class | unfrozen feature rescue.
Confidence: medium
Scope-risk: moderate
Reversibility: clean
Directive: Do not alter raw-alpha inference or diagnostic gates after this commit. A failed ablation margin closes the exact family.
Tested: Checkpoint loads as erasemap; raw alpha shape, range, finite values, nonconstancy, and exact blend identity verified on one frozen train tile before registration.
Not-tested: Full train275 alpha materialization, separability, training, checkpoint portability for a successor, candidate inference, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-12-source-output-support-diagnostic-kill.md
