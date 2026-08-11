# Spatial Primary Mask Support Preregistration

## Decision

`PREREQUISITE_NEEDED`. The previous 13-channel dual-pipeline support
diagnostic is KILLed: its full mean AUC was `0.648757` and its gain over the
second-stage-RGB ablation was only `0.004251`. This record does not reopen that
family or authorize a nonlinear rescue.

The next bounded uncertainty is materially different and pixel-aligned: do the
frozen primary model's `mb` and `ms` mask maps contain target-free support
evidence that separates residual handwriting from preserve pixels? The primary
pipeline already produces these maps and uses `mb` in its copy-mask path, but
the maps have not yet been materialized, hash-pinned, or evaluated under the
train-only leakage contract.

Only one train-only mask materialization and one fixed separability diagnostic
are authorized. No candidate model, checkpoint, quality gate, visual review,
reserved blind, promotion, or current-primary replacement is authorized.

## Frozen Mask Materialization

Run the existing primary full-page inference path with the exact current
primary config/checkpoint, train275 source manifest, `page_overlap=32`,
`batch_size=8`, `device=auto`, and the required label-free protocol marker.
Persist only the uint8 pixel-aligned `mb` and `ms` maps from the existing
overlap-fused inference output, one PNG per train page, plus a manifest and
content hashes. Do not read train labels during this stage and do not use the
postprocess output as a mask substitute.

The materializer must fail closed on source-role drift, duplicate names,
non-finite or shape-mismatched masks, config/checkpoint drift, output reuse,
target/label paths, or any quality/evaluation output. It must prove that the
mask maps came from `utils.page_inference.infer_full_page` and not a copied or
hand-built mask.

~~~text
current-primary checkpoint:
  e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae
current-primary config:
  8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e
train275 manifest:
  ba31900496161322f839f366fa40765d71182d99a59ddad2537786310aae432f
role plan:
  f2555ddec01981e44ad5ce965977ef2c88003bae3ca5966c60437c93f91a110a
second-stage prediction content:
  2ffa40fc0c9b2a7e721d560f6f12edfe2ccdc1c1988582fa7a8104665cdc088a
train label content:
  dfd459f552bd0828221c90258f33f4eacc54220494c7e02b21a179894853e99e
previous dual-input KILL:
  64c52693cc60bab60d33586e73a46a3789d0dcbd24a3b1f4492a88fc654f94d9
~~~

## Frozen Diagnostic Representation

The target-free mask representation has exactly four pixel-aligned channels,
all normalized to `[0, 1]`:

1. `mb`;
2. `ms`;
3. signed `mb - ms`;
4. product `mb * ms`.

No RGB, page-gate scalar, local context, threshold-selected component,
connected-component, or learned feature may be added in version 1. The
second-stage-RGB-only ablation from the previous diagnostic is retained as the
fixed cross-family reference; it is not retuned or re-selected.

## Train-Only Diagnostic Contract

Reuse the exact 275-page train role: 253 HW5K and 22 SCUT pages. All
inner-val15, development, SCUT115, holdout40, and reserved-blind roles remain
closed and mutually exclusive. After mask materialization, target decode is
permitted only for train-role labels and only to create diagnostics.

Use arithmetic-mean RGB luma. A target-lighter pixel is exactly
`target_luma - second_stage_luma > 2` gray; every other pixel is preserve.
Pages are assigned to five folds by `int(sha256(utf8_basename), 16) % 5` and
never split. Within each page and class, use the already frozen basename-seed
SplitMix64 coordinate ranking and retain
`min(1024, target_lighter_count, preserve_count)` pixels per class. Do not
rank patches by stroke density or create a new target-derived patch index.

Fit one class-balanced float64 closed-form ridge probe with `lambda=1.0`,
fitting-fold-only standardization, and an unpenalized intercept. Compare the
fixed four-channel mask representation against the fixed second-stage-RGB
ablation on the same coordinates and folds. No iterative optimizer,
hyperparameter search, nonlinear probe, product threshold, candidate
checkpoint, or quality selector is allowed.

Persist per-fold and per-page AUC, positive/preserve score means, mask
availability and content hashes, fold membership, sample counts, and four
fixed mask-channel strata. Strata are descriptive and cannot become product
thresholds.

## Acceptance Contract

The prerequisite is `PASS` only when all conditions hold:

- both masks are available, finite, pixel-aligned, and hash-stable for all 275
  pages;
- exact roles, source identities, fold assignment, coordinate sampling, and
  label content hashes pass;
- full mask mean held-out fold AUC is at least `0.65`;
- every full mask held-out fold AUC is at least `0.55`;
- macro median per-page full mask AUC is at least `0.60`;
- full mask mean AUC exceeds the fixed second-stage-RGB ablation by at least
  `0.03`;
- positive mean score exceeds preserve mean in at least four of five folds.

Any missing mask, mask provenance drift, non-finite value, leakage, failed
condition, output reuse, or unregistered artifact is `KILL` or
`PREREQUISITE_NEEDED`. There is no threshold, sampling, fold, lambda, channel,
or acceptance rescue.

A `PASS` authorizes only a later data/training preflight for a new mask-aware
support family. That preflight must require portable checkpoint metadata with
all `Path` values serialized as strings and an independent default
`weights_only=True` load test before candidate inference.

## Registered Surface

~~~text
plan:
  docs/spatial-primary-mask-support-prerequisite-v1.json
future materializer:
  scripts/analysis/materialize_primary_masks_train_only.py
future audit:
  scripts/analysis/audit_spatial_primary_mask_support.py
future test:
  tests/test_spatial_primary_mask_support_prerequisite.py
future materialization:
  outputs/spatial-primary-mask-support-materialization-20260811/
future audit output:
  outputs/spatial-primary-mask-support-prerequisite-20260811/audit.json
~~~

Intent: Test pixel-aligned frozen primary mask evidence after coarse page-context support failed.
Constraint: Only train275 source inference and train-label diagnostics are authorized; all quality surfaces stay closed.
Rejected: Rescue the 13-channel dual-input family | its independent ablation margin failed before training.
Rejected: Tune mask thresholds or components | mask support must be proven before any product selector is chosen.
Confidence: medium
Scope-risk: moderate
Reversibility: clean
Directive: Do not train a mask-aware candidate unless this exact materialization and separability contract passes and portable checkpoint metadata is separately preflighted.
Tested: Existing primary inference and mask provenance paths reviewed; no new mask materialization or target decode executed by this record.
Not-tested: Mask availability, content hashes, train-only separability, model training, checkpoint portability, candidate inference, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-11-dual-input-support-separation-diagnostic-kill.md
