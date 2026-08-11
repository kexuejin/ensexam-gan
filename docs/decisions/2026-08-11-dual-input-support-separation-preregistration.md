# Dual-Input Support-Separation Preregistration

## Decision

`PREREQUISITE_NEEDED`. Monotonic residual-erase v2 is closed after its only
real train-patch checkpoint produced zero pixels at the frozen 12-gray
application gate and assigned more mean brightening to preserve pixels than to
target-lighter pixels. No learning-rate, step-count, loss-weight,
patch-selection, or threshold rescue is authorized.

The next bounded uncertainty is whether a materially different, target-free
input representation can separate real target-lighter support from preserve
pixels before another candidate optimizer is allowed to run. This record
authorizes only the implementation and execution of one deterministic,
train-only feature diagnostic. It does not authorize a candidate model,
checkpoint, prediction, quality gate, visual review, reserved blind, promotion,
or replacement of `artifacts/current-primary`.

## Named Failure Bucket

`real_patch_monotonic_support_separation_collapse`

The killed v2 model consumed only the frozen second-stage RGB prediction. Its
synthetic gradient and reachability evidence did not prove that those pixels
contained enough target-free information to distinguish residual handwriting
from printed text and paper. The real checkpoint then collapsed below the
application gate on every audited patch.

The successor changes only the support representation at this boundary. It
does not reopen the killed model, trainer, patch ranking, losses, schedule, or
application protocol.

## Frozen Representation

The full diagnostic representation has exactly 13 target-free channels:

1. frozen current-primary RGB, normalized to `[0, 1]` (3 channels);
2. frozen current-second-stage RGB, normalized to `[0, 1]` (3 channels);
3. signed `second_stage_rgb - primary_rgb`, in `[-1, 1]` (3 channels);
4. the following per-page frozen-pipeline values broadcast over the page
   (4 channels):
   - `copy_mask_cov8`;
   - `primary_edit_px`;
   - `primary_p95_edit_delta`;
   - `second_stage_gate_ratio`.

The four page values are available without a target at candidate time. Their
diagnostic standardization statistics must come only from each fitting fold.
Any local context statistic considered by a later model must be a deterministic
derivative of these 13 channels; it may not introduce target, label, mask, or
split identity as an input. Version 1 of the diagnostic itself uses only the 13
registered channels, with no learned or hand-tuned extra feature.

## Frozen Roles And Evidence

The diagnostic reuses the exact 275-page train role: 253 HW5K pages and 22 SCUT
pages. All existing inner-val15, development, SCUT115, holdout40, and reserved
blind roles remain closed and mutually exclusive. Target decode is permitted
only for the train role and only to create diagnostic labels.

~~~text
current-primary checkpoint:
  e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae
current-primary config:
  8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e
current-second-stage checkpoint:
  36dd96a7efb8145a010b37a2e5351b6e1efda8fa329ee33a81e48511a8e400b7
train275 manifest:
  ba31900496161322f839f366fa40765d71182d99a59ddad2537786310aae432f
role plan:
  f2555ddec01981e44ad5ce965977ef2c88003bae3ca5966c60437c93f91a110a
train label content:
  dfd459f552bd0828221c90258f33f4eacc54220494c7e02b21a179894853e99e
primary metrics:
  efd58814583089e888482a7e1604efc1d19ee5f514085cbef0e0c6cabf479846
primary prediction content:
  6400c9413af963e3de280e348bd635cd962e5387c2e975e930036d320214274a
second-stage metrics:
  b800fdf385075bac46cc50db08a726dc2b9a6201b11a1229a164738b595a708d
second-stage prediction content:
  2ffa40fc0c9b2a7e721d560f6f12edfe2ccdc1c1988582fa7a8104665cdc088a
materialization audit:
  15ee947ed8abb877a2f0d3ea3ffc9392b40b6d800cb1d7372d801ba4b2366882
~~~

The diagnostic must fail closed if these identities, role counts, prediction
content hashes, pipeline settings, or metric semantics drift. Missing
`primary_edit_px` or `primary_p95_edit_delta` is not permission to drop those
channels: the frozen primary pipeline must recompute them without target access
using the existing `base_edit_threshold=12` definition.

## Train-Only Diagnostic Contract

For each train page, luma is the arithmetic mean of the three RGB channels.
Target-lighter pixels are exactly
`target_luma - second_stage_luma > 2` gray. Every other pixel is preserve.
Targets never enter the feature matrix. `primary_edit_px` and
`primary_p95_edit_delta` use the per-pixel arithmetic mean absolute RGB
difference between the original source and frozen primary prediction; edit
pixels are those at or above 12 gray.

Pages are assigned to five folds by interpreting the SHA-256 digest of the
UTF-8 basename as an integer and taking modulo 5. All pixels from one page stay
in one fold. For each page, seed SplitMix64 with the first 64 bits of that
basename SHA-256, hash the row-major coordinate index, rank by
`(unsigned_hash, y, x)`, and retain the lowest hashes separately by class. Let
`n = min(1024, target_lighter_count, preserve_count)` and retain exactly `n`
pixels from each class. A page without both classes is a diagnostic failure,
not a reason to move pixels across folds or change the cap.

Two class-balanced diagnostic probes use the identical retained coordinates:

- **full:** all 13 registered channels;
- **ablation:** frozen second-stage RGB only.

For each held-out fold, standardize features using fitting-fold mean and
standard deviation only. Encode target-lighter as `+1` and preserve as `-1`,
append an unpenalized intercept, and solve one float64 closed-form ridge system
with `lambda=1.0`. No iterative optimizer, hyperparameter search, threshold
search, candidate checkpoint, page prediction, or product selector is allowed.
AUC uses rank statistics with average ranks for tied scores.

The audit must persist per-fold and per-page AUC, positive and preserve score
means, channel availability, fold membership, sample counts, and quartile
strata for each of the four page gate features. Strata are descriptive only;
they must not become application thresholds or patch-selection rules.

## Acceptance Contract

The prerequisite is `PASS` only when all conditions hold:

- all 13 features are finite and available for every train page;
- exact role, artifact, prediction, pipeline, fold, and coordinate hashes pass;
- full-representation mean held-out fold AUC is at least `0.65`;
- every full-representation held-out fold AUC is at least `0.55`;
- macro median per-page full-representation AUC is at least `0.60`;
- full-representation positive mean score exceeds preserve mean score in at
  least four of five held-out folds;
- full-representation mean fold AUC exceeds the second-stage-RGB-only ablation
  by at least `0.03`.

Any unavailable channel, role leakage, fitting-fold leakage, non-finite value,
empty class, hash drift, failed condition, unregistered output, or changed
sampling/fold rule is `KILL` or `PREREQUISITE_NEEDED`. Training remains closed.
There is no post-result threshold, channel, fold, sample-cap, ridge-lambda, or
acceptance rescue.

## Next Boundary

A diagnostic `PASS` authorizes only a later metadata/data/training preflight
for a new dual-input support family. That preflight must freeze the model,
losses, schedule, application path, and candidate gate before training. It must
also require portable Torch checkpoint metadata: every `Path`-like value must
be serialized as a string, and an independent default `weights_only=True` load
must pass before candidate inference can be authorized.

A diagnostic `KILL` closes this exact 13-channel ridge-evidence direction. It
does not authorize direct training, another probe family, or a feature sweep.

## Registered Implementation Surface

~~~text
plan:
  docs/dual-input-support-separation-prerequisite-v1.json
future audit:
  scripts/analysis/audit_dual_input_support_separation.py
future test:
  tests/test_dual_input_support_separation_prerequisite.py
future output:
  outputs/dual-input-support-separation-prerequisite-20260811/audit.json
~~~

Intent: Prove that target-free dual-pipeline evidence separates real residual support before another candidate optimizer is allowed to run.
Constraint: Only the exact train275 diagnostic is authorized; all candidate, evaluation, visual, blind, and promotion surfaces remain closed.
Rejected: Rescue monotonic v2 | its only real checkpoint has zero reachable application support and the family is closed.
Rejected: Tune product selector thresholds from gate features | selector-only search is analysis infrastructure and cannot establish pixel support causality.
Confidence: medium-high
Scope-risk: moderate
Reversibility: clean
Directive: Do not train a dual-input candidate unless this exact page-grouped diagnostic passes and a later portable-checkpoint training preflight is separately recorded.
Tested: Existing KILL evidence, frozen roles, pipeline hashes, and target-free feature definitions reviewed; no data or implementation executed by this record.
Not-tested: Feature availability on all train pages, ridge separability, model training, candidate inference, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-11-monotonic-residual-erase-v2-checkpoint-kill.md
