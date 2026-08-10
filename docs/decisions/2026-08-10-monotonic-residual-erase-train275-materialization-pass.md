# Monotonic Residual Erase Train275 Materialization PASS

## Decision

`PASS`. The exact effective train275 role is available through the frozen
current-primary and current-second-stage pipeline, and the monotonic builder
materialized the registered top-256 target-lighter patch index. An independent
audit rebuilt all 52,645 eligible candidates from train-only predictions and
targets and matched every selected row and summary value exactly.

The primary and second-stage predictions were not recomputed. Their complete
275-file name sets and aggregate content hashes exactly match the prior
sign-separated materialization PASS, whose checkpoint, config, and inference
script hashes are identical to the monotonic plan. Registered monotonic paths
use relative symlinks to those immutable local archive directories. This
avoids a duplicate 550-page inference run without weakening provenance.

No monotonic training, checkpoint generation, inner-val15 evaluation, later
quality gate, visual review, reserved-blind access, promotion, or default
artifact replacement occurred.

## Materialized Evidence

~~~text
effective train pages           275
domains                         253 HW5K + 22 SCUT
primary predictions             275
frozen pipeline predictions     275
eligible target-lighter patches 52,645
selected patches                256
selected pages                  24
positive support ratio          0.310654 .. 0.945358
preserve-negative ratio         0.054642 .. 0.689346
~~~

Content identities:

~~~text
train filename SHA-256
e9ac4d6f700f41ef3a9b7c3f04ce0593f593324a881a0f9fc387901a497f9039

sample manifest SHA-256
ba31900496161322f839f366fa40765d71182d99a59ddad2537786310aae432f

primary prediction content SHA-256
6400c9413af963e3de280e348bd635cd962e5387c2e975e930036d320214274a

frozen pipeline prediction content SHA-256
2ffa40fc0c9b2a7e721d560f6f12edfe2ccdc1c1988582fa7a8104665cdc088a

train target content SHA-256
dfd459f552bd0828221c90258f33f4eacc54220494c7e02b21a179894853e99e

patch index SHA-256
2503616f2d94fd5bfd65be4ad61c7c53af8726dc4f2307745ef1da6f74033943
~~~

## Candidate Application Prerequisite

The model bound is `0.08`, approximately `20.4` gray levels. Reusing the
current second-stage `32`-gray candidate-delta threshold would make every
possible monotonic edit unreachable. Training is therefore not authorized
directly by this PASS.

The next preflight must freeze a candidate-specific third-stage application
protocol before optimization. It must use only the frozen pipeline prediction
as model input, choose a data-independent meaningful-delta threshold no larger
than the model bound, preserve exact identity as a no-op, prove nonnegative
bounded synthetic application, expose no target/split/domain override, and
keep every quality gate closed. This is representation compatibility, not a
post-result threshold sweep.

## Evidence Hashes

~~~text
scripts/analysis/materialize_monotonic_residual_erase_train_inputs.py
sha256 = fa63587d47d3d34f99d4976fbd9512ada076f23739e3b7b348d8e52ea6c28d8f

scripts/analysis/audit_monotonic_residual_erase_train_materialization.py
sha256 = 28e7cd35f26d94fa24648fb6cf1b3bf9476c7266a73f0819b0fe000d7cbcf4db

tests/test_monotonic_residual_erase_train_materialization.py
sha256 = 4063c8d1f5381956240813e08ea357c01360ae3be62b53de4e3e6709170da6ea

outputs/monotonic-residual-erase-train275-materialization-audit-20260810/audit.json
sha256 = 15ee947ed8abb877a2f0d3ea3ffc9392b40b6d800cb1d7372d801ba4b2366882

hardcase_lists/monotonic-residual-erase-train275-v1.txt
sha256 = ba31900496161322f839f366fa40765d71182d99a59ddad2537786310aae432f

hardcase_lists/monotonic-residual-erase-train-patches-v1.csv
sha256 = 2503616f2d94fd5bfd65be4ad61c7c53af8726dc4f2307745ef1da6f74033943

outputs/monotonic-residual-erase-train-patches-v1/summary.json
sha256 = 7373dfe313eb5c290b44c69950fdb0de6df9bec5b79bc5d4d79b9ba7949de2a2

outputs/monotonic-residual-erase-train275-materialization-v1/manifest.json
sha256 = 1115ccdeabc1b2afb9d2046964ef8a062b3d481a3a97a8e1fb282b5bd9028a2e

outputs/monotonic-residual-erase-train275-materialization-v1/primary.json
sha256 = 4cf932820424b862d620ab7e0f1fb245218e221d91b5d214b22f62372bdd5f0d

outputs/monotonic-residual-erase-train275-materialization-v1/second_stage.json
sha256 = 118bb89287e5dd9cbfe2fe009aa0999d6a40f02393049813a64a0a0aa8dac702

outputs/monotonic-residual-erase-train275-materialization-v1/patch_index.json
sha256 = 6522dc267629bb563a63e2271859194ac6cb7a003fe7bfb3a6648684651b0d64
~~~

Intent: Prove exact train275 brighten-only materialization before allowing optimization.
Constraint: The frozen baseline predictions are byte-identical to an existing audited local archive and must not be recomputed without baseline drift.
Rejected: Re-run 550 baseline inference pages | byte-identical audited predictions already exist for the same role, checkpoints, configs, and scripts.
Rejected: Train before freezing candidate application | the 0.08 model bound is analytically incompatible with the legacy 32-gray threshold.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Freeze and pass the monotonic candidate application preflight before invoking the trainer.
Tested: 275 primary and 275 second-stage prediction hashes, 52,645 independently rebuilt candidates, exact 256-patch comparison, train-only target decode, preserve coverage, stage records, and absent training/gate outputs.
Not-tested: Candidate application implementation, training, checkpoints, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-10-monotonic-residual-erase-training-preflight-pass.md
Related: docs/current-primary-quality-loop-ledger.json
