# Second-Stage Alpha Support Diagnostic KILL

## Decision

`KILL`. The exact preregistered train275 diagnostic evaluated 563,200
deterministically balanced pixels across the frozen five page folds. Raw
prethreshold erasemap alpha added directional signal to frozen second-stage
RGB, but it did not clear the independent support-separation requirement: the
mean AUC margin was only `0.011768`, below the required `0.03`.

All other registered gates passed. The full representation reached mean AUC
`0.656274`, minimum fold AUC `0.611754`, macro median page AUC `0.704466`,
and positive mean score exceeded preserve in all five folds. Every fold's
margin remained below `0.03`, so the result is a family-level failure rather
than a single-split anomaly.

No model training, checkpoint, candidate inference, quality gate, visual
review, reserved-blind access, promotion, or current-primary replacement
occurred.

## Frozen Result

~~~text
train pages / balanced samples             275 / 563,200
fold page counts                            54 / 62 / 47 / 50 / 62
full mean / minimum fold AUC                0.656274 / 0.611754
second-stage-RGB-only mean fold AUC         0.644506
full minus ablation mean AUC                0.011768
macro median per-page AUC                   0.704466
positive score above preserve folds         5 / 5

accept full mean AUC >= 0.65                PASS
accept every fold AUC >= 0.55               PASS
accept macro median page AUC >= 0.60        PASS
accept positive mean > preserve >= 4/5      PASS
accept full-ablation margin >= 0.03          FAIL
terminal                                   KILL
~~~

Held-out full-versus-RGB margins are `0.011072`, `0.012402`, `0.016270`,
`0.004429`, and `0.014668`. The exact raw-alpha representation therefore does
not establish independent support beyond the frozen second-stage RGB surface.

## Materialization

The label-free materializer produced exactly 275 aligned NPZ files with one
`float32` key named `raw_alpha`. Every map was finite, stayed in `[0,1]`, and
matched its registered shape, summary, source-prediction hash, and NPZ hash.
The output surface contained only `manifest.json` and `pages/`, and the
manifest records `target_access=false`.

~~~text
materialization manifest sha256
fa6cc3d0607a7b21e85e74884f4fa8bd6f09ae4135e477a3af8d467e8b3c46b3

materialization content sha256
c25bc0ec474020b2dd9f48898a5fefc970977e6f8418a815c10a59c6739a27ca
~~~

## Interpretation

The hidden prethreshold alpha head is informative for the registered labels,
but most of that information is already recoverable from final second-stage
RGB. Its incremental evidence is too small to justify an alpha-conditioned
optimizer or candidate. Lowering the ablation gate, thresholding alpha,
selecting another layer, adding transforms or neighborhoods, or fitting a
nonlinear probe after observing this result would be post-result rescue of the
exact family.

## Next Boundary

The long-lived quality loop remains active at `PREREQUISITE_NEEDED`.
`artifacts/current-primary` remains the product default, while promotion and
reserved blind remain disabled. A successor must be separately preregistered
and materially different from final-pipeline RGB/context, primary `mb`/`ms`,
primary reconstruction stages, raw source-plus-output RGB, and second-stage
raw alpha. It must name a new target-free causal support source and independent
train-only ablation before any data execution, training, or quality surface
can reopen.

## Evidence Hashes

~~~text
outputs/second-stage-alpha-support-prerequisite-20260812/audit.json
sha256 = db94152ccbcba11e4008e9ac31e3bdca821a336cbee5cc861861f0c99c4e2351

outputs/second-stage-alpha-support-materialization-20260812/manifest.json
sha256 = fa6cc3d0607a7b21e85e74884f4fa8bd6f09ae4135e477a3af8d467e8b3c46b3

scripts/analysis/materialize_second_stage_alpha_train_only.py
sha256 = 8a9314b9e880ad968aabab10b144d59d489488db4e83d0371e11bedb6db89f06

scripts/analysis/audit_second_stage_alpha_support.py
sha256 = 1142e5d160702c617ea5163fe61ecda47b9903d13d2f4ec45293e5b8e3496731

tests/test_second_stage_alpha_support_prerequisite.py
sha256 = 019dc7c4413dcd604d2da6f142dd5f72237fe6b5270aecef81baa70ec4d508a1

docs/second-stage-alpha-support-prerequisite-v1.json
sha256 = e0db918fa5641f063024b79c13686c63c1bf0df463a9fc7baa5cf56c4848c1e8
~~~

Intent: Close raw second-stage alpha support that does not clear its frozen RGB ablation margin.
Constraint: Alpha materialization was target-free; the diagnostic used only frozen train-role labels and kept all model and quality surfaces closed.
Rejected: Lower the independent-support margin | post-result acceptance rescue.
Rejected: Threshold alpha or select another layer, transform, neighborhood, or nonlinear probe | rescues the exact failed family.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not repeat or train the exact raw-alpha family. Require a materially new target-free source and independent ablation before successor execution.
Tested: Exact train275 roles and prediction/label hashes; target-free alpha materialization; NPZ schema, dtype, range, shape, and content hashes; deterministic balanced sampling; five page-grouped closed-form folds; full-versus-RGB AUC; no candidate or quality outputs.
Not-tested: A materially new successor representation, model training, checkpoint portability, candidate inference, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-12-second-stage-alpha-support-preregistration.md
