# Dual-Input Support-Separation Diagnostic KILL

## Decision

`KILL`. The exact preregistered train275 diagnostic completed on all 275
frozen train-role pages and 563,200 deterministically balanced pixels. The
13-channel dual-pipeline representation has measurable train-only signal, but
it fails the two gates that establish a materially different support source:
its mean held-out fold AUC is `0.648757`, below the frozen `0.65` minimum, and
its gain over second-stage RGB alone is only `0.004251`, far below the frozen
`0.03` ablation margin.

The exact full representation, second-stage-RGB ablation, fold assignment,
coordinate sampling, ridge lambda, and acceptance thresholds are closed. No
model training, checkpoint, candidate inference, quality gate, visual review,
reserved-blind access, promotion, or current-primary replacement occurred.

## Frozen Result

~~~text
train pages / sampled pixels             275 / 563,200
fold page counts                         54 / 62 / 47 / 50 / 62
full mean / minimum fold AUC             0.648757 / 0.608393
second-stage-RGB-only mean fold AUC      0.644506
full minus ablation mean AUC             0.004251
macro median per-page AUC                0.686132
positive score above preserve folds      5 / 5

accept full mean AUC >= 0.65             FAIL
accept every fold AUC >= 0.55            PASS
accept macro median page AUC >= 0.60     PASS
accept positive mean > preserve >= 4/5   PASS
accept full-ablation margin >= 0.03       FAIL
terminal                                  KILL
~~~

Per-fold evidence is consistent. Every full-representation fold exceeds its
ablation by only `0.001067` to `0.006902`; no single split hides a material
gain from primary RGB, signed RGB difference, or the four broadcast page gate
features. Lowering the aggregate AUC gate would not repair the failed ablation
claim.

## Interpretation

Second-stage RGB contains most of the linear support signal measured here.
Adding primary RGB, signed primary-to-second-stage difference, and page-level
`copy_mask_cov8`, `primary_edit_px`, `primary_p95_edit_delta`, and
`second_stage_gate_ratio` does not create the preregistered independent
separation. These page broadcasts are too coarse to establish pixel support
causality, even though the balanced per-page AUC indicates that local image
appearance is not entirely uninformative.

This result does not justify training a nonlinear model on the same 13
channels. That would bypass the exact pre-optimizer evidence gate after its
ablation claim failed.

## Next Boundary

Do not repeat or rescue this diagnostic through feature selection, local
context additions, fold changes, sampling changes, ridge-lambda changes,
nonlinear probes, or acceptance-threshold changes. A next iteration requires a
new preregistration for a materially different target-free support source.

The most defensible next uncertainty is pixel-aligned frozen primary mask
evidence (`mb` and `ms`) rather than page-broadcast summaries. Those maps are
already produced by the frozen primary pipeline and used by the product copy
mask path, but they are not materialized by this audit and have not been proven
train-only, hash-stable, or support-separating. A new preregistration may
authorize only a mask-evidence materialization and separability prerequisite;
it may not authorize model training.

Portable checkpoint metadata remains mandatory in any later training
preflight: `Path` values must become strings and a default weights-only load
must pass before candidate inference.

## Evidence Hashes

~~~text
outputs/dual-input-support-separation-prerequisite-20260811/audit.json
sha256 = 415dd3720bbb70db541979a9675c938730c5da6d62a223127fe40997a40b92ff

scripts/analysis/audit_dual_input_support_separation.py
sha256 = fd2dd33d7e730cbc0a38401ec95d21851016f3c328f84556f7136f58fda8a1ed

tests/test_dual_input_support_separation_prerequisite.py
sha256 = cd16796d33c25c49b0132c175a1dbd61fbc0838d6697ea225cecfb81760e157b

docs/dual-input-support-separation-prerequisite-v1.json
sha256 = 702b0f4cb68e371589177f5b0013a106ef38ff9c73528b6f2325f38c01245585
~~~

Intent: Close a dual-input representation that cannot prove material support separation over its second-stage-only ablation.
Constraint: The AUC and ablation gates were frozen before train-target decode; candidate and quality surfaces stayed closed.
Rejected: Lower the `0.65` aggregate AUC gate | the independent ablation-margin claim still fails by more than sevenfold.
Rejected: Add local features or a nonlinear probe after seeing the result | that is an unregistered feature-family rescue.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not repeat the exact 13-channel diagnostic or train on it. Preregister pixel-aligned frozen mask evidence before any successor execution.
Tested: Exact train275 hashes and roles; 275-page target-only label decode; deterministic balanced sampling; five page-grouped closed-form folds; full-versus-ablation AUC; no candidate or quality outputs.
Not-tested: Frozen `mb`/`ms` mask materialization, mask support separation, model training, checkpoint portability, candidate inference, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-11-dual-input-support-separation-preregistration.md
