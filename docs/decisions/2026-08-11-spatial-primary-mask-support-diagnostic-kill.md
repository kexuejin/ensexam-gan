# Spatial Primary Mask Support Diagnostic KILL

## Decision

`KILL`. The exact preregistered train275 materialization and diagnostic
completed on all 275 frozen train-role pages and 563,200 deterministically
balanced pixels. Label-free materialization itself passed: both `mb` and `ms`
were produced by `utils.page_inference.infer_full_page`, matched every source
shape, and remained hash-stable without target access.

The four-channel mask representation fails the support-separation contract.
Its mean held-out fold AUC is `0.580727`, below the frozen `0.65` minimum, its
macro median page AUC is `0.584623`, below `0.60`, and it underperforms the
fixed second-stage-RGB ablation by `0.063779` rather than exceeding it by the
required `0.03`.

The exact `mb`, `ms`, `mb-ms`, and `mb*ms` representation, materialized masks,
fold assignment, coordinate sampling, ridge lambda, and acceptance thresholds
are closed. No model training, checkpoint, candidate inference, quality gate,
visual review, reserved-blind access, promotion, or current-primary
replacement occurred.

## Frozen Result

~~~text
materialized train pages / target access       275 / false
mb content sha256                              6e36bee6841d678c4db83d0fb35c3046a95e08883e181d61cc45ad09b70ed4cc
ms content sha256                              677374bfed8c8bb1a831a248bdee7aae76dcdc21e70d58e29c18881590bc35e9
diagnostic train pages / sampled pixels        275 / 563,200
fold page counts                               54 / 62 / 47 / 50 / 62
full mean / minimum fold AUC                   0.580727 / 0.565196
second-stage-RGB-only mean fold AUC            0.644506
full minus ablation mean AUC                  -0.063779
macro median per-page AUC                      0.584623
positive score above preserve folds            5 / 5

accept full mean AUC >= 0.65                   FAIL
accept every fold AUC >= 0.55                  PASS
accept macro median page AUC >= 0.60           FAIL
accept positive mean > preserve >= 4/5         PASS
accept full-ablation margin >= 0.03             FAIL
terminal                                        KILL
~~~

Every held-out fold has a negative full-minus-ablation margin, from
`-0.042130` to `-0.078671`. The result is not caused by one split, a reversed
score direction, missing masks, or weak provenance. Lowering the aggregate AUC
gate would still leave the independent ablation claim decisively false.

## Interpretation

The frozen primary masks contain weak directional support signal: the positive
mean score exceeds preserve in all five folds and every fold AUC stays above
`0.55`. That signal is not strong enough for the registered purpose and is
materially worse than second-stage RGB on the same pages, coordinates, and
labels. Pixel alignment alone did not repair the support-identification
problem left by the previous coarse-context diagnostic.

This result does not authorize thresholding the masks, selecting one mask,
adding neighborhoods, fitting a nonlinear probe, or training a mask-aware
candidate. Those actions would be post-result rescue of the failed exact
family.

## Next Boundary

The current-primary checkpoint remains the product default. Promotion and
reserved blind remain disabled. A successor must be separately preregistered
and materially different from both closed support families: the 13-channel
dual-pipeline/context representation and the four-channel frozen `mb`/`ms`
representation. Before any optimizer or quality surface opens, it must name a
new target-free causal support source and an independent ablation that can
falsify the claim on train-only data.

The next authorized action is therefore preregistration and evidence review
only. No RGB/context feature rescue, mask transform, threshold, fold, sampling,
lambda, nonlinear-probe, training, or candidate execution is authorized by
this record.

## Evidence Hashes

~~~text
outputs/spatial-primary-mask-support-materialization-20260811/manifest.json
sha256 = 0ac8b8f0fbacb51bc39e2d37523565c8e18e381f73a91fb749e14e27fbcb778e

outputs/spatial-primary-mask-support-prerequisite-20260811/audit.json
sha256 = 88ed2f090bed3664cf0947faf273c066513a24e225e33f9846c7bb367a8fe1d7

scripts/analysis/materialize_primary_masks_train_only.py
sha256 = 76812498214ba2e6ebe387b10f9a65ab3c93d6f17e56599b91a5a8883f208b16

scripts/analysis/audit_spatial_primary_mask_support.py
sha256 = 45add93e2944590c3b9abbbd562bbdb09b08161edc35b221afecbe658d9e9c7a

tests/test_spatial_primary_mask_support_prerequisite.py
sha256 = 5075d7b8ada2296652e60a25f89212c1310616f00e2004653e1b47b6fe1966a9

docs/spatial-primary-mask-support-prerequisite-v1.json
sha256 = 577de835e7a37799b4feb4ee0f2b8ebd138bffdb766c3d41db143aacb7aa3082
~~~

Intent: Close frozen primary mask support that is weaker than the registered second-stage-RGB ablation.
Constraint: Mask materialization was label-free and the diagnostic decoded only frozen train-role targets while all quality surfaces stayed closed.
Rejected: Lower the aggregate or page AUC gates | the representation also loses to its ablation by 0.063779.
Rejected: Tune mask transforms, thresholds, neighborhoods, probe class, folds, sampling, or lambda | those are post-result rescues of the exact family.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not repeat or train the exact `mb`/`ms` support family. Require a materially new target-free causal source and frozen ablation before successor execution.
Tested: Exact train275 source/config/checkpoint provenance; 275 `mb` and 275 `ms` maps; content hashes; train-label-only target decode; deterministic balanced sampling; five page-grouped closed-form folds; full-versus-ablation AUC; no candidate or quality outputs.
Not-tested: A materially new successor representation, model training, checkpoint portability, candidate inference, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-11-spatial-primary-mask-support-preregistration.md
