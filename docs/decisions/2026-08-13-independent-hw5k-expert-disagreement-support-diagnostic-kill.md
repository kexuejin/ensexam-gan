# Independent HW5K Expert Disagreement Support Diagnostic KILL

## Decision

`KILL`. The exact preregistered diagnostic evaluated 251,904 deterministically
balanced pixels from 123 train-role HW5K pages that the specialist checkpoint
did not see during training. Frozen specialist RGB added measurable support
signal beyond frozen current-primary RGB, but the aggregate held-out AUC margin
was `0.029727`, below the frozen `0.03` independent-support requirement by
`0.000273`.

All other registered gates passed. The six-channel representation reached mean
AUC `0.664916`, minimum fold AUC `0.634484`, macro median page AUC `0.716586`,
and positive mean score exceeded preserve in all five folds. The margin gate is
not rounded or relaxed after observing the result, so this family does not
advance to training or any quality split.

No model training, checkpoint, candidate inference, inner-val15, development,
SCUT115, holdout40, visual review, reserved-blind access, promotion, or
current-primary replacement occurred.

## Frozen Result

~~~text
unseen HW5K train pages / balanced samples  123 / 251,904
fold page counts                            28 / 22 / 20 / 24 / 29
full mean / minimum fold AUC                0.664916 / 0.634484
current-primary-RGB-only mean fold AUC      0.635189
full minus ablation mean AUC                0.029727
required full minus ablation mean AUC       0.030000
shortfall                                   0.000273
macro median per-page AUC                   0.716586
positive score above preserve folds         5 / 5

accept full mean AUC >= 0.65                PASS
accept every fold AUC >= 0.55               PASS
accept macro median page AUC >= 0.60        PASS
accept positive mean > preserve >= 4/5      PASS
accept full-ablation margin >= 0.03          FAIL
terminal                                   KILL
~~~

Held-out full-versus-current-primary-RGB margins are `0.029751`, `0.027408`,
`0.025720`, `0.024722`, and `0.041037`. Only fold 4 cleared `0.03`; the result
therefore cannot be reinterpreted as a stable independent-margin pass.

## Materialization

The target-free materializer ran both frozen checkpoints sequentially over the
same ordered 123-page source list. It reproduced the frozen specialist-training
exclusion, eligible set, artifact hashes, and inference protocol before image
decode. The final manifest contains no target, domain, caller, route, split, or
expert-selection metadata and records `target_access=false` and
`routing_metadata=false`.

~~~text
current-primary predictions                  123
HW5K-expert predictions                      123
eligible basename sha256
e2921d717086c080606acd69dbec2de0e4a97281edc460aa6c0b74af41097698
eligible ordered-path sha256
ad7c794706edb1b832cb30af978663853fb10646531bc4cf83011023338d81e2
paired prediction content sha256
43c9152a9f0b724b76d5569dd1938f44c4d244c09178b736f6822f02a2a5b03d
materialization manifest sha256
05607257d424e2912c07bdc9bc28354875307210dfde21802f0cd9c8b403a09f
~~~

The first audit invocation stopped before page decode because the implementation
incorrectly treated the shared 383-file train label directory as if the frozen
275-label subset had exclusive directory ownership. Commit `520a7c6` repaired
only that provenance interpretation: the audit still verifies exactly the 275
manifest-selected label names and their frozen aggregate content hash, while
the other 108 shared labels remain outside the diagnostic. The repaired audit
then decoded only the 123 eligible train-role targets and completed to `KILL`.

## Interpretation

An independently trained specialist is the first tested source in this support
sequence to approach the frozen incremental-margin requirement, but approaching
the threshold is not passing it. Lowering the gate, rounding the result, changing
folds or sampling, selecting expert channels or layers, adding differences or
transforms, changing checkpoints, fitting a nonlinear probe, restoring routing,
or training after this observation would be post-result rescue of the family.

The HW5K-only diagnostic also could not establish SCUT safety even if its margin
had passed. Candidate 5 routing remains closed because its prior HW5K gain did
not preserve SCUT.

## Next Boundary

The long-lived quality loop remains active at `PREREQUISITE_NEEDED`.
`artifacts/current-primary` remains the product default, while promotion and
reserved blind remain disabled. No implementation or data execution is now
authorized. A successor must first preregister a materially different,
target-free causal source and an independent train-only ablation. It must not
rescue final-pipeline RGB/context, primary `mb`/`ms`, primary reconstruction
stages, raw source-plus-output RGB, second-stage raw alpha, or paired independent
expert RGB.

## Evidence Hashes

~~~text
outputs/independent-hw5k-expert-support-prerequisite-20260813/audit.json
sha256 = 81fce1de56a3e2d3090bb7fbd131d6de4be70c48b966ca58ea34b64c2dd857b4

outputs/independent-hw5k-expert-support-materialization-20260813/manifest.json
sha256 = 05607257d424e2912c07bdc9bc28354875307210dfde21802f0cd9c8b403a09f

scripts/analysis/materialize_independent_hw5k_expert_outputs_train_only.py
sha256 = 4e369f8d0b8d1e1da9a27db14e0799d85bf23d061a99bd3fc819a6ef9b29f4f9

scripts/analysis/audit_independent_hw5k_expert_disagreement_support.py
sha256 = 74c738c958aeef0ff0e7a5dd54b46a9d8c6197d71146ddc70ce853c44ec75923

tests/test_independent_hw5k_expert_disagreement_support_prerequisite.py
sha256 = bc89d669e77aba92f5f6c63d3b434ebbce0edb93df22ff02eb8579f3fd1aba3e

docs/independent-hw5k-expert-disagreement-support-prerequisite-v1.json
sha256 = 02b3f4081be88516b26ae69a8d2df346554bdd492310eda1ececdab398cf0355
~~~

Intent: Enforce the frozen independent-support margin even when paired specialist RGB narrowly misses it.
Constraint: The specialist saw 152 train275 pages, so only the frozen 123 unseen HW5K pages were eligible and the result cannot establish SCUT safety.
Rejected: Round 0.029727 to 0.03 or lower the margin | post-result acceptance rescue.
Rejected: Change channels, layers, transforms, folds, sampling, lambda, checkpoint, probe, or routing | rescues the exact failed family.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not repeat, rescue, route, or train paired independent-expert RGB. Require a materially new target-free causal source and independent ablation before successor execution.
Tested: Frozen artifact and exclusion hashes; exact 123-page unseen population; sequential target-free paired inference; prediction/metrics/command hashes; complete frozen 275-label subset identity; deterministic balanced sampling; five page-grouped closed-form folds; six-channel versus current-primary-RGB AUC; no candidate or quality outputs.
Not-tested: A materially new successor, model training, candidate inference, inner-val15, development, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-13-independent-hw5k-expert-disagreement-support-preregistration.md
Related: docs/decisions/2026-08-03-explicit-domain-dual-checkpoint-research-harness.md
