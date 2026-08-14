# Source-Output Support Diagnostic KILL

## Decision

`KILL`. The exact preregistered train275 diagnostic evaluated 563,200
deterministically balanced pixels across the frozen five page folds. Raw source
RGB added directional signal to frozen second-stage RGB, but it did not clear
the independent support-separation requirement: the mean AUC margin was only
`0.010171`, below the required `0.03`.

All other registered gates passed. The full representation reached mean AUC
`0.654677`, minimum fold AUC `0.626443`, macro median page AUC `0.702450`,
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
full mean / minimum fold AUC                0.654677 / 0.626443
second-stage-RGB-only mean fold AUC         0.644506
full minus ablation mean AUC                0.010171
macro median per-page AUC                   0.702450
positive score above preserve folds         5 / 5

accept full mean AUC >= 0.65                PASS
accept every fold AUC >= 0.55               PASS
accept macro median page AUC >= 0.60        PASS
accept positive mean > preserve >= 4/5      PASS
accept full-ablation margin >= 0.03          FAIL
terminal                                   KILL
~~~

Held-out full-versus-RGB margins are `0.016239`, `0.002861`, `0.004160`,
`0.019118`, and `0.008479`. The exact source/output representation therefore
does not establish independent support beyond the frozen second-stage RGB
surface.

## Interpretation

Pre-edit source appearance is informative for the registered labels, but most
of that information is already recoverable from the frozen second-stage RGB.
The raw source channels do not justify another optimizer or a source-aware
candidate. Lowering the ablation gate, selecting source channels, adding
differences or color transforms, or fitting a nonlinear probe after observing
this result would be post-result rescue of the exact family.

## Next Boundary

The long-lived quality loop remains active at `PREREQUISITE_NEEDED`.
`artifacts/current-primary` remains the product default, while promotion and
reserved blind remain disabled. A successor must be separately preregistered
and materially different from final-pipeline RGB/context, primary `mb`/`ms`,
reconstruction-stage disagreement, and raw source-plus-output RGB. It must name
a new target-free causal support source and independent train-only ablation
before any data execution, training, or quality surface can reopen.

## Evidence Hashes

~~~text
outputs/source-output-support-prerequisite-20260812/audit.json
sha256 = 29957a04e0ece3a4cdc0d0cdf97732536aa05caee96c5b571c64a191c953a2b6

scripts/analysis/audit_source_output_support_separation.py
sha256 = 5690bfcb04e9a69566ca67d8eb9ae479d98065926fd0d35c64a94e2d637d879e

tests/test_source_output_support_prerequisite.py
sha256 = e81dad38d09bb9d0912849a007d1482149bb39b98ff0f89594d07ba8c0722e9b

docs/source-output-support-prerequisite-v1.json
sha256 = fba4b1d333d5993d5741a044ed0dfec044c014a3d5c2f502c6ce7cfa25f86f5d
~~~

Intent: Close raw source-plus-output support that does not clear its frozen RGB ablation margin.
Constraint: The diagnostic used only frozen train-role labels and kept all model and quality surfaces closed.
Rejected: Relax the RGB-ablation margin | post-result threshold rescue.
Rejected: Select source channels, add transforms, neighborhoods, or nonlinear probes | rescues the exact failed family.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not repeat or train the exact source-output family. Require a materially new target-free source and independent ablation before successor execution.
Tested: Exact train275 roles and source/prediction/label hashes; deterministic balanced sampling; five page-grouped closed-form folds; full-versus-RGB AUC; no candidate or quality outputs.
Not-tested: A materially new successor representation, model training, checkpoint portability, candidate inference, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-12-source-output-support-preregistration.md
