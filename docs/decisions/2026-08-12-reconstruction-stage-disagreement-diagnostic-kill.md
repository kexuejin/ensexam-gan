# Reconstruction-Stage Disagreement Diagnostic KILL

## Decision

`KILL`. The exact preregistered label-free materialization completed for all
275 train-role sources, and the train-label-only diagnostic evaluated 563,200
deterministically balanced pixels across the frozen five page folds. The four
`Ic4`/`Ic2`/`Ic1`/`Ire` disagreement channels have real directional signal:
their mean fold AUC, minimum fold AUC, macro median page AUC, and score
direction all pass their registered gates.

The family nevertheless fails its independent RGB-ablation contract. Its mean
fold AUC is `0.657016`, only `0.012509` above the fixed second-stage-RGB result
of `0.644506`, rather than the required `0.03`. Fold 2 is `0.006025` worse than
RGB, and only fold 3 reaches a per-fold margin above `0.03`. The exact
representation therefore does not establish enough distinct support evidence
to justify reopening an optimizer.

No model training, checkpoint, candidate inference, quality gate, visual
review, reserved-blind access, promotion, or current-primary replacement
occurred.

## Frozen Result

~~~text
materialized train pages / target access       275 / false
materialization content sha256                 494285507902346db8e008716cb342f661731ff1aa5e4b4ff4a96b004467c365
diagnostic train pages / sampled pixels        275 / 563,200
fold page counts                               54 / 62 / 47 / 50 / 62
full mean / minimum fold AUC                   0.657016 / 0.639200
second-stage-RGB-only mean fold AUC            0.644506
full minus ablation mean AUC                   0.012509
macro median per-page AUC                      0.678154
positive score above preserve folds            5 / 5

accept full mean AUC >= 0.65                   PASS
accept every fold AUC >= 0.55                  PASS
accept macro median page AUC >= 0.60           PASS
accept positive mean > preserve >= 4/5         PASS
accept full-ablation margin >= 0.03             FAIL
terminal                                        KILL
~~~

Held-out full-versus-RGB margins are `0.012852`, `0.016578`, `-0.006025`,
`0.031875`, and `0.007268`. The rejection is therefore not caused by absent
stage maps, reversed score direction, a weak aggregate AUC, or provenance
drift. Lowering only the ablation-margin gate after observing the result would
be post-result rescue.

## Interpretation

Frozen reconstruction convergence is more informative than the previously
closed primary-mask family, but it is not materially stronger than the RGB
surface already available from the frozen second stage. The registered purpose
was to falsify whether stage disagreement supplied a sufficiently different
support source before paying for another training run. That claim failed its
only independent comparison.

Do not repeat this diagnostic with selected stages, transformed channels,
neighborhoods, changed folds, different sampling, another ridge lambda,
nonlinear probes, or relaxed thresholds. Those variants are rescues of the
observed result rather than the preregistered causal test.

## Next Boundary

The long-lived quality loop remains active at `PREREQUISITE_NEEDED`, while
`artifacts/current-primary` remains the product default and promotion plus
reserved blind remain disabled. A successor must first be separately
preregistered and materially different from final-pipeline RGB/context,
primary `mb`/`ms`, and reconstruction-stage disagreement. It must name a new
target-free causal source and an independent train-only ablation before data
execution, training, or any quality surface can reopen.

## Evidence Hashes

~~~text
outputs/reconstruction-stage-disagreement-materialization-20260812/manifest.json
sha256 = 1f17a884735614303a5340cefcfc5962b5c15677b288d56f65a4ed850d9147c6

outputs/reconstruction-stage-disagreement-prerequisite-20260812/audit.json
sha256 = c6564514041cde59e928032343d316d1f2a33c18e0931c79022a5368fbbbda1b

scripts/analysis/materialize_reconstruction_stage_disagreement_train_only.py
sha256 = dc7710e6f8b50224c2ad9ccaad9f85f62dcc904beee48d678af0c357f697ef25

scripts/analysis/audit_reconstruction_stage_disagreement.py
sha256 = 1ce37584ca6ec43a24887b5c36ba7a221ae09663d511480f0a641a06f755ebe8

docs/reconstruction-stage-disagreement-prerequisite-v1.json
sha256 = 90310fbc9c435bf714fc427bf1a1c1f1fe7440183896d01713230c09374166e0
~~~

Intent: Close reconstruction-stage support that does not clear its frozen RGB ablation margin.
Constraint: Materialization was label-free and the diagnostic decoded only frozen train-role targets while every model and quality surface stayed closed.
Rejected: Lower the ablation-margin gate | the result was observed after preregistration and one fold already loses to RGB.
Rejected: Select stages or tune transforms, neighborhoods, probe class, folds, sampling, lambda, or thresholds | these are post-result rescues of the exact family.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not repeat or train the exact reconstruction-stage disagreement family. Require a materially new target-free source and frozen independent ablation before successor execution.
Tested: Exact 275-page source/config/checkpoint provenance; four finite aligned float32 stage maps per page; content hashes; train-label-only target decode; deterministic balanced sampling; five page-grouped closed-form folds; full-versus-RGB AUC; no candidate or quality outputs.
Not-tested: A materially new successor representation, model training, checkpoint portability, candidate inference, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-12-reconstruction-stage-disagreement-preregistration.md
