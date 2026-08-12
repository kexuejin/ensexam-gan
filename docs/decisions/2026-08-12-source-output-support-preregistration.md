# Source-Output Support Preregistration

## Decision

`PREREQUISITE_NEEDED`. Three frozen support-evidence families have now failed
their independent train-only ablations: final-pipeline RGB/context, primary
`mb`/`ms`, and `Ic4`/`Ic2`/`Ic1`/`Ire` stage disagreement. Their exact
features, transforms, probes, folds, sampling, thresholds, training paths, and
quality surfaces remain closed.

The next bounded uncertainty is whether the raw input-to-output trajectory
contains independent support evidence that output appearance alone lacks. The
registered representation adds only unmodified source RGB to the unchanged
second-stage RGB baseline. It does not add primary RGB, masks, stages, page
scalars, edit thresholds, local neighborhoods, discriminator or encoder
transforms, OCR, metadata, or target-derived inputs.

Only one train275 separability diagnostic is authorized. No optimizer, model,
checkpoint, candidate inference, inner-val15, development gate, SCUT115,
holdout40, visual review, reserved blind, promotion, or current-primary
replacement is authorized.

## Causal Distinction

Residual handwriting and safe paper preservation depend on what the pipeline
changed, not only on the final cleaned RGB. Two output pixels with similar
appearance can represent different actions when one started as dark ink and
the other as paper. Raw source RGB preserves that pre-edit evidence. Unlike the
closed dual-pipeline diagnostic, this test excludes primary RGB, explicit
primary-to-second-stage signed differences, thresholded edit summaries, and
page-broadcast gate values. Unlike the mask and stage diagnostics, it executes
no model and derives no intermediate maps.

The exact full feature vector is six raw channels normalized by `255`:

1. source red;
2. source green;
3. source blue;
4. frozen second-stage red;
5. frozen second-stage green;
6. frozen second-stage blue.

The independent ablation is exactly the last three second-stage channels on
the same coordinates and page folds. Channel subtraction, ratios, products,
luma, color transforms, gradients, neighborhoods, thresholds, components,
feature selection, or nonlinear models are prohibited.

## Frozen Data And Diagnostic

Reuse the exact train275 source role, second-stage prediction set, and
train-label set already pinned by the closed support audits. Before decoding
targets, validate the role contract, manifest identity, source hashes recorded
by the frozen primary metrics, and prediction/label content hashes.

Target-lighter remains exactly `target_luma - second_stage_luma > 2` gray; all
other pixels are preserve. Assign pages by
`int(sha256(utf8_basename), 16) % 5`, then use the same deterministic
SplitMix64 ranking to select
`min(1024, target_lighter_count, preserve_count)` pixels per class and page.
Fit float64 closed-form ridge with `lambda=1.0`, fitting-fold-only
standardization, and an unpenalized intercept. No threshold is learned.

## Acceptance Contract

`PASS` requires every condition:

- exact 275 train sources, predictions, labels, roles, shapes, and hashes;
- mean held-out fold AUC at least `0.65`;
- every held-out fold AUC at least `0.55`;
- macro median per-page AUC at least `0.60`;
- mean AUC at least `0.03` above the fixed second-stage-RGB ablation;
- positive mean score above preserve in at least four of five folds.

Any metric failure is `KILL`. Missing or drifting implementation/provenance is
`PREREQUISITE_NEEDED`. `PASS` authorizes only a separate
data/training/application preflight with portable checkpoint metadata; it does
not authorize training or candidate inference directly.

## Terminal Successors

- `PASS`: freeze one source-conditioned data/training/application preflight.
- `KILL`: close the exact raw source-plus-output representation without
  transform, neighborhood, feature, probe, or threshold rescue.
- `PREREQUISITE_NEEDED`: repair provenance or implementation only; do not
  change the registered representation or gates.

## Registered Surface

~~~text
plan:
  docs/source-output-support-prerequisite-v1.json
future audit:
  scripts/analysis/audit_source_output_support_separation.py
future test:
  tests/test_source_output_support_prerequisite.py
future output:
  outputs/source-output-support-prerequisite-20260812/audit.json
~~~

## Evidence Hashes

~~~text
docs/decisions/2026-08-12-reconstruction-stage-disagreement-diagnostic-kill.md
sha256 = 01c6d854a7b1e70683c1697ecaf7b6357d12f5bbda0671cf96a1b5503c56ea9c

scripts/analysis/audit_dual_input_support_separation.py
sha256 = 4821a27c860aef54878fc15016458244a3cbd195fdbffacdc535ded1b1b032e5

hardcase_lists/monotonic-residual-erase-train275-v1.txt
sha256 = ba31900496161322f839f366fa40765d71182d99a59ddad2537786310aae432f
~~~

Intent: Test raw source appearance as independent support evidence after output-only, mask, and reconstruction-stage sources failed.
Constraint: Only a frozen train275 diagnostic is authorized; every model and quality surface remains closed.
Rejected: Frozen discriminator local logits | they are a nonlinear transform of the already closed final-RGB family.
Rejected: Reuse primary RGB, signed pipeline differences, or page gate features | those are part of the closed dual-input representation.
Rejected: Add luma, color transforms, thresholds, neighborhoods, gradients, or nonlinear probes | those would turn one raw causal change into an unfrozen feature search.
Confidence: medium
Scope-risk: narrow
Reversibility: clean
Directive: Do not alter channels, folds, sampling, probe, lambda, or acceptance after preregistration. A failed ablation margin closes this exact family.
Tested: Existing role, source hash, prediction hash, label hash, and train-only diagnostic utilities inspected; no data diagnostic executed.
Not-tested: Source-output separability, model training, checkpoint portability, candidate inference, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-12-reconstruction-stage-disagreement-diagnostic-kill.md
