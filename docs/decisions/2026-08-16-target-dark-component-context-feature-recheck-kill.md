# Target-Dark Component Context Feature Recheck Kill

## Decision

`KILL`. The `target_dark_or_overerase_risk` component-ranker successor is not
an available next route for the current quality loop.

The current-state recheck found existing train160 leave-one-page-out evidence
for both the original 18 scalar component features and the 26-feature
multiscale context / printed-line continuity extension. Both fail before any
generator fine-tune or validation surface opens.

## Evidence

- Recheck record:
  `docs/target-dark-component-context-feature-recheck-v1.json`
- Recheck record SHA256:
  `b94882379a22e9c152333a14621700d455e136ad9056cb197983720653fd9957`
- Train160 scalar-feature summary:
  `outputs/target_dark_component_page_holdout_preflight_20260719/summary.json`
- Train160 scalar-feature summary SHA256:
  `ab50439856b90b0cd18b89a59c2d5004a865bd532095c915db8501a481d500d7`
- Train160 context-feature summary:
  `outputs/target_dark_component_page_holdout_preflight_20260719/context_feature_preflight.json`
- Train160 context-feature summary SHA256:
  `b0fe8093702efc276d3260b6788a407a44b794f337b9d8d5e81b78835ef6bf21`
- Context-feature ablation:
  `outputs/target_dark_component_page_holdout_preflight_20260719/context_feature_ablation.json`
- Context-feature ablation SHA256:
  `0177bca125d292d18c5a741e11e428d4424d861720a89a8bcceb0173eb4c9f0c`
- Preflight decision JSON:
  `outputs/target_dark_component_page_holdout_preflight_20260719/decision.json`
- Preflight decision JSON SHA256:
  `3af307e19a5ffbd0044627490ec9f63e362c897320c54ce1e70dc50e7b65e48b`

## Gate Result

| Route | Feature count | ROC-AUC delta | AP delta | Reject-ratio delta | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Scalar component features | 18 | -0.09419069659268742 | -0.07793004418448701 | +0.08024691358024694 | fail |
| Context / printed-line extension | 26 | -0.11339957395126077 | -0.1057951931742872 | +0.09876543209876543 | fail |

The frozen train-only gate required reviewed labels to improve macro ROC-AUC
and average precision by at least `0.02` while lowering top-50-per-page reject
ratio by at least `0.02`. The rechecked evidence moves in the wrong direction
on all three criteria.

## Boundary

No generator training, checkpoint generation, candidate inference, `inner_val15`,
development gate, SCUT115, holdout40, visual review, reserved blind, promotion,
or `artifacts/current-primary` replacement is opened by this recheck.

The next move must select another named failure bucket, preregister a
materially different non-component-ranker family with train-only evidence, or
record broader durable exhaustion.

Intent: Close an already-failed component-ranker successor so the loop does not repeat old train160 selector evidence.
Constraint: The component route must pass the leakage-safe train160 leave-one-page-out preflight before any generator fine-tune or validation gate.
Rejected: Add more labels to the same scalar/context component-ranker family | reviewed labels already underperform weak labels after missing-page coverage was fixed.
Rejected: Choose a feature subset by the best ablation slice | every reported subset still fails the preregistered ROC-AUC, AP, and reject-ratio criteria.
Rejected: Run a generator fine-tune from this selector | train-only preflight failed and no candidate surface is authorized.
Confidence: high
Scope-risk: narrow
Directive: Do not revive `target_dark_component_context_feature_v1` with label-count, threshold, top-k, feature-subset, or generator-training rescue; choose a materially different route.
Tested: current-state recheck of existing train160 scalar-feature, context-feature, ablation, and decision JSON evidence.
Not-tested: new failure bucket, non-component-ranker successor, reserved blind, promotion.
Related: docs/decisions/2026-08-16-stroke-only-source-candidate-bucket-exhaustion.md
