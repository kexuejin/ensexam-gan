# Source-Chroma Primary-Edit Lift Source Candidate Pass

## Decision

`PASS`. The source-chroma primary-edit lift source candidate is a materially
different train-only successor after the missing selector-replay candidate PNGs,
the source-dark local-paper lift, and the source-dark thin-component lift all
failed to open a safe path.

The causal change is target-free during generation: it lifts only source pixels
that are high-chroma, were already brightened by current-primary versus the
original source, and still remain below local baseline paper tone. This uses
current-primary's existing edit as a permission signal instead of broad
source-dark lifting or thin-component selection.

## Evidence

- Proposal/result JSON:
  `docs/source-chroma-primary-edit-lift-source-candidate-v1.json`
- Proposal/result JSON SHA256:
  `08dc868229580b3a6699d129db415aed761091d755affd60cb5feafd784c484d`
- Materializer:
  `scripts/analysis/materialize_source_chroma_primary_edit_candidate.py`
- Materializer SHA256:
  `a26f0a7df22f799f804307cc9713c4389dde16f40741c827f521eaa08db7d4bd`
- Materializer test:
  `tests/test_materialize_source_chroma_primary_edit_candidate.py`
- Materializer test SHA256:
  `32c4b118b71ff0707ddc79e7986d9e5b7b44b1977e5d565e4818ea34952b66d8`
- Source-candidate summary:
  `outputs/source-chroma-primary-edit-candidate-train160-20260816-v2/summary.json`
- Source-candidate summary SHA256:
  `336b3a95d13279f05288cd58d1eca4b1217814520aff6f8d630c09eb1ad9a1ab`
- Source-candidate diagnostics:
  `outputs/source-chroma-primary-edit-candidate-train160-20260816-v2/diagnostics.csv`
- Source-candidate diagnostics SHA256:
  `384db2d8fcaad5ebe8a4b2b3a00c68299372b1473761985883892249746e0c4f`
- Source-candidate review rows:
  `outputs/source-chroma-primary-edit-candidate-train160-20260816-v2/review_rows.csv`
- Source-candidate review rows SHA256:
  `13c9e9ecbdd6a0ad00d226f00bd8f0db753bb3ee7a4c1d09fcd7f94d2a75350d`
- Stroke-only preflight summary:
  `outputs/stroke-only-patch-suppression-chroma-primary-edit-source-preflight-20260816-v1/summary.json`
- Stroke-only preflight summary SHA256:
  `9062271eabc0619fec8bc9a33b68e439fa68a919e67465811b52d5dddfa0d68e`
- Stroke-only preflight diagnostics:
  `outputs/stroke-only-patch-suppression-chroma-primary-edit-source-preflight-20260816-v1/diagnostics.csv`
- Stroke-only preflight diagnostics SHA256:
  `71d301a29a46fad9fb1180043678a562b69e09227e4c55404be8cfecef780b30`
- Stroke-only preflight review rows:
  `outputs/stroke-only-patch-suppression-chroma-primary-edit-source-preflight-20260816-v1/review_rows.csv`
- Stroke-only preflight review rows SHA256:
  `2813a1c6da5c0de850da077910faae98e75061e0d85111c4eae611b2cd7bdb75`
- Stroke-only metric summary:
  `outputs/stroke-only-patch-suppression-chroma-primary-edit-source-preflight-20260816-v1/metric_summary.json`
- Stroke-only metric summary SHA256:
  `8a961da23e037f802fae9ba840c70178eec811ee509d118dc6393b5d0684bacc`

## Train-Only Metric Result

Baseline on the three selected train160 rows was residual
`0.2421483934598291` and overerase `0.0025530178200578405`.

| Output | Residual Gain | Overerase Delta | Min Page Residual Gain | Max Page Overerase Delta |
| --- | ---: | ---: | ---: | ---: |
| source candidate | 0.0009158454912584815 | 0.0 | 0.0008349190322705646 | 0.0 |
| stroke strict | 0.0 | 0.0 | 0.0 | 0.0 |
| stroke balanced | 0.0001699201824825932 | 0.0 | 0.00010783298826777532 | 0.0 |
| stroke soft | 0.00023427296910196368 | 0.0 | 0.00019409937888198447 | 0.0 |

The source candidate clears the train-only preserve-first screen: every selected
row improves residual and no row increases overerase. The registered stroke-only
soft and balanced outputs preserve the no-overerase property with smaller but
positive residual gains; stroke strict is an exact no-op on this candidate.

## Boundary

Candidate generation used only source images and current-primary baseline
predictions. Targets were read only after candidate files were written for
train-only scoring. No model training, checkpoint generation, validation
candidate inference, inner-val15, development gate, SCUT115, holdout40,
reserved blind, visual review, promotion, or `artifacts/current-primary`
replacement occurred.

The only authorized successor is one leakage-safe `inner_val15` gate for the
fixed `source_chroma_primary_edit_lift_v1` candidate and the registered
`stroke_soft` output. Development gates, SCUT115, holdout40, visual review,
reserved blind, promotion, and current-primary replacement remain closed.

Intent: Advance the first available train-only source candidate that improves residual without overerase regression.
Constraint: The gate order still requires inner-val15 before any development, SCUT115, holdout40, reserved-blind, promotion, or current-primary replacement action.
Rejected: Tune saturation, primary-edit floor, alpha, lift cap, component size, kernel, or stroke-only variant thresholds | the fixed train-only PASS should move to the next gate or fail there, not become another parameter chase.
Rejected: Treat the train-only PASS as promotion evidence | it covers only three train160 source rows and does not prove validation or blind generalization.
Confidence: medium
Scope-risk: narrow
Directive: Run exactly one leakage-safe inner-val15 gate next; if it misses the calibrated residual/no-regression thresholds, KILL this family rather than tuning the source-chroma parameters.
Tested: py313 materializer tests 3/3 with warnings as errors.
Tested: py313 stroke-only tests 3/3 with warnings as errors.
Tested: py313 py_compile for materializer and tests.
Tested: train-only source-candidate materialization and metric scoring on rows 166, 190, and 192.
Tested: registered stroke-only strict/balanced/soft materialization and train-only metric scoring on rows 166, 190, and 192.
Not-tested: inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-16-source-dark-local-paper-lift-source-candidate-kill.md
Related: docs/decisions/2026-08-16-source-dark-thin-component-lift-source-candidate-kill.md
