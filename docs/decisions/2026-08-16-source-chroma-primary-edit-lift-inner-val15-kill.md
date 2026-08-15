# Source-Chroma Primary-Edit Lift Inner-Val15 Kill

## Decision

`KILL`. The fixed `source_chroma_primary_edit_lift_v1` candidate passed the
train-only preserve-first screen, but failed the first leakage-safe
`inner_val15` gate.

The failure is not overerase: both the source candidate and registered
`stroke_soft` output kept mean, max, p95, and page-level overerase unchanged.
The failure is insufficient generalizing residual lift. `stroke_soft` produced
only `0.000013328486620976104` mean residual gain versus the calibrated
`0.0005` minimum and introduced page-level residual regressions on four of the
15 frozen pages.

## Evidence

- Inner-val15 input rows:
  `outputs/source-chroma-primary-edit-inner-val15-gate-20260816-v1/inner_val15_review_rows.csv`
- Inner-val15 input rows SHA256:
  `927ea364709c6da2d3c7a7efb67d519662084b753feab475e209f568c93dca21`
- Source-candidate summary:
  `outputs/source-chroma-primary-edit-inner-val15-gate-20260816-v1/source_candidate/summary.json`
- Source-candidate summary SHA256:
  `28f18ffaee6661a9e2f34712909f85a66e32bf6b7dbedf7ab76882b51b12d7f4`
- Source-candidate diagnostics:
  `outputs/source-chroma-primary-edit-inner-val15-gate-20260816-v1/source_candidate/diagnostics.csv`
- Source-candidate diagnostics SHA256:
  `469145a5e9f8af4bc10c3e251952ae8dc90b2eaf9211dd76a6f0c8655034cd0f`
- Source-candidate review rows:
  `outputs/source-chroma-primary-edit-inner-val15-gate-20260816-v1/source_candidate/review_rows.csv`
- Source-candidate review rows SHA256:
  `dd890794a16d2eec629d29ce160c797f851742dc59b30e5d7d1734721a1e3fd9`
- Stroke-only summary:
  `outputs/source-chroma-primary-edit-inner-val15-gate-20260816-v1/stroke_only/summary.json`
- Stroke-only summary SHA256:
  `b315738a6b3a74f4273508a102beefd07ed4ffd18eeebd06e4d3ab1b1ba944c3`
- Stroke-only diagnostics:
  `outputs/source-chroma-primary-edit-inner-val15-gate-20260816-v1/stroke_only/diagnostics.csv`
- Stroke-only diagnostics SHA256:
  `9e5c025545a53bd9cb0182e42b28a98b6ffc85732b3574eb17fb83b55b16e207`
- Stroke-only review rows:
  `outputs/source-chroma-primary-edit-inner-val15-gate-20260816-v1/stroke_only/review_rows.csv`
- Stroke-only review rows SHA256:
  `9b71f70342648617e27481c4d18dea85d6c0f414540d8cb7dada9b549257e8d2`
- Inner-val15 metric summary:
  `outputs/source-chroma-primary-edit-inner-val15-gate-20260816-v1/inner_val15_metric_summary.json`
- Inner-val15 metric summary SHA256:
  `457f86e8b3c8da1e47c9c4f8f9086fde60f1b0471f2a40e6a413b6f55f5ba2f8`

## Gate Result

| Output | Residual Gain | Overerase Delta | Min Page Residual Gain | Max Page Overerase Delta |
| --- | ---: | ---: | ---: | ---: |
| source candidate | 0.0001606724783656197 | 0.0 | -0.00011430748714041927 | 0.0 |
| stroke soft | 0.000013328486620976104 | 0.0 | -0.00007620499142690917 | 0.0 |

`stroke_soft` passed overerase preservation, max residual, p95 residual, and
p95 overerase non-regression, but failed the calibrated residual-gain floor and
the no-page-residual-regression rule.

Residual-regression pages for `stroke_soft`:

- `250.jpg`
- `51.jpg`
- `517.jpg`
- `281.jpg`

## Boundary

The gate used the frozen `inner_val15` list and already-frozen current-primary
baseline predictions. Candidate generation was source-only before labels were
read for scoring. No development gate, SCUT115, holdout40, visual review,
reserved blind, promotion, or `artifacts/current-primary` replacement occurred.

Intent: Close the first train-only-passing source-candidate route when it failed to generalize through the calibrated inner-val15 gate.
Constraint: The registered gate requires at least 0.0005 mean residual gain and no page-level residual or overerase regressions before any development or held-out surface opens.
Rejected: Tune saturation, primary-edit floor, alpha, lift cap, component area, median kernel, or stroke-only blend | the family already had one authorized train-only PASS and one authorized inner-val15 gate; further tuning would become parameter search after validation feedback.
Rejected: Advance to development gate, SCUT115, holdout40, visual review, reserved blind, or promotion | inner-val15 failed.
Confidence: high
Scope-risk: narrow
Directive: Do not repeat or rescue `source_chroma_primary_edit_lift_v1`; the next route must be a materially different train-only source candidate or a formal exhaustion record for the bucket.
Tested: fixed source candidate materialized on 15 frozen inner_val15 pages from source plus current-primary baseline before scoring.
Tested: registered `stroke_soft` output materialized on the same 15 pages.
Tested: inner-val15 metric summary checked calibrated residual gain, aggregate/tail non-regression, and page-level regressions.
Not-tested: development gates, SCUT115, holdout40, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-16-source-chroma-primary-edit-lift-source-candidate-pass.md
