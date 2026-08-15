# Source-Edge Primary-Edit Lift SCUT115 Kill

## Decision

`KILL`. The fixed `source_edge_primary_edit_lift_v1` source candidate passed
train160, `inner_val15`, Dev40, and holdout40, but failed SCUT115 on the
required no-page-residual-regression rule.

The candidate improved SCUT115 aggregate and tail residual metrics and kept
overerase flat, but one page, `179.jpg`, regressed residual. This closes the
family without threshold, alpha, component, kernel, page-specific, visual, or
held-out rescue.

## Evidence

- SCUT115 result JSON:
  `docs/source-edge-primary-edit-lift-scut115-gate-v1.json`
- SCUT115 result JSON SHA256:
  `bc77de27c2d1bd162111ecebf3d16c17b155c059721c37ce475b55e2f73c5803`
- Rebuilt source-only baseline metrics:
  `outputs/source-edge-primary-edit-scut115-baseline-recovery-20260816-v2/frozen_predictions/metrics.csv`
- Rebuilt source-only baseline metrics SHA256:
  `5ebdb4705e206fe52b38fd43c2165196b44a13123c5de1a17cf7a5763bea023b`
- SCUT115 input rows:
  `outputs/source-edge-primary-edit-scut115-gate-20260816-v2/scut115_review_rows.csv`
- SCUT115 input rows SHA256:
  `e4cfa6df1b982fa5d4de47c6bb57706684973964d03894436cedce7d8911f96d`
- SCUT115 source summary:
  `outputs/source-edge-primary-edit-scut115-gate-20260816-v2/source_candidate/summary.json`
- SCUT115 source summary SHA256:
  `40a106c3c5115d968df36005017e4e3300b2e495683361933ab4a5273fcbd29e`
- SCUT115 source diagnostics:
  `outputs/source-edge-primary-edit-scut115-gate-20260816-v2/source_candidate/diagnostics.csv`
- SCUT115 source diagnostics SHA256:
  `4352719dbeb10c3d9964ea91627f8f6f7c1cded011fa25b90262b6247f377bea`
- SCUT115 source review rows:
  `outputs/source-edge-primary-edit-scut115-gate-20260816-v2/source_candidate/review_rows.csv`
- SCUT115 source review rows SHA256:
  `43a43b421034ed05115979e40c5ecf6d1488c21a5b18a0fe5d4b64c4bbf27636`
- SCUT115 baseline post-freeze metrics:
  `outputs/source-edge-primary-edit-scut115-gate-20260816-v2/baseline_post_freeze_metrics.csv`
- SCUT115 baseline post-freeze metrics SHA256:
  `e6ba2d693dce3e822557e3ef7f900ecc26c98e2701eac06405594a9ca547e851`
- SCUT115 candidate post-freeze metrics:
  `outputs/source-edge-primary-edit-scut115-gate-20260816-v2/source_candidate/post_freeze_metrics.csv`
- SCUT115 candidate post-freeze metrics SHA256:
  `e3c386d40ce3341b977858a756ee3812978468949b97708c93faadd4e85f00de`
- SCUT115 gate JSON:
  `outputs/source-edge-primary-edit-scut115-gate-20260816-v2/scut115_metric_gate.json`
- SCUT115 gate JSON SHA256:
  `fc34f85570a412f99823db64a4d368c557459afa513b50532155b3a45565a397`

## Baseline Recovery

The local SCUT115 current-primary baseline directories contained metrics but no
prediction PNGs, so the current-primary baseline surface was rebuilt with
source-only CPU inference:

- Source list: `docs/scut-test115-relative.txt`
- Prediction count: 115 PNG files
- Device: CPU
- Labels read during recovery: false
- Current-primary config: `artifacts/current-primary/config.yaml`
- Current-primary weights: `artifacts/current-primary/micro_region_probe_step0001.pth`

The rebuilt prediction filenames exactly matched the 115-page SCUT115 source
manifest before candidate generation or label scoring.

## Gate Result

| Metric | Baseline | Candidate | Result |
| --- | ---: | ---: | --- |
| Mean residual ratio | 0.11813039572882839 | 0.11674289306173857 | improves |
| Mean overerase ratio | 0.0030891541790281146 | 0.0030891541790281146 | does not regress |
| P95 residual ratio | 0.33793644579667487 | 0.3367818610042459 | does not regress |
| P95 overerase ratio | 0.007372123974219172 | 0.007372123974219172 | does not regress |
| Max residual ratio | 0.5958076064961345 | 0.5886473304623048 | does not regress |
| Max overerase ratio | 0.010381593714927048 | 0.010381593714927048 | does not regress |

Failed check:

- `no_page_residual_regressions`: 1 page, `179.jpg`

## Boundary

SCUT115 was run exactly once for the fixed source-edge candidate after
holdout40 had already passed. No reserved blind, visual review, promotion, or
`artifacts/current-primary` replacement occurred.

Intent: Close the source-edge primary-edit route after held-out page-level residual regression.
Constraint: The quality loop requires no page-level residual or overerase regression before reserved blind or promotion.
Rejected: Tune source-edge threshold, primary-edit floor, alpha, lift cap, component bounds, or median kernel | SCUT115 is a held-out gate, not a parameter-selection surface.
Rejected: Accept aggregate residual lift despite `179.jpg` regression | the no-page-regression rule is mandatory for promotion-safe generalization.
Rejected: Advance to reserved blind, visual review, promotion, or current-primary replacement | SCUT115 failed.
Confidence: high
Scope-risk: narrow
Directive: Do not repeat or rescue `source_edge_primary_edit_lift_v1`; the next route must be materially different or explicitly record bucket exhaustion.
Tested: source-only CPU baseline reconstruction over all 115 SCUT115 samples.
Tested: deterministic SCUT115 materialization for the fixed source-edge candidate.
Tested: fixed heldout metric gate rejected on page-level residual regression only.
Not-tested: reserved blind, visual review, promotion.
Related: docs/decisions/2026-08-16-source-edge-primary-edit-lift-holdout40-pass.md
