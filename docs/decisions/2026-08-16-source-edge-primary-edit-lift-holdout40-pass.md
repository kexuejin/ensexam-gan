# Source-Edge Primary-Edit Lift Holdout40 Pass

## Decision

`PASS`. The fixed `source_edge_primary_edit_lift_v1` source candidate passed
the holdout40 metric gate after passing train160, `inner_val15`, and Dev40.

This is a deterministic source/baseline postprocess gate. No checkpoint,
selector, threshold sweep, page-specific replacement, or visual acceptance was
introduced.

## Evidence

- Holdout40 result JSON:
  `docs/source-edge-primary-edit-lift-holdout40-gate-v1.json`
- Holdout40 result JSON SHA256:
  `27789b005a79cfd48490dd864c17b8986cbbd63d77191340494c058b991992d2`
- Holdout40 input rows:
  `outputs/source-edge-primary-edit-holdout40-gate-20260816-v1/holdout40_review_rows.csv`
- Holdout40 input rows SHA256:
  `996f5dd22372cf7e7319bbf65c555ed83510afcf46f3080d6f79d091522c828e`
- Holdout40 source summary:
  `outputs/source-edge-primary-edit-holdout40-gate-20260816-v1/source_candidate/summary.json`
- Holdout40 source summary SHA256:
  `64e5f45485f0ccfe84501c33f513c23a97fd054fd445cfd69c78e71f9832a268`
- Holdout40 source diagnostics:
  `outputs/source-edge-primary-edit-holdout40-gate-20260816-v1/source_candidate/diagnostics.csv`
- Holdout40 source diagnostics SHA256:
  `0b106e1e3845ccfdc1673cde5874339d31ff244d7350a44fbc941c0ca88dd407`
- Holdout40 source review rows:
  `outputs/source-edge-primary-edit-holdout40-gate-20260816-v1/source_candidate/review_rows.csv`
- Holdout40 source review rows SHA256:
  `d821cea3f4d2806c50a0853f0bdcd096c12f18aa8554811f3db480d75c3d2016`
- Holdout40 baseline metrics:
  `outputs/source-edge-primary-edit-holdout40-gate-20260816-v1/baseline_post_freeze_metrics.csv`
- Holdout40 baseline metrics SHA256:
  `04d89ca79b889e930adb42fa9061d3147caf92e79723911e97406c42919321eb`
- Holdout40 candidate metrics:
  `outputs/source-edge-primary-edit-holdout40-gate-20260816-v1/source_candidate/post_freeze_metrics.csv`
- Holdout40 candidate metrics SHA256:
  `838c8ad7046b4072f51609ab219e172f1e259baaee801799716a24f009cf64c9`
- Holdout40 gate JSON:
  `outputs/source-edge-primary-edit-holdout40-gate-20260816-v1/holdout40_metric_gate.json`
- Holdout40 gate JSON SHA256:
  `0a80f47fd246f1157753e998a33df499599e56e02d7b52b4605f69dbf407305f`

## Gate Result

| Metric | Baseline | Candidate | Result |
| --- | ---: | ---: | --- |
| Mean residual ratio | 0.13611082197479357 | 0.13487801403380395 | improves |
| Mean overerase ratio | 0.00248161303023898 | 0.00248161303023898 | does not regress |
| P95 residual ratio | 0.30574837989847886 | 0.303715157871504 | does not regress |
| P95 overerase ratio | 0.006366900583758761 | 0.006366900583758761 | does not regress |
| Max residual ratio | 0.4634964170207369 | 0.46216469021497875 | does not regress |
| Max overerase ratio | 0.010721820097559944 | 0.010721820097559944 | does not regress |

Page-level checks also passed: zero residual-regression pages and zero
overerase-regression pages across all 40 holdout40 samples.

## Boundary

The candidate used the existing `artifacts/current-holdout40-primary-pred`
baseline prediction surface and generated one fixed source-edge postprocess
candidate. Targets were read only after candidate files were written for
scoring. SCUT115, reserved blind, visual review, promotion, and
`artifacts/current-primary` replacement remain closed.

Intent: Preserve the fixed source-edge candidate after it generalized on holdout40 without page regressions.
Constraint: SCUT115 still lacks the current-primary prediction PNG surface needed to generate the same deterministic postprocess candidate.
Rejected: Tune source-edge threshold, alpha, component bounds, kernel, or page selection from holdout40 | holdout40 is a held-out gate, not a selector.
Rejected: Advance to reserved blind or promotion from holdout40 alone | SCUT115 remains incomplete.
Confidence: high
Scope-risk: moderate
Directive: Restore or rebuild SCUT115 current-primary baseline predictions, then run the same fixed source candidate once; do not change parameters.
Tested: deterministic holdout40 materialization over 40 frozen samples.
Tested: fixed heldout metric gate accepted all aggregate, tail, and page-level checks.
Not-tested: SCUT115, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-16-source-edge-primary-edit-lift-dev40-pass.md
