# External Text Layout Support Diagnostic Pass

## Decision

`PASS`. The recovered external text-layout support cache is now consumed by the
registered train-only separability diagnostic using the effective recovered-input
plan. This establishes that frozen PP-OCRv6 text occupancy and confidence add
target-aligned support beyond recovered second-stage RGB on train275.

## Evidence

- Audit output: `outputs/external-text-layout-support-prerequisite-20260813/audit.json`
- Audit SHA256: `b0e77e816afab4458f4064dd1431f24649b73af4ed04a3a483ff992f77980734`
- Effective plan: `outputs/external-text-layout-recovered-materializer-input-20260815/effective-plan.json`
- Effective plan SHA256: `39d5d801c0507dc965c970927b2d2ea6a2e7d9a2f3f04b27564956093fbca5d4`
- Materialization manifest SHA256: `2578fb78cc3188776d97fff2e1efd2fc8f91d224b4efb680ffa83e1847c96579`

## Results

| Metric | Result | Gate |
| --- | ---: | ---: |
| Full mean fold AUC | `0.6879765165680307` | `>= 0.65` |
| Minimum fold AUC | `0.6478868392944336` | `>= 0.55` |
| Second-stage RGB ablation mean AUC | `0.644506096958718` | reference |
| Full minus ablation mean AUC | `0.04347041960931275` | `>= 0.03` |
| Macro median page AUC | `0.7221097946166992` | `>= 0.60` |
| Positive mean above preserve folds | `5 / 5` | `>= 4 / 5` |

Fold AUCs were `0.6820619621394594`, `0.7018136804483436`,
`0.7227086824440427`, `0.6478868392944336`, and `0.6854114185138747`.
Fold page counts were `54`, `62`, `47`, `50`, and `62`.

## Boundary

This is train-role support evidence only. It did not start training, candidate
inference, SCUT115, holdout40, visual review, reserved blind access, promotion,
or current-primary replacement. The upstream detector corpus overlap with SCUT
or HW5K is still unverified, so the next authorized step is only a separate
leakage-aware text-layout-conditioned data/training/application preflight.

Intent: Open the external text-layout conditioned preflight only after the recovered support cache proves incremental train-role signal.
Constraint: The original support plan remains unchanged; the effective plan changes only the recovered second-stage metrics SHA256 from the historical missing-payload identity to the validated recovered identity.
Rejected: Treat this PASS as generalization evidence | upstream detector corpus overlap with SCUT/HW5K is unverified.
Rejected: Rerun materialization or tune detector thresholds | schema-v10 materialization already passed and threshold rescue is prohibited.
Confidence: high
Scope-risk: moderate
Directive: Do not open SCUT115 or holdout40 directly from this diagnostic; first freeze a leakage-aware conditioned preflight with explicit training and application boundaries.
Tested: py313 audit_external_text_layout_support.py with recovered effective plan, terminal PASS, 275 pages, 563200 sampled pixels.
Not-tested: candidate training, candidate inference, SCUT115, holdout40, visual review, reserved blind, promotion.
