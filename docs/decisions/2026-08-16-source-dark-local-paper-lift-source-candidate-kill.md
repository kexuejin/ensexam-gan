# Source-Dark Local-Paper Lift Source Candidate Kill

## Decision

`KILL`. The source-dark local-paper lift source candidate solved the immediate
input custody problem without using the missing relaxed-interval PNGs, but it
failed the train-only metric preflight before any validation surface opened.

The causal change was target-free: generate a train160 source candidate from
source image plus current-primary baseline only, lifting source-dark local
residual pixels toward local paper tone, then feed that candidate into the
registered stroke-only suppression variants. The source-candidate custody audit
passed with 12/12 required paths present, but all measured outputs produced
large overerase regressions.

## Evidence

- Proposal/result JSON:
  `docs/source-dark-local-paper-lift-source-candidate-v1.json`
- Proposal/result JSON SHA256:
  `2636ae6b77873828ac154b2ba703a538778e2de8d39f5e0ad6687e8596385226`
- Materializer:
  `scripts/analysis/materialize_source_dark_local_paper_candidate.py`
- Materializer SHA256:
  `3cb7760772c6ceae25405e3a1f18cbf38867bf6595825a2946e31ea3148026b5`
- Materializer test:
  `tests/test_materialize_source_dark_local_paper_candidate.py`
- Materializer test SHA256:
  `28ee33403865e1fb5862c70de40419032ad8dab4e4da81b54a1ff475769bd36a`
- Source-candidate summary:
  `outputs/source-dark-local-paper-candidate-train160-20260816/summary.json`
- Source-candidate summary SHA256:
  `a4b63a9841913768e36abb6f51016c61a2e9d812282d865cd5ae81a0e4b951f9`
- Source-candidate review rows:
  `outputs/source-dark-local-paper-candidate-train160-20260816/review_rows.csv`
- Source-candidate review rows SHA256:
  `e657b24f2484076e1b5a567113088e188709424b50d643d46e249e2e03b93874`
- Source-candidate selector rows:
  `outputs/source-dark-local-paper-candidate-train160-20260816/selector_replay_rows.csv`
- Source-candidate selector rows SHA256:
  `0112e8a827b6da4cbf62778def6120b294d9b8797056c112421de9e67ccb9d0e`
- Source-candidate diagnostics:
  `outputs/source-dark-local-paper-candidate-train160-20260816/diagnostics.csv`
- Source-candidate diagnostics SHA256:
  `714f71a4aff071eeb0ef539e68c3aa3222ca0329d4fbc7551c07b7646efbeba7`
- Source-candidate metric CSV:
  `outputs/source-dark-local-paper-candidate-train160-20260816/metrics.csv`
- Source-candidate metric CSV SHA256:
  `22feecfa7f0ec60eb4e606869d9a46cda8120a2aed3a148b4cec9c8f9496e586`
- Source-candidate custody audit:
  `outputs/stroke-only-patch-suppression-input-custody-audit-local-paper-source-20260816/audit.json`
- Source-candidate custody audit SHA256:
  `0cdd629ff440947dd1d58076bd564a545a9bd81cc6d126ee59ee7fb3cf4f85d4`
- Stroke-only preflight summary:
  `outputs/stroke-only-patch-suppression-local-paper-source-preflight-20260816/summary.json`
- Stroke-only preflight summary SHA256:
  `8e8e7230d89f63622442e64b95c482dc43ce6cdb84062eb6d80532ac16ba8d5a`
- Stroke-only preflight diagnostics:
  `outputs/stroke-only-patch-suppression-local-paper-source-preflight-20260816/diagnostics.csv`
- Stroke-only preflight diagnostics SHA256:
  `91333da1d7628fc463c16d4bb62eaae66c77d54ff1340b701e8112419950a19c`
- Stroke-only metric summary:
  `outputs/stroke-only-patch-suppression-local-paper-source-preflight-20260816/metric_summary.json`
- Stroke-only metric summary SHA256:
  `1d70db116caa22f5d7cb3d07896f172dde3714dcb3fb2910353b3c7f727624e7`

## Train-Only Metric Result

Baseline on the three selected train160 rows was residual
`0.2421483934598291` and overerase `0.0025530178200578405`.

| Output | Residual | Residual Gain | Overerase | Overerase Delta |
| --- | ---: | ---: | ---: | ---: |
| source candidate | 0.22422737191428913 | 0.017921021545539983 | 0.06528499339744981 | 0.06273197557739196 |
| stroke strict | 0.2273097790932771 | 0.014838614366552011 | 0.05775103461640374 | 0.0551980167963459 |
| stroke balanced | 0.22705216334351175 | 0.01509623011631736 | 0.06527473565454209 | 0.06272171783448424 |
| stroke soft | 0.23092359525537312 | 0.011224798204455988 | 0.06524194613474811 | 0.06268892831469026 |

Residual improves, but overerase is not remotely promotion-safe. The best
overerase variant still raises mean overerase by `0.0551980167963459`, and the
worst page-level overerase delta is `0.05821659331636842`.

## Boundary

No model training, checkpoint generation, candidate inference on validation,
inner-val15, SCUT115, holdout40, reserved blind, visual review, promotion, or
`artifacts/current-primary` replacement occurred. Target images were only used
after train-only materialization for metric scoring.

Intent: Close a materially different train-only source-candidate route instead of repeatedly waiting on missing historical relaxed-interval PNGs.
Constraint: The route must remain target-free during generation and cannot use validation/blind surfaces before passing train-only evidence.
Rejected: Tune alpha, dark thresholds, component limits, dilation, or median kernel | the first fixed causal change causes massive overerase and should not become threshold search.
Rejected: Advance residual-improving variants to inner-val15 | overerase regression violates the preserve-first gate.
Confidence: high
Scope-risk: narrow
Directive: Do not repeat or rescue source_dark_local_paper_lift_v1; choose a materially different source-candidate mechanism or restore the exact historical selector-replay candidate PNGs.
Tested: py313 materializer tests 3/3 with warnings as errors.
Tested: py313 input-custody tests 7/7 with warnings as errors.
Tested: py313 py_compile for materializer, audit script, and tests.
Tested: train-only source-candidate custody audit PASS with 12 present paths and zero missing paths.
Tested: train-only metric preflight for source candidate and all three stroke-only variants.
Not-tested: inner-val15, SCUT115, holdout40, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-16-stroke-only-patch-suppression-selector-path-custody.md
