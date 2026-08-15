# Source-Edge Primary-Edit Lift Inner-Val15 Pass

## Decision

`PASS`. The fixed `source_edge_primary_edit_lift_v1` source candidate passed
the train-only preserve-first screen and the first leakage-safe `inner_val15`
gate. It is now eligible for the next development gate as the fixed source
candidate only.

The registered `stroke_soft` derivative is `KILL` and must not be used as the
promoted output: it missed the calibrated residual-gain floor and regressed
`506.jpg`.

## Evidence

- Proposal/result JSON:
  `docs/source-edge-primary-edit-lift-source-candidate-v1.json`
- Proposal/result JSON SHA256:
  `45e81d50bb2b089e291952e2a86ace6ffed93ba5777f6a6c9c7a00a1ebd89412`
- Materializer:
  `scripts/analysis/materialize_source_edge_primary_edit_candidate.py`
- Materializer SHA256:
  `b598735a5a8cd55ccfdd101e5a3104cd7fc20e1eb1ed6c8868ef0ab69804676d`
- Materializer test:
  `tests/test_materialize_source_edge_primary_edit_candidate.py`
- Materializer test SHA256:
  `bfe558a4307b23edfb705506db75c917cd455cda815a7ba105a4bcff871b0969`
- Train-only source summary:
  `outputs/source-edge-primary-edit-candidate-train160-20260816-v1/summary.json`
- Train-only source summary SHA256:
  `e18312eddec78855e06bbcfbd72fe2ac8671390cb61c4debf105cecdff1440b4`
- Train-only source diagnostics:
  `outputs/source-edge-primary-edit-candidate-train160-20260816-v1/diagnostics.csv`
- Train-only source diagnostics SHA256:
  `94b76ee3ea4c6dd3207085286851b5bb2f13e6d3ed3c5079af154fffa2c72ffd`
- Train-only source review rows:
  `outputs/source-edge-primary-edit-candidate-train160-20260816-v1/review_rows.csv`
- Train-only source review rows SHA256:
  `7d59b59142f485185ad7646e0a1edbfd6517f522fdc9ba355ce14074f3ba4e5d`
- Train-only stroke-only summary:
  `outputs/stroke-only-patch-suppression-edge-primary-edit-source-preflight-20260816-v1/summary.json`
- Train-only stroke-only summary SHA256:
  `b32f967110a8d636270a1d896e61954a42265af0bff24e27196e56224c6e1e8c`
- Train-only stroke-only diagnostics:
  `outputs/stroke-only-patch-suppression-edge-primary-edit-source-preflight-20260816-v1/diagnostics.csv`
- Train-only stroke-only diagnostics SHA256:
  `9821a9c2332e0601513b3b0155a97c9bf0d1de1ebc41a4619f5342c27ab76740`
- Train-only stroke-only review rows:
  `outputs/stroke-only-patch-suppression-edge-primary-edit-source-preflight-20260816-v1/review_rows.csv`
- Train-only stroke-only review rows SHA256:
  `c09c7dc94ee902adaaf102539d6bd942a9e1cda5121d9a253afa4bd70f69fb68`
- Train-only stroke-only metric summary:
  `outputs/stroke-only-patch-suppression-edge-primary-edit-source-preflight-20260816-v1/metric_summary.json`
- Train-only stroke-only metric summary SHA256:
  `b032a51ad69d70c441b0e517af6e0d792902516b4549e7ce61fe66ebef09b8ca`
- Inner-val15 input rows:
  `outputs/source-edge-primary-edit-inner-val15-gate-20260816-v1/inner_val15_review_rows.csv`
- Inner-val15 input rows SHA256:
  `061a7d94e8c0ac9909c8f7a5bc242f198172c5355b724aa84283a3773198be6c`
- Inner-val15 source summary:
  `outputs/source-edge-primary-edit-inner-val15-gate-20260816-v1/source_candidate/summary.json`
- Inner-val15 source summary SHA256:
  `05dceaf795cbf25480f63a14f09909464c917a953efd4e12365aff9b86336f7a`
- Inner-val15 source diagnostics:
  `outputs/source-edge-primary-edit-inner-val15-gate-20260816-v1/source_candidate/diagnostics.csv`
- Inner-val15 source diagnostics SHA256:
  `7b9b3c219f0af3aad1579e6d0a834507824606c08a38f326e201af7bd7172e22`
- Inner-val15 source review rows:
  `outputs/source-edge-primary-edit-inner-val15-gate-20260816-v1/source_candidate/review_rows.csv`
- Inner-val15 source review rows SHA256:
  `9b727bab950bd765521a29392f50bc246bffe6c2a006b82287c4bddff4825a17`
- Inner-val15 stroke-only summary:
  `outputs/source-edge-primary-edit-inner-val15-gate-20260816-v1/stroke_only/summary.json`
- Inner-val15 stroke-only summary SHA256:
  `f4cd3fad10913fc1538d7c834e8bdbcd6343f401296ea4e9a9db8e027ee42f87`
- Inner-val15 stroke-only diagnostics:
  `outputs/source-edge-primary-edit-inner-val15-gate-20260816-v1/stroke_only/diagnostics.csv`
- Inner-val15 stroke-only diagnostics SHA256:
  `1cd9d0366b06244de36b9d2ad8c8287db4da3e6dfd3aa66db2a746111e54aa48`
- Inner-val15 stroke-only review rows:
  `outputs/source-edge-primary-edit-inner-val15-gate-20260816-v1/stroke_only/review_rows.csv`
- Inner-val15 stroke-only review rows SHA256:
  `6994c3ecefe2afd4191e1a0439dbf7fbdd7c07291cb46589ce5149c5dd01826c`
- Inner-val15 metric summary:
  `outputs/source-edge-primary-edit-inner-val15-gate-20260816-v1/inner_val15_metric_summary.json`
- Inner-val15 metric summary SHA256:
  `843517c743d2b0efdfec34d73c3e062657bdc0dcf91c1781456fdbd9da191ac2`

## Train-Only Result

Baseline on the three selected train160 rows was residual
`0.2421483934598291` and overerase `0.0025530178200578405`.

| Output | Residual Gain | Overerase Delta | Min Page Residual Gain | Max Page Overerase Delta |
| --- | ---: | ---: | ---: | ---: |
| source candidate | 0.0021707710248467377 | 0.0 | 0.0013586956521739052 | 0.0 |
| stroke soft | 0.0004963534150312412 | 0.0 | 0.0002803657694962075 | 0.0 |
| stroke balanced | 0.0 | 0.0 | 0.0 | 0.0 |
| stroke strict | 0.0 | 0.0 | 0.0 | 0.0 |

The source candidate passed preserve-first screening: all three rows improved
residual and no row increased overerase. The registered `stroke_soft` output
retained positive residual gain with zero page-level overerase delta, so one
frozen inner-val15 check was justified.

## Inner-Val15 Result

| Output | Residual Gain | Overerase Delta | Min Page Residual Gain | Max Page Overerase Delta | Terminal |
| --- | ---: | ---: | ---: | ---: | --- |
| source candidate | 0.001833141716084386 | 0.0 | 0.00004453192493698854 | 0.0 | PASS |
| stroke soft | 0.0002356696939108958 | 0.0 | -0.000004453192493703018 | 0.0 | KILL |

The fixed source candidate passed every calibrated inner-val15 check: mean
residual gain exceeded `0.0005`, every page improved residual, max and p95
residual were non-regressing, and mean/page/tail overerase stayed flat.

`stroke_soft` failed because it missed the `0.0005` residual-gain floor and
regressed `506.jpg`. Do not substitute it for the fixed source candidate.

## Boundary

Candidate generation used only source images and current-primary baseline
predictions. Targets were read only after candidate files were written for
train-only and inner-val15 scoring. No development gate, SCUT115, holdout40,
visual review, reserved blind, promotion, or `artifacts/current-primary`
replacement occurred.

Intent: Advance a source-edge primary-edit successor after it passed the first calibrated no-page-regression gate.
Constraint: Inner-val15 is the first gate; SCUT115, holdout40, reserved blind, visual review, promotion, and current-primary replacement remain closed until later gates pass.
Rejected: Promote `stroke_soft` instead of the source candidate | it missed the residual floor and regressed 506.jpg.
Rejected: Tune source-edge threshold, primary-edit floor, alpha, lift cap, component bounds, median kernel, or stroke-only blend | the fixed family already consumed its authorized train-only and inner-val15 evaluations.
Confidence: high
Scope-risk: narrow
Directive: Run only the next development gate for the fixed `source_edge_primary_edit_lift_v1` source candidate; do not repeat inner-val15 or tune parameters from this result.
Tested: py313 materializer tests 4/4 with warnings as errors.
Tested: py313 py_compile for materializer and tests.
Tested: train-only materialization and scoring for source candidate plus registered stroke-only variants.
Tested: one frozen inner-val15 gate for the fixed source candidate and registered `stroke_soft` output.
Not-tested: development gates, SCUT115, holdout40, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-16-source-achroma-primary-edit-lift-inner-val15-kill.md
