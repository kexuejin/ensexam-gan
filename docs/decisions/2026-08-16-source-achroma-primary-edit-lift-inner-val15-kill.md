# Source-Achroma Primary-Edit Lift Inner-Val15 Kill

## Decision

`KILL`. The fixed `source_achroma_primary_edit_lift_v1` route passed the
train-only preserve-first source-candidate screen, but failed the first
leakage-safe `inner_val15` gate.

This candidate was materially different from the prior source-chroma route: it
targeted low-saturation residual cores that current-primary had already
brightened, rather than high-saturation source marks. It also avoided the killed
source-dark local-paper and thin-component selectors by requiring existing
primary edit as the permission signal.

## Evidence

- Proposal/result JSON:
  `docs/source-achroma-primary-edit-lift-source-candidate-v1.json`
- Proposal/result JSON SHA256:
  `4829b06aaacb41b5c62b28cb1341b1a397037325b7b53dabd8c29878a25c620e`
- Materializer:
  `scripts/analysis/materialize_source_achroma_primary_edit_candidate.py`
- Materializer SHA256:
  `9e69663a1fcbc3efe5766d7c852eb034e26ec3979a0be349d1124b8547d0bae6`
- Materializer test:
  `tests/test_materialize_source_achroma_primary_edit_candidate.py`
- Materializer test SHA256:
  `8f7e260c8a1399ce5888302e00b1ee41bfcbcc8c945cb216e1d7ada526fc490b`
- Train-only source summary:
  `outputs/source-achroma-primary-edit-candidate-train160-20260816-v1/summary.json`
- Train-only source summary SHA256:
  `2851a8421ce8eb743e0187fb199cd1a69626b8d868f4c4b5a03c3a6a4fe2ef27`
- Train-only source diagnostics:
  `outputs/source-achroma-primary-edit-candidate-train160-20260816-v1/diagnostics.csv`
- Train-only source diagnostics SHA256:
  `d28a2e61e4f3ce1e9f5ec09dc6f86949dce906830e238a227520a724a3cfffb7`
- Train-only source review rows:
  `outputs/source-achroma-primary-edit-candidate-train160-20260816-v1/review_rows.csv`
- Train-only source review rows SHA256:
  `546d784f84951c6a2c4b24ed2a75eb7cdf8996593dd616a828511e3caaa08de9`
- Train-only stroke-only summary:
  `outputs/stroke-only-patch-suppression-achroma-primary-edit-source-preflight-20260816-v1/summary.json`
- Train-only stroke-only summary SHA256:
  `23ba33536f3654540d3a722d32c30886139e56350397b0e52bef18627be0c112`
- Train-only stroke-only diagnostics:
  `outputs/stroke-only-patch-suppression-achroma-primary-edit-source-preflight-20260816-v1/diagnostics.csv`
- Train-only stroke-only diagnostics SHA256:
  `1f87cb891ec8ac9b7175cf730a20bda90129f51f0c77f510e5ce1534fb3fca53`
- Train-only stroke-only review rows:
  `outputs/stroke-only-patch-suppression-achroma-primary-edit-source-preflight-20260816-v1/review_rows.csv`
- Train-only stroke-only review rows SHA256:
  `b663662f7705867c7cb1484313932f796e408d950c1f494d3922d8ed5102f031`
- Train-only stroke-only metric summary:
  `outputs/stroke-only-patch-suppression-achroma-primary-edit-source-preflight-20260816-v1/metric_summary.json`
- Train-only stroke-only metric summary SHA256:
  `5ca66109e181f9ac3f4c303a22592b1e68d8591c720795ad024150d85edfeb04`
- Inner-val15 input rows:
  `outputs/source-achroma-primary-edit-inner-val15-gate-20260816-v1/inner_val15_review_rows.csv`
- Inner-val15 input rows SHA256:
  `3bce3bf39f231230c5341343fc1d4466eea4b34433cacfc066606179eaf6ef1b`
- Inner-val15 source summary:
  `outputs/source-achroma-primary-edit-inner-val15-gate-20260816-v1/source_candidate/summary.json`
- Inner-val15 source summary SHA256:
  `97ed17b3b717d6861c31530181185fd3abcafa3e9936e7b41b50ae4951d7f910`
- Inner-val15 source diagnostics:
  `outputs/source-achroma-primary-edit-inner-val15-gate-20260816-v1/source_candidate/diagnostics.csv`
- Inner-val15 source diagnostics SHA256:
  `a5639ece583d780530356f592311037932c4955b458ef366af3ead9f45c706c7`
- Inner-val15 source review rows:
  `outputs/source-achroma-primary-edit-inner-val15-gate-20260816-v1/source_candidate/review_rows.csv`
- Inner-val15 source review rows SHA256:
  `3671ab32614f1faa5c153d7b4d1453c69933875534913353ce69a7456e9f7f2c`
- Inner-val15 stroke-only summary:
  `outputs/source-achroma-primary-edit-inner-val15-gate-20260816-v1/stroke_only/summary.json`
- Inner-val15 stroke-only summary SHA256:
  `e3d30e69e76f6ae732de190fd5acdc0d4392f18b12313bc035468140f603215f`
- Inner-val15 stroke-only diagnostics:
  `outputs/source-achroma-primary-edit-inner-val15-gate-20260816-v1/stroke_only/diagnostics.csv`
- Inner-val15 stroke-only diagnostics SHA256:
  `110c71c56f571ffefa6fdb2bd397bb301fc5ba551639c8a8aa1f630bf8ef5b08`
- Inner-val15 stroke-only review rows:
  `outputs/source-achroma-primary-edit-inner-val15-gate-20260816-v1/stroke_only/review_rows.csv`
- Inner-val15 stroke-only review rows SHA256:
  `2bb40b62010ccab0f694fcf5ce7ade3b556a71a3c147fc741bec44f4d358a9b7`
- Inner-val15 metric summary:
  `outputs/source-achroma-primary-edit-inner-val15-gate-20260816-v1/inner_val15_metric_summary.json`
- Inner-val15 metric summary SHA256:
  `0f0483f75940b531afdca223485ef279e54845a863cedbb7bbbdfc7ffb412788`

## Train-Only Result

Baseline on the three selected train160 rows was residual
`0.2421483934598291` and overerase `0.0025530178200578405`.

| Output | Residual Gain | Overerase Delta | Min Page Residual Gain | Max Page Overerase Delta |
| --- | ---: | ---: | ---: | ---: |
| source candidate | 0.001383257235171668 | 0.0 | 0.0003019323671497598 | 0.0 |
| stroke soft | 0.00048300826841829 | 0.0 | 0.00017253278122843219 | 0.0 |
| stroke balanced | 0.00036349478131109114 | 0.0 | 0.00010783298826777532 | 0.0 |
| stroke strict | 0.0 | 0.0 | 0.0 | 0.0 |

The train-only source candidate passed preserve-first screening: all three rows
improved residual and no row increased overerase. The registered `stroke_soft`
output retained positive residual gain with zero page-level overerase delta, so
one frozen inner-val15 check was justified.

## Inner-Val15 Result

| Output | Residual Gain | Overerase Delta | Min Page Residual Gain | Max Page Overerase Delta | Terminal |
| --- | ---: | ---: | ---: | ---: | --- |
| source candidate | 0.0018780111460169327 | 0.0 | -0.000013359577481095175 | 0.0 | KILL |
| stroke soft | 0.0004868933442909539 | 0.0 | -0.000008906384987399096 | 0.0 | KILL |

The source candidate passed the calibrated mean residual-gain floor and all
aggregate/tail overerase checks, but failed the no-page-residual-regression
rule on `506.jpg`. `stroke_soft` also regressed `506.jpg` and missed the
`0.0005` calibrated residual-gain floor by `0.000013106655709046143`.

## Boundary

Candidate generation used only source images and current-primary baseline
predictions. Targets were read only after candidate files were written for
train-only and inner-val15 scoring. No development gate, SCUT115, holdout40,
visual review, reserved blind, promotion, or `artifacts/current-primary`
replacement occurred.

Intent: Close a low-saturation primary-edit successor after it failed the first calibrated no-page-regression gate.
Constraint: Inner-val15 requires measurable residual lift with no page-level residual or overerase regression before any development or held-out surface opens.
Rejected: Tune saturation, primary-edit floor, alpha, lift cap, component bounds, median kernel, or stroke-only blend | the fixed family already consumed its authorized train-only and inner-val15 evaluations.
Rejected: Advance to development gate, SCUT115, holdout40, visual review, reserved blind, or promotion | both eligible outputs failed inner-val15.
Confidence: high
Scope-risk: narrow
Directive: Do not repeat or rescue `source_achroma_primary_edit_lift_v1`; the next route must be materially different or explicitly record bucket exhaustion.
Tested: py313 materializer tests 3/3 with warnings as errors.
Tested: py313 stroke-only tests 3/3 with warnings as errors.
Tested: py313 py_compile for materializer and tests.
Tested: train-only materialization and scoring for source candidate plus registered stroke-only variants.
Tested: one frozen inner-val15 gate for the fixed source candidate and registered `stroke_soft` output.
Not-tested: development gates, SCUT115, holdout40, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-16-source-chroma-primary-edit-lift-inner-val15-kill.md
