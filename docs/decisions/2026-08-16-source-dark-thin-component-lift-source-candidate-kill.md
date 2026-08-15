# Source-Dark Thin-Component Lift Source Candidate Kill

## Decision

`KILL`. The source-dark thin-component lift source candidate is materially
different from the earlier local-paper lift, and it is available without the
missing selector-replay PNGs, but it still failed the train-only preserve-first
preflight.

The strict source candidate produced only a `0.00002356978216212317` mean
residual gain while increasing mean overerase by
`0.00011767388035046628`. Wider variants gained more residual but increased
overerase even more. Running the strict source candidate through the registered
stroke-only suppression variants did not recover the family: the soft and
balanced variants reduced overerase somewhat, but both lost residual versus the
baseline.

## Evidence

- Proposal/result JSON:
  `docs/source-dark-thin-component-lift-source-candidate-v1.json`
- Proposal/result JSON SHA256:
  `4af6d2620564a08e3938c8b164ac603ff118e42ed9061093519c3117a0b912d0`
- Materializer:
  `scripts/analysis/materialize_source_dark_thin_component_candidate.py`
- Materializer SHA256:
  `997c5ac660bdd57df0d656597eff13540480217a50b0bb68fabea200a5f302d7`
- Materializer test:
  `tests/test_materialize_source_dark_thin_component_candidate.py`
- Materializer test SHA256:
  `312526fa6536ac6485505c92e8f90f3d66d91b4e922ebbd2bc070271c78ba959`
- Source-candidate summary:
  `outputs/source-dark-thin-component-candidate-train160-20260816-v1/summary.json`
- Source-candidate summary SHA256:
  `e7fabe1e35b4dd8c1de319decb37d41c872d5b54e298d7d2b7daef4db65651fd`
- Source-candidate diagnostics:
  `outputs/source-dark-thin-component-candidate-train160-20260816-v1/diagnostics.csv`
- Source-candidate diagnostics SHA256:
  `57f7ba93f1cae315d3d65f518a9ea62376128ded7b7fb9d86cd33c9c3745fb58`
- Source-candidate review rows:
  `outputs/source-dark-thin-component-candidate-train160-20260816-v1/review_rows.csv`
- Source-candidate review rows SHA256:
  `e7931ae5cb448e006687847e94f1225fa886595e5776fa592d4efa9dfcb6733b`
- Thin-strict review rows:
  `outputs/source-dark-thin-component-candidate-train160-20260816-v1/thin_strict_review_rows.csv`
- Thin-strict review rows SHA256:
  `a77288badbd43c078292279ef45c5794ad2604604350191da816f40a12a80c0e`
- Stroke-only preflight summary:
  `outputs/stroke-only-patch-suppression-thin-strict-source-preflight-20260816/summary.json`
- Stroke-only preflight summary SHA256:
  `ffc1bb2a36f100926590e01d4cb40b873f3f275868ba577d2c7c2e4be4400f48`
- Stroke-only preflight diagnostics:
  `outputs/stroke-only-patch-suppression-thin-strict-source-preflight-20260816/diagnostics.csv`
- Stroke-only preflight diagnostics SHA256:
  `25f286b1ce976115e6e3efba344533e248ec39a6396cb52ddaf8bdfef893bc43`
- Stroke-only preflight review rows:
  `outputs/stroke-only-patch-suppression-thin-strict-source-preflight-20260816/review_rows.csv`
- Stroke-only preflight review rows SHA256:
  `3f9f402ab92ee845d9729ac445e055b246dda9e11a580d2e78540774f38b2687`
- Stroke-only metric summary:
  `outputs/stroke-only-patch-suppression-thin-strict-source-preflight-20260816/metric_summary.json`
- Stroke-only metric summary SHA256:
  `b7d91757eca77a8d1f006420547493146b68efb429980d4d07bb2fe4be3d9284`

## Train-Only Metric Result

Baseline on the three selected train160 rows was residual
`0.2421483934598291` and overerase `0.0025530178200578405`.

| Source Candidate | Residual Gain | Overerase Delta | Max Page Overerase Delta |
| --- | ---: | ---: | ---: |
| thin_strict | 0.00002356978216212317 | 0.00011767388035046628 | 0.00016253379804712434 |
| thin_mid | 0.00023919631405886396 | 0.00040135030180927416 | 0.0005440250442406334 |
| thin_broad | 0.00053704756837084 | 0.0005898335588607433 | 0.0008508461863226827 |

| Stroke-Only On Thin-Strict | Residual Gain | Overerase Delta |
| --- | ---: | ---: |
| stroke_strict | 0.00002356978216212317 | 0.00011767388035046628 |
| stroke_balanced | -0.0000013754103019384771 | 0.00008679586642687243 |
| stroke_soft | -0.000000916940201295402 | 0.000052701866981266776 |

This is not a promotion-safe tradeoff. The only positive-gain path regresses
overerase, while the reduced-overerase stroke-only paths give up the residual
gain and still regress overerase.

## Boundary

Candidate generation used only source images and current-primary baseline
predictions. Targets were read only after candidate files were written for
train-only scoring. No model training, checkpoint generation, candidate
inference on validation, inner-val15, SCUT115, holdout40, reserved blind,
visual review, promotion, or `artifacts/current-primary` replacement occurred.

Intent: Close a materially different available train-only source-candidate route after exact selector-replay PNG recovery proved unavailable.
Constraint: Preserve-first scoring must reject residual gains that cost more overerase than they buy.
Rejected: Increase component size, source-dark threshold, alpha, or max lift | the trend is monotonic toward larger overerase deltas.
Rejected: Tune stroke-only strict/balanced/soft variants | the registered variants already show the tradeoff collapses to either tiny unsafe gain or no gain.
Confidence: high
Scope-risk: narrow
Directive: Do not repeat or rescue source_dark_thin_component_lift_v1; pick a qualitatively different train-only source candidate mechanism.
Tested: py313 materializer tests 4/4 with warnings as errors.
Tested: py313 py_compile for materializer and tests.
Tested: train-only source-candidate materialization on 3 selector false-positive train160 rows.
Tested: registered stroke-only suppression on the thin_strict source candidate.
Tested: train-only metric scoring for source candidate variants and stroke-only variants.
Not-tested: inner-val15, SCUT115, holdout40, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-16-stroke-only-patch-suppression-exact-candidate-local-absence.md
Related: docs/decisions/2026-08-16-source-dark-local-paper-lift-source-candidate-kill.md
