# hw5k-mixed candidate 4: Gate B Rejected — HW5K Adaptation Trades SCUT Source Domain

## Scope

Gate B (SCUT source-domain guard) for candidate 4, which passed the HW5K-domain
Gate A. Both current-primary and candidate 4 were run on SCUT `inner_val15` with
identical frozen inference settings (mb copy-mask + mb_cov8_step auto,
page_overlap 32, batch 8), source-only first, labels only after predictions
existed. `inner_val15` is disjoint from the candidate's `inner_train_130`
training pages.

- Evidence:
  `outputs/scut_innerval15_current_primary_20260730/post_freeze_summary.json`
  `outputs/scut_innerval15_candidate4_20260730/post_freeze_summary.json`
  `outputs/hw5k_train_intake_20260729/gate_b_result_candidate4.json`

## Gate B Result (SCUT inner_val15, 15 pages) — FAIL

| Metric | current-primary | candidate 4 | Result |
| --- | ---: | ---: | --- |
| Mean residual ratio | 0.176948604902 | 0.254132127840 | fails (+43.6%) |
| Mean overerase ratio | 0.002324671360 | 0.004538025317 | fails (+95.2%) |
| Max residual ratio | 0.386987730998 | 0.503494035195 | regresses |

Per-page residual regression: **15/15 pages**. The drift is systematic, not a
few outliers.

Decision: **candidate 4 is rejected as a universal-checkpoint replacement.**
`artifacts/current-primary` remains unchanged. Do not run Gate C / SCUT115.

## Interpretation

This is the central finding of the HW5K adaptation program so far, and it is a
genuine result rather than a bug:

- Candidate 4 reduces HW5K-domain mean residual by 20.5% (Gate A pass) but
  raises SCUT mean residual by 43.6%. With only 33% SCUT in the mix, strong
  residual pressure, and a 6400-step budget, the shared generator moved toward
  HW5K erasure behavior at the SCUT domain's expense.
- Candidates 1-3 traced the HW5K side of this same axis: weaker adaptation
  preserved SCUT better but failed Gate A. The two domains are in direct
  tension under a single shared-weight generator at this mix.

Per the predeclared rule in
`2026-07-29-hw5k-train-domain-adaptation-proposal.md`, a Gate B failure does
NOT authorize weakening the SCUT gate. It opens a separate product decision
about a domain-adapted variant versus one universal checkpoint. That decision
is recorded/branched separately (see "Strategic fork" below); this document
only records the faithful gate result.

## Strategic Fork (for the next decision)

Three admissible directions, none of which is a threshold rescue:

1. **Push for a universal checkpoint** — raise SCUT share substantially
   (e.g. 50-67%) and add the existing train-only baseline-solved residual
   guard (`cache_train_baseline_tail_support.py` +
   `lambda_cached_baseline_tail_nonregress`) so SCUT-solved pixels are
   explicitly protected. Retry Gate A then Gate B. Risk: a point passing both
   gates may not exist; higher compute.

2. **Domain-routed dual checkpoint** — keep current-primary for the SCUT/exam
   distribution and use the candidate-4 family for the HW5K distribution, with
   an inference-time domain router. Candidate 4 already clears the HW5K screen.
   This sidesteps the single-weight tension but is a product-architecture
   change requiring its own decision doc plus a domain classifier.

3. **Scope and pause** — record that HW5K adaptation is feasible (-20.5%
   residual) but trades SCUT under a shared generator; keep current-primary
   frozen; revisit when a real reserved blind set or an explicit product
   routing requirement exists.

The candidate-4 checkpoint and all frozen evidence are retained as the HW5K
adaptation reference for whichever direction is chosen.
