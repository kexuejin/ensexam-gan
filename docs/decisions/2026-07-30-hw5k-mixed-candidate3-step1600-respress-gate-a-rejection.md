# hw5k-mixed candidate 3 (step1600 respress): Gate A Rejected — 5x Aggregate Gain, Easy-Page Interference Is The Remaining Blocker

## Scope

Third bounded mixed-domain candidate under
`2026-07-29-hw5k-train-domain-adaptation-proposal.md`. Identical to candidate 2
except the predeclared loss rebalance (`lambda_input_preserve` 13.5 -> 6.75,
`lambda_eval_changed_residual` 2.15 -> 4.30). Source-only frozen inference
first; labels only after predictions existed. No HW5K-test, Dev40, SCUT115,
holdout40, selector, or page-specific choice was used.

- Config: `configs/local/config.local-hw5k-mixed-scut130-hw5k260-jointtail-lite-step1600-respress-bs4-mps.yaml`
- Checkpoint SHA-256: `d26c3072381fdde89b536477cd329eafe9b19f3395e8321fbe4d7686dd49f9c4`
- Gate A evidence:
  `outputs/hw5k_dev_candidate3_full232_20260730/post_freeze_summary.json`
  `outputs/hw5k_train_intake_20260729/gate_a_result_candidate3.json`

## Gate A Result (hw5k_dev, 232 pages)

| Metric | Baseline | Cand 2 | Cand 3 |
| --- | ---: | ---: | ---: |
| Mean residual ratio | 0.723852852720 | 0.720157524355 (-0.51%) | 0.704371800834 (**-2.69%**) |
| Mean overerase ratio | 0.064444891090 | 0.009053019074 | 0.011629165284 (-82.0%) |
| Max residual ratio | 0.954781812446 | 0.917802402522 | 0.906322472794 |
| Max overerase ratio | 0.455976452017 | 0.119512885868 | 0.165873690173 |

Decision: **reject candidate 3; do not run Gate B / Dev40 / SCUT115**.
A1 requires <= 0.579082 (-20% relative); candidate 3 reached -2.69%.

## Tier Structure: The Rebalance Worked Where Predicted

```text
tier            cand2 mean delta    cand3 mean delta
high >0.8       -0.0659             -0.0865   (deeper hard-page erasure)
mid 0.6-0.8     +0.0033             -0.0073   (flipped to improvement)
low <0.6        +0.1037             +0.0827   (still regressing)
improved/flat/worse: 102/28/102
```

## Diagnostic: Easy-Page Regression Is In The Generator, Not The Gate

The 12 worst-regressing low-tier pages were re-inferred with
`--copy-input-outside-mask none`:

```text
12-page mean residual: baseline(mb)=0.484  cand3(mb)=0.655  cand3(none)=0.647
evidence: outputs/hw5k_easyreg12_candidate3_nocopy_20260730/post_freeze_summary.json
```

Disabling the copy gate changes almost nothing, so the fine-tuned generator
genuinely erases less than current-primary on this content subset. Combined
with the candidate-2 diagnostic (gate-off ~no change on generic pages), the
copy-mask gate is now excluded twice as the bottleneck. The remaining blocker
is training interference: short-budget adaptation trades previously-working
erasure behavior for new-domain behavior. The cand2 -> cand3 trend (easy-page
regression shrinking while hard-page gains deepen) indicates deeper adaptation
reduces the interference.

## Next candidate (predeclared)

Candidate 4 = identical respress config with `max_steps_per_epoch` 1600 ->
6400 (single changed variable: budget; cosine schedule stretches
accordingly; ~1.7 h MPS). Gate A/B/C bounds unchanged.

If candidate 4 still fails A1 with persisting low-tier regression, the next
revision is a train-only baseline-solved-residual guard (the cached
current-primary support machinery already in the codebase) to explicitly
protect pixels the baseline already solved — not another budget or scalar
change.
