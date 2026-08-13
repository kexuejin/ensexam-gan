# hw5k-mixed candidate 4 (step6400 respress): Gate A PASSED — First Candidate Through The Domain Screen

## Scope

Fourth bounded mixed-domain candidate under
`2026-07-29-hw5k-train-domain-adaptation-proposal.md`. Identical to candidate 3
except `max_steps_per_epoch` 1600 -> 6400 (budget-only change). Source-only
frozen inference first; labels only after predictions existed. No HW5K-test,
Dev40, SCUT115, holdout40, selector, or page-specific choice was used.

- Config: `configs/local/config.local-hw5k-mixed-scut130-hw5k260-jointtail-lite-step6400-respress-bs4-mps.yaml`
- Checkpoint SHA-256: `f5a5833ac4c27b06848a7b371dd8bb08ac45ae2fe901df42de36ae65b195400d`
- Gate A evidence:
  `outputs/hw5k_dev_candidate4_full232_20260730/post_freeze_summary.json`
  `outputs/hw5k_train_intake_20260729/gate_a_result_candidate4.json`

## Gate A Result (hw5k_dev, 232 pages) — PASS

| Metric | Baseline | Cand 3 | Cand 4 | Gate A |
| --- | ---: | ---: | ---: | --- |
| Mean residual ratio | 0.723852852720 | 0.704371800834 | 0.575717807255 | **pass** (-20.47%, needs -20%) |
| Mean overerase ratio | 0.064444891090 | 0.011629165284 | 0.018720976784 | pass (-70.95%) |
| Max residual ratio | 0.954781812446 | 0.906322472794 | 0.918578361249 | pass |
| Max overerase ratio | 0.455976452017 | 0.165873690173 | 0.398753894081 | (reported) |

## Tier Trajectory Confirms The Interference Hypothesis

```text
tier            cand2 (1600)   cand3 (1600 rb)   cand4 (6400 rb)
high >0.8       -0.0659        -0.0865           -0.2885
mid 0.6-0.8     +0.0033        -0.0073           -0.0849
low <0.6        +0.1037        +0.0827           -0.0434   (regression eliminated)
improved/flat/worse: cand4 = 173 / 12 / 47
```

The candidate-2/3 diagnosis was that short-budget adaptation traded away
already-working easy-page erasure (training interference), not a gate or
direction problem. Extending the budget 4x eliminated the low-tier regression
(+0.083 -> -0.043) while deepening hard-page gains, exactly as predicted. All
three tiers now improve.

## Status

This is the first candidate to clear the HW5K-domain screen. It does NOT yet
authorize any promotion:

- Gate B (SCUT inner-val15 source-domain guard) is running next; the mix is
  only 33% SCUT, so source-domain retention must be proven.
- Gate C (frozen SCUT Dev40) only if Gate B passes.
- Any unknown-paper generalization claim still requires a separately reserved,
  unseen final blind set per `docs/blind-generalization-evaluation-protocol.md`.
  The scored HW5K test partition cannot serve as that set.

`artifacts/current-primary` remains unchanged pending the full gate chain.
