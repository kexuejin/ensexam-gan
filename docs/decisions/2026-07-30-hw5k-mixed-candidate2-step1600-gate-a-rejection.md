# hw5k-mixed candidate 2 (step1600): Gate A Rejected — Residual Learning Suppressed By Preservation-Heavy Loss

## Scope

Second bounded mixed-domain candidate under
`2026-07-29-hw5k-train-domain-adaptation-proposal.md`. Identical to candidate 1
except `max_steps_per_epoch` 160 -> 1600. Source-only frozen inference first;
labels only after predictions existed. No HW5K-test, Dev40, SCUT115, holdout40,
selector, or page-specific choice was used.

- Config: `configs/local/config.local-hw5k-mixed-scut130-hw5k260-jointtail-lite-step1600-bs4-mps.yaml`
- Checkpoint SHA-256: `b30b044215407d728dde0153d41300f0c1a9f88e8cd771583bf5db29bc6156e8`
- Training: 1600 updates, physical batch 4, MPS, no AMP, finite losses
  (G 23.4 -> 21.6, D 1.81 -> 1.58 vs candidate 1)
- Gate A evidence:
  `outputs/hw5k_dev_candidate2_full232_20260730/post_freeze_summary.json`
  `outputs/hw5k_train_intake_20260729/gate_a_result_candidate2.json`

## Gate A Result (hw5k_dev, 232 pages)

| Metric | Baseline | Candidate 1 (step160) | Candidate 2 (step1600) |
| --- | ---: | ---: | ---: |
| Mean residual ratio | 0.723852852720 | 0.720372051446 (-0.48%) | 0.720157524355 (-0.51%) |
| Mean overerase ratio | 0.064444891090 | 0.031286731476 (-51.5%) | 0.009053019074 (-86.0%) |
| Max residual ratio | 0.954781812446 | 0.912810297126 | 0.917802402522 |
| Max overerase ratio | 0.455976452017 | 0.225214906120 | 0.119512885868 |

Decision: **reject candidate 2; do not run Gate B / Dev40 / SCUT115**.

## Diagnostic: The Bottleneck Is Not The Copy-Mask Gate

A 10x training-budget increase left mean residual essentially unchanged
(-0.48% -> -0.51%) while mean overerase kept collapsing (-51% -> -86%). Two
structural hypotheses were tested on the first 8 `hw5k_dev` pages:

```text
copy-mask gating hypothesis (rejected):
  candidate 2 with mb gating:      residual 0.6324, overerase 0.0219
  candidate 2 with gating disabled: residual 0.6276, overerase 0.0221
  evidence: outputs/hw5k_diag8_candidate2_nocopy_20260730/post_freeze_summary.json
  removing the gate changes residual by only ~0.005; the raw generator output
  itself leaves ~63% residual.

loss-suppression hypothesis (supported):
  the jointtail-lite loss stack carries strong preservation pressure tuned for
  SCUT tail safety (lambda_input_preserve=13.5, lambda_eval_outside_edit=4.75,
  lambda_mb_leak=0.70). On a domain where the model must learn to edit far
  more than on SCUT, these terms suppress erasure learning: preservation
  behavior converges quickly (overerase -86%) while residual stays immobile.
  A budget increase cannot fix this; the erasure gradient is being outweighed.
```

Additional observation: inference `copy_mask_cov8` distribution shifted from
mean 0.80 (baseline) to 0.30 (candidate 2), consistent with the model becoming
broadly more conservative rather than more selective.

## Page-Level Delta Structure (232 dev pages)

The flat aggregate hides a bimodal trade, computed from the two post-freeze
metrics CSVs:

```text
pages improved (>0.5pp): 93    flat: 27    regressed (>0.5pp): 112
mean delta -0.0037, median +0.0028

by baseline-residual tier:
  high  (>0.8,   n=80):  mean delta -0.0659  (best page -0.43: 127.jpg 0.712->0.282)
  mid   (0.6-0.8, n=113): mean delta +0.0033
  low   (<0.6,   n=39):  mean delta +0.1037  (worst page +0.38: 3652.jpg 0.441->0.825)
```

The candidate is genuinely learning to erase on the hardest pages while
simultaneously regressing pages the baseline already handled, via the same
conservativeness shift that collapsed overerase. This makes the candidate-3
rebalance (lower preservation pressure, higher residual-event pressure) the
directly indicated lever: it should reduce the easy-page conservative
regression while keeping the hard-page erasure gains.

## Next candidate (predeclared)

Candidate 3 = identical data/mix/steps (1600) with a named loss rebalance
targeting the diagnosed suppression, trading back part of the large overerase
headroom (-86%) for residual coverage:

```text
lambda_input_preserve:        13.5 -> 6.75  (halve preservation pressure)
lambda_eval_changed_residual: 2.15 -> 4.30  (double residual-event pressure)
all other loss terms, data, sampling, steps, thresholds: unchanged
```

Gate A/B/C bounds unchanged. If candidate 3 still fails A1, the next revision
must consider the SCUT-pretrained feature stack itself (e.g. longer schedule
with warm restarts or trainable-scope widening), not further scalar loss
tuning beyond this named pair.
