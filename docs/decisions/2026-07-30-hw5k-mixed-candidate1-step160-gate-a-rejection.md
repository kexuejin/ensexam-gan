# hw5k-mixed candidate 1 (step160): Gate A Rejected — Under-Trained, Direction Confirmed

## Scope

First bounded mixed-domain candidate under
`2026-07-29-hw5k-train-domain-adaptation-proposal.md`. Whole-checkpoint
continuation from current-primary; source-only frozen inference on `hw5k_dev`
first, labels only after predictions existed. No HW5K-test, Dev40, SCUT115,
holdout40, selector, threshold rescue, or page-specific choice was used.

- Config: `configs/local/config.local-hw5k-mixed-scut130-hw5k260-jointtail-lite-step160-bs4-mps.yaml`
- Checkpoint SHA-256: `65b41ce23eaf89be72bc8ec7eea0f1a520ddfa8b6802fa36766163d7995ead1a`
- Train data: merged `SCUT-HW5K-mixed-20260729` root, 130 SCUT inner_train_130
  + 253 HW5K train pages (7 size-mismatched HW5K pairs excluded and recorded),
  SCUT share 33.94%
- Training: 160 updates, physical batch 4, MPS, no AMP, finite losses
- Gate A evidence:
  `outputs/hw5k_dev_candidate1_full232_20260730/post_freeze_summary.json`
  `outputs/hw5k_train_intake_20260729/gate_a_result.json`

## Gate A Result (hw5k_dev, 232 pages)

| Metric | Baseline | Candidate | Result |
| --- | ---: | ---: | --- |
| Mean residual ratio | 0.723852852720 | 0.720372051446 | fails A1 (-0.48% rel; needs >= -20%) |
| Mean overerase ratio | 0.064444891090 | 0.031286731476 | improves (-51.5% rel) |
| Max residual ratio | 0.954781812446 | 0.912810297126 | improves |
| Max overerase ratio | 0.455976452017 | 0.225214906120 | improves |

Decision: **reject candidate 1; do not run Gate B / Dev40 / SCUT115**.

## Interpretation

All four frozen metrics moved in the right direction and none regressed, but
the residual movement is negligible. 160 updates x batch 4 = 640 crops over
383 pages (<2 crops/page) is far too small a budget for a domain where the
baseline leaves ~72% of handwriting. This is an under-training signature, not
a wrong-direction signature. Per the predeclared interpretation rule, the
revision is to the candidate (training budget), not to the gate.

## Data intake fix recorded

The first launch of this candidate crashed at step 32 because a small fraction
(~2.7%) of HW5K train pairs have targets a few pixels smaller than their
inputs; the training `__getitem__` assumes paired sizes match. The mixed-root
builder now excludes and records size-mismatched pairs. This filter is part of
the frozen intake path for all future HW5K candidates.

## Next candidate (predeclared)

Candidate 2 = identical config and data except `max_steps_per_epoch` raised
160 -> 1600 (still bounded, ~30 min MPS), new `save_dir`, same predeclared
Gate A/B/C bounds. No loss-weight, sampling, threshold, or mix changes, so the
only changed variable remains the training budget.
