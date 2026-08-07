# Current-Primary Inner-Val15 Replay-Noise Calibration

```text
calibration_terminal = PASS
calibration_scope = scut_inner_val15_current_primary_matched_copy
run_count = 3
pages_per_run = 15
prediction_hashes_identical = 15/15
residual_replay_stddev = 0.0
overerase_replay_stddev = 0.0
calibrated_minimum_residual_gain = 0.0005
product_default = artifacts/current-primary
promotion = disabled
```

## Purpose

This bounded measurement mission calibrates the minimum detectable quality
lift before the long-lived quality program admits another candidate. It does
not train a model or establish generalization. The calibration is specific to
the frozen current-primary, exact SCUT inner-val15 manifest, and matched-copy
protocol described below.

## Frozen Inputs And Protocol

```text
config = artifacts/current-primary/config.yaml
config_sha256 = 8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e
checkpoint = artifacts/current-primary/micro_region_probe_step0001.pth
checkpoint_sha256 = e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae
manifest = hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt
manifest_sha256 = fb25bb2aef2f9285403f908deb3da6d88b07b5d1c2c812965ce9e0636ddc172e
device = mps
page_overlap = 32
batch_size = 8
copy_input_outside_mask = mb
copy_mask_threshold_auto = mb_cov8_step
copy_mask_dilate = 0
change_threshold = 12
eval_threshold = 12
```

Each of `replay01`, `replay02`, and `replay03` ran source-only inference into a
new directory. Labels were read only by the separate post-freeze evaluator.
The fail-closed calibration utility then verified exact sample order, image
hashes, config/checkpoint hashes, global protocol fields, and per-page
copy-mask thresholds before computing noise.

Reusable tooling:

```text
scripts/analysis/calibrate_prediction_replay_noise.py
tests/test_calibrate_prediction_replay_noise.py
```

Focused verification:

```text
6 passed, 4 subtests passed
py_compile = pass
git diff --check = pass
```

## Evidence

Local ignored evidence:

```text
outputs/current_primary_inner_val15_replay_calibration_20260807/
  replay01/frozen_predictions/metrics.csv
  replay01/post_freeze_metrics.csv
  replay02/frozen_predictions/metrics.csv
  replay02/post_freeze_metrics.csv
  replay03/frozen_predictions/metrics.csv
  replay03/post_freeze_metrics.csv
  calibration_summary.json
  calibration_summary_sha256 = e1df8a9c68bf5d883928e1cb0465ce91f1ad93f526435c88870261220ebadd6e
```

Aggregate metrics were identical across all runs:

| Metric | Replay01 | Replay02 | Replay03 | Sample Stddev | 3 Sigma |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual_ratio | 0.17694860490191527 | 0.17694860490191527 | 0.17694860490191527 | 0.0 | 0.0 |
| overerase_ratio | 0.0023246713604556293 | 0.0023246713604556293 | 0.0023246713604556293 | 0.0 | 0.0 |

Page-level evidence:

```text
nonzero_metric_files = []
residual_max_page_replay_stddev = 0.0
overerase_max_page_replay_stddev = 0.0
prediction_hashes.identical_files = 15
prediction_hashes.different_files = 0
```

## Decision

The current-primary inner-val15 matched-copy replay is deterministic at both
the prediction-byte and accepted metric levels in this three-run MPS
calibration. The long-lived quality program therefore freezes its minimum
measurable residual gain for this gate as:

```text
max(0.0005, 3 * aggregate_replay_stddev) = 0.0005
```

A candidate with no page delta or aggregate residual gain below `0.0005` is a
no-lift result at this gate. This does not relax the stricter source guard: any
positive page-level residual or overerase delta remains an immediate KILL.

Do not transfer the zero-noise result to train160, next120, SCUT115, holdout40,
or reserved blind without an equivalent split-specific replay calibration.
No promotion split or blind evidence was consumed in this mission, and
current-primary remains unchanged.

Intent: Freeze an evidence-based minimum measurable lift before admitting another quality candidate.
Constraint: Calibration is limited to current-primary, exact inner-val15, and the fixed matched-copy protocol; it may not train or select a candidate.
Rejected: Treat three identical means as sufficient without provenance validation | page order, hashes, protocol fields, and per-page thresholds also had to match.
Rejected: Apply the 0.0005 threshold to every split | replay noise is protocol- and split-specific.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Require residual gain >= 0.0005 and zero page regressions on this exact gate; calibrate other splits independently.
Tested: Three independent source-only MPS replays, separate post-freeze scoring, strict provenance/protocol calibration, six focused tests, syntax compilation, and whitespace checks.
Not-tested: Train160, next120, SCUT115, holdout40, reserved blind, visual quality, or any candidate checkpoint.
