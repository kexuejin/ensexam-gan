# Universal Residual Adapter Sidecar D2 Step80 Kill

```text
d2_terminal = KILL
d2_preflight_terminal = PASS
d2_step80_smoke_terminal = PASS
d2_scut15_source_guard_terminal = KILL
d2_result = measurable_page_delta_with_residual_source_guard_regression
product_default = artifacts/current-primary
fresh_blind = disabled
promotion = disabled
```

## Scope

D2 tested whether the D1 gradient-alive sidecar could move beyond the V1
step20 mechanism smoke into a slightly longer step80 sidecar-only candidate.
The run preserved the universal boundary:

- single external `clean(image)`-style inference surface
- no domain label, caller hint, source selector, path selector, expert selector,
  or hard route
- current-primary initialization and product default unchanged
- sidecar-only trainable generator parameters
- no optimizer/scheduler state saved
- fresh blind and promotion out of scope

## Preflight Evidence

Config:

```text
configs/local/config.local-universal-sidecar-d2-d1-mixed-scut130-hw5k260-step80-mps.yaml
```

Observed preflight:

```text
d2_config_validator = pass
max_steps_per_epoch = 80
train_file_count = 383
data_root_exists = true
init_checkpoint_exists = true
train_dataset_patches = 70518
init_missing_keys = 17
unexpected_init_keys = 0
trainable_tensors = 17
frozen_tensors = 226
trainable_params = 7437
total_params = 24690655
sidecar_only_freeze = pass
```

Verification:

```text
tests/test_universal_residual_adapter_sidecar.py = 10 passed, 4 subtests passed
scripts/analysis/audit_universal_sidecar_structure.py = status: pass
py_compile(train.py, audit_universal_sidecar_checkpoint.py, run_primary_full_page.py, evaluate_prediction_directory.py) = pass
```

## Step80 Smoke Evidence

Command:

```bash
source /Volumes/Tool/source/ensexam-gan/.env
$ENSEXAM_PYTHON train.py \
  --config configs/local/config.local-universal-sidecar-d2-d1-mixed-scut130-hw5k260-step80-mps.yaml
```

Run directory:

```text
artifacts/trials/universal-sidecar-d2-d1-mixed-scut130-hw5k260-step80-20260806/ensexam/20260806_221426
```

Training log evidence:

```text
seed = 20260806 strict
device = mps
init_checkpoint = artifacts/current-primary/micro_region_probe_step0001.pth
missing_init_keys = 17
unexpected_init_keys = 0
trainable = 17/243 tensors
trainable_params = 7437/24690655
BatchNorm frozen = 40 modules
max_steps_per_epoch = 80
validation = skipped
final_test = skipped
avg_loss_G = 30.171286010742186
avg_loss_D = 1.8397056579589843
```

Checkpoint:

```text
artifacts/trials/universal-sidecar-d2-d1-mixed-scut130-hw5k260-step80-20260806/ensexam/20260806_221426/epoch_1.pth
```

Checkpoint audit:

```text
outputs/universal_sidecar_d2_d1_step80_20260806/checkpoint_audit.json
```

Audit result:

```text
status = pass
sidecar_key_count = 17
moved_final_projection_key_count = 8
global_residual_scale.maxabs = 0.0024741601664572954
base_changed_count_vs_current_primary = 0
base_missing_count_vs_current_primary = 0
unexpected_non_sidecar_key_count = 0
has_optimizer_state = false
has_scheduler_state = false
failures = []
```

## Matched-Copy SCUT15 Source Guard

Inference used the current-primary matched-copy protocol:

```text
copy_input_outside_mask = mb
copy_mask_threshold_auto = mb_cov8_step
page_overlap = 32
batch_size = 8
```

Outputs:

```text
outputs/universal_sidecar_d2_d1_inner_val15_step80_matched_copy_20260806/frozen_predictions
outputs/universal_sidecar_d2_d1_inner_val15_step80_matched_copy_20260806/post_freeze_metrics.csv
outputs/universal_sidecar_d2_d1_inner_val15_step80_matched_copy_20260806/source_guard_summary.json
```

Aggregate scoring output:

```text
pages = 15
baseline_residual = 0.176949
candidate_residual = 0.176949
residual_gain = -0.000001
baseline_overerase = 0.002325
candidate_overerase = 0.002325
overerase_delta = 0.000000
```

Detailed guard summary:

| Metric | Baseline Mean | Candidate Mean | Candidate - Baseline Mean | Candidate - Baseline P95 | Candidate - Baseline Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual_ratio | 0.17694860490191527 | 0.17694920338356732 | +0.0000005984816520476777 | +0.0000026931674342145557 | +0.000008977224780715165 |
| overerase_ratio | 0.0023246713604556293 | 0.0023246713604556293 | 0.0000000000000000 | 0.0000000000000000 | 0.0000000000000000 |

Only one page changed:

```text
file = 301.jpg
candidate_minus_baseline_residual_ratio = +0.000008977224780715165
candidate_minus_baseline_overerase_ratio = 0.0
```

## Decision

D2 is killed at the SCUT15 matched-copy source guard.

The run proves that a longer sidecar-only continuation can produce a measurable
page-level delta while preserving the current-primary base and clean checkpoint
contract. However, the first measured delta is in the wrong direction: residual
increased on `301.jpg` and raised residual p95/max above baseline. That violates
the D2 requirement to produce measurable movement without residual or overerase
source-guard regression.

Do not evaluate this D2 checkpoint on SCUT115, holdout40, fresh blind, or
promotion gates. Do not tune copy-mask thresholds as a rescue for this checkpoint.

## Handoff

Recommended next goal, if continuing this direction:

```text
D2B bounded repair: test a more conservative sidecar-only candidate that keeps
gradient liveness but explicitly suppresses residual-source regression before
longer training. Candidate options include shorter step40 source-guard probe,
lower learning rate, or a source-guard-weighted residual penalty. Keep the same
universal interface and matched-copy source-guard protocol.
```
