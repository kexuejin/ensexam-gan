# Universal Residual Adapter Sidecar V1 Bounded Smoke

```text
v1_preflight_terminal = PASS
v1_step20_smoke_terminal = PASS
v1_scut15_source_guard_terminal = PASS
v1_result = gradient_alive_but_no_measurable_scut15_page_delta
product_default = artifacts/current-primary
fresh_blind = disabled
promotion = disabled
```

## Scope

V1 tested whether the D1 gradient-alive universal residual adapter sidecar can
actually train while preserving the universal contract:

- single external `clean(image)`-style inference surface
- no domain label, caller hint, source selector, path selector, expert selector,
  or hard route
- current-primary initialization and product default unchanged
- sidecar-only trainable generator parameters
- no optimizer/scheduler state saved in the smoke checkpoint
- fresh blind and promotion remain separate future goals

## Preflight Evidence

Config:

```text
configs/local/config.local-universal-sidecar-v1-d1-mixed-scut130-hw5k260-step20-mps.yaml
```

Training list:

```text
hardcase_lists/mixed_scut130_hw5k260_20260729.txt
```

Observed preflight:

```text
config_validator = pass
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

Tests:

```text
tests/test_universal_residual_adapter_sidecar.py = 10 passed, 4 subtests passed
scripts/analysis/audit_universal_sidecar_structure.py = status: pass
py_compile(train.py, run_primary_full_page.py, audit_universal_sidecar_checkpoint.py) = pass
```

## Step20 Smoke Evidence

Command:

```bash
source /Volumes/Tool/source/ensexam-gan/.env
$ENSEXAM_PYTHON train.py \
  --config configs/local/config.local-universal-sidecar-v1-d1-mixed-scut130-hw5k260-step20-mps.yaml
```

Run directory:

```text
artifacts/trials/universal-sidecar-v1-d1-mixed-scut130-hw5k260-step20-20260806/ensexam/20260806_215603
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
max_steps_per_epoch = 20
validation = skipped
final_test = skipped
avg_loss_G = 26.5289794921875
avg_loss_D = 1.8383193969726563
```

Checkpoint:

```text
artifacts/trials/universal-sidecar-v1-d1-mixed-scut130-hw5k260-step20-20260806/ensexam/20260806_215603/epoch_1.pth
```

Checkpoint audit:

```text
outputs/universal_sidecar_v1_d1_step20_20260806/checkpoint_audit.json
```

Audit result:

```text
status = pass
sidecar_key_count = 17
moved_final_projection_key_count = 8
global_residual_scale.maxabs = 0.0014159473357722163
base_changed_count_vs_current_primary = 0
base_missing_count_vs_current_primary = 0
unexpected_non_sidecar_key_count = 0
has_optimizer_state = false
has_scheduler_state = false
failures = []
```

## Matched-Copy SCUT15 Source Guard

Inference command used the current-primary matched-copy postprocess protocol:

```bash
source /Volumes/Tool/source/ensexam-gan/.env
$ENSEXAM_PYTHON scripts/infer/run_primary_full_page.py \
  --samples-file hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt \
  --image-dir data-links/samples/SCUT-EnsExam/train/all_images \
  --output-dir outputs/universal_sidecar_v1_d1_inner_val15_step20_matched_copy_20260806_run2/frozen_predictions \
  --primary-config artifacts/trials/universal-sidecar-v1-d1-mixed-scut130-hw5k260-step20-20260806/ensexam/20260806_215603/config.yaml \
  --primary-weights artifacts/trials/universal-sidecar-v1-d1-mixed-scut130-hw5k260-step20-20260806/ensexam/20260806_215603/epoch_1.pth \
  --device auto \
  --page-overlap 32 \
  --batch-size 8 \
  --copy-input-outside-mask mb \
  --copy-mask-threshold-auto mb_cov8_step \
  --skip-label-metrics
```

Scoring command:

```bash
source /Volumes/Tool/source/ensexam-gan/.env
$ENSEXAM_PYTHON scripts/eval/evaluate_prediction_directory.py \
  --baseline-metrics /Volumes/Tool/source/ensexam-gan/outputs/scut_innerval15_current_primary_20260802/post_freeze_metrics.csv \
  --pred-dir outputs/universal_sidecar_v1_d1_inner_val15_step20_matched_copy_20260806_run2/frozen_predictions/pred \
  --output-csv outputs/universal_sidecar_v1_d1_inner_val15_step20_matched_copy_20260806_run2/post_freeze_metrics.csv \
  --change-threshold 12 \
  --eval-threshold 12
```

Source guard output:

```text
pages = 15
baseline_residual = 0.176949
candidate_residual = 0.176949
residual_gain = 0.000000
baseline_overerase = 0.002325
candidate_overerase = 0.002325
overerase_delta = 0.000000
```

Detailed summary:

```text
outputs/universal_sidecar_v1_d1_inner_val15_step20_matched_copy_20260806_run2/source_guard_summary.json
```

Summary values:

```text
residual_delta_mean = 0.0
residual_delta_p95 = 0.0
residual_delta_max = 0.0
overerase_delta_mean = 0.0
overerase_delta_p95 = 0.0
overerase_delta_max = 0.0
nonzero_delta_files = 0
source_guard_status = pass
```

## Decision

V1 passes. D1 fixed the original no-learning mechanism: the sidecar final
projection path moved while the base generator stayed byte-identical to
current-primary and the checkpoint omitted optimizer/scheduler state.

The SCUT15 matched-copy source guard also passes, but this step20 candidate has
no measurable page-level delta on SCUT15 after matched-copy postprocess. It is a
mechanism smoke pass, not a quality candidate and not a promotion candidate.

## Handoff

Recommended next goal:

```text
D2 bounded continuation: run a slightly longer sidecar-only candidate only if
the objective is to test whether the now-gradient-alive sidecar can produce
measurable residual lift without source-guard regression. Keep fresh blind and
promotion out of scope until SCUT115 and holdout40 source guards pass.
```

