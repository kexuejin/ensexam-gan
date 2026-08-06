# Universal Residual Adapter Sidecar D2C Step60 Kill

```text
d2c_terminal = KILL
d2c_preflight_terminal = PASS
d2c_step60_smoke_terminal = PASS
d2c_scut15_source_guard_terminal = KILL
d2c_result = step60_reproduces_step80_residual_source_guard_regression
product_default = artifacts/current-primary
fresh_blind = disabled
promotion = disabled
```

## Scope

D2C tested the smallest fixed interpolation between D2B step40 safe/no-delta
and D2 step80 measurable-delta/source-guard-KILL. The only intentional training
change from D2B was increasing the sidecar-only continuation from 40 to 60
steps while keeping the same learning rate, seed, data, loss, initialization,
and checkpoint contract.

The run preserved the approved universal boundary:

- single external `clean(image)`-style inference surface
- no domain label, caller hint, source selector, path selector, expert selector,
  or hard route
- internal image-feature-conditioned continuous soft residual adapter mixing
  only
- current-primary initialization and product default unchanged
- sidecar-only trainable generator parameters
- fallback returns the same-call current-primary baseline output
- no optimizer/scheduler state saved
- fresh blind and promotion out of scope

## Reproducible Source-Guard Summary

Earlier D2/D2B runs persisted `source_guard_summary.json` without preserving a
generator command. D2C adds a bounded analysis utility and focused tests:

```text
scripts/analysis/summarize_universal_sidecar_source_guard.py
tests/test_summarize_universal_sidecar_source_guard.py
```

The utility binds `post_freeze_metrics.csv` to the exact expected sample
manifest, rejects blank, duplicate, missing, unexpected, malformed, non-finite,
or out-of-range ratio inputs, computes candidate-minus-baseline page deltas, and
fails closed when any page has positive residual or overerase delta. It exits
zero for a passing guard and one for a failing guard.

Verification:

```text
tests/test_summarize_universal_sidecar_source_guard.py = 6 passed, 8 subtests passed
```

## Preflight Evidence

Config:

```text
configs/local/config.local-universal-sidecar-d2c-d1-mixed-scut130-hw5k260-step60-mps.yaml
```

The parsed D2C config was compared with D2B after removing only
`train.max_steps_per_epoch` and `train.save_dir`; the remaining objects were
equal.

Observed preflight:

```text
d2c_config_validator = pass
max_steps_per_epoch = 60
lr = 2.5e-05
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

Additional verification:

```text
tests/test_universal_residual_adapter_sidecar.py = 10 passed, 4 subtests passed
scripts/analysis/audit_universal_sidecar_structure.py = status: pass
py_compile(train.py, checkpoint audit, inference, evaluator, source summarizer) = pass
```

## Step60 Smoke Evidence

Command:

```bash
source /Volumes/Tool/source/ensexam-gan/.env
$ENSEXAM_PYTHON train.py \
  --config configs/local/config.local-universal-sidecar-d2c-d1-mixed-scut130-hw5k260-step60-mps.yaml
```

Run directory:

```text
artifacts/trials/universal-sidecar-d2c-d1-mixed-scut130-hw5k260-step60-20260806/ensexam/20260806_225653
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
max_steps_per_epoch = 60
validation = skipped
final_test = skipped
avg_loss_G = 26.183856201171874
avg_loss_D = 1.8604110717773437
```

Checkpoint:

```text
artifacts/trials/universal-sidecar-d2c-d1-mixed-scut130-hw5k260-step60-20260806/ensexam/20260806_225653/epoch_1.pth
```

Checkpoint audit:

```text
outputs/universal_sidecar_d2c_d1_step60_20260806/checkpoint_audit.json
```

Audit result:

```text
status = pass
sidecar_key_count = 17
moved_final_projection_key_count = 8
global_residual_scale.maxabs = 0.0018903231248259544
base_changed_count_vs_current_primary = 0
base_missing_count_vs_current_primary = 0
unexpected_non_sidecar_key_count = 0
has_optimizer_state = false
has_scheduler_state = false
failures = []
```

## Matched-Copy SCUT15 Source Guard

Inference retained the frozen current-primary protocol:

```text
copy_input_outside_mask = mb
copy_mask_threshold_auto = mb_cov8_step
page_overlap = 32
batch_size = 8
```

Outputs:

```text
outputs/universal_sidecar_d2c_d1_inner_val15_step60_matched_copy_20260806/frozen_predictions
outputs/universal_sidecar_d2c_d1_inner_val15_step60_matched_copy_20260806/post_freeze_metrics.csv
outputs/universal_sidecar_d2c_d1_inner_val15_step60_matched_copy_20260806/source_guard_summary.json
```

Summary command:

```bash
$ENSEXAM_PYTHON scripts/analysis/summarize_universal_sidecar_source_guard.py \
  --post-freeze-metrics outputs/universal_sidecar_d2c_d1_inner_val15_step60_matched_copy_20260806/post_freeze_metrics.csv \
  --expected-samples-file hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt \
  --output-json outputs/universal_sidecar_d2c_d1_inner_val15_step60_matched_copy_20260806/source_guard_summary.json
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

Detailed guard result:

| Metric | Baseline Mean | Candidate Mean | Candidate - Baseline Mean | Candidate - Baseline P95 | Candidate - Baseline Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual_ratio | 0.17694860490191527 | 0.17694920338356732 | +0.0000005984816520476777 | +0.00000269316743421454 | +0.000008977224780715165 |
| overerase_ratio | 0.0023246713604556293 | 0.0023246713604556293 | 0.0000000000000000 | 0.0000000000000000 | 0.0000000000000000 |

Only one page changed at the metric gate:

```text
file = 301.jpg
candidate_minus_baseline_residual_ratio = +0.000008977224780715165
candidate_minus_baseline_overerase_ratio = 0.0
measurable_page_delta = true
source_guard_status = fail
failures = [residual_source_guard_regression]
```

Frozen provenance:

```text
config_sha256 = 8d9dc9d78abbba7d7f45409ccca6a2c3076ff1f7a7c95aa304e86a01a6de1369
checkpoint_sha256 = 38756e89417a841a6053f1981919bd916b488d841556aed955e1c65a2c1fefb4
checkpoint_audit_sha256 = 67d64204a09cb7be3dfb3ec95d16cf3be34e193b970fa81fe0b410a71571bb29
expected_samples_sha256 = fb25bb2aef2f9285403f908deb3da6d88b07b5d1c2c812965ce9e0636ddc172e
post_freeze_metrics_sha256 = 2db374d7c4109b8b1595661387d76330a49f564055901d49f68151a1e1cc98de
source_guard_summary_sha256 = d67f03763c23e837efc818e2509070f1fe4ef94e4cb9c4669239c6640f2545c1
expected_pages = 15
```

This is the same page and the same thresholded residual regression magnitude
that killed D2 step80. The underlying prediction images are not byte-identical,
but the accepted page-level residual/overerase counts and terminal guard result
are the same.

## Decision

D2C is killed at the first SCUT15 source guard. The step60 run satisfies the
mechanism and checkpoint-isolation contracts but its first measurable page
movement is again a source residual regression on `301.jpg`.

This result does not prove that every step between 40 and 80 fails. It does show
that simple same-LR training-length interpolation has not produced a safe lift:

```text
step20 = PASS, no measurable SCUT15 delta
step40 = PASS, no measurable SCUT15 delta
step60 = KILL, measurable residual regression on 301.jpg
step80 = KILL, measurable residual regression on 301.jpg
```

Do not continue step-count-only interpolation without a new causal reason. Do
not evaluate this checkpoint on SCUT115, holdout40, fresh blind, or promotion
gates. Do not rescue it with copy-mask threshold tuning.

## Handoff

If this direction continues, open a separately preregistered bounded
causal-control goal, not another training-length interpolation goal. A future
candidate must change exactly one named control. Examples include a lower-LR
probe or a uniformly defined, source-label-free train-safe residual penalty;
neither may use domain identity, source metadata, caller hints, or per-domain
loss routing. Retain SCUT15 matched-copy as the first kill gate and keep the
universal interface plus current-primary default fixed.
