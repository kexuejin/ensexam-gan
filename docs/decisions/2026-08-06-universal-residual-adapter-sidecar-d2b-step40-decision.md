# Universal Residual Adapter Sidecar D2B Step40 Decision

```text
d2b_terminal = PASS
d2b_preflight_terminal = PASS
d2b_step40_smoke_terminal = PASS
d2b_scut15_source_guard_terminal = PASS
d2b_result = conservative_step40_source_guard_pass_no_measurable_page_delta
product_default = artifacts/current-primary
fresh_blind = disabled
promotion = disabled
```

## Scope

D2B tested the conservative repair suggested after D2 step80 was killed on the
SCUT15 matched-copy residual source guard. The only intentional change from D2
is shortening the sidecar-only continuation from step80 to step40.

The run preserved the approved universal boundary:

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
configs/local/config.local-universal-sidecar-d2b-d1-mixed-scut130-hw5k260-step40-mps.yaml
```

Observed preflight:

```text
d2b_config_validator = pass
max_steps_per_epoch = 40
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

## Step40 Smoke Evidence

Command:

```bash
source /Volumes/Tool/source/ensexam-gan/.env
$ENSEXAM_PYTHON train.py \
  --config configs/local/config.local-universal-sidecar-d2b-d1-mixed-scut130-hw5k260-step40-mps.yaml
```

Run directory:

```text
artifacts/trials/universal-sidecar-d2b-d1-mixed-scut130-hw5k260-step40-20260806/ensexam/20260806_223216
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
max_steps_per_epoch = 40
validation = skipped
final_test = skipped
avg_loss_G = 29.334506225585937
avg_loss_D = 1.8604305267333985
```

Checkpoint:

```text
artifacts/trials/universal-sidecar-d2b-d1-mixed-scut130-hw5k260-step40-20260806/ensexam/20260806_223216/epoch_1.pth
```

Checkpoint audit:

```text
outputs/universal_sidecar_d2b_d1_step40_20260806/checkpoint_audit.json
```

Audit result:

```text
status = pass
sidecar_key_count = 17
moved_final_projection_key_count = 8
global_residual_scale.maxabs = 0.001882148440927267
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
outputs/universal_sidecar_d2b_d1_inner_val15_step40_matched_copy_20260806/frozen_predictions
outputs/universal_sidecar_d2b_d1_inner_val15_step40_matched_copy_20260806/post_freeze_metrics.csv
outputs/universal_sidecar_d2b_d1_inner_val15_step40_matched_copy_20260806/source_guard_summary.json
```

Aggregate scoring output:

```text
pages = 15
baseline_residual = 0.176949
candidate_residual = 0.176949
residual_gain = 0.000000
baseline_overerase = 0.002325
candidate_overerase = 0.002325
overerase_delta = 0.000000
```

Detailed guard summary:

| Metric | Baseline Mean | Candidate Mean | Candidate - Baseline Mean | Candidate - Baseline P95 | Candidate - Baseline Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual_ratio | 0.17694860490191527 | 0.17694860490191527 | 0.0000000000000000 | 0.0000000000000000 | 0.0000000000000000 |
| overerase_ratio | 0.0023246713604556293 | 0.0023246713604556293 | 0.0000000000000000 | 0.0000000000000000 | 0.0000000000000000 |

No page changed after matched-copy scoring:

```text
nonzero_delta_files = 0
measurable_page_delta = false
source_guard_status = pass
failures = []
```

## Decision

D2B passes as a bounded repair probe. The step40 sidecar-only continuation keeps
the D1 gradient-alive property, moves the sidecar final projections, preserves
the frozen current-primary base, saves no optimizer/scheduler state, and avoids
the D2 step80 residual source-guard regression on SCUT15 matched-copy.

This is not a quality candidate. The source guard shows no measurable page-level
delta after matched-copy postprocess, so D2B proves that shorter training is
safe under this guard but does not prove HW5K lift or promotion readiness.

Do not evaluate this checkpoint on fresh blind or promotion gates. Broader
SCUT115/holdout40 source guards should only be opened by a follow-on goal if the
next objective is to confirm safety of this conservative sidecar state; they are
not sufficient to promote because D2B has no measurable development lift.

## Handoff

Recommended next goal:

```text
D2C bounded lift probe: keep the same universal interface and sidecar-only
contract, but introduce one conservative mechanism for measurable movement
without source-guard regression, such as step60 with the same LR, a lower-LR
step80, or an explicit residual-source penalty. Pick one candidate at a time,
retain matched-copy SCUT15 as the first kill gate, and keep fresh blind plus
promotion disabled until SCUT115 and holdout40 source guards pass with material
development lift.
```
