# Universal Residual Adapter Sidecar U4B Step20 Smoke

```text
u4b_terminal = STEP20_SIDECAR_ONLY_SMOKE_COMPLETE
dataset_validation = not_run
dataset_inference = not_run
fresh_blind_handoff = disabled
promotion_handoff = disabled
product_default = artifacts/current-primary
```

## Scope

U4B ran exactly one bounded sidecar-only step20 training smoke using:

```text
configs/local/config.local-universal-sidecar-u4a-mixed-scut130-hw5k260-step20-mps.yaml
```

Allowed artifact mutation was limited to:

```text
artifacts/trials/universal-sidecar-u4a-mixed-scut130-hw5k260-step20-20260806
```

No validation loader, final test, dataset inference, fresh blind evidence,
target-image manual review, consumed HW5K official-test use, promotion, or
`artifacts/current-primary` mutation was authorized or performed.

## Training Evidence

Command:

```bash
source .env
$ENSEXAM_PYTHON train.py \
  --config configs/local/config.local-universal-sidecar-u4a-mixed-scut130-hw5k260-step20-mps.yaml
```

Runtime evidence:

```text
train-only list pages = 383
patches = 70518
max_steps_per_epoch = 20
validation = skipped
final_test = skipped
trainable_generator_patterns = ['^universal_residual_adapter_sidecar\\.']
trainable_tensors = 17
frozen_tensors = 226
trainable_params = 7437 / 24690655
frozen_batchnorm_layers = 40
avg_loss_G = 22.454846
avg_loss_D = 2.054694
```

Generated files:

```text
artifacts/trials/universal-sidecar-u4a-mixed-scut130-hw5k260-step20-20260806/ensexam/20260806_100131/config.yaml
artifacts/trials/universal-sidecar-u4a-mixed-scut130-hw5k260-step20-20260806/ensexam/20260806_100131/epoch_1.pth
artifacts/trials/universal-sidecar-u4a-mixed-scut130-hw5k260-step20-20260806/ensexam/20260806_100131/loss_history.csv
artifacts/trials/universal-sidecar-u4a-mixed-scut130-hw5k260-step20-20260806/ensexam/20260806_100131/train.log
```

## Checkpoint Verification

Checkpoint:

```text
artifacts/trials/universal-sidecar-u4a-mixed-scut130-hw5k260-step20-20260806/ensexam/20260806_100131/epoch_1.pth
```

Verification result:

```text
sidecar_key_count = 17
has_optimizer_state = false
has_scheduler_state = false
base_changed_count_vs_current_primary = 0
```

This proves the smoke updated only sidecar checkpoint keys and preserved the
loaded current-primary base generator weights.

## Next Boundary

U4C development metric screening is not started by this decision. Before any
dataset inference/evaluation runs, U4C must freeze exact commands, output paths,
baseline/candidate comparison inputs, and the status of mechanism controls
(`current-primary`, single adapter, uniform mixture, continuous mixture).
