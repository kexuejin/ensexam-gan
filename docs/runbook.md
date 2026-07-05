# Runbook

## Environment

Use the existing torch environment until this fork gets its own isolated environment:

```bash
cp .env.example .env
source .env
$ENSEXAM_PYTHON --version
```

See `docs/environment.md`.

## Compile Smoke

```bash
$ENSEXAM_PYTHON -m py_compile \
  scripts/micro_train_region_probe.py \
  scripts/run_second_stage_residual_repair.py \
  scripts/eval_hardcase_worst_pages.py \
  scripts/batch_eval_hardcase_checkpoints.py \
  scripts/cached_sweep_hardcase_postprocess.py
```

## Current Main Inference Enhancement

Use `scripts/run_second_stage_residual_repair.py` with the primary checkpoint and the erasemap cleanup checkpoint registered in `docs/model-registry.md`.

## Training Continuation Policy

Do not restart full training by default. The one-day full-training result is already registered in this fork and can be used directly:

```text
artifacts/full-training-best.pth
```

For current-best product continuation, continue from:

```text
artifacts/current-primary/micro_region_probe_step0001.pth
```

Use this config for direct current-primary continuation:

```text
configs/local/config.local-current-primary-continuation-mps.yaml
```

Use broad full training only as an explicit reset experiment after documenting why the existing full-training base, targeted fine-tune, gate tuning, and second-stage repair are insufficient.

## Migration Smoke

The forked project has been verified with a real second-stage inference smoke using registered artifact symlinks.

Readiness verification completed:

```text
git history: retained from https://github.com/xiaozhejiya/ensexam-gan
upstream remote: configured as upstream
full-training checkpoint: torch.load OK
current-primary checkpoint: torch.load OK
second-stage checkpoint: torch.load OK
clean-doc script import dependency: removed from second-stage runner
py_compile: main migrated train/eval/infer entries OK
```

Command shape:

```bash
cd /Volumes/Tool/source/ensexam-gan
$ENSEXAM_PYTHON \
  scripts/infer/run_second_stage_residual_repair.py \
  --samples-file docs/smoke-holdout3-absolute.txt \
  --output-dir outputs/readiness_second_stage_holdout3_20260705 \
  --primary-pred-dir artifacts/current-holdout40-primary-pred \
  --cleanup-checkpoint artifacts/current-second-stage-best.pt \
  --cleanup-alpha-threshold 0.3 \
  --cleanup-tile-size 160 \
  --cleanup-stride 160 \
  --base-edit-threshold 12 \
  --second-delta-threshold 32 \
  --dark-threshold 0 \
  --device auto
```

Result:

```text
summary residual=0.125765 overerase=0.002500
metrics_csv: /Volumes/Tool/source/ensexam-gan/outputs/readiness_second_stage_holdout3_20260705/metrics.csv
```

Generated predictions:

```text
outputs/readiness_second_stage_holdout3_20260705/pred/27.png
outputs/readiness_second_stage_holdout3_20260705/pred/346.png
outputs/readiness_second_stage_holdout3_20260705/pred/516.png
```

## Holdout40 Reproduction

The current second-stage pipeline has also been reproduced on the full holdout40 list from this fork:

```bash
$ENSEXAM_PYTHON scripts/run_second_stage_residual_repair.py \
  --samples-file docs/holdout40-relative.txt \
  --output-dir outputs/holdout40_second_stage_readiness_20260705 \
  --primary-pred-dir artifacts/current-holdout40-primary-pred \
  --cleanup-checkpoint artifacts/current-second-stage-best.pt \
  --cleanup-alpha-threshold 0.3 \
  --cleanup-tile-size 160 \
  --cleanup-stride 160 \
  --base-edit-threshold 12 \
  --second-delta-threshold 32 \
  --dark-threshold 0 \
  --device auto
```

Result:

```text
summary residual=0.134026 overerase=0.002482
metrics_csv: outputs/holdout40_second_stage_readiness_20260705/metrics.csv
```
