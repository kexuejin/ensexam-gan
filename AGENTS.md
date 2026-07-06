# EnsExam-GAN Fork Instructions

This repository is the active model-engineering workspace for the clean document handwriting-removal pipeline.

## Working Directory

Run training, evaluation, and inference from:

```text
/Volumes/Tool/source/ensexam-gan
```

Do not add new model-training or model-inference scripts under:

```text
/Volumes/Tool/source/clean-doc/scripts
/Volumes/Tool/source/clean-doc/external/ensexam-gan
```

`clean-doc` remains a historical product/research workspace only. Active datasets and model artifacts should be local to this repository under `data-links/` and `artifacts/`.

## Environment

Use the local environment variable instead of hardcoding Python paths in commands:

```bash
cp .env.example .env
source .env
$ENSEXAM_PYTHON --version
```

The current validated Python is still shared from clean-doc:

```text
/Volumes/Tool/source/clean-doc/.venv-torch310-mps-stable/bin/python
```

## Artifacts And Data

Use registered symlinks:

```text
artifacts/full-training-best.pth
artifacts/current-primary
artifacts/current-second-stage-best.pt
artifacts/current-holdout40-primary-pred
data-links/samples
```

Do not copy large checkpoint, prediction, or dataset payloads into git.

## Training Policy

Do not rerun full training by default. Reuse:

```text
artifacts/full-training-best.pth
```

For current-best continuation, start from:

```text
artifacts/current-primary/micro_region_probe_step0001.pth
```

Prefer bounded hardcase fine-tunes, local target-vs-prediction evaluation, gate tuning, and second-stage residual repair before considering another broad retrain.

For patch-sensitivity experiments, do not rank patches by stroke density alone. Prefer local
target-difference and gate-feature diagnostics, especially `copy_mask_cov8`, `primary_edit_px`,
`primary_p95_edit_delta`, and `second_stage_gate_ratio`.

Treat selector-only threshold search as analysis infrastructure, not a product-quality path by
itself. Promote a candidate only when it improves residual without cross-split overerase regression
on SCUT115 and holdout40; otherwise document the result as rejected/not-default instead of tuning
thresholds indefinitely.

## Evaluation Policy

Prefer local comparisons against target/label images and reproducible metrics. Do not rely on uploading images to visual AI unless local metrics and review packs are insufficient.

Default smoke:

```bash
$ENSEXAM_PYTHON scripts/run_second_stage_residual_repair.py \
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

Expected readiness smoke:

```text
summary residual=0.125765 overerase=0.002500
```

## Rejected Defaults

Whiteout/correction-fluid inpaint repair is not default. It improved some metrics but produced visually dirtier paper-tone restoration. Keep clean white correction-fluid patches unless a better paper-tone harmonization method passes visual review.
