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
  scripts/run_hybrid_second_stage_gate.py \
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

## Hybrid Gate Product Candidate

The best current product candidate is a page-level hybrid gate. It keeps the current
baseline second-stage output for risky pages and uses the `nearworst_safe_step1`
candidate only when inference-time safety features pass:

```text
copy_mask_cov8 >= 0.18436555
primary_edit_px <= 107112
```

Command:

```bash
$ENSEXAM_PYTHON scripts/run_hybrid_second_stage_gate.py \
  --samples-file docs/holdout40-relative.txt \
  --output-dir outputs/hybrid_gate_nearworst_safe_step1_t98_holdout40_20260705 \
  --baseline-pred-dir outputs/holdout40_second_stage_readiness_20260705/pred \
  --candidate-config configs/local/config.local-current-primary-continuation-mps.yaml \
  --candidate-weights outputs/exp_current_primary_nearworst_safe_step1_20260705/micro_region_probe.pth \
  --cleanup-checkpoint artifacts/current-second-stage-best.pt \
  --candidate-copy-mask mb \
  --candidate-copy-threshold 98 \
  --candidate-copy-threshold-auto none \
  --candidate-copy-dilate 0 \
  --min-copy-mask-cov8 0.18436555 \
  --max-primary-edit-px 107112 \
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
summary residual=0.131148 overerase=0.002547 selected=14/40
metrics_csv: outputs/hybrid_gate_nearworst_safe_step1_t98_holdout40_20260705/metrics.csv
```

SCUT test115 validation:

```bash
$ENSEXAM_PYTHON scripts/run_second_stage_residual_repair.py \
  --samples-file docs/scut-test115-relative.txt \
  --output-dir outputs/scut_test115_second_stage_baseline_20260705 \
  --primary-config artifacts/current-primary/config.yaml \
  --primary-weights artifacts/current-primary/micro_region_probe_step0001.pth \
  --primary-copy-mask mb \
  --primary-copy-threshold 70 \
  --primary-copy-threshold-auto mb_cov8_step \
  --primary-copy-dilate 0 \
  --cleanup-checkpoint artifacts/current-second-stage-best.pt \
  --cleanup-alpha-threshold 0.3 \
  --cleanup-tile-size 160 \
  --cleanup-stride 160 \
  --base-edit-threshold 12 \
  --second-delta-threshold 32 \
  --dark-threshold 0 \
  --device auto \
  --save-primary

$ENSEXAM_PYTHON scripts/run_hybrid_second_stage_gate.py \
  --samples-file docs/scut-test115-relative.txt \
  --output-dir outputs/scut_test115_hybrid_gate_nearworst_safe_step1_t98_20260705 \
  --baseline-pred-dir outputs/scut_test115_second_stage_baseline_20260705/pred \
  --candidate-config configs/local/config.local-current-primary-continuation-mps.yaml \
  --candidate-weights outputs/exp_current_primary_nearworst_safe_step1_20260705/micro_region_probe.pth \
  --cleanup-checkpoint artifacts/current-second-stage-best.pt \
  --candidate-copy-mask mb \
  --candidate-copy-threshold 98 \
  --candidate-copy-threshold-auto none \
  --candidate-copy-dilate 0 \
  --min-copy-mask-cov8 0.18436555 \
  --max-primary-edit-px 107112 \
  --cleanup-alpha-threshold 0.3 \
  --cleanup-tile-size 160 \
  --cleanup-stride 160 \
  --base-edit-threshold 12 \
  --second-delta-threshold 32 \
  --dark-threshold 0 \
  --device auto
```

SCUT test115 result:

```text
baseline residual=0.114225 overerase=0.003048
hybrid residual=0.112203 overerase=0.003125 selected=29/115
review pack: outputs/review_scut_test115_hybrid_gate_nearworst_safe_step1_t98_20260705
```

Strict SCUT test115 gate:

```bash
$ENSEXAM_PYTHON scripts/run_hybrid_second_stage_gate.py \
  --samples-file docs/scut-test115-relative.txt \
  --output-dir outputs/scut_test115_hybrid_gate_strict_cov806_edit98868_20260705 \
  --baseline-pred-dir outputs/scut_test115_second_stage_baseline_20260705/pred \
  --candidate-config configs/local/config.local-current-primary-continuation-mps.yaml \
  --candidate-weights outputs/exp_current_primary_nearworst_safe_step1_20260705/micro_region_probe.pth \
  --cleanup-checkpoint artifacts/current-second-stage-best.pt \
  --candidate-copy-mask mb \
  --candidate-copy-threshold 98 \
  --candidate-copy-threshold-auto none \
  --candidate-copy-dilate 0 \
  --min-copy-mask-cov8 0.806133 \
  --max-primary-edit-px 98868 \
  --cleanup-alpha-threshold 0.3 \
  --cleanup-tile-size 160 \
  --cleanup-stride 160 \
  --base-edit-threshold 12 \
  --second-delta-threshold 32 \
  --dark-threshold 0 \
  --device auto
```

Strict result:

```text
baseline residual=0.114225 overerase=0.003048
strict hybrid residual=0.113956 overerase=0.003047 selected=6/115
selected pages: 17.jpg 156.jpg 254.jpg 303.jpg 370.jpg 371.jpg
review pack: outputs/review_scut_test115_hybrid_gate_strict_cov806_edit98868_20260705
manual contact-sheet pass: no obvious large-area overerase regression, but visual gain is subtle
diff-crop review: most selected-page changes are low-contrast texture / gray-balance shifts; only a subset shows visible cleanup benefit
```

Visible-delta analysis:

```bash
$ENSEXAM_PYTHON scripts/analysis/analyze_candidate_visible_delta.py \
  --baseline-metrics outputs/scut_test115_second_stage_baseline_20260705/metrics.csv \
  --candidate-metrics outputs/scut_test115_hybrid_gate_strict_cov806_edit98868_20260705/metrics.csv \
  --output-csv outputs/analysis_visible_delta_strict_scut_test115_20260705_rerun/components.csv \
  --summary-csv outputs/analysis_visible_delta_strict_scut_test115_20260705_rerun/summary.csv \
  --crops-dir outputs/analysis_visible_delta_strict_scut_test115_20260705_rerun/crops \
  --contact-sheet outputs/analysis_visible_delta_strict_scut_test115_20260705_rerun/contact_sheet_components.png \
  --max-crops 60 \
  --change-threshold 12 \
  --gain-threshold 8 \
  --min-area 20
```

Visible-delta result:

```text
improve_visible_target_region: components=11 area=464
regress_low_contrast_target: components=1 area=36
regress_visible_target_region: components=7 area=193
```

Decision:

```text
Loose gate is not a default replacement because overerase rose on SCUT test115. Strict gate is
safer on aggregate metrics, but diff-crop review shows most changes are subtle texture /
gray-balance shifts rather than clear product-visible cleanup. Keep it as a research candidate,
not an optional product mode, until full-size manual review proves consistent page-level gains.
```

## Current-Primary Continuation Step4 Evaluation

The 2026-07-05 four-step continuation from `artifacts/current-primary/micro_region_probe_step0001.pth`
was evaluated on holdout40 and rejected for promotion. All candidate checkpoints increased residual
versus the current primary baseline.

Command:

```bash
RUN=outputs/exp_current_primary_continuation_step4_20260705
EVAL=outputs/eval_current_primary_continuation_step4_holdout40_20260705

$ENSEXAM_PYTHON scripts/batch_eval_hardcase_checkpoints.py \
  --items-csv "$RUN/holdout40_candidate_items.csv" \
  --samples-file docs/holdout40-relative.txt \
  --output-root "$EVAL" \
  --summary-csv "$EVAL/summary.csv" \
  --baseline-pred-dir artifacts/current-holdout40-primary-pred \
  --device auto \
  --copy-input-outside-mask mb \
  --copy-mask-threshold-auto mb_cov8_step \
  --copy-mask-threshold 70 \
  --copy-mask-dilate 0 \
  --page-overlap 32 \
  --batch-size 8
```

Result:

```text
baseline primary residual=0.136111 overerase=0.002482
step0001 residual=0.138113 overerase=0.002797 score=-0.004524
step0002 residual=0.146358 overerase=0.002602 score=-0.011211
step0003 residual=0.145277 overerase=0.002515 score=-0.009433
step0004 residual=0.145792 overerase=0.002285 score=-0.009681
summary_csv: outputs/eval_current_primary_continuation_step4_holdout40_20260705/summary.csv
```

Decision:

```text
No promotion. Do not update artifacts/current-primary from this run.
```
