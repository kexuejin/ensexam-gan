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

## Hybrid Gate Research Candidate

The current page-level hybrid gate is a research candidate, not a promoted product
default. It keeps the current baseline second-stage output for risky pages and uses
the `nearworst_safe_step1` candidate only when inference-time safety features pass.

The original loose rule was useful for finding residual-reduction headroom, but it
is not safe enough for default use because overerase rises on both SCUT test115 and
holdout40:

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

Visible-delta training patch index:

```bash
$ENSEXAM_PYTHON scripts/experimental/convert_visible_delta_to_patch_index.py \
  --components-csv outputs/analysis_visible_delta_strict_scut_test115_20260705_rerun/components.csv \
  --output-csv outputs/visible_delta_patch_index_strict_scut_test115_20260705/improve_patch_index.csv \
  --reject-csv outputs/visible_delta_patch_index_strict_scut_test115_20260705/regress_reject_components.csv \
  --region-type improve \
  --reason-contains visible_target_region \
  --img-size 256 \
  --overlap 96 \
  --patch-pad 96 \
  --max-tiles-per-component 4 \
  --min-area 20
```

Patch-index result:

```text
improve patch-index rows=24 files=2
reject regress components=8 files=5
output: outputs/visible_delta_patch_index_strict_scut_test115_20260705/improve_patch_index.csv
reject: outputs/visible_delta_patch_index_strict_scut_test115_20260705/regress_reject_components.csv
```

Visible-delta smoke dataset:

```bash
$ENSEXAM_PYTHON scripts/experimental/materialize_visible_delta_dataset.py \
  --components-csv outputs/analysis_visible_delta_strict_scut_test115_20260705_rerun/components.csv \
  --output-root data-links/samples/visible-delta-strict-scut-test115 \
  --split train \
  --file-list outputs/visible_delta_patch_index_strict_scut_test115_20260705/visible_delta_train_files.txt \
  --region-type improve \
  --reason-contains visible_target_region
```

Visible-delta one-step train smoke:

```bash
$ENSEXAM_PYTHON scripts/micro_train_region_probe.py \
  --config configs/local/config.local-visible-delta-smoke-mps.yaml \
  --output-dir outputs/smoke_visible_delta_patch_index_step1_20260705 \
  --max-steps 1 \
  --batch-size 1 \
  --train-pages 2 \
  --train-file-list outputs/visible_delta_patch_index_strict_scut_test115_20260705/visible_delta_train_files.txt \
  --patch-index-file outputs/visible_delta_patch_index_strict_scut_test115_20260705/improve_patch_index.csv \
  --disable-augmentation \
  --trace-batches-file outputs/smoke_visible_delta_patch_index_step1_20260705/trace_batches.csv \
  --log-every 1 \
  --save-every 1 \
  --loss-override lambda_input_preserve=12.0 \
  --device-override mps \
  --box-class-mode all
```

Smoke result:

```text
MPS preflight: passed
dataset patches: 234
patch-index filter: 234->24
step=1/1 G=15.062152 D=1.886716 lr_part=10.803489 sn=0.425416 block=0.130858
trace sample: 156.jpg x1=960 y1=480 x2=1216 y2=736
output: outputs/smoke_visible_delta_patch_index_step1_20260705/micro_region_probe.pth
```

Visible-delta 10-step probe:

```bash
$ENSEXAM_PYTHON scripts/micro_train_region_probe.py \
  --config configs/local/config.local-visible-delta-smoke-mps.yaml \
  --output-dir outputs/exp_visible_delta_patch_index_step10_20260705 \
  --max-steps 10 \
  --batch-size 1 \
  --train-pages 2 \
  --train-file-list outputs/visible_delta_patch_index_strict_scut_test115_20260705/visible_delta_train_files.txt \
  --patch-index-file outputs/visible_delta_patch_index_strict_scut_test115_20260705/improve_patch_index.csv \
  --disable-augmentation \
  --trace-batches-file outputs/exp_visible_delta_patch_index_step10_20260705/trace_batches.csv \
  --log-every 1 \
  --save-every 5 \
  --loss-override lambda_input_preserve=12.0 \
  --device-override mps \
  --box-class-mode all
```

Step10 strict-gate evaluation:

```text
output: outputs/scut_test115_hybrid_gate_visible_delta_step10_strict_20260705
baseline residual=0.114225 overerase=0.003048
original strict hybrid residual=0.113956 overerase=0.003047 selected=6/115
visible-delta step10 residual=0.114225 overerase=0.003048 selected=0/115
```

Failure mode:

```text
The 10-step patch-only full-generator update broke the candidate safety features. On the original
six selected pages, copy_mask_cov8 dropped and primary_edit_px increased enough that every page
failed the strict gate. Do not continue this exact full-generator patch-only direction.
```

Decision:

```text
Loose gate is not a default replacement because overerase rose on SCUT test115. Strict gate is
safer on aggregate metrics, but diff-crop review shows most changes are subtle texture /
gray-balance shifts rather than clear product-visible cleanup. Keep it as a research candidate,
not an optional product mode, until full-size manual review proves consistent page-level gains.
```

### Joint Selector Replay

Use the replay script to compare label-free selector rules across SCUT test115 and
holdout40 from saved candidate outputs:

```bash
$ENSEXAM_PYTHON scripts/analysis/replay_hybrid_selector.py \
  --split scut115:outputs/scut_test115_second_stage_baseline_20260705/metrics.csv:outputs/scut_test115_hybrid_gate_strict_cov806_edit98868_savecand_20260705/metrics.csv \
  --split holdout40:outputs/holdout40_second_stage_readiness_20260705/metrics.csv:outputs/holdout40_hybrid_gate_strict_cov806_edit98868_savecand_20260705/metrics.csv \
  --output-dir outputs/selector_replay_joint_strict_candidate_20260705 \
  --candidate-subdir candidate \
  --max-overerase-regret 0 \
  --min-selected-total 1 \
  --max-thresholds-per-feature 22 \
  --pin-min-copy-mask-cov8 0.806133 \
  --pin-min-copy-mask-cov8 0.65 \
  --pin-max-primary-edit-px 98868 \
  --pin-max-primary-p95-edit-delta 5 \
  --pin-max-second-stage-gate-ratio 0.0015 \
  --pin-max-second-stage-gate-ratio 1 \
  --named-rule strict_cov806_edit98868:0.806133:98868:1000000000:1000000000 \
  --named-rule scut7_cov65_edit98868_p95_5_gate0015:0.65:98868:5:0.0015 \
  --named-rule best_safe_joint:0.4584378323676181:101340:5:0.0002489434157317036 \
  --top-n 80
```

Joint replay result:

```text
output: outputs/selector_replay_joint_strict_candidate_20260705
rules scored=212187
safe rules=15057
best safe joint rule: copy_mask_cov8 >= 0.458438, primary_edit_px <= 101340,
  primary_p95_edit_delta <= 5, second_stage_gate_ratio <= 0.000248943
best safe joint selected=2 total pages
best safe joint total residual gain=0.000074976101
best safe joint max split overerase regret=-0.000000187820
selected pages: scut115/254.jpg, holdout40/477.jpg
```

Named-rule comparison:

```text
strict_cov806_edit98868:
  selected=9 total pages
  total residual gain=0.000699581706
  max split overerase regret=+0.000010712336
  scut115 selected=6 residual_gain=0.000269417458 overerase_regret=-0.000001672613
  holdout40 selected=3 residual_gain=0.000430164248 overerase_regret=+0.000010712336

scut7_cov65_edit98868_p95_5_gate0015:
  selected=10 total pages
  total residual gain=0.000702181859
  max split overerase regret=+0.000010712336
  scut115 selected=7 residual_gain=0.000272017610 overerase_regret=-0.000000667079
  holdout40 selected=3 residual_gain=0.000430164248 overerase_regret=+0.000010712336

best_safe_joint:
  selected=2 total pages
  total residual gain=0.000074976101
  max split overerase regret=-0.000000187820
  scut115 selected=1 residual_gain=0.000001562550 overerase_regret=-0.000000187820
  holdout40 selected=1 residual_gain=0.000073413551 overerase_regret=-0.000000322772
```

Decision:

```text
Do not promote the loose, strict, SCUT7, or best-safe joint selector as the default
product path yet. The only rule that is non-worse on overerase across both splits
selects just 2/155 pages and gives negligible residual gain. The strict and SCUT7
rules give more residual improvement, but holdout40 overerase still rises slightly.
Treat inference-side selector tuning as useful analysis infrastructure, not a
product-quality solution.
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
