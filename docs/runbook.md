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

Strict nearworst-safe reproduction:

```text
registered patch index: hardcase_lists/nearworst_safe_step1_exact129_patch_index.csv
exact patch: 129.jpg x1=320 y1=160 x2=576 y2=416
```

Use the registered patch index when reproducing the one-step candidate. A random one-step rerun is
not equivalent because patch identity changes gate features.

```bash
$ENSEXAM_PYTHON scripts/train/micro_train_region_probe.py \
  --config configs/local/config.local-current-primary-continuation-mps.yaml \
  --output-dir outputs/exp_current_primary_nearworst_safe_step1_exact129_YYYYMMDD \
  --max-steps 1 \
  --batch-size 1 \
  --train-pages 16 \
  --patch-index-file hardcase_lists/nearworst_safe_step1_exact129_patch_index.csv \
  --loss-override lambda_input_preserve=24.0 \
  --loss-override lambda_mb_leak=2.0 \
  --trace-batches-file outputs/exp_current_primary_nearworst_safe_step1_exact129_YYYYMMDD/trace_batches.csv \
  --log-every 1 \
  --save-every 1
```

Validation on 2026-07-06:

```text
exact129 strict scut115: selected=5/115 residual=0.113987486262 overerase=0.003046587193
exact129 strict holdout40: selected=3/40 residual=0.133642377143 overerase=0.002492527893
exact129 cov806/edit98908 scut115: selected=6/115 residual=0.113988387733 overerase=0.003046310539
exact129 cov806/edit98908 holdout40: selected=3/40 residual=0.133642377143 overerase=0.002492527893
random one-step rerun scut115: selected=1/115 residual=0.114234747159 overerase=0.003057036945
random one-step rerun holdout40: selected=2/40 residual=0.134134815794 overerase=0.002521871278
```

The `edit98908` threshold only admits the SCUT `254.jpg` near-miss relative to `edit98868`; it
does not add holdout pages. Treat it as a reproducibility variant, not a product default.

Joint selector replay on the exact129 candidate:

```text
output: outputs/selector_replay_joint_exact129_cov806_edit98908_20260706
best safe rule: selected=3/155 total residual gain=0.000144868289
exact129_cov806_edit98868: selected=8 total residual gain=0.000621405155 max overerase regret=+0.000010921776
exact129_cov806_edit98908: selected=9 total residual gain=0.000620503683 max overerase regret=+0.000010921776
```

The selector replay reaches the same conclusion as earlier candidates: label-free selector tuning
alone is not enough for a product default. The safe rule is too small, while the larger strict rules
still carry a small holdout overerase regression.

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

### Candidate-Only Overerase Delta

Use the local component analyzer to inspect where the candidate introduces
background edits that the baseline does not:

```bash
$ENSEXAM_PYTHON scripts/analysis/analyze_candidate_overerase_delta.py \
  --page-choices outputs/selector_replay_joint_strict_candidate_20260705/page_choices.csv \
  --output-dir outputs/analysis_joint_candidate_overerase_delta_script_20260705 \
  --positive-gain-only \
  --max-crops 40
```

Diagnostic result:

```text
pages with residual gain and overerase increase=106
new candidate-only overerase components=649
components summary:
  small_background_edit: 431 components, area=21354
  page_edge_artifact: 127 components, area=7363
  near_changed_region_halo: 57 components, area=2492
  near_target_boundary_or_low_contrast_label: 28 components, area=1286
```

A quick top24 risk-page protection probe tested reverting candidate-only
background components back to baseline. It reduced overerase only modestly and
kept every tested high-risk page unsafe:

```text
none:           residual_gain=0.176509 overerase_regret=0.015874 safe_pages=0/24
edge_only:      residual_gain=0.176170 overerase_regret=0.015132 safe_pages=0/24
small_nonlarge: residual_gain=0.154762 overerase_regret=0.012318 safe_pages=0/24
small_or_edge:  residual_gain=0.154762 overerase_regret=0.012291 safe_pages=0/24
all_components: residual_gain=0.154762 overerase_regret=0.012248 safe_pages=0/24
```

Decision:

```text
Do not add a simple candidate-only component protection switch to product
inference. The overerase increase is distributed across many small background
edits and page-edge artifacts; reverting those components sacrifices residual
gain but does not make high-risk pages safe. A useful protection method likely
needs better model-side confidence/background preservation, not only connected
component filtering after inference.
```

### Patch Sensitivity Queue

Use this when testing whether the exact one-step patch identity, rather than a
loss or selector change, explains strict-gate behavior. It creates one patch CSV,
one training command, and two gate-eval commands per candidate patch, so the
experiment is reproducible and does not depend on random `DataLoader` order.

```bash
$ENSEXAM_PYTHON scripts/experimental/build_patch_sensitivity_queue.py \
  --patch-index-csv outputs/<ranked_patch_index>/patch_index.csv \
  --output-dir outputs/patch_sensitivity_queue_YYYYMMDD \
  --limit 16 \
  --experiment-prefix patch_sensitivity \
  --date-tag YYYYMMDD
```

Outputs:

```text
manifest.csv
patch_indices/*.csv
commands/01_train.sh
commands/02_eval.sh
commands/run_all.sh
```

This tool only prepares deterministic local experiment queues. Commit the script
and resulting decision records, not failed sweep outputs or large checkpoints.

### Mask-Confidence Candidate Diagnostics

Use this when a saved candidate reduces residual on many pages but fails product promotion because
overerase rises. The diagnostic re-runs the primary model to recover `ms` / `mb` masks, compares the
saved candidate against baseline metrics, and writes per-page features for local selector analysis.

```bash
$ENSEXAM_PYTHON scripts/analysis/analyze_mask_confidence_features.py \
  --config configs/local/config.local-lowdiff-outside-edit-mps.yaml \
  --weights outputs/<experiment>/micro_region_probe_step0001.pth \
  --samples-file docs/holdout40-relative.txt \
  --baseline-metrics outputs/holdout40_second_stage_readiness_20260705/metrics.csv \
  --candidate-dir outputs/<eval>/candidate \
  --output-dir outputs/analysis_mask_confidence_<tag> \
  --device mps \
  --batch-size 8
```

Treat this as diagnostic evidence, not a product selector by itself. A threshold rule from the CSV
still needs joint SCUT115 + holdout40 replay before promotion, and failed threshold sweeps should be
rolled into consolidated rejected-direction records rather than committed one by one.

Top4 train-split high-stroke sweep:

```text
patch index: outputs/train_hard_patch_index_for_sensitivity_20260706/patch_index.csv
queue: outputs/patch_sensitivity_train_top4_20260706
summary: outputs/patch_sensitivity_train_top4_20260706/summary.csv
patches:
  001_362_x1280_y320
  002_362_x1280_y640
  003_362_x1280_y480
  004_362_x1440_y480
result:
  scut115 selected=0/115 residual=0.114224963938 overerase=0.003048296717
  holdout40 selected=0/40 residual=0.134026304621 overerase=0.002481606117
```

Decision: do not expand the naive high-stroke train-patch sweep. The top ranked
training patches all collapse to baseline under the strict gate, so this is not
a useful product-quality route by itself. Future patch selection should target
gate-feature preservation explicitly (`copy_mask_cov8`, `primary_edit_px`) rather
than only high local handwriting/stroke density.

Top4 exact129 low-diff anchor sweep:

```bash
$ENSEXAM_PYTHON scripts/experimental/build_anchor_similar_patch_list.py \
  --config configs/local/config.local-current-primary-continuation-mps.yaml \
  --anchor-patch-csv hardcase_lists/nearworst_safe_step1_exact129_patch_index.csv \
  --train-file-list hardcase_lists/scut_train_hard_proxy_160.txt \
  --output-csv outputs/anchor_similar_lowdiff_exact129_patch_index_20260706/patch_index.csv \
  --train-pages 160 \
  --top-k 16 \
  --exclude-anchor
```

```text
queue: outputs/patch_sensitivity_anchor_lowdiff_top4_20260706
summary: outputs/patch_sensitivity_anchor_lowdiff_top4_20260706/summary.csv
001_161_x1120_y0:   scut selected=1/115 residual=0.114160943919 overerase=0.003060213443; holdout selected=2/40 residual=0.134691305786 overerase=0.002524365006
002_130_x2240_y960: scut selected=3/115 residual=0.114388655784 overerase=0.003050635477; holdout selected=2/40 residual=0.133796433250 overerase=0.002500568850
003_84_x800_y480:   scut selected=5/115 residual=0.114007786589 overerase=0.003061889507; holdout selected=3/40 residual=0.133760842647 overerase=0.002508892113
004_378_x0_y960:    scut selected=1/115 residual=0.114839309779 overerase=0.003155448069; holdout selected=0/40 residual=0.134026304621 overerase=0.002481606117
```

Decision: low-diff anchor-similar patches are a useful signal because they can
restore strict-gate eligibility, unlike high-stroke patches. They are not a
product default yet: the best SCUT residual case still increases SCUT and
holdout overerase, and one patch worsens SCUT residual. Next step should tune
selector thresholds or training loss around the low-diff family while explicitly
penalizing edit-size growth.

### Preservation Weight Probes

The existing loss already has `input_preserve` and `mb_leak` terms. A logging
update exposed those parts in `micro_loss_history.csv`, then two 2-step MPS
probes tested stronger preservation:

```bash
$ENSEXAM_PYTHON scripts/train/micro_train_region_probe.py \
  --config configs/local/config.local-current-primary-continuation-mps.yaml \
  --output-dir outputs/exp_preserve16_leak0p75_step2_20260705 \
  --max-steps 2 \
  --batch-size 1 \
  --train-pages 8 \
  --disable-augmentation \
  --trace-batches-file outputs/exp_preserve16_leak0p75_step2_20260705/trace_batches.csv \
  --log-every 1 \
  --save-every 1 \
  --device-override mps \
  --loss-override lambda_input_preserve=16.0 \
  --loss-override lambda_mb_leak=0.75

$ENSEXAM_PYTHON scripts/train/micro_train_region_probe.py \
  --config configs/local/config.local-current-primary-continuation-mps.yaml \
  --output-dir outputs/exp_preserve24_leak1_step2_20260705 \
  --max-steps 2 \
  --batch-size 1 \
  --train-pages 8 \
  --disable-augmentation \
  --trace-batches-file outputs/exp_preserve24_leak1_step2_20260705/trace_batches.csv \
  --log-every 1 \
  --save-every 1 \
  --device-override mps \
  --loss-override lambda_input_preserve=24.0 \
  --loss-override lambda_mb_leak=1.0
```

Both probes trained stably, but strict-gate evaluation got too conservative:

```text
preserve16/leak0.75:
  scut115 residual=0.114210374926 overerase=0.003048663423 selected=1/115 files=303.jpg
  holdout40 residual=0.134427251448 overerase=0.002498697889 selected=3/40 files=466.jpg,268.jpg,341.jpg

preserve24/leak1:
  scut115 residual=0.114210460744 overerase=0.003048568711 selected=1/115 files=303.jpg
  holdout40 residual=0.134429857005 overerase=0.002498202272 selected=3/40 files=466.jpg,268.jpg,341.jpg

original strict:
  scut115 residual=0.113955546480 overerase=0.003046624105 selected=6/115 files=17.jpg,156.jpg,254.jpg,303.jpg,370.jpg,371.jpg
  holdout40 residual=0.133596140373 overerase=0.002492318453 selected=3/40 files=193.jpg,466.jpg,268.jpg
```

Decision:

```text
Do not continue simply increasing lambda_input_preserve/lambda_mb_leak from the
current checkpoint. It suppresses gate eligibility and loses the original strict
residual gains while still not beating baseline overerase. Future model-side work
needs a more selective preserve target, not a global preservation-weight increase.
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
