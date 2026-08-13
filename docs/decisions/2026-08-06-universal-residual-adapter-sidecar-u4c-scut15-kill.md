# Universal Residual Adapter Sidecar U4C SCUT15 Source Guard Kill

```text
u4c_terminal = KILL
candidate = universal-sidecar-u4a-mixed-scut130-hw5k260-step20
kill_reason = SCUT_INNER_VAL15_SOURCE_GUARD_REGRESSION
hw5k_dev_evaluation = not_run
scut115_evaluation = not_run
holdout40_evaluation = not_run
fresh_blind_handoff = disabled
promotion_handoff = disabled
product_default = artifacts/current-primary
```

## Scope

U4C evaluated the U4B step20 sidecar-only checkpoint on the first SCUT source
guard only:

```text
hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt
```

The evaluation used label-free prediction generation followed by CSV metric
scoring. It did not generate target review crops, manually inspect target
images, use consumed HW5K official test evidence, open fresh blind evidence, or
mutate `artifacts/current-primary`.

## Commands

Candidate prediction generation:

```bash
source .env
$ENSEXAM_PYTHON scripts/infer/run_primary_full_page.py \
  --samples-file hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt \
  --image-dir data-links/samples/SCUT-EnsExam/train/all_images \
  --output-dir outputs/universal_sidecar_u4c_inner_val15_step20_20260806/frozen_predictions \
  --primary-config artifacts/trials/universal-sidecar-u4a-mixed-scut130-hw5k260-step20-20260806/ensexam/20260806_100131/config.yaml \
  --primary-weights artifacts/trials/universal-sidecar-u4a-mixed-scut130-hw5k260-step20-20260806/ensexam/20260806_100131/epoch_1.pth \
  --device auto \
  --page-overlap 32 \
  --batch-size 8 \
  --skip-label-metrics
```

Metric scoring:

```bash
source .env
$ENSEXAM_PYTHON scripts/eval/evaluate_prediction_directory.py \
  --baseline-metrics outputs/scut_innerval15_current_primary_20260802/post_freeze_metrics.csv \
  --pred-dir outputs/universal_sidecar_u4c_inner_val15_step20_20260806/frozen_predictions/pred \
  --output-csv outputs/universal_sidecar_u4c_inner_val15_step20_20260806/post_freeze_metrics.csv \
  --change-threshold 12 \
  --eval-threshold 12
```

## Results

Aggregate command output:

```text
pages = 15
baseline_residual = 0.176949
candidate_residual = 0.176724
residual_gain = 0.000224
baseline_overerase = 0.002325
candidate_overerase = 0.002379
overerase_delta = 0.000054
```

Frozen inner-val15 guard summary:

| Metric | Baseline | Candidate | Delta candidate - baseline | Guard |
| --- | ---: | ---: | ---: | --- |
| residual mean | 0.1769486049 | 0.1767243098 | -0.0002242951 | pass |
| residual p95 | 0.2850987046 | 0.2853410897 | +0.0002423851 | fail |
| residual max | 0.3869877310 | 0.3868621562 | -0.0001255748 | pass |
| overerase mean | 0.0023246714 | 0.0023789961 | +0.0000543247 | fail |
| overerase p95 | 0.0035794460 | 0.0035802018 | +0.0000007558 | pass within `1e-6` |
| overerase max | 0.0061057815 | 0.0063860320 | +0.0002802505 | fail |

Failing guard items:

```text
residual_ratio_p95
overerase_ratio_mean
overerase_ratio_max
```

## Decision

The U4B step20 sidecar-only checkpoint is killed. It must not be evaluated on
HW5K development, SCUT115, holdout40, fresh blind, or promotion gates. The
positive mean residual movement is not actionable because the first SCUT source
guard regressed.

## Follow-Up Boundary

No threshold rescue, postprocess rescue, hard routing, domain labels, source
selectors, base unfreezing, or `current-primary` default mutation is authorized.
Any successor must be a new bounded candidate or cleanup/hygiene task with its
own pass/kill/prerequisite terminal.
