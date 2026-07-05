# Model Registry

## Current Primary EnsExam-GAN Checkpoint

Checkpoint:

```text
artifacts/current-primary/micro_region_probe_step0001.pth
```

Config:

```text
artifacts/current-primary/config.yaml
```

Inference gate:

```text
copy mask: mb
auto threshold: mb_cov8_step
base threshold fallback: 70
dilation: 0
```

## Full-Training Base Checkpoint

The previous one-day full-training run is directly reusable in this fork. Do not repeat that full training by default.

Registered local checkpoint:

```text
artifacts/full-training-best.pth
```

Source run:

```text
artifacts/full-training/20260702_070153/
```

Local copied source:

```text
artifacts/full-training-best.pth
```

Use this checkpoint as the reusable full-training base / baseline for comparison, rollback, and new targeted continuation experiments. Rerun broad full training only if the run artifacts are invalid, incompatible, or later evidence shows a reset is better than continuing from the existing full-training base.

Default continuation path:

```text
start from artifacts/full-training-best.pth when a clean full-training base is needed
or start from artifacts/current-primary/micro_region_probe_step0001.pth for current best continuation
-> targeted hardcase fine-tune / probe
-> local target-vs-prediction evaluation
-> promote only if residual improves without visible overerase regression
```

Current-best continuation config:

```text
configs/local/config.local-current-primary-continuation-mps.yaml
```

## Current Second-Stage Residual Repair Checkpoint

Checkpoint:

```text
artifacts/current-second-stage-best.pt
```

Default promoted gate:

```text
cleanup_alpha_threshold: 0.3
base_edit_threshold: 12
second_delta_threshold: 32
dark_threshold: 0
```

## Hybrid Gate Product Candidate

This is the strongest current product candidate, but it is not promoted as the default pipeline yet.
It chooses per page between the current baseline second-stage output and the `nearworst_safe_step1`
candidate using inference-time features only:

```text
script: scripts/run_hybrid_second_stage_gate.py
candidate weights: outputs/exp_current_primary_nearworst_safe_step1_20260705/micro_region_probe.pth
baseline pred dir: outputs/holdout40_second_stage_readiness_20260705/pred
candidate copy mask: mb
candidate copy threshold: 98
min_copy_mask_cov8: 0.18436555
max_primary_edit_px: 107112
```

Holdout40 result:

```text
output: outputs/hybrid_gate_nearworst_safe_step1_t98_holdout40_20260705
summary residual=0.131148 overerase=0.002547 selected=14/40
baseline second-stage residual=0.134026 overerase=0.002482
candidate-all second-stage residual=0.130543 overerase=0.002732
```

SCUT test115 result:

```text
samples: docs/scut-test115-relative.txt
baseline output: outputs/scut_test115_second_stage_baseline_20260705
hybrid output: outputs/scut_test115_hybrid_gate_nearworst_safe_step1_t98_20260705
review pack: outputs/review_scut_test115_hybrid_gate_nearworst_safe_step1_t98_20260705
baseline second-stage residual=0.114225 overerase=0.003048
hybrid residual=0.112203 overerase=0.003125 selected=29/115
delta residual=-0.002022 overerase=+0.000077
```

Strict SCUT test115 gate:

```text
rule: copy_mask_cov8 >= 0.806133 and primary_edit_px <= 98868
output: outputs/scut_test115_hybrid_gate_strict_cov806_edit98868_20260705
review pack: outputs/review_scut_test115_hybrid_gate_strict_cov806_edit98868_20260705
visible-delta analysis: outputs/analysis_visible_delta_strict_scut_test115_20260705_rerun
baseline second-stage residual=0.114225 overerase=0.003048
strict hybrid residual=0.113956 overerase=0.003047 selected=6/115
delta residual=-0.000269 overerase=-0.000002
selected pages: 17.jpg 156.jpg 254.jpg 303.jpg 370.jpg 371.jpg
manual contact-sheet pass: no obvious large-area overerase regression, but visual gain is subtle
diff-crop review: most selected-page changes are low-contrast texture / gray-balance shifts; only a subset shows visible cleanup benefit
visible-delta components: improve area=464 across 11 components; regress area=229 across 8 components
visible-delta patch index: outputs/visible_delta_patch_index_strict_scut_test115_20260705/improve_patch_index.csv
visible-delta rejects: outputs/visible_delta_patch_index_strict_scut_test115_20260705/regress_reject_components.csv
```

Promotion gate:

```text
Do not promote either gate as the default pipeline. The stricter gate is safer on aggregate
metrics, but diff-crop review shows most gains are subtle texture / gray-balance shifts rather
than clear product-visible cleanup. Treat it as a research candidate, not a product candidate,
unless a future visual review proves consistent full-size page improvements.
```

## Validation Anchors

```text
hardcase7: residual 0.263922, overerase 0.001125
holdout40: residual 0.134026, overerase 0.002482
scut-test115 second-stage: residual 0.114225, overerase 0.003048
```

## New-Project Readiness Reproduction

The forked workspace reproduced the registered holdout40 second-stage anchor using only new-project paths and registered local payloads:

```text
samples: docs/holdout40-relative.txt
output: outputs/holdout40_second_stage_readiness_20260705
summary residual=0.134026 overerase=0.002482
metrics_csv: outputs/holdout40_second_stage_readiness_20260705/metrics.csv
```

## Rejected / Not Default

Whiteout inpaint repair is metric-positive but visually rejected. Do not enable it in the default product pipeline without a better paper-tone restoration method and manual visual approval.

The 2026-07-05 current-primary four-step continuation run is also rejected for promotion:

```text
run: outputs/exp_current_primary_continuation_step4_20260705
eval: outputs/eval_current_primary_continuation_step4_holdout40_20260705/summary.csv
baseline primary residual=0.136111 overerase=0.002482
best candidate by score: step0001 residual=0.138113 overerase=0.002797 score=-0.004524
decision: no promotion; keep artifacts/current-primary unchanged
```
