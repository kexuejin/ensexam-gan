# Universal Residual Adapter Sidecar D2D Step80 Half-LR Decision

```text
d2d_terminal = PASS
d2d_preflight_terminal = PASS
d2d_step80_smoke_terminal = PASS
d2d_scut15_source_guard_terminal = PASS
d2d_result = lower_lr_safe_no_lift_causal_evidence
product_default = artifacts/current-primary
fresh_blind = disabled
scut115 = disabled
holdout40_promotion = disabled
current_primary_replacement = disabled
```

## Scope

D2D is the bounded lower-learning-rate causal control after D2 step80 and D2C
step60 both produced the same SCUT15 matched-copy residual regression on
`301.jpg`, while D2B step40 was safe but had no measurable page-level change.
It holds the D2 step80 protocol fixed and halves only the execution-affecting
learning rate from `2.5e-05` to `1.25e-05`. The changed `save_dir` isolates
artifacts; comments describe the causal-control intent.

The approved universal boundary remains unchanged:

- one external clean-image inference surface
- no domain label, caller hint, source/path selector, expert selector, or hard
  routing
- internal image-feature-conditioned continuous soft residual-adapter mixing
  only
- fallback is the same-call `current-primary` baseline output
- current-primary initialization, data, losses, seed, scheduler, and
  matched-copy inference protocol fixed
- sidecar-only generator updates; no optimizer or scheduler state in the
  checkpoint

## Preflight And Training

Config:

```text
configs/local/config.local-universal-sidecar-d2d-d1-mixed-scut130-hw5k260-step80-halflr-mps.yaml
sha256 = c74c5f91085f0556d9898ef79fe467c120f4a771ec99a1a14cd0c3fdcc86ede4
```

The preflight exact-diff audit passed. Apart from comments and the isolated
`save_dir`, D2D differs from D2 only in:

```text
train.lr: 2.5e-05 -> 1.25e-05
```

It found clean data with `383` train files and `70518` patches, loaded
`artifacts/current-primary/micro_region_probe_step0001.pth` with only the
expected `17` new sidecar keys missing, and confirmed `17` trainable sidecar
tensors (`7437 / 24690655` parameters) with `226` frozen tensors. Seed is
`20260806`, scheduler is cosine with `eta_min=1e-06`, and the bounded run is
one epoch with `80` maximum steps.

Run:

```text
artifacts/trials/universal-sidecar-d2d-d1-mixed-scut130-hw5k260-step80-halflr-20260806/ensexam/20260806_232236
```

Training invocation:

```bash
source /Volumes/Tool/source/ensexam-gan/.env
$ENSEXAM_PYTHON train.py \
  --config configs/local/config.local-universal-sidecar-d2d-d1-mixed-scut130-hw5k260-step80-halflr-mps.yaml
```

Training completed on MPS with strict seed control, frozen BatchNorm running
statistics, skipped validation/final test as configured, and checkpoint:

```text
epoch_1.pth
sha256 = 63b66d36349ebed628c9d8855f5a5944c13fe4f1b838f7721da5f23dfbcc315c
```

The checkpoint audit passed:

```text
sidecar_key_count = 17
moved_final_projection_key_count = 8
global_residual_scale.maxabs = 0.001650137361139059
base_changed_count_vs_current_primary = 0
base_missing_count_vs_current_primary = 0
unexpected_non_sidecar_key_count = 0
has_optimizer_state = false
has_scheduler_state = false
failures = []
```

## Matched-Copy SCUT15 Source Guard

The first and only evaluation gate used the exact
`hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt` manifest
(`15` pages; SHA-256
`fb25bb2aef2f9285403f908deb3da6d88b07b5d1c2c812965ce9e0636ddc172e`).
It used the fixed current-primary matched-copy protocol:

```text
copy_input_outside_mask = mb
copy_mask_threshold_auto = mb_cov8_step
copy_mask_dilate = 0
page_overlap = 32
batch_size = 8
change_threshold = 12
eval_threshold = 12
```

Materialization and scoring invocation:

```bash
source /Volumes/Tool/source/ensexam-gan/.env
$ENSEXAM_PYTHON scripts/infer/run_primary_full_page.py \
  --samples-file hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt \
  --image-dir data-links/samples/SCUT-EnsExam/train/all_images \
  --output-dir outputs/universal_sidecar_d2d_d1_step80_halflr_20260806/frozen_predictions \
  --primary-config configs/local/config.local-universal-sidecar-d2d-d1-mixed-scut130-hw5k260-step80-halflr-mps.yaml \
  --primary-weights artifacts/trials/universal-sidecar-d2d-d1-mixed-scut130-hw5k260-step80-halflr-20260806/ensexam/20260806_232236/epoch_1.pth \
  --device mps --page-overlap 32 --batch-size 8 \
  --copy-input-outside-mask mb \
  --copy-mask-threshold-auto mb_cov8_step \
  --skip-label-metrics

$ENSEXAM_PYTHON scripts/eval/evaluate_prediction_directory.py \
  --baseline-metrics /Volumes/Tool/source/ensexam-gan/outputs/scut_innerval15_current_primary_20260802/post_freeze_metrics.csv \
  --pred-dir outputs/universal_sidecar_d2d_d1_step80_halflr_20260806/frozen_predictions/pred \
  --output-csv outputs/universal_sidecar_d2d_d1_step80_halflr_20260806/post_freeze_metrics.csv \
  --change-threshold 12 --eval-threshold 12

$ENSEXAM_PYTHON scripts/analysis/summarize_universal_sidecar_source_guard.py \
  --post-freeze-metrics outputs/universal_sidecar_d2d_d1_step80_halflr_20260806/post_freeze_metrics.csv \
  --expected-samples-file hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt \
  --output-json outputs/universal_sidecar_d2d_d1_step80_halflr_20260806/source_guard_summary.json
```

The current-primary baseline metrics came from the frozen
`outputs/scut_innerval15_current_primary_20260802/post_freeze_metrics.csv`
record. Candidate predictions, post-freeze metrics, and the strict summary are
ignored local artifacts under:

```text
outputs/universal_sidecar_d2d_d1_step80_halflr_20260806/
```

Guard result:

| Metric | Baseline Mean | Candidate Mean | Candidate - Baseline Mean | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual_ratio | 0.17694860490191527 | 0.17694860490191527 | 0.0 | 0.0 | 0.0 |
| overerase_ratio | 0.0023246713604556293 | 0.0023246713604556293 | 0.0 | 0.0 | 0.0 |

```text
source_guard_status = pass
failures = []
measurable_page_delta = false
nonzero_delta_files = []
post_freeze_metrics_sha256 = 368e23d52ae677b4e8837c3ce667319b848693293be5c833f3802baca159ec61
source_guard_summary_sha256 = ea97c5464f9b2d9bc2b37dccc766782141eb84f037784198cff1ed8067738dc6
```

## Decision

D2D passes the first source guard as lower-LR causal evidence: halving the
step80 learning rate removes the D2/D2C thresholded SCUT15 source regression.
It does not produce a measurable page-level delta, so it is a safe/no-lift
result, not a quality candidate and not evidence for any promotion decision.

Do not run fresh blind, SCUT115, or holdout40 promotion evaluation for D2D and
do not replace `current-primary`. Do not use copy-mask threshold tuning to turn
this no-lift result into a lift claim. A future development-lift/source-guard
goal must be independently registered and retain the same universal interface
and first-gate failure rule.

Intent: Isolate cumulative update magnitude as the cause of the D2/D2C SCUT15 residual regression.
Constraint: D2D may change only the D2 learning rate; it must preserve the universal interface, sidecar-only trainable set, data, losses, seed, scheduler, initialization, and inference protocol.
Rejected: Broader SCUT115, holdout40, fresh blind, and promotion evaluation | no measurable development delta exists to justify consuming those gates.
Rejected: Copy-mask threshold tuning | would alter the frozen matched-copy source-guard protocol rather than test the learning-rate control.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Treat D2D as safe/no-lift causal evidence only; open a new independently scoped goal before testing a lift mechanism.
Tested: Exact-diff/data/checkpoint/trainable-pattern preflight; bounded 80-step training; checkpoint isolation audit; 15-page matched-copy SCUT15 inference; fixed post-freeze scoring; strict source-guard summarizer.
Not-tested: Fresh blind, SCUT115, holdout40 promotion, current-primary replacement, or any product-quality claim; all remain intentionally disabled.
Related: docs/decisions/2026-08-06-universal-residual-adapter-sidecar-d2-step80-kill.md; docs/decisions/2026-08-06-universal-residual-adapter-sidecar-d2b-step40-decision.md; docs/decisions/2026-08-06-universal-residual-adapter-sidecar-d2c-step60-kill.md
