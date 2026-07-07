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

## Region Component Reviewed-Label Workflow

Page-level selector tuning and weak component labels have reached a low-safety
ceiling. Use this workflow when continuing toward a learned region selector.

Generate component features and rule-probe outputs from the residual-delta t4
candidate:

```bash
$ENSEXAM_PYTHON scripts/analysis/evaluate_region_component_selector.py \
  --split scut115:outputs/scut_test115_second_stage_baseline_20260705/metrics.csv:outputs/sweep_scut115_residual_delta_t4_20260707/metrics.csv \
  --split holdout40:outputs/holdout40_second_stage_nearworst_safe_step1_t98_20260705/metrics.csv:outputs/sweep_holdout40_residual_delta_t4_20260707/metrics.csv \
  --split train160:outputs/scut_train160_nonholdout_second_stage_baseline_20260706/metrics.csv:outputs/sweep_train160_residual_delta_t4_20260707/metrics.csv \
  --split next120:outputs/scut_next120_nonoverlap_second_stage_baseline_20260706/metrics.csv:outputs/sweep_next120_residual_delta_t4_20260707/metrics.csv \
  --train-split train160 \
  --train-split next120 \
  --test-split scut115 \
  --test-split holdout40 \
  --output-dir outputs/region_component_selector_t4_ratio05_train160_next120_to_scut115_holdout40_20260707 \
  --max-conditions 2 \
  --max-single-candidates 40 \
  --max-rules 50 \
  --min-train-components 500 \
  --min-test-pages 1 \
  --max-train-reject-components 100000 \
  --max-train-reject-ratio 0.05
```

Build a balanced held-out review pack for component labels:

```bash
$ENSEXAM_PYTHON scripts/analysis/build_region_component_review_pack.py \
  --components-csv outputs/region_component_selector_t4_ratio05_train160_next120_to_scut115_holdout40_20260707/components.csv \
  --split scut115:outputs/scut_test115_second_stage_baseline_20260705/metrics.csv:outputs/sweep_scut115_residual_delta_t4_20260707/metrics.csv \
  --split holdout40:outputs/holdout40_second_stage_nearworst_safe_step1_t98_20260705/metrics.csv:outputs/sweep_holdout40_residual_delta_t4_20260707/metrics.csv \
  --split train160:outputs/scut_train160_nonholdout_second_stage_baseline_20260706/metrics.csv:outputs/sweep_train160_residual_delta_t4_20260707/metrics.csv \
  --split next120:outputs/scut_next120_nonoverlap_second_stage_baseline_20260706/metrics.csv:outputs/sweep_next120_residual_delta_t4_20260707/metrics.csv \
  --allowed-split scut115 \
  --allowed-split holdout40 \
  --output-dir outputs/region_component_review_pack_t4_heldout_20260707 \
  --max-total 60 \
  --max-per-page 3 \
  --crop-size 220 \
  --thumb-size 180
```

Review these generated artifacts:

```text
outputs/region_component_review_pack_t4_heldout_20260707/contact_sheet_high_gain_accept.png
outputs/region_component_review_pack_t4_heldout_20260707/contact_sheet_borderline_review.png
outputs/region_component_review_pack_t4_heldout_20260707/contact_sheet_hard_reject.png
outputs/region_component_review_pack_t4_heldout_20260707/component-labels-template.csv
```

Build an impact-ranked held-out review pack when review time is limited. This
is preferred for the next selector pass because rows are prioritized by actual
page-level residual / overerase pixel deltas rather than weak component verdict
balance:

```bash
$ENSEXAM_PYTHON scripts/analysis/build_region_component_impact_review_pack.py \
  --components-csv outputs/region_component_selector_t4_ratio05_train160_next120_to_scut115_holdout40_20260707/components.csv \
  --score-csv outputs/region_component_ranker_t4_train160_next120_to_scut115_holdout40_20260707/predictions.csv \
  --min-score 0.5130248089880388 \
  --split scut115:outputs/scut_test115_second_stage_baseline_20260705/metrics.csv:outputs/sweep_scut115_residual_delta_t4_20260707/metrics.csv \
  --split holdout40:outputs/holdout40_second_stage_nearworst_safe_step1_t98_20260705/metrics.csv:outputs/sweep_holdout40_residual_delta_t4_20260707/metrics.csv \
  --allowed-split scut115 \
  --allowed-split holdout40 \
  --output-dir outputs/region_component_impact_review_pack_weak_t0513_20260707 \
  --max-total 80 \
  --max-per-page 4 \
  --crop-size 220 \
  --thumb-size 180
```

Review these generated impact artifacts:

```text
outputs/region_component_impact_review_pack_weak_t0513_20260707/contact_sheet.png
outputs/region_component_impact_review_pack_weak_t0513_20260707/contact_sheet_residual_help.png
outputs/region_component_impact_review_pack_weak_t0513_20260707/contact_sheet_residual_hurt.png
outputs/region_component_impact_review_pack_weak_t0513_20260707/contact_sheet_overerase_risk.png
outputs/region_component_impact_review_pack_weak_t0513_20260707/contact_sheet_large_noop.png
outputs/region_component_impact_review_pack_weak_t0513_20260707/component-impact-labels-template.csv
```

The 2026-07-07 weak-threshold impact pack selected 76 components from 10,041
impact-scored rows: 30 residual_help, 30 residual_hurt, 14 overerase_risk, and
2 large_noop. Use this pack before spending more time on weak-label threshold
micro-tuning.

Measure the target-aware oracle ceiling before investing in another selector
loop for the same candidate family:

```bash
$ENSEXAM_PYTHON scripts/analysis/evaluate_region_component_oracle_ceiling.py \
  --components-csv outputs/region_component_selector_t4_ratio05_train160_next120_to_scut115_holdout40_20260707/components.csv \
  --split scut115:outputs/scut_test115_second_stage_baseline_20260705/metrics.csv:outputs/sweep_scut115_residual_delta_t4_20260707/metrics.csv \
  --split holdout40:outputs/holdout40_second_stage_nearworst_safe_step1_t98_20260705/metrics.csv:outputs/sweep_holdout40_residual_delta_t4_20260707/metrics.csv \
  --output-csv outputs/region_component_oracle_ceiling_t4_20260707/oracle_ceiling.csv
```

The 2026-07-07 t4 oracle ceiling is low even with target-aware component
selection: SCUT115 best residual_gain is 0.000941709 and holdout40 best
residual_gain is 0.000452334, both with zero worse pages. Treat this as the
stop condition for more selector micro-tuning on this candidate family.

Fill `component-labels-template.csv` using the contract in
`docs/region-component-labeling.md`. Prefer labels `keep`, `drop`, and
`review`; only `keep` and `drop` train the ranker by default.

Validate reviewed labels before training:

```bash
$ENSEXAM_PYTHON scripts/analysis/validate_region_component_labels.py \
  --components-csv outputs/region_component_selector_t4_ratio05_train160_next120_to_scut115_holdout40_20260707/components.csv \
  --label-csv outputs/region_component_review_pack_t4_heldout_20260707/component-labels-template.csv \
  --require-positive 1 \
  --require-negative 1 \
  --fail-on-unknown-label \
  --fail-on-unmatched \
  --fail-on-duplicates \
  --output-csv outputs/region_component_review_pack_t4_heldout_20260707/label-validation.csv
```

Train a reviewed-label region ranker:

```bash
$ENSEXAM_PYTHON scripts/analysis/train_region_component_ranker.py \
  --components-csv outputs/region_component_selector_t4_ratio05_train160_next120_to_scut115_holdout40_20260707/components.csv \
  --label-csv outputs/region_component_review_pack_t4_heldout_20260707/component-labels-template.csv \
  --train-split scut115 \
  --train-split holdout40 \
  --test-split train160 \
  --test-split next120 \
  --output-dir outputs/region_component_ranker_reviewed_20260707 \
  --epochs 2500 \
  --lr 0.01 \
  --l2 0.05 \
  --positive-mode accept \
  --min-train-selected 20 \
  --min-test-selected 100 \
  --max-rows 80
```

Do not promote a selector from weak local-proxy labels alone. Require reviewed
held-out labels with near-zero reject rate at useful coverage before product
gating.

Materialize a component selector into page-level predictions after choosing a
ranker threshold or fixed component rule:

```bash
THRESHOLD=$($ENSEXAM_PYTHON - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(Path("outputs/region_component_ranker_reviewed_20260707/threshold_summary.csv").open()))
print(rows[0]["threshold"])
PY
)

$ENSEXAM_PYTHON scripts/infer/materialize_region_component_selector.py \
  --components-csv outputs/region_component_selector_t4_ratio05_train160_next120_to_scut115_holdout40_20260707/components.csv \
  --predictions-csv outputs/region_component_ranker_reviewed_20260707/predictions.csv \
  --score-threshold "$THRESHOLD" \
  --split scut115:outputs/scut_test115_second_stage_baseline_20260705/metrics.csv:outputs/sweep_scut115_residual_delta_t4_20260707/metrics.csv \
  --split holdout40:outputs/holdout40_second_stage_nearworst_safe_step1_t98_20260705/metrics.csv:outputs/sweep_holdout40_residual_delta_t4_20260707/metrics.csv \
  --output-dir outputs/region_component_materialized_reviewed_20260707 \
  --write-empty-pages
```

The materializer starts from each baseline prediction and copies candidate
pixels only inside selected connected components. It writes
`selection.csv`, `component-selection.csv`, and per-split `pred/*.png`
outputs for downstream page-level metric evaluation.

Evaluate a materialized `pred` directory without rerunning a model:

```bash
$ENSEXAM_PYTHON scripts/eval/evaluate_prediction_directory.py \
  --baseline-metrics outputs/holdout40_second_stage_nearworst_safe_step1_t98_20260705/metrics.csv \
  --pred-dir outputs/region_component_materialized_reviewed_20260707/holdout40/pred \
  --output-csv outputs/region_component_materialized_reviewed_20260707/holdout40-evaluated-metrics.csv
```

The evaluator reuses the same SCUT target-derived residual and overerase
metrics as the hardcase evaluation scripts. It is for offline validation only;
selector decisions should still use inference-time rules or reviewed ranker
scores, not target-derived component fields.

Sweep component-score thresholds without writing per-threshold PNGs:

```bash
$ENSEXAM_PYTHON scripts/analysis/evaluate_region_component_threshold_sweep.py \
  --components-csv outputs/region_component_selector_t4_ratio05_train160_next120_to_scut115_holdout40_20260707/components.csv \
  --predictions-csv outputs/region_component_ranker_reviewed_20260707/predictions.csv \
  --threshold-summary-csv outputs/region_component_ranker_reviewed_20260707/threshold_summary.csv \
  --split scut115:outputs/scut_test115_second_stage_baseline_20260705/metrics.csv:outputs/sweep_scut115_residual_delta_t4_20260707/metrics.csv \
  --split holdout40:outputs/holdout40_second_stage_nearworst_safe_step1_t98_20260705/metrics.csv:outputs/sweep_holdout40_residual_delta_t4_20260707/metrics.csv \
  --output-csv outputs/region_component_reviewed_threshold_sweep_20260707/page_threshold_summary.csv
```

Use this before materializing many threshold candidates. It evaluates threshold
effects by summing component-level residual and overerase pixel deltas in memory,
which avoids filling the disk with repeated `pred/*.png` directories.

Summarize residual-delta candidate families before launching another training
or selector loop:

```bash
$ENSEXAM_PYTHON scripts/analysis/summarize_candidate_metric_families.py \
  --baseline scut115:outputs/scut_test115_second_stage_baseline_20260705/metrics.csv \
  --baseline holdout40:outputs/holdout40_second_stage_nearworst_safe_step1_t98_20260705/metrics.csv \
  --baseline train160:outputs/scut_train160_nonholdout_second_stage_baseline_20260706/metrics.csv \
  --baseline next120:outputs/scut_next120_nonoverlap_second_stage_baseline_20260706/metrics.csv \
  --outputs-root outputs \
  --name-contains residual_delta \
  --output-csv outputs/candidate_family_summary_residual_delta_20260707/summary.csv \
  --top-n-per-split 20
```

The 2026-07-07 residual-delta triage keeps `sweep_*_residual_delta_t4` as the
best existing candidate benchmark, not a product default. A new candidate
objective should beat t4 on SCUT115 residual and reduce the holdout40 overerase
penalty; do not promote variants that mainly buy lower overerase by worsening
SCUT115 residual.

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

Exact129 outside-edit interval gate on 2026-07-06:

```text
candidate: outputs/exp_exact129_outside_edit_lam16_step1_20260706/micro_region_probe.pth
analysis replay: outputs/selector_replay_exact129_outside_edit_lam16_interval_relaxed_20260706
real SCUT115 gate: outputs/eval_scut115_exact129_outside_edit_lam16_interval_relaxed_gate_20260706
real holdout40 gate: outputs/eval_holdout40_exact129_outside_edit_lam16_interval_relaxed_gate_20260706

selector interval:
  copy_mask_cov8: 0.807064 <= cov8 <= 0.881053
  primary_edit_px: 12580 <= edit_px <= 98699
  primary_p95_edit_delta: 0 <= p95 <= 4.667
  second_stage_gate_ratio: 0.0004928 <= gate <= 0.0010997

SCUT115 baseline: residual=0.114224963938 overerase=0.003048296717
SCUT115 interval: selected=5/115 residual=0.113964590999 overerase=0.003046575437
SCUT115 selected: 17.jpg, 156.jpg, 303.jpg, 370.jpg, 371.jpg

holdout40 baseline: residual=0.134026304621 overerase=0.002481606117
holdout40 interval: selected=1/40 residual=0.133963087233 overerase=0.002481294748
holdout40 selected: 466.jpg
```

Decision: this relaxed interval was useful for finding selector headroom, but it is no longer a
productization candidate after the train160 follow-up below. The improvement came from
inference-time interval gating, not lambda-only retraining; keep generated output directories out
of commits.

Extended train160 non-holdout validation:

```text
sample list: docs/scut-train160-nonholdout-relative.txt
baseline: outputs/scut_train160_nonholdout_second_stage_baseline_20260706
relaxed interval eval: outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_interval_relaxed_gate_20260706
refined replay: outputs/selector_replay_exact129_outside_edit_lam16_interval_train160_refined_20260706

train160 baseline: residual=0.144527160040 overerase=0.002314662644
relaxed interval: selected=3/160 residual=0.144524066276 overerase=0.002322399968
relaxed selected: 166.jpg, 190.jpg, 192.jpg
relaxed decision: unsafe; overerase regresses by +0.000007737324 on train160.

refined interval adds an edit floor:
  copy_mask_cov8: 0.807064 <= cov8 <= 0.881053
  primary_edit_px: 67887 <= edit_px <= 98699
  primary_p95_edit_delta: 0 <= p95 <= 4.667
  second_stage_gate_ratio: 0.0004928 <= gate <= 0.0010997

refined replay across SCUT115 + holdout40 + train160:
  scut115 selected=4/115 residual_gain=0.000048330735 overerase_regret=-0.000003493826
  holdout40 selected=1/40 residual_gain=0.000063217389 overerase_regret=-0.000000311370
  train160 selected=0/160 residual_gain=0 overerase_regret=0
```

Decision update: the earlier relaxed interval is not broad-validation safe. The refined edit-floor
rule is safer but drops the large SCUT 156.jpg gain, so it should be treated as a conservative
selector hypothesis rather than a strong productization candidate. Larger validation and visual
review are still required before default use.

Exact129 outside-edit OR-of-intervals union gate:

```text
date: 2026-07-06
candidate: outputs/exp_exact129_outside_edit_lam16_step1_20260706/micro_region_probe.pth
replay: outputs/selector_replay_exact129_outside_edit_lam16_union_train160_20260706
real SCUT115 gate: outputs/eval_scut115_exact129_outside_edit_lam16_union_gate_20260706
real holdout40 gate: outputs/eval_holdout40_exact129_outside_edit_lam16_union_gate_20260706
real train160 gate: outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_union_gate_20260706

low156 interval:
  copy_mask_cov8: 0.807 <= cov8 <= 0.8072
  primary_edit_px: 0 <= edit_px <= 13000
  primary_p95_edit_delta: 0 <= p95 <= 1.7
  second_stage_gate_ratio: 0 <= gate <= 0.0011

normal interval:
  copy_mask_cov8: 0.807064 <= cov8 <= 0.881053
  primary_edit_px: 67887 <= edit_px <= 98699
  primary_p95_edit_delta: 0 <= p95 <= 4.667
  second_stage_gate_ratio: 0.0004928 <= gate <= 0.0010997

SCUT115 union: selected=5/115 residual=0.113964591000 overerase=0.003046575437
SCUT115 selected: 17.jpg, 156.jpg, 303.jpg, 370.jpg, 371.jpg
holdout40 union: selected=1/40 residual=0.133963087233 overerase=0.002481294748
holdout40 selected: 466.jpg
train160 union: selected=0/160 residual=0.144527160040 overerase=0.002314662644

delta vs baseline:
  SCUT115 residual_delta=-0.000260372939 overerase_delta=-0.000001721280
  holdout40 residual_delta=-0.000063217389 overerase_delta=-0.000000311370
  train160 residual_delta=+0.000000000000 overerase_delta=+0.000000000000
```

Decision update: the two-box OR union gate is the current best selector hypothesis. It restores the
large SCUT 156.jpg gain that the refined single interval dropped, keeps the holdout40 466.jpg gain,
and avoids selecting the train160 bad pages from the relaxed interval. It is still a selector
hypothesis rather than product default until larger non-overlapping validation and manual visual
review pass.

Union visible-delta local review:

```text
SCUT115 review pack: outputs/analysis_visible_delta_union_scut115_20260706
holdout40 review pack: outputs/analysis_visible_delta_union_holdout40_20260706

SCUT115:
  improve_visible_target_region: components=11 area=458
  regress_visible_target_region: components=5 area=144
  regress_low_contrast_target: components=1 area=37
  by page:
    156.jpg improve=8:304 regress_low_contrast=1:37
    17.jpg improve=3:154 regress_visible=3:85
    303.jpg regress_visible=1:27
    370.jpg regress_visible=1:32

holdout40:
  improve_visible_target_region: components=5 area=501
  regress_visible_target_region: components=7 area=249
  by page:
    466.jpg improve=5:501 regress_visible=7:249

strict SCUT115 comparison:
  old strict review: improve=11:464 regress_visible=7:193 regress_low_contrast=1:36
  union review: improve=11:458 regress_visible=5:144 regress_low_contrast=1:37
```

Review decision: local visible-delta evidence supports the union gate over the old strict selector
because it removes the `254.jpg` selected-page regression and reduces SCUT visible-regress area
from 193 to 144 while keeping the same 17/156 improvement pattern. Holdout40 `466.jpg` remains
mixed, with larger improve area than regress area but enough local visible regressions that full
manual review is still required before product default promotion.

Next120 non-overlap validation:

```text
sample list: docs/scut-next120-nonoverlap-relative.txt
construction: first 120 SCUT train pages not present in scut-test115, holdout40, train160,
              or smoke-holdout3 lists
baseline output: outputs/scut_next120_nonoverlap_second_stage_baseline_20260706
union output: outputs/eval_scut_next120_nonoverlap_exact129_outside_edit_lam16_union_gate_20260706

baseline: n=120 residual=0.161840916843 overerase=0.002782668402
union: selected=0/120 residual=0.161840916843 overerase=0.002782668402
delta: residual=+0.000000000000 overerase=+0.000000000000
```

Decision update: next120 does not add product-visible coverage for the union selector, but it is a
useful safety check because the gate selected no extra pages and caused no metric regression on a
new non-overlapping SCUT train slice. Treat this as safety evidence only, not quality improvement.

Micro-tuning stop rule:

```text
Do not keep running one-step probe + selector replay loops on the current exact129/outside-edit
candidate family. The latest safe selector covers only 6/435 validated pages and next120 adds
0/120 new pages. Further tuning needs a named failure bucket, expected coverage lift, and page-level
visual acceptance criteria before it runs.
```

Next high-leverage task: build a compact product-quality benchmark with page labels:

```text
clear win / slight win / no-op / slight loss / clear loss
failure buckets: correction-fluid white patch, gray paper tone, residual handwriting,
                 printed-text damage, halo/edge artifacts
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
