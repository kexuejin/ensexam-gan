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

## Hybrid Gate Research Candidate

This is a research candidate, not a promoted product default. It chooses per page
between the current baseline second-stage output and the `nearworst_safe_step1`
candidate using inference-time features only. The loose rule below finds residual
headroom but increases overerase, so it must not be enabled by default:

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
visible-delta smoke dataset: data-links/samples/visible-delta-strict-scut-test115
visible-delta one-step smoke: outputs/smoke_visible_delta_patch_index_step1_20260705
visible-delta smoke result: patch-index filter 234->24, MPS step=1/1 passed
visible-delta step10 output: outputs/exp_visible_delta_patch_index_step10_20260705
visible-delta step10 strict eval: outputs/scut_test115_hybrid_gate_visible_delta_step10_strict_20260705
visible-delta step10 result: residual=0.114225 overerase=0.003048 selected=0/115
visible-delta step10 decision: rejected; full-generator patch-only training broke safety-gate features
```

Nearworst safe-step reproducibility:

```text
original trace patch: 129.jpg x1=320 y1=160 x2=576 y2=416
registered patch index: hardcase_lists/nearworst_safe_step1_exact129_patch_index.csv
random one-step rerun: scut selected=1/115, holdout selected=2/40
exact129 strict result:
  scut selected=5/115 residual=0.113987486262 overerase=0.003046587193
  holdout selected=3/40 residual=0.133642377143 overerase=0.002492527893
decision: use the exact patch index for nearworst_safe_step1 reproduction; random one-step reruns
are not comparable because sampled patch identity changes gate features
```

Joint selector replay:

```text
script: scripts/analysis/replay_hybrid_selector.py
output: outputs/selector_replay_joint_strict_candidate_20260705
top_rules: outputs/selector_replay_joint_strict_candidate_20260705/top_rules.csv
named_rules: outputs/selector_replay_joint_strict_candidate_20260705/named_rules.csv

rules scored=212187
safe rules=15057

strict_cov806_edit98868:
  selected=9 total pages
  total residual gain=0.000699581706
  max split overerase regret=+0.000010712336

scut7_cov65_edit98868_p95_5_gate0015:
  selected=10 total pages
  total residual gain=0.000702181859
  max split overerase regret=+0.000010712336

best_safe_joint:
  rule: copy_mask_cov8 >= 0.458438 and primary_edit_px <= 101340 and
        primary_p95_edit_delta <= 5 and second_stage_gate_ratio <= 0.000248943
  selected=2 total pages: scut115/254.jpg, holdout40/477.jpg
  total residual gain=0.000074976101
  max split overerase regret=-0.000000187820
```

Candidate-only overerase diagnostic:

```text
script: scripts/analysis/analyze_candidate_overerase_delta.py
output: outputs/analysis_joint_candidate_overerase_delta_script_20260705

positive-gain pages with overerase increase=106
new candidate-only overerase components=649
dominant component buckets:
  small_background_edit: 431 components, area=21354
  page_edge_artifact: 127 components, area=7363
  near_changed_region_halo: 57 components, area=2492

top24 component-protection probe:
  none residual_gain=0.176509 overerase_regret=0.015874 safe_pages=0/24
  all_components residual_gain=0.154762 overerase_regret=0.012248 safe_pages=0/24
decision: do not add simple candidate-only connected-component protection as product inference
```

Preservation weight probes:

```text
loss telemetry: input_preserve, mb_leak, and box_preserve are logged by train.py and micro_train_region_probe.py

preserve16/leak0.75:
  run: outputs/exp_preserve16_leak0p75_step2_20260705
  scut eval: outputs/scut_test115_hybrid_gate_preserve16_leak0p75_step2_strict_20260705
  holdout eval: outputs/holdout40_hybrid_gate_preserve16_leak0p75_step2_strict_20260705
  result: selected=1/115 on scut115 and 3/40 on holdout40; worse than original strict

preserve24/leak1:
  run: outputs/exp_preserve24_leak1_step2_20260705
  scut eval: outputs/scut_test115_hybrid_gate_preserve24_leak1_step2_strict_20260705
  holdout eval: outputs/holdout40_hybrid_gate_preserve24_leak1_step2_strict_20260705
  result: selected=1/115 on scut115 and 3/40 on holdout40; worse than original strict

decision: do not promote either probe; global preservation-weight increases are too conservative
```

Selective box-preserve probes:

```text
class2 preserve patch index: outputs/box_preserve_class2_patch_index_20260705/class2_preserve_patch_index.csv
class2-only lambda_box_preserve=4:
  run: outputs/exp_box_preserve4_class2patch_step2_20260705
  result: scut115 selected=0/115; holdout40 selected=0/40; exactly baseline metrics

class2-only lambda_box_preserve=1:
  run: outputs/exp_box_preserve1_class2patch_step2_20260706
  result: scut115 selected=0/115; holdout40 selected=1/40 but residual worsened vs baseline

class2-mix50 lambda_box_preserve=1 step4:
  run: outputs/exp_box_preserve1_class2mix50_step4_20260706
  result: scut115 selected=0/115; holdout40 selected=1/40 but residual worsened vs baseline

decision: do not promote selective class2 box-preserve continuation; it lowers copy_mask_cov8 on
the original strict-gate wins and loses gate eligibility
```

Promotion gate:

```text
Do not promote any current hybrid gate as the default pipeline. The loose gate improves
residual but increases overerase. The stricter and SCUT7 rules improve residual less and
still increase holdout40 overerase slightly. The only jointly non-worse rule selects just
2/155 pages and has negligible residual gain. Treat selector tuning as analysis
infrastructure unless a future candidate gives meaningful residual improvement without
cross-split overerase regression.
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
