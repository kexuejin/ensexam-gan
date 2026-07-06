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
exact129 cov806/edit98908 result:
  scut selected=6/115 residual=0.113988387733 overerase=0.003046310539
  holdout selected=3/40 residual=0.133642377143 overerase=0.002492527893
  selected pages match original strict on both splits
exact129 joint selector replay:
  output: outputs/selector_replay_joint_exact129_cov806_edit98908_20260706
  best safe rule: selected=3/155 total residual gain=0.000144868289
  rule: copy_mask_cov8 >= 0.389202, primary_edit_px <= 101434,
        primary_p95_edit_delta <= 5, second_stage_gate_ratio <= 0.000245596
  exact129_cov806_edit98868: selected=8 total residual gain=0.000621405155,
                             max overerase regret=+0.000010921776
  exact129_cov806_edit98908: selected=9 total residual gain=0.000620503683,
                             max overerase regret=+0.000010921776
decision: use the exact patch index for nearworst_safe_step1 reproduction; random one-step reruns
are not comparable because sampled patch identity changes gate features
```

Train high-stroke patch sensitivity:

```text
patch index: outputs/train_hard_patch_index_for_sensitivity_20260706/patch_index.csv
queue: outputs/patch_sensitivity_train_top4_20260706
summary: outputs/patch_sensitivity_train_top4_20260706/summary.csv
top4 patches: 362.jpg at (1280,320), (1280,640), (1280,480), (1440,480)
scut115: selected=0/115 residual=0.114224963938 overerase=0.003048296717
holdout40: selected=0/40 residual=0.134026304621 overerase=0.002481606117
decision: rejected as a direct route; high stroke density alone does not preserve or restore
strict-gate eligibility
```

Exact129 low-diff anchor patch sensitivity:

```text
script: scripts/experimental/build_anchor_similar_patch_list.py
patch index: outputs/anchor_similar_lowdiff_exact129_patch_index_20260706/patch_index.csv
queue: outputs/patch_sensitivity_anchor_lowdiff_top4_20260706
summary: outputs/patch_sensitivity_anchor_lowdiff_top4_20260706/summary.csv
best SCUT residual candidate: 003_84_x800_y480
  scut115 selected=5/115 residual=0.114007786589 overerase=0.003061889507
  holdout40 selected=3/40 residual=0.133760842647 overerase=0.002508892113
decision: useful direction, not product default; low-diff anchor-like patches restore gate
eligibility but still increase overerase and edit-size risk
```

Low-diff anchor selector replay:

```text
script: scripts/analysis/replay_hybrid_selector.py
outputs:
  outputs/selector_replay_anchor_lowdiff_001_161_x1120_y0_20260706
  outputs/selector_replay_anchor_lowdiff_002_130_x2240_y960_20260706
  outputs/selector_replay_anchor_lowdiff003_20260706
  outputs/selector_replay_anchor_lowdiff_004_378_x0_y960_20260706

001_161_x1120_y0:
  best safe rule selected=1/155 total residual gain=0.000276124066
  max split overerase regret=0
  selected: scut115/153.jpg only

002_130_x2240_y960:
  best safe rule selected=4/155 total residual gain=0.000119013592
  max split overerase regret=-0.000000740458
  selected: scut115/490.jpg, scut115/491.jpg, holdout40/475.jpg, holdout40/477.jpg

003_84_x800_y480:
  best safe rule selected=1/155 total residual gain=-0.000000895287
  max split overerase regret=0
  selected: holdout40/477.jpg only

004_378_x0_y960:
  no safe rules under max-overerase-regret=0
  best unsafe rule selected=7/155 total residual gain=0.003780324651
  max split overerase regret=0.000317846011

decision: selector-only tuning on low-diff anchor candidates is analysis-only. It can find
small zero-regret windows for 001/002, but the gains are too small and unstable for product
promotion. The next useful route is model-side edit-size/background preservation while
retaining strict-gate eligibility, not more threshold search on the current candidates.
```

Exact129 outside-edit interval gate:

```text
date: 2026-07-06
candidate: outputs/exp_exact129_outside_edit_lam16_step1_20260706/micro_region_probe.pth
script support:
  scripts/infer/run_hybrid_second_stage_gate.py supports interval bounds for cov8/edit/p95/gate
  scripts/analysis/replay_hybrid_selector.py supports fixed named interval replay

selector interval:
  0.807064 <= copy_mask_cov8 <= 0.881053
  12580 <= primary_edit_px <= 98699
  0 <= primary_p95_edit_delta <= 4.667
  0.0004928 <= second_stage_gate_ratio <= 0.0010997

replay: outputs/selector_replay_exact129_outside_edit_lam16_interval_relaxed_20260706
real eval scut115: outputs/eval_scut115_exact129_outside_edit_lam16_interval_relaxed_gate_20260706
real eval holdout40: outputs/eval_holdout40_exact129_outside_edit_lam16_interval_relaxed_gate_20260706

scut115: selected=5/115 residual=0.113964590999 overerase=0.003046575437
holdout40: selected=1/40 residual=0.133963087233 overerase=0.002481294748
selected pages: scut115/17.jpg, scut115/156.jpg, scut115/303.jpg,
                scut115/370.jpg, scut115/371.jpg, holdout40/466.jpg

decision: useful for finding selector headroom, but no longer a productization candidate after
train160 validation. Do not describe it as model-side retraining quality improvement: lam8/lam16
alone did not reduce holdout overerase. The improvement came from inference-time interval gating.

train160 follow-up:
  sample list: docs/scut-train160-nonholdout-relative.txt
  baseline: outputs/scut_train160_nonholdout_second_stage_baseline_20260706
  relaxed interval eval: outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_interval_relaxed_gate_20260706
  relaxed result: selected=3/160 residual=0.144524066276 overerase=0.002322399968
  baseline result: residual=0.144527160040 overerase=0.002314662644
  selected pages: train160/166.jpg, train160/190.jpg, train160/192.jpg
  decision update: relaxed interval is unsafe on train160 because overerase regresses
                   by +0.000007737324.

refined train160-safe interval:
  0.807064 <= copy_mask_cov8 <= 0.881053
  67887 <= primary_edit_px <= 98699
  0 <= primary_p95_edit_delta <= 4.667
  0.0004928 <= second_stage_gate_ratio <= 0.0010997
  replay: outputs/selector_replay_exact129_outside_edit_lam16_interval_train160_refined_20260706
  scut115 selected=4/115 residual_gain=0.000048330735 overerase_regret=-0.000003493826
  holdout40 selected=1/40 residual_gain=0.000063217389 overerase_regret=-0.000000311370
  train160 selected=0/160 residual_gain=0 overerase_regret=0
  decision: safer but much more conservative; drops SCUT 156.jpg, so keep as a
            selector hypothesis needing larger validation, not a promoted default.

two-box OR union interval gate:
  script support:
    scripts/infer/run_hybrid_second_stage_gate.py supports repeated --candidate-interval-rule
    scripts/analysis/replay_hybrid_selector.py supports --named-interval-union-rule
  replay: outputs/selector_replay_exact129_outside_edit_lam16_union_train160_20260706
  real eval scut115: outputs/eval_scut115_exact129_outside_edit_lam16_union_gate_20260706
  real eval holdout40: outputs/eval_holdout40_exact129_outside_edit_lam16_union_gate_20260706
  real eval train160: outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_union_gate_20260706
  low156 box: 0.807 <= cov8 <= 0.8072, 0 <= edit_px <= 13000,
              0 <= p95 <= 1.7, 0 <= gate <= 0.0011
  normal box: 0.807064 <= cov8 <= 0.881053, 67887 <= edit_px <= 98699,
              0 <= p95 <= 4.667, 0.0004928 <= gate <= 0.0010997
  scut115 selected=5/115 residual=0.113964591000 overerase=0.003046575437
  holdout40 selected=1/40 residual=0.133963087233 overerase=0.002481294748
  train160 selected=0/160 residual=0.144527160040 overerase=0.002314662644
  selected pages: scut115/17.jpg, scut115/156.jpg, scut115/303.jpg,
                  scut115/370.jpg, scut115/371.jpg, holdout40/466.jpg
  delta vs baseline:
    scut115 residual_delta=-0.000260372939 overerase_delta=-0.000001721280
    holdout40 residual_delta=-0.000063217389 overerase_delta=-0.000000311370
    train160 residual_delta=+0.000000000000 overerase_delta=+0.000000000000
  decision: current best selector hypothesis. It restores SCUT 156.jpg, keeps holdout40 466.jpg,
            and avoids train160 relaxed-interval bad pages. Still not product default until larger
            non-overlapping validation and manual visual review pass.
  visible-delta local review:
    scut115 output: outputs/analysis_visible_delta_union_scut115_20260706
    holdout40 output: outputs/analysis_visible_delta_union_holdout40_20260706
    scut115 improve_visible_target_region=11 components / 458 area
    scut115 regress_visible_target_region=5 components / 144 area
    scut115 regress_low_contrast_target=1 component / 37 area
    holdout40 improve_visible_target_region=5 components / 501 area
    holdout40 regress_visible_target_region=7 components / 249 area
    strict scut115 comparison: old strict regress_visible=7 components / 193 area;
                               union regress_visible=5 components / 144 area
    decision: supports union over old strict selector because it removes the 254.jpg selected-page
              regression and lowers SCUT visible-regress area, but holdout40 466.jpg remains mixed
              and still needs full manual review before default promotion.
```

Low-diff outside-edit preservation probe:

```text
objective: add a default-off outside-edit-size loss that penalizes large edits outside Mb_gt
config: configs/local/config.local-lowdiff-outside-edit-mps.yaml
train output: outputs/exp_lowdiff002_outside_edit_step1_20260706
eval outputs:
  outputs/eval_scut115_lowdiff002_outside_edit_step1_strict_20260706
  outputs/eval_holdout40_lowdiff002_outside_edit_step1_strict_20260706
selector replay: outputs/selector_replay_lowdiff002_outside_edit_step1_20260706

strict gate result:
  scut115 selected=3/115 residual=0.114371078175 overerase=0.003050526697
  holdout40 selected=2/40 residual=0.133809982834 overerase=0.002500990117

comparison to lowdiff002 without outside-edit:
  scut115 residual 0.114388655784 -> 0.114371078175, overerase 0.003050635477 -> 0.003050526697
  holdout40 residual 0.133796433250 -> 0.133809982834, overerase 0.002500568850 -> 0.002500990117

safe replay:
  best safe rule selected=4/155 total residual gain=0.000027968598
  max split overerase regret=0
  selected split coverage: scut115=4, holdout40=0

decision: keep the outside-edit-size loss as reusable tooling because it is a direct,
targeted way to penalize large background edits, but do not promote the lowdiff002
step1 candidate. The first probe is slightly more conservative on SCUT but still has
strict-gate overerase regression and no meaningful safe joint gain.

mix25 step2 follow-up:
  train output: outputs/exp_lowdiff002_outside_edit_mix25_step2_20260706
  eval outputs:
    outputs/eval_scut115_lowdiff002_outside_edit_mix25_step2_strict_20260706
    outputs/eval_holdout40_lowdiff002_outside_edit_mix25_step2_strict_20260706
  selector replay: outputs/selector_replay_lowdiff002_outside_edit_mix25_step2_20260706

  strict gate:
    scut115 selected=4/115 residual=0.114540661017 overerase=0.003039349003
    holdout40 selected=3/40 residual=0.134347379815 overerase=0.002487066735

  safe replay:
    best safe rule selected=1/155 total residual gain=0.000105093877
    max split overerase regret=0
    selected split coverage: scut115=1, holdout40=0

  decision: also not a product candidate. Mixed sampling reduces SCUT overerase but
  worsens residual on both SCUT115 and holdout40. The safe replay window is too small
  to justify further tuning of this exact anchor/mix setting.

lowdiff001 step1 follow-up:
  train output: outputs/exp_lowdiff001_outside_edit_step1_20260706
  eval outputs:
    outputs/eval_scut115_lowdiff001_outside_edit_step1_strict_20260706
    outputs/eval_holdout40_lowdiff001_outside_edit_step1_strict_20260706
  selector replay: outputs/selector_replay_lowdiff001_outside_edit_step1_20260706

  strict gate:
    scut115 selected=1/115 residual=0.114160171559 overerase=0.003060470866
    holdout40 selected=2/40 residual=0.134636204411 overerase=0.002526421284

  comparison to lowdiff001 without outside-edit:
    scut115 residual 0.114160943919 -> 0.114160171559, overerase 0.003060213443 -> 0.003060470866
    holdout40 residual 0.134691305786 -> 0.134636204411, overerase 0.002524365006 -> 0.002526421284

  safe replay:
    best safe rule selected=1/155 total residual gain=0.000275221700
    max split overerase regret=0
    selected split coverage: scut115=1, holdout40=0

  decision: not a product candidate. The outside-edit objective slightly improves
  residual versus the original 001 probe, but overerase is worse and holdout remains
  unsafe. The only safe replay window is the same tiny one-page SCUT-only pattern.

lowdiff003 step1 follow-up:
  train output: outputs/exp_lowdiff003_outside_edit_step1_20260706
  eval outputs:
    outputs/eval_scut115_lowdiff003_outside_edit_step1_strict_20260706
    outputs/eval_holdout40_lowdiff003_outside_edit_step1_strict_20260706
  selector replay: outputs/selector_replay_lowdiff003_outside_edit_step1_20260706

  strict gate:
    scut115 selected=5/115 residual=0.113975166648 overerase=0.003064461676
    holdout40 selected=3/40 residual=0.133709426395 overerase=0.002514275694

  comparison to lowdiff003 without outside-edit:
    scut115 residual 0.114007786589 -> 0.113975166648, overerase 0.003061889507 -> 0.003064461676
    holdout40 residual 0.133760842647 -> 0.133709426395, overerase 0.002508892113 -> 0.002514275694

  safe replay:
    best safe rule selected=1/155 total residual gain=0.000015518312
    max split overerase regret=0
    selected split coverage: scut115=0, holdout40=1

  decision: not a product candidate. This anchor gives the strongest strict residual
  improvement among the low-diff outside-edit probes, but it increases overerase on
  both splits. The safe replay window has negligible gain and no SCUT coverage.

mask-confidence diagnostic on lowdiff003 outside-edit:
  script: scripts/analysis/analyze_mask_confidence_features.py
  outputs:
    outputs/analysis_mask_confidence_lowdiff003_outside_scut115_20260706
    outputs/analysis_mask_confidence_lowdiff003_outside_holdout40_20260706

  summary:
    scut115 pages=115 positive_gain=71 safe_positive=1
    scut115 avg_residual_gain=0.003083072386 avg_overerase_regret=0.000349649965
    scut115 only safe positive: 397.jpg gain=0.002036560758 overerase_regret=-0.000174437785
    holdout40 pages=40 positive_gain=26 safe_positive=1
    holdout40 avg_residual_gain=0.003024969740 avg_overerase_regret=0.000350554299
    holdout40 only safe positive: 477.jpg gain=0.000620732464 overerase_regret=-0.000002347434

  selector sweep:
    features tested: mb_cov8, mb_cov98, primary_edit_px, primary_p95_edit_delta, ms_cov98
    safe_rules=0

  decision: keep the script as reusable diagnostic tooling. Mask-confidence features show
  that the candidate often reduces residual, but they do not separate safe useful pages
  well enough for a product selector on this candidate family.
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
Do not promote the hybrid gate as the default pipeline without larger validation and visual
review. The previous loose/strict gates either increased overerase or had negligible safe gain.
The relaxed exact129 outside-edit interval gate fixed SCUT115/holdout40 but regressed train160
overerase, while the refined train160-safe interval is much more conservative and loses the large
SCUT 156.jpg gain. The two-box OR union gate is the best current selector hypothesis because it
restores SCUT 156.jpg, keeps holdout40 466.jpg, and selects 0/160 train160 pages. Treat it as a
selector hypothesis, not a product default.
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
