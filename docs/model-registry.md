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

## Explicit-Domain Dual-Checkpoint Research Harness

This is a caller-routed research surface, not an automatic router and not a
product promotion. The caller must provide a strict UTF-8 CSV with exactly
`image_path,domain`; the immutable mapping is:

```text
default -> artifacts/current-primary/micro_region_probe_step0001.pth
unknown -> artifacts/current-primary/micro_region_probe_step0001.pth
hw5k    -> Candidate 5 epoch_1.pth (requires --ack-research-specialist)
```

Registered entry point and artifact identities:

```text
script: scripts/infer/run_explicit_domain_dual_checkpoint.py

default config:
  artifacts/current-primary/config.yaml
  sha256=8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e
default checkpoint:
  artifacts/current-primary/micro_region_probe_step0001.pth
  sha256=e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae

hw5k research config:
  artifacts/trials/hw5k-mixed-scut130-hw5k130-50pct-guard-jointtail-lite-step6400-respress-bs4-20260730/ensexam/20260801_183409/config.yaml
  sha256=c0ab5cc2a96dcaffa86dc75754c2a9bb9bfdc741c8ff7319e93bf8e2abc8adf8
hw5k research checkpoint:
  artifacts/trials/hw5k-mixed-scut130-hw5k130-50pct-guard-jointtail-lite-step6400-respress-bs4-20260730/ensexam/20260801_183409/epoch_1.pth
  sha256=8da25117dd883f95059b6d7067e3dc3580da11339de365ef904f711db4a1f490
```

Command template:

```bash
source .env
$ENSEXAM_PYTHON scripts/infer/run_explicit_domain_dual_checkpoint.py \
  --manifest-csv /absolute/path/to/caller-domain.csv \
  --output-dir outputs/explicit_domain_dual_checkpoint_<run-id> \
  --ack-research-specialist \
  --device auto
```

The acknowledgement is required only when an `hw5k` row is present. Both
non-empty branches run strictly serially with the frozen current-primary
inference gate above. The harness fails closed on invalid or missing domains,
artifact drift, partial branch output, provenance mismatch, and prediction SHA
mismatch; it never reads labels and never silently falls back between branches.

Candidate 5 remains `research_only/gate_qualified_nonpromotion`. Product use of
an explicit HW5K checkpoint requires a separate preregistered domain gate,
source-risk report, caller contract, contamination audit, and fresh unseen
HW5K-domain blind set. Automatic routing remains `not_authorized` and requires
an independent routing contract. See
`docs/decisions/2026-08-03-explicit-domain-dual-checkpoint-research-harness.md`.

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
  next120 non-overlap validation:
    sample list: docs/scut-next120-nonoverlap-relative.txt
    construction: first 120 SCUT train pages not present in scut-test115, holdout40, train160,
                  or smoke-holdout3 lists
    baseline output: outputs/scut_next120_nonoverlap_second_stage_baseline_20260706
    union output: outputs/eval_scut_next120_nonoverlap_exact129_outside_edit_lam16_union_gate_20260706
    baseline n=120 residual=0.161840916843 overerase=0.002782668402
    union selected=0/120 residual=0.161840916843 overerase=0.002782668402
    delta residual=+0.000000000000 overerase=+0.000000000000
    decision: safety evidence only. The union selector caused no regression on this new
              non-overlapping train slice, but also added no coverage or quality gain.
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

## Sign-Separated Residual v2 Structural KILL

```text
plan: docs/sign-separated-residual-candidate-plan-v2.json
checkpoint: artifacts/archive/sign-separated-residual-repair-20260810/training-output/sign_separated_probe.pt
checkpoint sha256: e9d75a525173a7ddf913f01765d7b5bdbc2bdb228deebfc742a24607292d05fc
audit: outputs/archive/sign-separated-residual-repair-20260810/checkpoint-audit/audit.json
terminal: KILL before inner-val15
route argmax identity / brighten / darken: 0 / 0 / 33,554,432
application-eligible brighten / darken: 0 / 8,514,478
```

This checkpoint is local rejected evidence, not a product artifact. Keep
`artifacts/current-primary` and `artifacts/current-second-stage-best.pt` unchanged. Do not run the
candidate on inner-val15, SCUT115, holdout40, or reserved blind, and do not rescue it with threshold
or optimization sweeps.

## Monotonic Residual Erase Synthetic Prerequisite

```text
model type: monotonic_residual_erase
representation: identity-initialized preserve-or-brighten luminance delta
bound: 0.08
synthetic audit: outputs/monotonic-residual-erase-synthetic-prerequisite-20260810/audit.json
terminal: PASS
training CLI enabled: false
real data accessed: false
next action: metadata/data-role preflight only
```

This is not a candidate or a measured quality lift. It removes the competing
darken output that collapsed sign-separated v2, while treating target-darker
and identity pixels as preserve negatives. Keep current-primary and the current
second stage unchanged; no training, prediction, quality gate, visual review,
reserved blind, or promotion is authorized by the synthetic result.

Metadata/data-role preflight subsequently passed without pixel decode:

```text
plan: docs/monotonic-residual-erase-data-roles.json
train: 275 pages (253 HW5K + 22 SCUT)
inner/development/SCUT115/holdout40: 15 / 156+112 / 115 / 40
pairwise effective-role overlap: 0
target access: train only
preserve negatives: target-darker, identity, submargin target-lighter
training CLI enabled: false
next action: training/config preflight only
```

The metadata-only training/config preflight also passed:

```text
plan: docs/monotonic-residual-erase-training-plan.json
preflight: outputs/monotonic-residual-erase-training-preflight-20260810/preflight.json
train role: 275 pages (253 HW5K + 22 SCUT)
model: exact-identity, nonnegative delta only, bound 0.08
schedule: 80 steps, batch 1, lr 0.00002, seed 42, MPS
support loss: separate per-sample target-lighter and preserve means
patch selection: top 256 target-lighter-support train patches
real pixels decoded / training / checkpoint: false / false / false
terminal: PASS
next action: exact train275 materialization audit only
```

This PASS is configuration evidence, not a quality result. It authorizes only
the named train-role prediction and patch-index materialization. Training,
inner-val15, later gates, visual review, reserved blind, promotion, and changes
to the default artifacts remain prohibited until their preceding gates pass.

Train275 materialization subsequently passed:

```text
manifest: 275 pages (253 HW5K + 22 SCUT)
primary / frozen second-stage predictions: 275 / 275
prediction provenance: byte-identical reuse of the prior audited frozen archive
target-lighter candidates: 52,645
selected patches / pages: 256 / 24
positive support ratio: 0.310654 .. 0.945358
preserve-negative ratio: 0.054642 .. 0.689346
training / checkpoint / quality gate: false / false / false
terminal: PASS
next action: candidate application preflight only
```

The `0.08` output bound cannot satisfy the legacy `32`-gray delta threshold,
so this PASS does not authorize training. Freeze a target-free third-stage
application protocol before optimization; no post-result threshold rescue is
allowed.

Candidate application preflight subsequently passed with a v2-only
reachability correction:

```text
legacy learning rate / max delta: 2e-5 / 0.023543 gray
registered learning rate / steps: 1e-4 / 80
registered max delta / bound: 20.399681 / 20.4 gray
edit probability / delta gate: 0.5 / 12 gray
direction guard: reject any candidate channel darker than baseline
identity / target-darker: exact no-op / exact no-op
real training / checkpoint / quality gate: false / false / false
terminal: PASS
```

The next authorized action is one exact v2 training run on the audited
train275 patch index. Candidate inference and inner-val15 remain closed until
the resulting checkpoint passes a separate structural audit.

The exact v2 checkpoint was subsequently killed before candidate inference:

```text
real train patches / reachable gates: 256 / 0
maximum real brighten delta: 2.667739 gray
positive / preserve delta mean: 1.482286 / 2.030916 gray
positive / preserve support >= 0.5: 0.024918 / 0.001735
negative deltas: 0
candidate inference / inner-val15: not started / not started
terminal: KILL
```

Do not repeat this v2 family or rescue it with learning-rate, step-count,
loss-weight, patch-selection, or threshold sweeps. A successor must prove
real-patch target-lighter versus preserve support separation before training
and must serialize portable checkpoint metadata.

## Dual-Input Support-Separation Preregistration

```text
family: dual_input_support_separation
state: PREREQUISITE_NEEDED
representation: primary RGB + second-stage RGB + signed RGB delta + four broadcast gate features
feature count: 13
diagnostic: page-grouped closed-form ridge, train275 only, full vs second-stage-RGB ablation
plan: docs/dual-input-support-separation-prerequisite-v1.json
decision: docs/decisions/2026-08-11-dual-input-support-separation-preregistration.md
implementation / data / training: false / false / false
candidate inference / quality gates / promotion: disabled / disabled / disabled
next action: exact train-only support-separation diagnostic
```

This is a preregistered evidence boundary, not a model implementation or
quality result. Keep the current-primary checkpoint as the product default.
Only a passing diagnostic may authorize a separate dual-input data/training
preflight; it cannot authorize training, candidate inference, or evaluation by
itself. Future checkpoint metadata must serialize `Path` values as strings and
pass the default weights-only load before candidate admission.

The exact train275 diagnostic subsequently returned KILL:

```text
train pages / balanced samples: 275 / 563,200
full mean / minimum fold AUC: 0.648757 / 0.608393
second-stage-RGB-only mean AUC: 0.644506
full minus ablation mean AUC: 0.004251
macro median page AUC: 0.686132
aggregate AUC gate / ablation-margin gate: FAIL / FAIL
training / checkpoint / candidate: false / false / false
terminal: KILL
```

Do not repeat the 13-channel diagnostic or rescue it by changing features,
folds, sampling, ridge lambda, probe class, or thresholds. The next eligible
uncertainty is a separately preregistered train-only audit of pixel-aligned
frozen primary `mb`/`ms` mask evidence. That direction is not yet implemented
or authorized.

## Spatial Primary Mask Support Preregistration

```text
family: spatial_primary_mask_support
state: PREREQUISITE_NEEDED
materialization: frozen current-primary mb/ms, train275 sources only, no labels
representation: mb + ms + signed mb-ms + mb*ms
diagnostic: page-grouped closed-form ridge, full masks vs fixed second-stage-RGB ablation
plan: docs/spatial-primary-mask-support-prerequisite-v1.json
decision: docs/decisions/2026-08-11-spatial-primary-mask-support-preregistration.md
implementation / data / training: false / false / false
candidate inference / quality gates / promotion: disabled / disabled / disabled
next action: exact label-free mask materialization and train-only support diagnostic
```

This is a preregistered evidence boundary, not a mask result or candidate. The
current-primary product default is unchanged. A PASS can authorize only a later
mask-aware data/training preflight with portable checkpoint metadata; it cannot
authorize training or evaluation directly.

The exact materialization passed and the train275 diagnostic subsequently
returned KILL:

```text
materialized mb / ms pages: 275 / 275
materialization target access: false
train pages / balanced samples: 275 / 563,200
full mean / minimum fold AUC: 0.580727 / 0.565196
second-stage-RGB-only mean AUC: 0.644506
full minus ablation mean AUC: -0.063779
macro median page AUC: 0.584623
aggregate / page / ablation-margin gates: FAIL / FAIL / FAIL
training / checkpoint / candidate: false / false / false
terminal: KILL
```

Do not repeat or rescue the four-channel `mb`/`ms` family through mask
selection, transforms, thresholds, neighborhoods, folds, sampling, ridge
lambda, nonlinear probes, or training. A successor requires separate
preregistration of a materially new target-free causal support source and an
independent train-only ablation. The current-primary product default,
promotion closure, and reserved-blind closure remain unchanged.

## Reconstruction-Stage Disagreement Preregistration

```text
family: reconstruction_stage_disagreement
state: PREREQUISITE_NEEDED
materialization: frozen Ic4/Ic2/Ic1/Ire stage disagreement, train275 sources only, no labels
representation: refine-coarse signed luma + refine-coarse abs RGB + Ic2/Ic1 abs RGB + Ic4/Ic1 abs RGB
diagnostic: page-grouped closed-form ridge, stage disagreement vs fixed second-stage-RGB ablation
plan: docs/reconstruction-stage-disagreement-prerequisite-v1.json
decision: docs/decisions/2026-08-12-reconstruction-stage-disagreement-preregistration.md
implementation / data / training: false / false / false
candidate inference / quality gates / promotion: disabled / disabled / disabled
next action: exact label-free stage-disagreement materialization and train-only support diagnostic
```

This family measures convergence inside the frozen primary reconstruction
hierarchy rather than reusing final RGB, masks, page scalars, or selectors. Its
four channels, interpolation, per-patch derivation, overlap fusion, folds,
sampling, ridge probe, RGB ablation, and acceptance gates are frozen before
data execution. A PASS may authorize only a later data/training/application
preflight with portable checkpoint metadata; it cannot authorize training or
candidate inference directly.

The exact label-free materialization passed and the train275 diagnostic
subsequently returned KILL:

```text
materialized stage-disagreement pages: 275 / 275
materialization target access: false
train pages / balanced samples: 275 / 563,200
full mean / minimum fold AUC: 0.657016 / 0.639200
second-stage-RGB-only mean AUC: 0.644506
full minus ablation mean AUC: 0.012509
macro median page AUC: 0.678154
aggregate / page / direction gates: PASS / PASS / PASS
ablation-margin gate: FAIL
training / checkpoint / candidate: false / false / false
terminal: KILL
```

Do not repeat or rescue the four-channel reconstruction-stage family through
stage selection, transforms, neighborhoods, thresholds, folds, sampling,
ridge lambda, nonlinear probes, or training. A successor requires separate
preregistration of a materially new target-free causal support source and an
independent train-only ablation. The current-primary product default,
promotion closure, and reserved-blind closure remain unchanged.

## Source-Output Support Preregistration

```text
family: source_output_support
state: PREREQUISITE_NEEDED
representation: raw source RGB + frozen second-stage RGB
ablation: frozen second-stage RGB only
diagnostic: train275 page-grouped closed-form ridge with frozen sampling and AUC gates
plan: docs/source-output-support-prerequisite-v1.json
decision: docs/decisions/2026-08-12-source-output-support-preregistration.md
data execution / training / candidate: false / false / false
inner-val15 / development / promotion: disabled / disabled / disabled
next action: implement and run the exact train-only six-channel diagnostic once
```

This family tests whether pre-edit source appearance supplies support evidence
that final cleaned appearance lacks. It deliberately excludes the closed
dual-pipeline additions: primary RGB, signed pipeline differences, masks,
reconstruction stages, page scalars, thresholds, neighborhoods, and model
transforms. A PASS may authorize only a later source-conditioned
data/training/application preflight with portable checkpoint metadata; it
cannot authorize training or candidate inference directly.

The exact train275 diagnostic subsequently returned KILL:

```text
train pages / balanced samples: 275 / 563,200
full mean / minimum fold AUC: 0.654677 / 0.626443
second-stage-RGB-only mean AUC: 0.644506
full minus ablation mean AUC: 0.010171
macro median page AUC: 0.702450
aggregate / page / direction gates: PASS / PASS / PASS
ablation-margin gate: FAIL
training / checkpoint / candidate: false / false / false
terminal: KILL
```

Do not repeat or rescue the raw source-plus-output family through channel
selection, differences, color transforms, neighborhoods, thresholds, folds,
sampling, ridge lambda, nonlinear probes, or training. A successor requires
separate preregistration of a materially new target-free causal support source
and an independent train-only ablation. The current-primary product default,
promotion closure, and reserved-blind closure remain unchanged.

## Second-Stage Alpha Support Preregistration

```text
family: second_stage_alpha_support
state: KILL
representation: frozen final second-stage RGB + raw erasemap sigmoid alpha before every threshold
ablation: frozen final second-stage RGB only
diagnostic: label-free train275 alpha materialization, then five-fold page-grouped closed-form ridge
plan: docs/second-stage-alpha-support-prerequisite-v1.json
decision: docs/decisions/2026-08-12-second-stage-alpha-support-preregistration.md
data execution / training / candidate: alpha materialization only / false / false
inner-val15 / development / promotion: disabled / disabled / disabled
next action: do not repeat or rescue; preregister only a materially new target-free causal source
```

This family tests the frozen erasemap edit-confidence surface that the product
removes with `cleanup_alpha_threshold=0.3` before its later gates. It excludes
clean-candidate RGB, threshold masks, primary/source signals, page scalars,
neighborhoods, transforms, alternative model layers, nonlinear probes, and
all optimizer or quality work. A PASS can authorize only a separately
preregistered alpha-conditioned data/training/application preflight with
portable checkpoint metadata; it cannot authorize training, candidate
inference, or any quality split directly.

The label-free materialization completed with exactly 275 finite, aligned,
hash-stable `float32` raw-alpha maps and `target_access=false`. The exact
train275 diagnostic subsequently returned KILL:

```text
train pages / balanced samples: 275 / 563,200
full mean / minimum fold AUC: 0.656274 / 0.611754
second-stage-RGB-only mean AUC: 0.644506
full minus ablation mean AUC: 0.011768
macro median page AUC: 0.704466
aggregate / page / direction gates: PASS / PASS / PASS
ablation-margin gate: FAIL
training / checkpoint / candidate: false / false / false
terminal: KILL
```

Do not repeat or rescue the raw-alpha family through thresholding, layer or
channel selection, transforms, neighborhoods, folds, sampling, ridge lambda,
nonlinear probes, or training. A successor requires separate preregistration
of a materially new target-free causal support source and an independent
train-only ablation. The current-primary product default, promotion closure,
and reserved-blind closure remain unchanged.

## Independent HW5K Expert Disagreement Support Preregistration

```text
family: independent_hw5k_expert_disagreement_support_v1
state: KILL
population: 123 HW5K train-role pages not present in the specialist training manifest
materialization: both frozen checkpoints on every same source page, no targets, no routing
representation: current-primary RGB + frozen HW5K expert RGB
ablation: current-primary RGB only
diagnostic: five-fold page-grouped closed-form ridge with frozen sampling and AUC gates
plan: docs/independent-hw5k-expert-disagreement-support-prerequisite-v1.json
decision: docs/decisions/2026-08-13-independent-hw5k-expert-disagreement-support-preregistration.md
data execution / training / candidate: paired materialization only / false / false
inner-val15 / development / promotion: disabled / disabled / disabled
next action: do not repeat or rescue; preregister only a materially new target-free causal source
```

This prerequisite does not reopen Candidate 5 routing. Its prior routed product
path remains closed because HW5K improvement did not preserve SCUT. The new
causal source is paired same-page evidence from an independently trained frozen
model, evaluated only after excluding all 152 train275 pages seen by that
specialist. The remaining diagnostic set is exactly 123 HW5K pages with frozen
basename and ordered-path content hashes.

Both checkpoints must run on all 123 pages without domain metadata or expert
selection. Version 1 compares six raw RGB channels with the identical
current-primary-RGB-only probe. Difference channels, transforms, masks, page
scalars, alternative checkpoints, nonlinear probes, and every optimizer or
quality surface remain closed. A PASS may authorize only a later
expert-conditioned data/training/application preflight; it cannot authorize
routing, training, candidate inference, or quality evaluation directly. The
current-primary product default, promotion closure, and reserved-blind closure
remain unchanged.

The target-free paired materialization completed with exactly 123 aligned
predictions from each checkpoint and no routing metadata. The exact train-only
diagnostic subsequently returned KILL:

```text
unseen HW5K pages / balanced samples: 123 / 251,904
full mean / minimum fold AUC: 0.664916 / 0.634484
current-primary-RGB-only mean AUC: 0.635189
full minus ablation mean AUC: 0.029727
macro median page AUC: 0.716586
aggregate / page / direction gates: PASS / PASS / PASS
ablation-margin gate: FAIL by 0.000273
training / checkpoint / candidate: false / false / false
terminal: KILL
```

Do not round or relax the frozen margin, and do not rescue paired independent
expert RGB through channel/layer selection, differences, transforms,
neighborhoods, folds, sampling, ridge lambda, alternate checkpoints, nonlinear
probes, routing, or training. A successor requires separate preregistration of
a materially new target-free causal source and an independent train-only
ablation. The current-primary product default, promotion closure, and
reserved-blind closure remain unchanged.

## External Text Layout Support Preregistration

```text
family: external_printed_text_layout_support_v1
state: PREREQUISITE_NEEDED
producer: official PP-OCRv6_medium_det safetensors, frozen CPU Transformers inference
population: exact 275 train-role raw source pages
materialization: sorted quadrilaterals/scores plus occupancy/confidence grids, no targets or recognition
representation: frozen second-stage RGB + text occupancy + text confidence
ablation: frozen second-stage RGB only
diagnostic: five-fold page-grouped closed-form ridge with frozen sampling and AUC gates
plan: docs/external-text-layout-support-prerequisite-v1.json
decision: docs/decisions/2026-08-13-external-text-layout-support-preregistration.md
runtime prerequisite: docs/decisions/2026-08-13-external-text-layout-runtime-safety-prerequisite.md
runtime result: docs/external-text-layout-runtime-safety-probe-20260813.json
static memory risk: docs/external-text-layout-static-memory-risk-20260814.json
runtime repair contract: docs/external-text-layout-runtime-equivalence-repair-v1.json
runtime repair decision: docs/decisions/2026-08-14-external-text-layout-runtime-equivalence-repair-preregistration.md
repaired probe result: docs/external-text-layout-runtime-equivalence-repair-probe-20260814.json
repaired probe decision: docs/decisions/2026-08-14-external-text-layout-runtime-equivalence-repair-probe-kill.md
tiled 9x9 static feasibility: docs/external-text-layout-tiled-9x9-feasibility-20260814.json
tiled 9x9 repair contract: docs/external-text-layout-tiled-9x9-runtime-repair-v1.json
tiled 9x9 repair preregistration: docs/decisions/2026-08-14-external-text-layout-tiled-9x9-runtime-repair-preregistration.md
tiled 9x9 fake verification: docs/external-text-layout-tiled-9x9-runtime-repair-verification-20260814.json
tiled 9x9 verification decision: docs/decisions/2026-08-14-external-text-layout-tiled-9x9-runtime-repair-verification-pass.md
tiled 9x9 one-page contract: docs/external-text-layout-tiled-9x9-one-page-safety-probe-v1.json
tiled 9x9 one-page decision: docs/decisions/2026-08-14-external-text-layout-tiled-9x9-one-page-safety-probe-preregistration.md
tiled 9x9 integration verification: docs/external-text-layout-tiled-9x9-one-page-integration-verification-20260814.json
tiled 9x9 integration decision: docs/decisions/2026-08-14-external-text-layout-tiled-9x9-one-page-integration-pass.md
tiled probe cache reconstruction v2: docs/external-text-layout-tiled-probe-cache-reconstruction-v2.json
tiled probe cache reconstruction v2 decision: docs/decisions/2026-08-14-external-text-layout-tiled-probe-cache-reconstruction-v2-preregistration.md
v2 cache validator: scripts/analysis/reconstruct_external_text_layout_frozen_caches.py (sha256 699adf9b2abcbdbfb36913220a4decc5debaf77f1252bfb2b1b52e4a4bdde712)
v2 cache validator tests: tests/test_external_text_layout_frozen_cache_reconstruction.py (sha256 c2277f72e70b2a8da835ea1f2952f9f52a619260ee9c8e68f2a81a4b90d97faf)
historical cache reconstruction contract (immutable v1): docs/external-text-layout-frozen-cache-reconstruction-v1.json
cache reconstruction decision: docs/decisions/2026-08-14-external-text-layout-frozen-cache-reconstruction-preregistration.md
runtime restoration report: docs/external-text-layout-historical-runtime-restoration-20260814.json
runtime restoration decision: docs/decisions/2026-08-14-external-text-layout-historical-runtime-restoration.md
historical cache runtime: Python 3.10.11 / Torch 2.5.1 / NumPy 2.2.6 / OpenCV runtime 5.0.0 / OpenCV wheel 5.0.0.93
data execution / training / candidate: exact layout materialization only / false / false
inner-val15 / development / promotion: disabled / disabled / disabled
runtime status: the only clean-baseline repaired page crossed the free-memory floor at 26.0% and swap cap at 2,973,562,306 bytes; peak process-tree RSS 5,643,206,656 bytes; no formal evidence or residual model process
static finding: limit_type=min preserves the large page at 2432x1728 after 32-pixel rounding; full-resolution 9x9 neck work and duplicate upsample construction are concrete but unproven memory-risk contributors
runtime repair: hash/version/AST-bound in-memory forward replacement removes only the overwritten first upsample construction and remains statically equivalent, but its only authorized clean-baseline probe KILLed the exact repeat path as empirically unsafe
tiled 9x9 successor: four-row spatial tiles reduce the highest projection unfold comparison bound from 21,785,739,264 to 201,719,808 bytes; four exact CPU float32 fake cases covering 256-to-64 and 64-to-64 are bitwise equal with maximum error 0.0; real detector execution remains disabled
tiled one-page authorization: bounded integration is complete; exactly one target-free hw5k_1011.jpg attempt is allowed only after every launch gate passes; any non-PASS terminal closes the exact tiled path
tiled integration: PASS with 61/61 external-text-layout tests under Python 3.13.1 and 61/61 under Python 3.10.11; contract/source/result hashes, stricter limits, thread caps, parent-plus-child Simulator checks, RUNNING sentinel, non-overwrite, cleanup, and synthetic PASS behavior are verified without detector execution
tiled cache handoff: v1 cannot consume the tiled result path, probe identity, or strict safety schema; v2 is preregistered to bind the tiled PASS while preserving historical cache hashes, runtime identity, helper ordering, and publication boundaries
v2 cache validator integration: PASS with 7/7 compatibility tests under Python 3.13.1 and 7/7 under Python 3.10.11; old result path/identity/schema, incomplete attempt/completion/process fields, thread-cap drift, split probe/reconstruction health gates, helper ordering, exact historical runtime, and static preflight are verified without detector or cache execution
cache recovery: original build paths, archived manifest, exact historical runtime and metrics/prediction hashes, then relative archive symlinks; static preflight PASS with historical_runtime_ready=true and execution_authorized=false
current host gate: 85% free memory, 1,450.62 MiB swap used, zero Booted iOS Simulators, no model process, and tiled result absent; model and cache execution prohibited while swap exceeds 512 MiB
next action: wait for swap <=512 MiB and every one-page launch gate; then run exactly one target-free hw5k_1011.jpg attempt before any historical cache reconstruction
```

This is the first registered support producer trained outside the EnsExam-GAN
pipeline for generic text localization. Version 1 persists deterministic text
quadrilaterals and scores for provenance, but exposes only binary polygon
occupancy and pixelwise maximum detection confidence to the diagnostic. OCR
recognition, recognized content, polygon geometry features, source RGB, masks,
alpha, reconstruction stages, expert outputs, routing metadata, transforms,
and detector or probe searches are excluded.

The external model was not trained or tuned in this repository and receives no
train275 targets. Its published training-corpus overlap with SCUT or HW5K is
unverified, so this is only an incremental train-role support screen. Even a
PASS cannot establish product generalization; it may authorize only a separate
leakage-aware data/training/application preflight. The current-primary product
default, promotion closure, and reserved-blind closure remain unchanged.

The family is not KILLed. Its train-only diagnostic remains pending behind two
fail-closed prerequisites: a safe detector runtime and reconstruction of the
missing frozen primary/second-stage prediction caches with exact historical
content hashes. The repository-local duplicate-upsample repair is statically
equivalent, but its only clean-baseline probe crossed both memory safeguards;
the exact detector/runtime/geometry/repair repeat path is therefore KILLed.
Clearing swap does not authorize another retry, and the registered safeguards
must not be relaxed. The cache reconstruction path remains recovery
infrastructure only: its successful-probe prerequisite is unmet and neither
archive may be published before both reconstructed caches match every
registered historical hash. The exact historical cache runtime remains
restored. Static analysis and the implementation-only gate for a materially
lower-memory full-resolution `9x9` path now PASS: four-row tiles preserve
tested CPU float32 values bitwise while reducing the highest projection unfold
comparison bound by `108x`. The result does not establish full-map backend
identity, timeout, or host peak memory. A separate one-page contract now binds
the exact implementation and test hashes. Its bounded integration now PASSes
under both registered Python environments without detector execution. The
probe requires stricter resource limits, fixed thread caps, parent and child
Simulator checks, and an atomic `RUNNING` sentinel before page start. Detector
execution remains disabled until every host launch gate passes. The resulting
probe is exactly one attempt; any non-PASS terminal or existing result closes
this tiled path without retry.
