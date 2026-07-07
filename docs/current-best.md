# Current Best Pipeline

Current recommended pipeline:

```text
primary EnsExam-GAN fine-tune
-> auto_cov8_step copy-mask inference
-> second-stage residual repair
-> local metrics + review pack
```

Training policy:

```text
do not default to full retraining
reuse artifacts/full-training-best.pth as the full-training base
reuse the current primary checkpoint for current-best continuation
run bounded hardcase fine-tunes / probes before considering any broad retrain
```

The previous one-day full-training model is reusable in this new project through `artifacts/full-training-best.pth`. It should be copied or symlinked as an asset, not regenerated. Do not return to broad full retraining until the current quality bottleneck is measured with page-level labels and failure buckets.

## Current Productization Gap

The recent union interval selector is the best current selector hypothesis, but it should not drive
more threshold-only work:

```text
validated pages: SCUT115 + holdout40 + train160 + next120 = 435
candidate-selected pages: 6/435
new next120 coverage: 0/120
status: safe and narrow, not broadly useful
```

This means the project has moved past "find a safe tiny gate" and into "increase reliable coverage
without visible regressions." Repeating one-step probes, hand-tuned intervals, or selector replay on
the same candidate family is low-leverage unless it is tied to a new failure bucket, a new training
objective, or a labeled page-level acceptance set.

## Stop Rule For Micro-Tuning

Stop adding new threshold-only or one-step micro-probe experiments unless the experiment can name all
of the following before it runs:

```text
target failure bucket: e.g. correction-fluid white patch, gray background residue, low-contrast handwriting
expected coverage lift: which pages should newly pass, not just aggregate residual
regression guard: printed text, paper tone, edge artifacts, and overerase limits
promotion metric: page-level visual label improvement, not only residual/overerase average
```

If those are not available, the next task is to improve the evaluation set, not run another training
or selector sweep.

## Higher-Leverage Next Direction

Build a compact page-level product-quality benchmark before further model tuning:

```text
1. Label selected SCUT/holdout/correction-fluid pages as clear win / slight win / no-op / slight loss / clear loss.
2. Bucket failures by visible cause: residual handwriting, whiteout patch, gray paper tone, printed-text damage, halo/edge artifact.
3. Require every candidate to report coverage and win/loss counts per bucket.
4. Promote only candidates that expand clear/slight wins without adding clear losses.
```

Only after this benchmark exists should the project choose between a learned selector, a revised
generator objective, or a specialized repair branch for correction-fluid / paper-tone restoration.

## Region Selector Track

The selector work has moved from page-level hand thresholds to a reviewed region/component workflow.
The current evidence says weak target-derived component labels are not enough:

```text
region component rule probe:
  components=106489
  train=79442
  test=27047
  strict rules=0
  ratio<=5% rules=0

best simple pair:
  train reject ratio=18.2%
  held-out reject ratio=36.1%

weak-label component ranker:
  test_selected=230/27047
  test_reject_ratio=37.0%

weak-label ranker materialized to page predictions:
  script: scripts/infer/materialize_region_component_selector.py
  evaluator: scripts/eval/evaluate_prediction_directory.py
  threshold: 0.7588800487801861
  SCUT115: materialized_pages=33/115, components=181, selected_pixels=754
  SCUT115: residual_gain=0.000001815, overerase_delta=0.000000000
  holdout40: materialized_pages=9/40, components=49, selected_pixels=217
  holdout40: residual_gain=0.000007298, overerase_delta=0.000000000
```

Reusable workflow now exists:

```text
component extraction/rule probe: scripts/analysis/evaluate_region_component_selector.py
review pack builder: scripts/analysis/build_region_component_review_pack.py
label validator: scripts/analysis/validate_region_component_labels.py
reviewed-label ranker: scripts/analysis/train_region_component_ranker.py --label-csv
label contract: docs/region-component-labeling.md
commands: docs/runbook.md
```

Generated held-out review pack:

```text
outputs/region_component_review_pack_t4_heldout_20260707/
  component-labels-template.csv
  contact_sheet_high_gain_accept.png
  contact_sheet_borderline_review.png
  contact_sheet_hard_reject.png
```

Next selector progress should come from reviewed `keep` / `drop` / `review` labels and held-out
reviewed validation, not more weak-label or hand-threshold tuning.
The materialization/evaluation bridge is now proven usable, but the weak-label
ranker result is effectively a no-op at page level. Treat it as workflow
infrastructure only; real progress requires reviewed labels or a better
candidate family before promotion.

Do not add the current whiteout inpaint repair to the default pipeline. It can reduce residual metrics on correction-fluid pages, but visual review shows the repaired area may look dirtier than leaving a clean white patch.

The current optimization roadmap is in `docs/optimization-roadmap.md`. The highest-leverage next
experiment is conservative paper-tone harmonization for correction-fluid pages, followed by better
candidate generation and calibrated selector analysis. Do not resume threshold-only micro-tuning
unless it is tied to a named failure bucket and page-level acceptance criteria.

## Preliminary Candidate: Identity-Safe Erasemap Cleanup

A separate second-stage `EraseMapCleanupNet` training probe is now available in
`scripts/train/train_patch_cleanup_erasemap_probe.py`. It is intentionally independent from the
main EnsExam generator and initializes the cleanup branch as an identity mapping, so an untrained or
undertrained checkpoint does not randomly rewrite the page.

Initial evidence is positive but not enough for promotion:

```text
training data: ExamInk-Seg smoke4 explicit-mask patches
checkpoint: outputs/smoke_examink_cleanup_erasemap_identity_step100_20260707/cleanup_probe.pt
ExamInk smoke4, base_edit=12, second_delta=2:
  residual 0.193198 -> 0.168635
  overerase 0.004355 -> 0.004293
SCUT holdout4, base_edit=12, second_delta=4:
  residual ~= 0.1562 -> 0.1550
  overerase ~= 0.00202 -> 0.00198

training data: frozen ExamInk-Seg train31 explicit-mask patches
checkpoint: outputs/train_examink_cleanup_erasemap_identity_train31_step500_20260707/cleanup_probe.pt
ExamInk train31, base_edit=12, second_delta=2:
  residual 0.198286 -> 0.193751
  overerase 0.004114 -> 0.004060
SCUT holdout4, base_edit=12, second_delta=2:
  residual 0.156151 -> 0.154797
  overerase 0.002022 -> 0.001983
SCUT115 replacement check, base_edit=12, second_delta=2:
  primary input residual 0.118313, overerase 0.003048
  current second-stage baseline residual 0.114225, overerase 0.003048
  train31 cleanup residual 0.136329, overerase 0.003003
SCUT115 third-stage check after current second-stage, base_edit=12, second_delta=2:
  current second-stage baseline residual 0.114225, overerase 0.003048
  train31 cleanup residual 0.132060, overerase 0.003009

training data: frozen ExamInk-Seg train24 with held-out val7
checkpoint: outputs/train_examink_cleanup_erasemap_identity_train24_val7_step500_20260707/cleanup_best.pt
validation loss:
  1.727791 at step100 -> 1.573239 at step500
ExamInk val7, base_edit=12, second_delta=2:
  residual 0.182208 -> 0.159401
  overerase 0.004473 -> 0.004433
SCUT115 replacement check, base_edit=12, second_delta=2:
  current second-stage baseline residual 0.114225, overerase 0.003048
  train24/val7 cleanup residual 0.162796, overerase 0.003009

training data: mixed frozen ExamInk-Seg + SCUT target-diff explicit patches
checkpoint: outputs/train_mixed_cleanup_erasemap_step500_20260707/cleanup_best.pt
validation loss:
  best step100 = 1.229255; later steps did not improve
SCUT115 replacement check, base_edit=12, second_delta=2:
  current second-stage baseline residual 0.114225, overerase 0.003048
  mixed cleanup residual 0.150084, overerase 0.003008
SCUT115 conservative gate sweep:
  best checked gate base_edit=12, second_delta=12
  residual 0.116928, overerase 0.003021, gate 0.002447
  still worse than current baseline by +0.002703 residual
```

Do not treat this as a product default. The ExamInk results are same-sample train/eval and the
holdout4 / val7 results do not survive SCUT115 validation. A small mixed-domain run improves over
ExamInk-only cleanup but still fails against the current second-stage baseline, and conservative gate
tuning does not close the gap. The useful part is the identity-safe training and validation
infrastructure; the current checkpoints are rejected for replacement or third-stage promotion. The
next useful step is to change the cleanup objective itself, not tune selectors around this candidate
family.

## Metric-Aligned Cleanup Objective

The cleanup training probe now includes differentiable proxy terms for the page-level SCUT metrics:
`residual_proxy` penalizes above-threshold residual delta inside the erase mask, while
`overerase_proxy` penalizes above-threshold changes outside the erase mask. This keeps the
optimization closer to the promotion gate than patch L1/BCE alone.

Validation smoke:

```text
script: scripts/train/train_patch_cleanup_erasemap_probe.py
environment: source .env; $ENSEXAM_PYTHON
device: mps
smoke data: frozen ExamInk train24/val7 patches, current primary pred input
result: 2-step train+val completed; history CSV includes residual_proxy and overerase_proxy
```

This is infrastructure only, not a promoted checkpoint. The next meaningful experiment is a bounded
SCUT-calibrated cleanup run using these proxy terms, followed by SCUT115 validation against the
current second-stage baseline.

SCUT-calibrated cleanup follow-up:

```text
data: SCUT target-diff explicit patches, file-level split 15 train pages / 6 val pages
no-op checkpoint:
  alpha_init_bias=-6, lr=1e-5
  SCUT115 residual=0.118313, overerase=0.003048, gate=0.000000
  diagnosis: alpha ~= 0.002476; max second_delta ~= 0.335/255
aggressive checkpoint:
  alpha_init_bias=-2, lr=5e-5, balanced alpha BCE
  SCUT115 residual=0.278154, overerase=0.003030, gate=0.023974
safer checkpoint:
  alpha_init_bias=-3, lr=5e-5, stronger outside/overerase/negative-alpha constraints
  SCUT115 residual=0.209663, overerase=0.003024, gate=0.018118
residual-delta checkpoint:
  model_type=residual_delta, residual_delta_scale=0.08
  SCUT115 residual=0.118880, overerase=0.003014, gate=0.016595
  diagnosis: max sampled second_delta ~= 5.48/255; no sampled pixels exceeded 12/255
```

The current cleanup branch now has two failure modes: conservative initialization stays identity,
while stronger alpha training produces broad edits that worsen residual badly. This points away from
more scalar alpha/bias tuning and toward a better candidate architecture/objective, such as masked
residual-only deltas, explicit paper-tone preservation, or a selector trained on page-level win/loss
labels.

A bounded residual-delta candidate is now implemented and is materially safer than the full clean
image head, but it still does not beat the current second-stage baseline. Keep it as a reusable
candidate-generation branch, not a product default. The next iteration should combine this
representation with page-level win/loss selection or stronger residual-localization targets rather
than increasing edit strength globally.

Residual-delta selector mining:

```text
script: scripts/analysis/analyze_residual_delta_selector_features.py
candidate: outputs/train_scut_residual_delta_bias3_scale008_step150_20260707/cleanup_best.pt
SCUT115 direct candidate:
  baseline residual=0.114225, overerase=0.003048
  candidate residual=0.118880, overerase=0.003014
  wins=40/115, losses=75/115
  oracle residual gain=0.002361903162
SCUT115 best local rule:
  active_gray_p25 >= 123.9 AND active_baseline_edit_p95 <= 153.703333333
  selected=12, wins=12, losses=0
  residual_gain=0.000982513297, overerase_delta=-0.000006683623
holdout40 direct candidate:
  baseline residual=0.130543, overerase=0.002732
  candidate residual=0.134065, overerase=0.002694
  wins=15/40, losses=25/40
joint SCUT115 + holdout40 best safe rule:
  active_gray_p25 >= 111.6 AND candidate_delta_max <= 200.133333333
  selected=7 total, wins=7, losses=0
  SCUT115 selected=6, wins=6, residual_gain=0.000670315057
  holdout40 selected=1, wins=1, residual_gain=0.000826318693
```

This shows the residual-delta branch has real but narrow page-level wins. The safe transfer rules are
too low-coverage for productization, so the next useful path is a labeled page-level selector or a
better candidate family, not more hand-mined threshold tightening.

Source-of-truth details for the historical migration remain in:

```text
/Volumes/Tool/source/clean-doc/docs/current-best-scut-hardcase.md
```

New model work should continue in this repository. The clean-doc workspace is a historical product/research workspace and temporary artifact source, not the active model-engineering entrypoint.
