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

The recent union interval selector was safe but too narrow:

```text
validated pages: SCUT115 + holdout40 + train160 + next120 = 435
candidate-selected pages: 6/435
new next120 coverage: 0/120
status: safe and narrow, not broadly useful
```

The latest selector-expansion pass materially improves coverage while preserving the current hard
safety constraints:

```text
validated pages: SCUT115 + holdout40 + train160 + next120 = 435
current clean high-coverage candidate: zero_reject_veto_125_accept_clean
selected pages: 125/435
coverage: 28.74%
accept/review/reject: 102/23/0
metric losses: 0
residual_gain: 1.519751211
overerase_delta: -0.003181883
split coverage: holdout40=9, next120=41, scut115=22, train160=53
```

`zero_reject_veto_125_accept_clean` is the current best high-coverage selector candidate. It is built
from the prior zero-reject veto stack plus deployable page-feature/ranker buckets, then a final
accept-only union after `zero_reject_veto_121`. The final post-121 accept-only search found four
additional accept pages and then exhausted: after selecting 125 pages, the remaining 310 pages had no
new `accept`-only, zero-reject, zero-metric-loss deployable increment in the searched ranker/feature
space.

Do not interpret selector coverage as "fully solved" page quality. The project has local source,
baseline, candidate, and target images, so target-aware quality scoring should be the first-line
quality gate and manual review should focus on borderline/contradictory pages:

```bash
$ENSEXAM_PYTHON scripts/analysis/score_target_comparison_quality.py \
  --review-csv outputs/balanced007_full_page_review_20260707/local_target_comparison.csv \
  --output-csv outputs/balanced007_ranker_expansion_source_eval_20260708/target_quality_scoring_20260708/full435_target_quality_v2.csv \
  --summary-json outputs/balanced007_ranker_expansion_source_eval_20260708/target_quality_scoring_20260708/full435_target_quality_v2_summary.json

$ENSEXAM_PYTHON scripts/analysis/summarize_selector_target_quality.py \
  --quality-csv outputs/balanced007_ranker_expansion_source_eval_20260708/target_quality_scoring_20260708/full435_target_quality_v2.csv \
  --selector zero_reject_veto_125_accept_clean:outputs/balanced007_ranker_expansion_source_eval_20260708/final_candidate_selectors/zero_reject_veto_125_accept_clean_selected.csv \
  --selector zero_reject_veto_134_manual_accept_whitelist:outputs/balanced007_ranker_expansion_source_eval_20260708/final_candidate_selectors/zero_reject_veto_134_manual_accept_whitelist_selected.csv \
  --selector zero_reject_veto_auto_triage_promote_candidate:outputs/balanced007_ranker_expansion_source_eval_20260708/final_candidate_selectors/zero_reject_veto_auto_triage_promote_candidate_selected.csv \
  --selector zero_reject_veto_147_visual_round2_candidate:outputs/balanced007_ranker_expansion_source_eval_20260708/final_candidate_selectors/zero_reject_veto_145_visual_round2_candidate_selected.csv \
  --output-csv outputs/balanced007_ranker_expansion_source_eval_20260708/target_quality_scoring_20260708/selector_target_quality_summary.csv
```

Current target-aware scorer results on all 435 pages:

```text
slight_win: 58
borderline: 73
slight_loss: 42
clear_loss: 262
clear_win/noop: 0
```

Selector coverage vs. target-confirmed quality:

```text
zero_reject_veto_125_accept_clean:
  selected=125/435 selected_coverage=28.74%
  target_confirmed_wins=54/435 target_win_coverage=12.41%
  target_borderline=58 target_losses=13

zero_reject_veto_134_manual_accept_whitelist:
  selected=134/435 selected_coverage=30.80%
  target_confirmed_wins=58/435 target_win_coverage=13.33%
  target_borderline=63 target_losses=13

zero_reject_veto_auto_triage_promote_candidate:
  selected=137/435 selected_coverage=31.49%
  target_confirmed_wins=58/435 target_win_coverage=13.33%
  target_borderline=65 target_losses=14

zero_reject_veto_147_visual_round2_candidate:
  selected=147/435 selected_coverage=33.79%
  target_confirmed_wins=58/435 target_win_coverage=13.33%
  target_borderline=72 target_losses=17
```

This reframes the current bottleneck: growing selected coverage beyond `134` mostly adds borderline
or loss-risk pages under the target-aware scorer. The next quality-improving step is not broader
threshold search; it is reducing target-aware losses and converting borderline pages into confirmed
wins through better candidate generation, calibrated scoring, or formal visual labels.

For quality-first selector variants, derive reproducible CSVs from the target-aware labels instead of
manually filtering outputs:

```bash
$ENSEXAM_PYTHON scripts/analysis/derive_target_quality_selectors.py \
  --base-selected-csv outputs/balanced007_ranker_expansion_source_eval_20260708/final_candidate_selectors/zero_reject_veto_125_accept_clean_selected.csv \
  --quality-csv outputs/balanced007_ranker_expansion_source_eval_20260708/target_quality_scoring_20260708/full435_target_quality_v2.csv \
  --output-dir outputs/balanced007_ranker_expansion_source_eval_20260708/final_candidate_selectors \
  --name-prefix zero_reject_veto
```

Current quality-first derived selectors:

```text
zero_reject_veto_54_target_confirmed_wins:
  selected=54/435 selected_coverage=12.41%
  target_confirmed_wins=54 target_borderline=0 target_losses=0
  use as the strict lower-bound quality baseline

zero_reject_veto_112_zero_target_loss:
  selected=112/435 selected_coverage=25.75%
  target_confirmed_wins=54 target_borderline=58 target_losses=0
  use as the quality-first default candidate before broader coverage work
```

The `112_zero_target_loss` variant is the immediate quality improvement over `125`: it removes the
13 selected pages that the target-aware scorer flags as loss-risk while preserving all 54
target-confirmed wins. The remaining 58 borderline pages are the next review/calibration target.

To make that borderline pass reproducible, bucket the 58 pages with:

```bash
$ENSEXAM_PYTHON scripts/analysis/triage_target_quality_borderline.py \
  --quality-csv outputs/balanced007_ranker_expansion_source_eval_20260708/target_quality_scoring_20260708/full435_target_quality_v2.csv \
  --selected-csv outputs/balanced007_ranker_expansion_source_eval_20260708/final_candidate_selectors/zero_reject_veto_112_zero_target_loss_selected.csv \
  --output-dir outputs/balanced007_ranker_expansion_source_eval_20260708/target_quality_borderline_triage_20260708 \
  --name zero_reject_veto_112_borderline_triage
```

Current borderline triage:

```text
auto_win_candidate: 4
ratio_noise_review: 25
accept_weak_win_review: 17
manual_review_strong_metric: 1
manual_review_mixed_metric: 9
keep_borderline: 2

local accept/review among borderline: 48/10
```

The first high-leverage quality pass is not more selector threshold search. It is reviewing the
4 `auto_win_candidate` pages plus the 25 `ratio_noise_review` pages where local verdicts and target
metrics are positive but conservative changed-area risk ratios kept them borderline. Confirmed pages
can then be promoted into durable quality labels or a reviewed whitelist.

Build reproducible page/crop review packs for those 29 high-priority pages with:

```bash
$ENSEXAM_PYTHON scripts/analysis/build_target_quality_bucket_review_packs.py \
  --triage-csv outputs/balanced007_ranker_expansion_source_eval_20260708/target_quality_borderline_triage_20260708/zero_reject_veto_112_borderline_triage.csv \
  --output-dir outputs/balanced007_ranker_expansion_source_eval_20260708/target_quality_borderline_triage_20260708/high_priority_review_packs \
  --name high_priority \
  --max-crops-per-row 4 \
  --max-contact-crops 120
```

This writes a combined 29-page pack plus per-bucket packs for `auto_win_candidate` and
`ratio_noise_review`, including page-level contact sheets, crop-level contact sheets, and crop label
templates. Treat the generated packs as local review evidence only; commit the reusable commands and
scripts, not the generated images/CSVs.

After reviewing the `auto_win_candidate` pack, apply that reviewed promotion as an overlay instead of
mutating the scorer output:

```bash
$ENSEXAM_PYTHON scripts/analysis/apply_target_quality_promotions.py \
  --quality-csv outputs/balanced007_ranker_expansion_source_eval_20260708/target_quality_scoring_20260708/full435_target_quality_v2.csv \
  --triage-csv outputs/balanced007_ranker_expansion_source_eval_20260708/target_quality_borderline_triage_20260708/zero_reject_veto_112_borderline_triage.csv \
  --output-csv outputs/balanced007_ranker_expansion_source_eval_20260708/target_quality_promotions_20260708/full435_target_quality_v2_auto_win_promoted.csv \
  --summary-json outputs/balanced007_ranker_expansion_source_eval_20260708/target_quality_promotions_20260708/full435_target_quality_v2_auto_win_promoted_summary.json \
  --promote-bucket auto_win_candidate \
  --promoted-label slight_win \
  --expect-promoted 4 \
  --promotion-note "local page/crop review pack auto-win promotion"
```

Then re-summarize the selector with the promoted quality CSV. Current verified result:

```text
zero_reject_veto_112_zero_target_loss with auto_win_candidate promotion:
  selected=112/435 selected_coverage=25.75%
  target_confirmed_wins=58/435 target_confirmed_win_pct=13.33%
  target_borderline=54 target_losses=0
```

Key comparison points:

```text
zero_reject_veto_84:
  selected=84/435 coverage=19.31% accept/review/reject=79/5/0 metric_losses=0

zero_reject_veto_108:
  selected=108/435 coverage=24.83% accept/review/reject=95/13/0 metric_losses=0

zero_reject_veto_121:
  selected=121/435 coverage=27.82% accept/review/reject=98/23/0 metric_losses=0

zero_reject_veto_125_accept_clean:
  selected=125/435 coverage=28.74% accept/review/reject=102/23/0 metric_losses=0
```

Post-125 coverage search found one additional deployable single-page accept rule:

```text
zero_reject_veto_126_deployable_single_accept:
  selected=126/435 coverage=28.97% accept/review/reject=103/23/0 metric_losses=0
  added=scut115/228.jpg
  rule=active_edge_mean >= 636.286865234 AND score_accept_lr0.005_l20.01 <= 0.095072925224
```

This `126` candidate is useful as a measured upper-bound probe, not as the cleaner default: the
increment is only one page, and the rule is too narrow to treat as a robust product selector without
additional labels. Page and crop review for `scut115/228.jpg` did not show obvious visible
regression, but the practical default remains `zero_reject_veto_125_accept_clean` until the 23
review pages are labeled or candidate generation improves.

To speed up that labeling loop, use the reusable queue triage helper:

```bash
$ENSEXAM_PYTHON scripts/analysis/triage_selector_label_queue.py \
  --queue-csv outputs/balanced007_ranker_expansion_source_eval_20260708/labeling_queue_post125_coverage_20260708/labeling_queue.csv \
  --output-dir outputs/balanced007_ranker_expansion_source_eval_20260708/labeling_queue_post125_coverage_20260708/auto_triage \
  --accept-verdict-promotes
```

Current triage on the 53-page labeling queue:

```text
promote_candidate: 16
borderline_review: 26
keep_review: 11
reject_candidate: 0

auto-triage promote candidate:
  selected=137/435 coverage=31.49% accept/review/reject=110/27/0 metric_losses=0
  added pages beyond zero_reject_veto_125_accept_clean: 12
```

The auto-triage `137` candidate is not a product default. It is a faster review target: inspect the
16 promote candidates and 26 borderline pages, then convert confirmed pages into durable visual
labels or a reviewed whitelist. Keep `zero_reject_veto_125_accept_clean` as the practical default
until that confirmation step is complete.

Generated evidence is intentionally local-only and should not be committed:

```text
outputs/balanced007_ranker_expansion_source_eval_20260708/final_candidate_selectors/
  zero_reject_veto_125_accept_clean_selected.csv
  zero_reject_veto_125_accept_clean_summary.json
  zero_reject_veto_125_accept_clean_post121_page_review/contact_sheet.png
  zero_reject_veto_125_accept_clean_post121_crop_review/contact_sheet.png
  zero_reject_veto_126_deployable_single_accept_summary.json
  zero_reject_veto_126_deployable_single_accept_page_review/contact_sheet.png
  zero_reject_veto_126_deployable_single_accept_crop_review/contact_sheet.png
  zero_reject_veto_auto_triage_promote_candidate_summary.json
  target_quality_scoring_20260708/full435_target_quality_v2.csv
  target_quality_scoring_20260708/selector_target_quality_summary.csv
  zero_reject_veto_54_target_confirmed_wins_summary.json
  zero_reject_veto_112_zero_target_loss_summary.json
  target_quality_borderline_triage_20260708/zero_reject_veto_112_borderline_triage_summary.json
  target_quality_borderline_triage_20260708/high_priority_review_packs/high_priority_29/page_pack/contact_sheet.png
  target_quality_borderline_triage_20260708/high_priority_review_packs/high_priority_29/crop_pack/contact_sheet.png
  target_quality_borderline_triage_20260708/high_priority_review_packs/auto_win_candidate/page_pack/contact_sheet.png
  target_quality_borderline_triage_20260708/high_priority_review_packs/auto_win_candidate/crop_pack/contact_sheet.png
  target_quality_borderline_triage_20260708/high_priority_review_packs/ratio_noise_review/page_pack/contact_sheet.png
  target_quality_borderline_triage_20260708/high_priority_review_packs/ratio_noise_review/crop_pack/contact_sheet.png
  target_quality_promotions_20260708/full435_target_quality_v2_auto_win_promoted_summary.json
  target_quality_promotions_20260708/selector_target_quality_auto_win_promoted_summary.json
```

This means the project has moved past "find a safe tiny gate" and into "increase reliable coverage
without visible regressions." Repeating one-step probes, hand-tuned intervals, or selector replay on
the same candidate family is low-leverage unless it is tied to a new failure bucket, a new training
objective, or a labeled page-level acceptance set. Further selector work should either label the 23
review pages or improve candidate generation; post-125 accept-only search is exhausted, and the wider
deployable feature/ranker search found only a single-page `126` upper-bound increment for the current
feature family.

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

weak-label ranker threshold sweep without writing PNGs:
  script: scripts/analysis/evaluate_region_component_threshold_sweep.py
  best shared threshold in sampled ranker summary: 0.5130248089880388
  SCUT115: materialized_pages=115/115, components=9144, selected_pixels=53746
  SCUT115: residual_gain=0.000231559, overerase_delta=-0.000000055
  SCUT115: improved_pages=75, worse_pages=32, over_reg_pages=0
  holdout40: materialized_pages=40/40, components=2898, selected_pixels=19102
  holdout40: residual_gain=0.000227216, overerase_delta=-0.000000027
  holdout40: improved_pages=22, worse_pages=13, over_reg_pages=0

t4 residual-delta oracle ceiling:
  script: scripts/analysis/evaluate_region_component_oracle_ceiling.py
  SCUT115 best oracle residual_gain=0.000941709, worse_pages=0
  holdout40 best oracle residual_gain=0.000452334, worse_pages=0
  status: even perfect target-aware component selection has limited headroom
```

Reusable workflow now exists:

```text
component extraction/rule probe: scripts/analysis/evaluate_region_component_selector.py
review pack builder: scripts/analysis/build_region_component_review_pack.py
impact review pack builder: scripts/analysis/build_region_component_impact_review_pack.py
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

outputs/region_component_impact_review_pack_weak_t0513_20260707/
  component-impact-labels-template.csv
  contact_sheet_residual_help.png
  contact_sheet_residual_hurt.png
  contact_sheet_overerase_risk.png
  contact_sheet_large_noop.png
```

Next selector progress should come from reviewed `keep` / `drop` / `review` labels and held-out
reviewed validation, not more weak-label or hand-threshold tuning.
The materialization/evaluation bridge is now proven usable, but the weak-label
ranker result is effectively too small at page level. Lowering the score
threshold increases metric gain slightly, but only by selecting every page and
still causing many page-level residual regressions. Treat this as workflow
infrastructure only; real progress requires reviewed labels or a better
candidate family before promotion.

The impact-ranked pack is the preferred next labeling source because it ranks
components by actual page-level residual and overerase pixel deltas. The first
pack selected 76 high-impact components from 10,041 scored components:

```text
residual_help: 30
residual_hurt: 30
overerase_risk: 14
large_noop: 2
```

Use it to spend review effort on components that can move page metrics. The
older balanced weak-verdict pack remains useful for selector sanity checks, but
it is less efficient for improving product-quality decisions.

The oracle ceiling check is now the stop condition for this t4 candidate family:
selector work can still validate infrastructure, but it should not be expected
to close the product-quality gap because the target-aware upper bound is below
0.001 residual gain on SCUT115 and below 0.0005 on holdout40. Further quality
work should prioritize a stronger candidate/generator objective before another
selector-training loop.

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

Residual-delta candidate-family triage:

```text
script: scripts/analysis/summarize_candidate_metric_families.py
output: outputs/candidate_family_summary_residual_delta_20260707/summary.csv

best current t4 sweep:
  SCUT115 residual_delta=-0.000193191, overerase_delta=-0.000004219
  holdout40 residual_delta=-0.004008179, overerase_delta=+0.000244709
  train160 residual_delta=-0.001401817, overerase_delta=-0.000008722
  next120 residual_delta=-0.001163004, overerase_delta=-0.000003471

darkpreserve_w2_t4:
  SCUT115 residual_delta=-0.000189791, overerase_delta=-0.000003162
  holdout40 residual_delta=-0.003806695, overerase_delta=+0.000246228
  status: slightly safer-looking but not materially better than t4

signed_delta_w1_t4:
  SCUT115 residual_delta=+0.000598372, overerase_delta=-0.000007248
  holdout40 residual_delta=-0.002714538, overerase_delta=+0.000242788
  status: improves some train/holdout behavior but regresses SCUT115 residual

mixed_td_w2_t4:
  SCUT115 residual_delta=+0.003735417, overerase_delta=-0.000178287
  holdout40 residual_delta=-0.000530685, overerase_delta=+0.000092958
  status: overerase-oriented tradeoff, not a handwriting-removal improvement path
```

The current t4 residual-delta sweep remains the best reusable candidate among
existing residual-delta outputs, but it is not strong enough to be the next
product-quality lever. Use it as a regression benchmark for future objectives:
a new candidate should beat t4 on SCUT115 residual without introducing the
holdout40 overerase penalty.

Overguard residual-delta follow-up:

```text
training output: outputs/train_scut_residual_delta_overguard_scale006_step80_20260707/
checkpoint: outputs/train_scut_residual_delta_overguard_scale006_step80_20260707/cleanup_best.pt
analysis output: outputs/overguard_scale006_step80_analysis_20260707/
evaluation gate: cleanup_alpha_threshold=0.3, second_delta_threshold=2

objective changes from t4 source checkpoint:
  init: outputs/train_scut_residual_delta_bias3_scale008_step150_20260707/cleanup_best.pt
  residual_delta_scale: 0.06
  outside_weight: 8.0
  alpha_negative_weight: 6.0
  alpha_sparsity_weight: 0.2
  overerase_proxy_weight: 40.0
  dark_preserve_weight: 1.0
  signed_delta_weight: 0.5 inside mask

SCUT115:
  t4 residual_delta=-0.000193191, overerase_delta=-0.000004219, wins=78, losses=34
  overguard residual_delta=-0.000786008, overerase_delta=-0.000009437, wins=89, losses=26
  verdict: materially better than t4 on aggregate metrics and page win/loss count

holdout40:
  t4 residual_delta=-0.004008179, overerase_delta=+0.000244709, wins=30, losses=10, over_reg_pages=36
  overguard residual_delta=-0.000831641, overerase_delta=-0.000009240, wins=31, losses=9, over_reg_pages=0
  verdict: fixes t4 overerase regression but sacrifices most holdout residual gain

gate sweep:
  alpha=0.3, delta=2 is the only useful tested setting
  alpha=0.5 or delta>=4 collapses to no-op on SCUT115 and holdout40
```

Treat overguard as the new conservative residual-delta branch to investigate,
not a product default. It is the first residual-delta variant in this fork that
beats t4 on SCUT115 while avoiding holdout40 overerase regression. The remaining
gap is holdout40 residual coverage: a product candidate needs to recover more of
t4's holdout residual gain without reintroducing overerase.

Balanced residual-delta follow-up:

```text
training output: outputs/train_scut_residual_delta_balanced_scale007_step80_20260707/
checkpoint: outputs/train_scut_residual_delta_balanced_scale007_step80_20260707/cleanup_best.pt
evaluation gate: cleanup_alpha_threshold=0.3, base_edit_threshold=12, second_delta_threshold=2

objective changes from overguard:
  residual_delta_scale: 0.07
  outside_weight: 6.0
  alpha_negative_weight: 5.0
  alpha_sparsity_weight: 0.15
  overerase_proxy_weight: 30.0
  dark_preserve_weight: 0.75
  signed_delta_weight: 0.35 inside mask

SCUT115:
  residual_delta=-0.001349386843
  overerase_delta=-0.000020558496
  gate=0.014204064

holdout40:
  residual_delta=-0.001418918619
  overerase_delta=-0.000018467665
  gate=0.011316136

train160:
  residual_delta=-0.004913059464
  overerase_delta=-0.000024327035
  gate=0.007661010

next120:
  residual_delta=-0.004139072483
  overerase_delta=-0.000023137928
  gate=0.017372110
```

Treat balanced scale0.07 as the current best residual-delta candidate, not yet
as a product default. It is the first checked residual-delta branch that improves
both aggregate residual and aggregate overerase across SCUT115, holdout40,
train160, and next120. It beats t4's SCUT115 residual gain, removes t4's
holdout40 overerase regression, and recovers more holdout residual coverage than
overguard. The next required step is visual/page-level review on high-gate pages
and worst residual-loss pages before promotion.

Balanced007 review triage:

```text
review queue: outputs/product_quality_review_balanced007_four_split_20260707/
scope: 96 pressure-test rows, 24 per split
page buckets:
  metric wins: 32
  metric losses / overerase risks: 32
  high-activity risks: 32
crop pack: 384 crops
local target-comparison proxy:
  accept=38
  review=3
  reject=55
by split:
  SCUT115: accept=10, reject=14
  holdout40: accept=6, review=2, reject=16
  train160: accept=13, review=1, reject=10
  next120: accept=9, reject=15
by bucket:
  metric_win: accept=28, review=2, reject=2
  metric_loss_or_overrisk: reject=32
  high_activity_risk: accept=10, review=1, reject=21
```

Interpret this carefully: the queue intentionally oversamples wins, losses,
and high-activity risk pages, so the 55 rejects are not a population reject
rate. The important signal is that metric wins mostly pass the local proxy, but
high-activity and metric-loss pages contain many obvious target-distance hurts.
Balanced007 remains the best aggregate candidate and the right benchmark for
review, but it is not product-ready without a selector/veto or visible-review
pass that filters the high-risk pages.

Balanced007 selector/veto checkpoint:

```text
rule:
  active_baseline_edit_mean >= 96.6185399391
  AND active_gray_p75 <= 192

full four-split metrics:
  pages=435
  selected=26
  coverage=6.0%
  wins/losses=26/0
  residual_gain=0.001247317
  overerase_delta=-0.000002939

split detail:
  SCUT115: selected=4/115, wins/losses=4/0, residual_gain=0.000458754
  holdout40: selected=2/40, wins/losses=2/0, residual_gain=0.000717984
  train160: selected=12/160, wins/losses=12/0, residual_gain=0.001549166
  next120: selected=8/120, wins/losses=8/0, residual_gain=0.001777002

pressure-review proxy:
  selected=20/96
  accept=20
  review=0
  reject=0
```

This is the best current selector-shaped result for balanced007: it is much
safer than applying the candidate broadly, but coverage is still only about 6%
across the four validation splits. Treat it as a safe optional review benchmark,
not enough for product-level coverage.

Balanced007 full-proxy ranker check:

```text
full local proxy queue:
  rows=435
  accept=111
  review=45
  reject=279

metric buckets:
  metric_loss_or_overrisk: reject=90, review=1
  metric_win: accept=111, review=44, reject=189

ranker train=train160+next120, test=SCUT115+holdout40:
  best listed held-out row selected=12/155
  test metric_losses=0
  test accept/review/reject=9/2/1
  train selected=42/280, train reject=3

ranker train=SCUT115+holdout40, test=train160+next120:
  best listed held-out row selected=34/280
  test metric_losses=6
  test accept/review/reject=25/3/6
  train selected=16/155, train reject=2
```

The learned page ranker is not yet better than the hand-mined safe26 rule. It
can select more pages, but reject and metric-loss transfer is unstable across
split directions. Keep safe26 as the current conservative selector benchmark
and use full-proxy labels only for further feature analysis or a larger reviewed
label set, not for product gating.

Safe26 expansion review queue:

```text
output: outputs/balanced007_safe26_expansion_review_20260707/
source: ranker-selected pages not selected by safe26
rows=67
local proxy:
  accept=46
  review=10
  reject=11
by split:
  SCUT115: accept=9, review=4, reject=2
  holdout40: accept=2, review=1, reject=0
  train160: accept=16, review=4, reject=6
  next120: accept=19, review=1, reject=3
focused packs:
  expansion_accepts: 32 pages, 128 crops
  expansion_reviews: 10 pages, 40 crops
  expansion_rejects: 11 pages, 44 crops
```

This is the next useful manual/visual-AI review surface. If the 32 expansion
accepts are visually clean, safe26 can potentially widen materially without
another training run. The 11 rejects should be mined as explicit veto patterns
before accepting ranker-selected pages wholesale.

2026-07-08 continuation:

The safe26 expansion search is now reproducible in the active repository:

```text
script: scripts/analysis/search_expansion_keep_rules.py
repro output: outputs/balanced007_safe26_expansion_rule_search_repro_20260707/
best expansion rule: help_hurt_ratio >= 1.515555555556
expansion-only: 46 pages, accept/review/reject=46/0/0
base safe26: 26 pages, accept/review/reject=21/2/3
combined: 72 pages, accept/review/reject=67/2/3
metric_losses=0
residual_gain=0.002444313
```

Important interpretation: the combined 3 rejects are inherited from the base
safe26 set, not introduced by the expansion rule. Use `added_reject` /
`expansion_reject`, not raw `combined_reject`, when judging whether an
incremental expansion source adds new reject pages.

A second ranker-score expansion source was evaluated with
`scripts/analysis/evaluate_ranker_expansion_sources.py`:

```text
source: t4_enh@0.88 + safe001@0.80
output: outputs/balanced007_ranker_expansion_source_eval_20260708/
incremental: 12 pages, accept/review/reject=8/4/0
metric_losses=0
residual_gain=0.000284985
visual spot-check: 4 review pages pass_with_caution; no obvious printed-structure damage
```

The current recommended product-safe candidate is the zero-reject vetoed
selector:

```text
selector: zero_reject_veto_79
rule stack:
  safe26 base gated by active_target_delta_mean > 96.00890552911
  + best expansion rule above
  + ranker-score bump above
selected=79/435
coverage=18.16%
accept/review/reject=74/5/0
metric_losses=0
residual_gain=0.002609363
```

The higher-gain no-veto variant is not recommended as a product default:

```text
selector: no_veto_84
selected=84/435
coverage=19.31%
accept/review/reject=75/6/3
metric_losses=0
residual_gain=0.002729299
```

Reason for rejection: it keeps the three known base safe26 reject pages
(`scut115/17.jpg`, `train160/91.jpg`, `train160/160.jpg`). Prefer
`zero_reject_veto_79` unless a later manual review overturns those local
reject labels.

Further single-feature mining after `zero_reject_veto_79` found only tiny clean
third-bucket opportunities. The best was `active_edge_p95 <= 524`, adding just
2 accept pages (`scut115/395.jpg`, `scut115/43.jpg`) with no rejects or metric
losses. Do not complicate the default selector for this small gain unless a
later packaging step needs every zero-risk page.

Source-of-truth details for the historical migration remain in:

```text
/Volumes/Tool/source/clean-doc/docs/current-best-scut-hardcase.md
```

New model work should continue in this repository. The clean-doc workspace is a historical product/research workspace and temporary artifact source, not the active model-engineering entrypoint.
