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

Source-of-truth details for the historical migration remain in:

```text
/Volumes/Tool/source/clean-doc/docs/current-best-scut-hardcase.md
```

New model work should continue in this repository. The clean-doc workspace is a historical product/research workspace and temporary artifact source, not the active model-engineering entrypoint.
