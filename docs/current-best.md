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

Initial smoke evidence is positive but not enough for promotion:

```text
training data: ExamInk-Seg smoke4 explicit-mask patches
checkpoint: outputs/smoke_examink_cleanup_erasemap_identity_step100_20260707/cleanup_probe.pt
ExamInk smoke4, base_edit=12, second_delta=2:
  residual 0.193198 -> 0.168635
  overerase 0.004355 -> 0.004293
SCUT holdout4, base_edit=12, second_delta=4:
  residual ~= 0.1562 -> 0.1550
  overerase ~= 0.00202 -> 0.00198
```

Do not treat this as a product default yet. The ExamInk result is same-sample train/eval and the SCUT
result is small. The next useful step is to train this cleanup branch on a larger ExamInk-Seg subset
and validate on fixed SCUT/holdout splits before selector tuning.

Source-of-truth details for the historical migration remain in:

```text
/Volumes/Tool/source/clean-doc/docs/current-best-scut-hardcase.md
```

New model work should continue in this repository. The clean-doc workspace is a historical product/research workspace and temporary artifact source, not the active model-engineering entrypoint.
