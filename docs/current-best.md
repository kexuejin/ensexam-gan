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

The previous one-day full-training model is reusable in this new project through `artifacts/full-training-best.pth`. It should be copied or symlinked as an asset, not regenerated. The current product direction should spend time on targeted continuation, gate tuning, second-stage repair, and local visual/metric review before considering another full-training run.

Do not add the current whiteout inpaint repair to the default pipeline. It can reduce residual metrics on correction-fluid pages, but visual review shows the repaired area may look dirtier than leaving a clean white patch.

Source-of-truth details for the historical migration remain in:

```text
/Volumes/Tool/source/clean-doc/docs/current-best-scut-hardcase.md
```

New model work should continue in this repository. The clean-doc workspace is a historical product/research workspace and temporary artifact source, not the active model-engineering entrypoint.
