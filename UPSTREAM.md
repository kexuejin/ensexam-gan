# Upstream Boundary

This project is a productized fork / continuation of:

```text
https://github.com/xiaozhejiya/ensexam-gan
```

The original upstream repository did not include a `scripts/` directory. The `scripts/` tree in this fork contains local engineering extensions for SCUT hardcase fine-tuning, inference, evaluation, residual repair, and experiment analysis.

Tracked upstream-style areas:

```text
train.py
test.py
meta_train.py
tune.py
config.yaml
data/
losses/
networks/
tools/
utils/
```

Local fork responsibilities:

```text
scripts/
configs/
docs/
hardcase_lists/
```

Do not treat this repository as a clean vendor checkout. It is now the main model-engineering workspace for the clean-doc handwriting-removal pipeline.
