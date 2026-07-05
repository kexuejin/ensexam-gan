# Configs

This directory contains local run configurations for the productized EnsExam-GAN fork.

Mainline configs:

```text
configs/local/config.local-full-mps-fast.yaml
configs/local/config.local-hardcase-mps.yaml
configs/local/config.local-hardcase-region-mps.yaml
configs/local/config.local-hardcase-region-preserve12-leak0p5-mps.yaml
```

Historical / exploratory configs are archived under `configs/archive/`:

```text
configs/archive/config.local-batch4-probe.yaml
configs/archive/config.local-batch8-probe.yaml
configs/archive/config.local-compare.yaml
configs/archive/config.local-full-mps.yaml
configs/archive/config.local-hardcase-region-b2-probe-mps.yaml
configs/archive/config.local-hardcase-region-preserve-*.yaml
configs/archive/config.local-hardcase-region-probe-mps.yaml
configs/archive/config.local-hardcase-region-short-mps.yaml
configs/archive/config.local-smoke.yaml
```

Path policy:

```text
data_root: ./data-links/samples/SCUT-EnsExam
full-training resume_path: ./artifacts/full-training-best.pth
```

Do not point active configs back to `clean-doc/external/ensexam-gan`. Large checkpoints and datasets should stay registered through `artifacts/` and `data-links/`.
