# Local Config Index

Use these configs from the repository root.

Recommended commands:

```bash
$ENSEXAM_PYTHON train.py --config configs/local/config.local-full-mps-fast.yaml
$ENSEXAM_PYTHON scripts/micro_train_region_probe.py --config configs/local/config.local-current-primary-continuation-mps.yaml
$ENSEXAM_PYTHON train.py --config configs/local/config.local-hardcase-mps.yaml
$ENSEXAM_PYTHON scripts/micro_train_region_probe.py --config configs/local/config.local-hardcase-region-mps.yaml
```

Default roles:

```text
config.local-full-mps-fast.yaml: reusable full-training recipe; do not rerun by default if artifacts/full-training-best.pth is available
config.local-current-primary-continuation-mps.yaml: direct continuation from artifacts/current-primary/micro_region_probe_step0001.pth
config.local-hardcase-mps.yaml: hardcase continuation from artifacts/full-training-best.pth
config.local-hardcase-region-mps.yaml: bounded region-focused hardcase continuation
config.local-hardcase-region-preserve12-leak0p5-mps.yaml: preserve/low-leak hardcase branch used for targeted probes
```

The remaining configs are retained for reproducibility of previous probes and should not become default without a fresh evaluation note in `docs/model-registry.md`.
