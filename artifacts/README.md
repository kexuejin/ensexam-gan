# Local Artifacts

This directory is intentionally local-first. Large checkpoints, prediction folders, and dataset-derived outputs should be symlinked here instead of committed.

Registered production/current assets are documented in:

```text
docs/model-registry.md
```

Current expected local symlinks:

```text
artifacts/full-training-best.pth
artifacts/full-training/20260702_070153
artifacts/current-primary
artifacts/current-second-stage-best.pt
artifacts/current-holdout40-primary-pred
```
