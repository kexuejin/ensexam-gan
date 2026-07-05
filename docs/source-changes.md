# Source Changes From Upstream

This fork retains upstream history from `https://github.com/xiaozhejiya/ensexam-gan` and adds productization changes needed for local SCUT hardcase training, MPS runs, and residual repair.

## Data And Masks

- `data/dataset.py` supports explicit masks under `all_masks`, so datasets such as ExamInk-Seg can provide direct mask supervision instead of only target-diff heuristics.
- `data/dataset.py` can infer erase box classes from local input/target differences and optionally preserve non-erase box classes.
- `data/augmentation.py` treats `box_preserve` as a mask target so preservation masks receive the same spatial transforms as the image.
- `data/mask_utils.py` parses box classes, filters `generate_mb_from_boxes` by target classes, and infers changed box classes using deterministic local target comparison.

## Losses

- `losses/losses.py` adds optional region-normalized LR loss.
- `losses/losses.py` adds optional input preservation, mask-leak, and box-preserve penalties for over-erasure control.
- New loss terms default to zero unless enabled by config, preserving upstream behavior for default configs.

## Training

- `train.py` supports MPS device selection and non-CUDA batch transfer.
- `train.py` supports explicit train/val file lists, max train/val file limits, max steps per epoch, validation cadence, and skip-final-test for bounded probes.
- `train.py` accepts dataset batches with optional `Box_preserve_gt`.
- `train.py` records skipped validation cleanly in CSV/W&B paths.

## Testing

- `test.py` supports MPS auto-selection.
- `test.py` supports `--max-test-files` and `--max-test-patches` for bounded smoke checks.

## Projectization

- New scripts under `scripts/` provide hardcase eval, checkpoint sweeps, micro training probes, second-stage residual repair, analysis helpers, and experimental utilities.
- `scripts/infer/patch_cleanup_erasemap.py` internalizes the second-stage cleanup inference code so the main runner no longer imports `clean-doc/scripts`.
- Configs under `configs/local/` use `data-links/` and `artifacts/` paths rather than old `clean-doc/external/ensexam-gan` paths.

## Commit Guidance

Split future commits by concern where possible:

```text
1. upstream source capability changes: data/loss/train/test
2. projectization scripts and configs
3. documentation and local artifact registration
```

Do not mix new experiments with migration cleanup unless the experiment needs the migration to run.

