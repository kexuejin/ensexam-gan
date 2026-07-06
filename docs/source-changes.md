# Source Changes From Upstream

This fork retains upstream history from `https://github.com/xiaozhejiya/ensexam-gan` and adds productization changes needed for local SCUT hardcase training, MPS runs, and residual repair.

## Data And Masks

- `data/dataset.py` supports explicit masks under `all_masks`, so datasets such as ExamInk-Seg can provide direct mask supervision instead of only target-diff heuristics.
- `data/dataset.py` can infer erase box classes from local input/target differences and optionally preserve non-erase box classes.
- `data/augmentation.py` treats `box_preserve` as a mask target so preservation masks receive the same spatial transforms as the image.
- `data/mask_utils.py` parses box classes, filters `generate_mb_from_boxes` by target classes, and infers changed box classes using deterministic local target comparison.
- `scripts/experimental/materialize_target_diff_masks.py` creates local `all_masks` datasets from paired input/target images without copying large image payloads by default.
- `scripts/experimental/build_explicit_mask_patch_index.py` ranks explicit-mask patches using coordinates that match `EnsExamRealDataset` exactly, including edge patches.
- `scripts/experimental/prepare_examink_seg_dataset.py` downloads the current `ynyg/ExamInk-Seg`
  Hugging Face layout through the tree API, including pagination, source/target/mask stem matching,
  per-triplet progress, train-only probes, and curl fallback for transient SSL EOFs.

## Losses

- `losses/losses.py` adds optional region-normalized LR loss.
- `losses/losses.py` adds optional input preservation, mask-leak, and box-preserve penalties for over-erasure control.
- New loss terms default to zero unless enabled by config, preserving upstream behavior for default configs.

## Training

- `train.py` supports MPS device selection and non-CUDA batch transfer.
- `train.py` supports explicit train/val file lists, max train/val file limits, max steps per epoch, validation cadence, and skip-final-test for bounded probes.
- `train.py` accepts dataset batches with optional `Box_preserve_gt`.
- `train.py` records skipped validation cleanly in CSV/W&B paths.
- `scripts/train/micro_train_region_probe.py` can mix patch-index-selected patches with ordinary
  patches via `--patch-index-mix-ratio`, and trace whether sampled patches matched the index.
- `scripts/experimental/build_hard_patch_list.py` can be run directly from this repository to
  generate ranked train-split patch-index CSVs for sensitivity sweeps.
- `hardcase_lists/nearworst_safe_step1_exact129_patch_index.csv` pins the single patch needed to
  reproduce the nearworst-safe one-step probe without relying on DataLoader shuffle order.
- `scripts/experimental/build_anchor_similar_patch_list.py` ranks train-split patches by local
  similarity to an anchor patch, which is useful when patch identity changes gate features.
- `scripts/experimental/build_patch_sensitivity_queue.py` turns ranked patch-index CSVs into
  per-patch index files plus train/eval command queues for deterministic sensitivity sweeps.
- `configs/local/config.local-scut-targetdiff-maskonly-smoke-mps.yaml` provides a bounded MPS smoke
  config for target-difference explicit-mask mask-only probes.
- `configs/local/config.local-examink-seg-maskonly-smoke-mps.yaml` provides the matching bounded
  MPS smoke config for ExamInk-Seg explicit-mask probes.

## Testing

- `test.py` supports MPS auto-selection.
- `test.py` supports `--max-test-files` and `--max-test-patches` for bounded smoke checks.

## Analysis

- `scripts/analysis/compare_gate_feature_drift.py` compares strict-gate feature drift across
  candidate metrics, highlighting whether rejected pages missed `copy_mask_cov8`,
  `primary_edit_px`, or both.

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
