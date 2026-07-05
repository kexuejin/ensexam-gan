# EnsExam-GAN Fork Migration Plan

## Decision

Promote the current local `external/ensexam-gan` work into a first-class fork/project under:

```text
/Volumes/Tool/source/ensexam-gan
```

Keep `/Volumes/Tool/source/clean-doc` as the higher-level product/research workspace for datasets, comparisons, review packs, historical experiments, and product-quality documentation.

The upstream project `xiaozhejiya/ensexam-gan` did not contain a `scripts/` directory. The local `external/ensexam-gan/scripts/` tree is our engineering extension layer for SCUT hardcase training, evaluation, residual repair, and productization. It should not remain hidden under `external/`.

## Current Upstream Boundary

Tracked upstream-style files in `external/ensexam-gan` include:

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

Local modifications currently exist in:

```text
data/augmentation.py
data/dataset.py
data/mask_utils.py
losses/losses.py
test.py
train.py
```

Local untracked extension work includes:

```text
scripts/
config.local-*.yaml
hardcase_lists/
checkpoints-*/
```

## Target Project Layout

Recommended new layout:

```text
/Volumes/Tool/source/ensexam-gan/
  README.md
  UPSTREAM.md
  configs/
    scut/
    local/
    archive/
  data/
  losses/
  networks/
  utils/
  tools/
  scripts/
    train/
    infer/
    eval/
    analysis/
    experimental/
    archive/
  docs/
    current-best.md
    model-registry.md
    runbook.md
    rejected-directions.md
    migration-notes.md
  tests/
```

## Mainline Scope

Promote these paths first:

```text
external/ensexam-gan/train.py
external/ensexam-gan/test.py
external/ensexam-gan/config_loader.py
external/ensexam-gan/data/
external/ensexam-gan/losses/
external/ensexam-gan/networks/
external/ensexam-gan/utils/
external/ensexam-gan/tools/
external/ensexam-gan/scripts/micro_train_region_probe.py
external/ensexam-gan/scripts/run_second_stage_residual_repair.py
external/ensexam-gan/scripts/eval_hardcase_worst_pages.py
external/ensexam-gan/scripts/batch_eval_hardcase_checkpoints.py
external/ensexam-gan/scripts/cached_sweep_hardcase_postprocess.py
```

Current effective product path:

```text
primary EnsExam-GAN fine-tune
-> auto_cov8_step copy-mask inference
-> second-stage residual repair
-> local metrics + review pack
```

Do not promote whiteout inpaint repair into the default pipeline. It is metric-positive but visually rejected.

## Extension Script Classification

### Mainline

```text
scripts/train/micro_train_region_probe.py
scripts/infer/run_second_stage_residual_repair.py
scripts/eval/eval_hardcase_worst_pages.py
scripts/eval/batch_eval_hardcase_checkpoints.py
scripts/eval/cached_sweep_hardcase_postprocess.py
scripts/analysis/analyze_residual_components.py
scripts/analysis/extract_overerase_components.py
```

### Experimental

```text
scripts/experimental/prepare_examink_seg_dataset.py
scripts/experimental/run_gated_whiteout_repair.py
scripts/experimental/build_explicit_mask_patch_index.py
scripts/experimental/convert_overerase_patches_to_dataset_index.py
scripts/experimental/select_copy_mask_threshold_map.py
scripts/experimental/select_hardcase_checkpoint.py
scripts/experimental/select_top_overerase_patches.py
```

### Archive Candidates

Old one-off configs, hardcase checkpoint folders, and rejected probe outputs should be documented in `docs/rejected-directions.md` and left outside the clean mainline.

## Model Registry Seed

Current primary checkpoint:

```text
/Volumes/Tool/source/clean-doc/outputs/micro_seed999_secondring_up_left_1440_640_lr1e6_step1_20260705/micro_region_probe_step0001.pth
```

Current primary config:

```text
/Volumes/Tool/source/clean-doc/outputs/micro_seed999_secondring_up_left_1440_640_lr1e6_step1_20260705/config.yaml
```

Current second-stage checkpoint:

```text
/Volumes/Tool/source/clean-doc/outputs/patch-cleanup-erasemap-mixed6k-dense1x-lctx-e8/best.pt
```

Current validation anchors:

```text
hardcase7: residual 0.263922, overerase 0.001125
holdout40: residual 0.134026, overerase 0.002482
scut-test115 second-stage: residual 0.114225, overerase 0.003048
```

## Migration Phases

### Phase 1: Copy Fork Skeleton

Create `/Volumes/Tool/source/ensexam-gan` and copy the current `external/ensexam-gan` source tree into it.

Do not move large outputs/checkpoints yet. Reference them by absolute path or symlink only after smoke validation passes.

### Phase 2: Normalize Script Layout

Move extension scripts into `scripts/train`, `scripts/infer`, `scripts/eval`, `scripts/analysis`, and `scripts/experimental`.

Keep compatibility wrappers only where historical commands are still useful.

### Phase 3: Register Current Best

Create:

```text
docs/current-best.md
docs/model-registry.md
docs/runbook.md
docs/rejected-directions.md
```

Move the relevant conclusions from `clean-doc/docs/current-best-scut-hardcase.md` into these split documents.

### Phase 4: Verify Mainline

Required checks before treating the new project as active:

```bash
.venv-torch310-mps-stable/bin/python -m py_compile \
  scripts/train/micro_train_region_probe.py \
  scripts/infer/run_second_stage_residual_repair.py \
  scripts/eval/eval_hardcase_worst_pages.py \
  scripts/eval/batch_eval_hardcase_checkpoints.py \
  scripts/eval/cached_sweep_hardcase_postprocess.py
```

Then run a small smoke evaluation using existing predictions before any retraining.

### Phase 5: Update Clean-Doc References

After the new fork works:

- Update `clean-doc/docs/current-best-scut-hardcase.md` to point to `/Volumes/Tool/source/ensexam-gan`.
- Keep `clean-doc` outputs and review packs as historical/product artifacts.
- Stop adding new model-training scripts under `clean-doc/scripts` unless they are product-level orchestration.

## Safety Rules

- Copy before moving.
- Do not delete `external/ensexam-gan` until the new project reproduces the current best validation.
- Do not migrate large checkpoints blindly; register them first.
- Do not promote whiteout inpaint repair despite positive metrics.
- Do not use target-derived features in any product inference gate.
- Keep `clean-doc` as the product/research workspace and `ensexam-gan` as the model engineering workspace.

