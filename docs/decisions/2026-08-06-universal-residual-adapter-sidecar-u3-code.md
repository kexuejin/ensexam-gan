# Universal Residual Adapter Sidecar U3 Code Decision

```text
u3_terminal = CODE_IMPLEMENTED_SYNTHETIC_TESTS_PASS
training_handoff = disabled
dataset_inference_handoff = disabled
artifact_mutation = none
product_default = artifacts/current-primary
commit_status = deferred_clean_split_required
```

## Scope

U3 implemented the code-only and synthetic/unit-test surface authorized by
`docs/plans/2026-08-06-universal-residual-adapter-sidecar-implementation-plan.md`.
It did not train, fine-tune, run dataset inference, inspect target images,
download data, mutate checkpoints/artifacts, open fresh blind evidence, or
change the default `artifacts/current-primary` product path.

## Implemented Surface

- `networks/generator.py` now has a disabled-by-default
  `UniversalResidualAdapterSidecar`.
- Default `Generator()` creates no sidecar parameters and keeps the legacy
  return shape.
- Enabled sidecar uses only `reconstruction_feature` plus same-call baseline
  `Icomp`.
- Gate weights are continuous softmax simplex weights.
- Sidecar residual is residual-only, bounded, and zero-initialized.
- Zero-init enabled output is exactly equal to the same baseline generator.
- Structural fallback returns the same-call baseline output.
- Optional telemetry is numeric/statistical only.
- `train.py` includes `validate_universal_sidecar_config(cfg)` and fails closed
  on unsafe sidecar training configs:
  - non-current-primary initialization;
  - resume or optimizer/scheduler state;
  - base-trainable patterns;
  - unfrozen BatchNorm running stats;
  - non-strict reproducibility;
  - adapter count other than 3;
  - residual bound above `12/255`;
  - routing-like sidecar config keys.
- `scripts/analysis/audit_universal_sidecar_structure.py` performs synthetic
  structure checks only.

## Verification

```text
source .env
$ENSEXAM_PYTHON -m pytest \
  tests/test_universal_residual_adapter_sidecar.py \
  tests/test_universal_sidecar_config_validation.py \
  tests/test_audit_universal_sidecar_structure.py

result: 16 passed
```

```text
source .env
$ENSEXAM_PYTHON -m pytest \
  tests/test_generator_activation_checkpointing.py \
  tests/test_train_checkpoint_initialization.py \
  tests/test_train_runtime_config.py

result: 17 passed
```

```text
source .env
$ENSEXAM_PYTHON -m pytest \
  tests/test_collision_paired_reconstruction_feature_invariance.py \
  tests/test_underprint_semantic_conditioning.py \
  tests/test_semantic_separation_auxiliary.py \
  tests/test_three_class_semantic_separation_auxiliary.py \
  tests/test_three_class_semantic_mb_logit_fusion.py

result: 25 passed
```

```text
git diff --check -- \
  networks/generator.py \
  train.py \
  scripts/analysis/audit_universal_sidecar_structure.py \
  tests/test_universal_residual_adapter_sidecar.py \
  tests/test_universal_sidecar_config_validation.py \
  tests/test_audit_universal_sidecar_structure.py

result: pass
```

Subagent checks:

- generator/train mapping: completed read-only before implementation;
- boundary verifier: no key artifact/checkpoint mutation found, no training or
  dataset inference process found;
- architecture critic: initial defects found, then fixed; final verdict `PASS`.

## Commit Handling

U3 code is intentionally not committed in the current worktree because
`networks/generator.py` and `train.py` already contained large pre-existing
uncommitted changes before U3 began. A broad commit of those files would mix U3
with unrelated historical work. The correct next repository-hygiene step is a
clean-split commit or a clean worktree replay that stages only the U3 sidecar
implementation, tests, audit helper, and this decision record.

## Next Stage Boundary

U4 remains disabled until a separate bounded development-validation plan is
opened. U4 must not use consumed HW5K official test evidence, fresh blind
evidence, promotion paths, or checkpoint mutation unless separately authorized
by that bounded stage.
