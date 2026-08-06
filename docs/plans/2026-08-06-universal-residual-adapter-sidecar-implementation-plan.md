# Universal Residual Adapter Sidecar Implementation Plan

## U2 Terminal

```text
u2_terminal = IMPLEMENTATION_PLAN_READY
implementation_handoff = enabled_for_U3_code_only
training_handoff = disabled
inference_handoff = disabled
product_default = artifacts/current-primary
```

This U2 plan converts the admitted U1 mechanism into a bounded code/test plan.
It does not implement code, train, run model inference on datasets, inspect
target images, download data, mutate checkpoints, open fresh blind evidence, or
promote a product path.

U3 may implement only the code and synthetic/unit-test surfaces listed below.
U4 development-only validation remains a later separate stage.

## Entry Evidence

- U1 design:
  `docs/plans/2026-08-06-universal-residual-adapter-sidecar-design.md`.
- U1 admission:
  `docs/decisions/2026-08-06-universal-residual-adapter-sidecar-admission.md`.
- Relevant code anchors:
  - `networks/generator.py`: `RefineNet.forward(..., return_reconstruction_feature=True)`
    already returns the feature tap needed by the sidecar.
  - `networks/generator.py`: `Generator.forward(..., return_reconstruction_feature=True)`
    already propagates the reconstruction feature.
  - `train.py`: `apply_generator_trainable_patterns` and
    `freeze_batchnorm_running_stats` already support conservative sidecar-only
    trainability and BN immutability controls.
  - `scripts/analysis/gate_dev_candidate_metrics.py` and
    `scripts/analysis/gate_scut_fixed_regression_metrics.py` already provide
    whole-candidate residual/overerase gate patterns.

## U3 Scope

U3 is code implementation only. It may:

- add disabled-by-default sidecar modules;
- add config parsing and validation for the sidecar;
- add synthetic unit tests and static checks;
- add a structure-audit helper that uses synthetic tensors and state dicts;
- update docs to describe the disabled-by-default code surface.

U3 must not:

- train or fine-tune;
- run dataset inference;
- load or inspect target images;
- download data;
- mutate checkpoints or artifacts;
- run development gates on real splits;
- use consumed HW5K official test;
- implement router/domain-label behavior;
- alter the default `current-primary` path when the sidecar is disabled.

## Code Design

### 1. `networks/generator.py`

Add a small, disabled-by-default sidecar under `Generator`:

```text
UniversalResidualAdapterSidecar
  input: reconstruction_feature f, baseline output y0
  outputs:
    candidate output y
    telemetry dict
```

Required behavior:

- config namespace:
  `model.universal_residual_adapter_sidecar`;
- default `enabled = false`;
- when disabled, no sidecar parameters are created and the legacy
  `Generator()` state dict surface remains unchanged;
- when enabled:
  - `adapter_count = 3`;
  - conditioner reads only `reconstruction_feature`;
  - gate emits continuous softmax simplex weights;
  - no domain/caller/path/source arguments exist;
  - adapter final projections and global residual scale initialize to zero;
  - initial enabled output equals the baseline output;
  - residual is bounded before output clamp;
  - structural fallback returns same-call baseline `y0`;
  - telemetry is numeric-only and contains no path/domain/source text.

Implementation notes:

- Treat `Icomp` as `y0`, because it is the current product output returned by
  `Generator.forward`.
- Use `reconstruction_feature` from `RefineNet` as the only conditioner/adapter
  input.
- Keep existing return shapes by default. Only return telemetry when an explicit
  `return_universal_sidecar_telemetry=True` flag is passed.
- Do not add a `domain`, `source`, `caller`, `route`, or `expert_id` argument to
  public model calls.

### 2. `train.py`

Do not add a training run. Add only configuration validation needed to keep
future U4 safe:

- reject sidecar-enabled configs unless `init_checkpoint` points at
  `artifacts/current-primary/micro_region_probe_step0001.pth`;
- require `train.trainable_generator_patterns` to match only sidecar parameter
  names for sidecar training configs;
- require `freeze_generator_batchnorm_stats = true`;
- require no resume optimizer state;
- require strict reproducibility mode for future sidecar training configs;
- record trainable/frozen parameter counts in existing trainability summary.

If these checks are too invasive for U3, implement them as a separate
`validate_universal_sidecar_config(cfg)` helper plus tests, and call it only
when the sidecar config is enabled.

### 3. New Structure Audit Helper

Add a lightweight analysis helper, for example:

```text
scripts/analysis/audit_universal_sidecar_structure.py
```

The helper may use synthetic tensors and local state dict inspection only. It
must not read datasets, labels, target images, predictions, or checkpoints
unless a later U3 substage explicitly authorizes current-primary SHA checks.

Required checks:

- sidecar disabled leaves default generator state-dict keys unchanged;
- sidecar enabled creates only sidecar-prefixed trainable parameters when base
  freeze patterns are applied;
- no public model argument contains `domain`, `source`, `caller`, `route`, or
  `expert`;
- zero-init sidecar output matches baseline on synthetic tensors;
- telemetry keys are numeric-only and domain-free;
- fallback returns baseline output for synthetic invalid-residual cases.

## Test Plan

U3 must add focused tests before any development validation:

```text
tests/test_universal_residual_adapter_sidecar.py
tests/test_universal_sidecar_config_validation.py
tests/test_audit_universal_sidecar_structure.py
```

Required unit tests:

1. **Default compatibility.**
   `Generator()` with default config has no sidecar params and preserves legacy
   forward return shapes.
2. **Zero-init equivalence.**
   Sidecar-enabled generator returns identical `Icomp` output to the same
   generator with the sidecar disabled on synthetic fixed-seed tensors.
3. **No external routing surface.**
   Public forward/config surfaces expose no domain/source/caller/path routing
   argument.
4. **Simplex gate.**
   Gate weights are finite, non-negative, sum to one, and are continuous tensor
   values.
5. **Residual bound.**
   Mixed residual is clamped to the configured L-infinity bound before output
   clamp.
6. **Fallback.**
   Synthetic invalid gate/residual conditions return same-call baseline output
   and emit structural fallback reason codes.
7. **Telemetry hygiene.**
   Telemetry contains only approved numeric/statistical keys and no image,
   path, source, label, or domain text.
8. **Base freeze validation.**
   Sidecar config validation rejects base-trainable, BN-mutating, resume, and
   non-current-primary initialization configs.
9. **Audit helper.**
   Structure audit passes on a compliant synthetic sidecar config and fails on a
   deliberately domain-like config or trainable-base config.

U3 verification commands:

```bash
source .env
$ENSEXAM_PYTHON -m pytest \
  tests/test_universal_residual_adapter_sidecar.py \
  tests/test_universal_sidecar_config_validation.py \
  tests/test_audit_universal_sidecar_structure.py

$ENSEXAM_PYTHON -m pytest \
  tests/test_generator_activation_checkpointing.py \
  tests/test_train_checkpoint_initialization.py \
  tests/test_train_runtime_config.py

git diff --check -- \
  networks/generator.py \
  train.py \
  scripts/analysis/audit_universal_sidecar_structure.py \
  tests/test_universal_residual_adapter_sidecar.py \
  tests/test_universal_sidecar_config_validation.py \
  tests/test_audit_universal_sidecar_structure.py
```

These commands use synthetic/unit-test fixtures only. They are not training or
dataset inference.

## U3 Pass / Kill / Prerequisite

```text
PASS = CODE_IMPLEMENTED_SYNTHETIC_TESTS_PASS
KILL = CODE_SURFACE_REJECTED
PREREQUISITE_NEEDED = ENVIRONMENT_OR_TEST_INFRA_BLOCKED
```

Pass requires:

- all U3 scoped tests pass;
- default `Generator()` remains legacy-compatible;
- sidecar-enabled zero-init output equals baseline;
- no domain/router surface exists;
- train/config validation prevents base mutation;
- no artifact/checkpoint/data payload changes exist;
- no training or dataset inference process ran.

Kill if:

- sidecar requires domain labels, hard routing, trunk unfreezing, or default
  path changes;
- zero-init equivalence cannot be proven with fixed tolerances;
- fallback cannot return same-call baseline output;
- implementation requires broad train-loop rewrite before synthetic proof;
- telemetry cannot be made domain-free.

Prerequisite-needed if:

- local test environment cannot import torch or project modules;
- existing unrelated worktree conflicts touch the same files and cannot be
  separated safely;
- current-primary SHA/config registry is unavailable for validation.

## U4 Preview

U4 is not authorized by U2. If U3 passes, U4 should be a separate
development-only validation Goal. U4 may authorize bounded training/evaluation
only after it freezes:

- exact train/dev manifests;
- sidecar config;
- control baselines;
- no-consumed-blind ledger;
- output directories;
- stop-on-first-fail order;
- metric thresholds from the U1 design.

U4 still cannot use fresh blind evidence or make promotion claims.

## Multi-Agent Execution Design

When U3 opens, use bounded parallel lanes:

- `executor`: owns `networks/generator.py` sidecar implementation.
- `executor`: owns `train.py` config validation and no-base-mutation checks.
- `test-engineer`: owns synthetic tests and audit helper fixtures.
- `verifier`: owns no-training/no-inference/artifact mutation verification.
- `critic`: reviews hidden router risk and default-path compatibility before
  any U3 pass claim.

Each lane must stay inside its file ownership and report conflicts upward.

Confidence: medium
Scope-risk: moderate
Reversibility: clean
Directive: U2 authorizes U3 code-only implementation next; it does not authorize
  training, dataset inference, checkpoint mutation, fresh blind use, or product
  promotion.
Tested: Read U1 design/admission, `networks/generator.py`, `train.py`, existing
  trainability helpers, and current gate/test surfaces.
Not-tested: No code implementation, training, dataset inference, target-image
  review, data download, checkpoint mutation, development validation, or
  promotion.
Related: docs/decisions/2026-08-06-universal-residual-adapter-sidecar-admission.md
