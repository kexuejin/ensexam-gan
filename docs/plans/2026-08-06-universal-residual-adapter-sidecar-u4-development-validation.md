# Universal Residual Adapter Sidecar U4 Development Validation Plan

```text
u4_scope = development_only
product_default = artifacts/current-primary
fresh_blind_handoff = disabled
promotion_handoff = disabled
consumed_hw5k_official_test = prohibited
```

U4 is the first bounded stage that may authorize development-only training and
evaluation for the universal residual adapter sidecar. It does not authorize
fresh blind evidence, release claims, default-path replacement, or promotion.

## Entry Evidence

- U1 architecture admission:
  `docs/decisions/2026-08-06-universal-residual-adapter-sidecar-admission.md`.
- U2 implementation/test plan:
  `docs/plans/2026-08-06-universal-residual-adapter-sidecar-implementation-plan.md`.
- U3 code decision:
  `docs/decisions/2026-08-06-universal-residual-adapter-sidecar-u3-code.md`.
- U4A static config:
  `configs/local/config.local-universal-sidecar-u4a-mixed-scut130-hw5k260-step20-mps.yaml`.

U3 passed synthetic/unit verification, but its code commit is deferred because
`networks/generator.py` and `train.py` had large pre-existing uncommitted
changes before U3 began. U4 may proceed only as development evidence; durable
claiming still requires a clean code split before results are promoted beyond
working evidence.

## Evidence Ledger

| Evidence | Status | Allowed U4 use | Prohibited use |
| --- | --- | --- | --- |
| `artifacts/current-primary/micro_region_probe_step0001.pth` | current default checkpoint | Frozen initialization and baseline comparator | Mutation or replacement |
| `hardcase_lists/mixed_scut130_hw5k260_20260729.txt` | train/development mix, 383 listed pages by current line count | U4A sidecar-only step20 smoke training | Promotion or final claim |
| `hardcase_lists/hw5k_dev_232_20260729.txt` | HW5K development, 232 listed pages | Development materiality screen after training | Fresh blind or release claim |
| `hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt` | SCUT inner validation, 15 listed pages | First source guard after training | Training or threshold tuning |
| `hardcase_lists/scut_val_holdout_40.txt` | SCUT holdout40, 40 listed pages | Later source guard only after inner guard passes | Training, tuning, or promotion |
| HW5K official consumed test | consumed blind | Historical context only | Training, tuning, selection, rescue, promotion |

U4 must not inspect target images manually. Any metric computation must use
registered scripts and persist aggregate CSV/JSON outputs.

## Stage U4A — Static Readiness

U4A is the current bounded stage. It authorizes only static config validation,
synthetic structure audit, and manifest/path checks.

```text
u4a_terminal_pass = STATIC_READY_FOR_BOUNDED_STEP20
u4a_terminal_kill = STATIC_SURFACE_REJECTED
u4a_terminal_prerequisite_needed = CLEAN_U3_SPLIT_OR_CONFIG_INFRA_NEEDED
```

Pass requires:

- U3 synthetic/unit tests still pass;
- U4A config parses and passes `validate_universal_sidecar_config(cfg)`;
- sidecar config remains:
  - `enabled=true`;
  - `adapter_count=3`;
  - `residual_bound<=12/255`;
  - no routing-like keys;
- trainable patterns match only `universal_residual_adapter_sidecar.*`;
- training controls are frozen:
  - `init_checkpoint=./artifacts/current-primary/micro_region_probe_step0001.pth`;
  - `resume=false`;
  - empty `resume_path`;
  - `freeze_generator_batchnorm_stats=true`;
  - `save_optimizer_state=false`;
  - `save_scheduler_state=false`;
  - `reproducibility_mode=strict`;
- no training, dataset inference, checkpoint mutation, target inspection, data
  download, or fresh blind use occurs during U4A.

Kill if static validation requires domain labels, caller hints, hard routing,
base-generator trainability, wider residual bounds, non-current-primary
initialization, or default-path mutation.

Prerequisite-needed if U3 code cannot be cleanly separated for durable
development evidence, or if the local config/test environment cannot import the
project modules.

## Stage U4B — Step20 Sidecar-Only Smoke

U4B is not started by this plan write. If U4A passes, U4B may authorize one
bounded train-only smoke:

```bash
source .env
$ENSEXAM_PYTHON train.py \
  --config configs/local/config.local-universal-sidecar-u4a-mixed-scut130-hw5k260-step20-mps.yaml
```

Allowed mutation in U4B is limited to the configured trial directory:

```text
artifacts/trials/universal-sidecar-u4a-mixed-scut130-hw5k260-step20-20260806
```

No validation, final test, dataset inference, fresh blind, target-image manual
inspection, or promotion is authorized in U4B. Its only purpose is to prove the
sidecar-only training path runs, emits model-only checkpoint payloads, and does
not unfreeze the base generator.

```text
u4b_terminal_pass = STEP20_SIDECAR_ONLY_SMOKE_COMPLETE
u4b_terminal_kill = TRAINING_SURFACE_REJECTED
u4b_terminal_prerequisite_needed = ENVIRONMENT_OR_CLEAN_SPLIT_NEEDED
```

## Stage U4C — Development Metric Screen

U4C is not started by this plan write. If U4B passes, U4C must be opened with
exact prediction/evaluation commands before any dataset inference is run.

Minimum requirements:

- compare whole-candidate outputs against `current-primary`;
- include HW5K development materiality and SCUT source guards;
- include mechanism controls before mechanism attribution is claimed:
  - `current-primary`;
  - equal-parameter single residual adapter;
  - uniform three-adapter mixture;
  - image-conditioned continuous three-adapter mixture;
- persist metric CSV/JSON summaries;
- do not use consumed HW5K official test, fresh blind evidence, promotion gates,
  or default-path mutation.

U4C cannot pass the mechanism claim until single-adapter and uniform-mixture
controls exist and are evaluated under matched training/resource settings.

## Stop Boundary

Even if all U4 development gates pass, the only allowed terminal is a
development-stage result. Fresh blind registration, end-to-end system
validation, and promotion must be separate future goals.
