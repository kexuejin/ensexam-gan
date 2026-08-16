# Spatial Continuous Reconstruction Mixture Implementation Plan

## Status

```text
implementation_plan_status = USER_APPROVED_NUMBERS_FROZEN
source_design = docs/plans/2026-08-16-spatial-continuous-reconstruction-mixture-design.md
implementation_handoff = enabled_for_code_and_synthetic_preflight_only
phase0_training_handoff = disabled_pending_implementation_and_gate_a_pass
quality_split_access = prohibited
product_default = artifacts/current-primary
```

This plan freezes the implementation surface, architecture, effective capacity
budget, train-role folds, loss weights, execution budget, and numerical Phase 0
gates for the spatial continuous reconstruction mixture program. It does not
itself authorize Phase 0 training. The next authorized action is code, manifest,
and synthetic Gate A implementation only. A separate implementation-complete
record must pass before the sealed Phase 0 command matrix may run.

## Immutable Entry Evidence

- Approved design: `docs/plans/2026-08-16-spatial-continuous-reconstruction-mixture-design.md`.
- Frozen product config: `artifacts/current-primary/config.yaml`, SHA-256
  `8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e`.
- Frozen product checkpoint:
  `artifacts/current-primary/micro_region_probe_step0001.pth`, SHA-256
  `e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae`.
- Existing full-resolution feature tap:
  `networks/generator.py::RefineNet.forward(..., return_reconstruction_feature=True)`
  returns the 16-channel tensor immediately before the current RGB output head.
- Existing train-role registry: `docs/sign-separated-residual-data-roles.json`.
- Existing 12-gray metric implementation:
  `scripts/eval/eval_hardcase_worst_pages.py::compute_residual_metrics`.

`artifacts/current-primary` remains independently runnable and immutable. Its
parameters, buffers, config, checkpoint, and default inference path must not be
modified.

## Frozen Architecture

### Inference-Available Feature Bundle

At full resolution, construct exactly:

```text
F = concat(
  source image Iin,                 #  3
  current-primary output y0,        #  3
  current-primary Ms,               #  1
  current-primary Mb,               #  1
  current-primary Ic1,              #  3
  RefineNet reconstruction feature  # 16
)                                   # total 27 channels
```

No target, label, target-derived mask, dataset identity, split identity, file
path, caller hint, domain label, or quality metric may enter `F` or any model
forward call.

### Shared Reconstruction Trunk

All learned controls share one active trunk:

```text
Conv2d(27, 64, kernel=3, padding=1, bias=true)
BatchNorm2d(64)
ReLU
ResBlock(64, 64, stride=1)
```

Frozen trainable count: **89,728**.

### Canonical Terminal Reconstruction Head

Used identically for E1 and E2 as independent modules:

```text
ResBlock(64, 64, stride=1)
Conv2d(64, 64, kernel=3, padding=1, bias=true)
BatchNorm2d(64)
ReLU
Conv2d(64, 3, kernel=1, bias=true)  # zero initialized
```

Trainable count per head: **111,235**. The final RGB projection is zero initialized.

The head produces raw correction `r_i`. The bounded expert output is:

```text
correction_bound = 25 / 127.5
c_i = correction_bound * tanh(r_i)
y_i = clamp(y0 + c_i, -1, 1)
```

The 25-gray bound is 2.0833 times the fixed 12-gray event threshold. No learned
global scalar, nonnegative clamp, page-level scale, or hard edit threshold is
allowed.

### Spatial Soft Gate

Use the same 27-channel `F` with this multi-scale tower:

```text
Conv2d(27, 16, kernel=3, padding=1, bias=true) + BN + ReLU
DownSample(16, 16)
DownSample(16, 32)
UpSample(32, 16)
add skip from first downsample feature
UpSample(16, 16)
Conv2d(16, 3, kernel=1, bias=true)
softmax(dim=1)
```

Trainable count: **28,723**. Output is `[B,3,H,W]`; weights must be finite,
nonnegative, and sum to one per pixel. Final candidate:

```text
y = clamp(w0 * y0 + w1 * y1 + w2 * y2, -1, 1)
```

Hard routing, top-k, `argmax`, threshold-selected experts, and whole-page gate
weights are prohibited.

### Initialization Contract

- The final RGB projections of E1 and E2 are all-zero initialized.
- Gate logits initialize to zero, so `w0=w1=w2=1/3` per pixel.
- Because `y1=y2=y0`, the initialized mixed output must equal `y0`.
- CPU tolerance: max absolute delta `<=1e-7` in float32 when clamp is inactive.
- MPS tolerance: max absolute delta `<=1e-6` in float32.
- Any initialized pixel outside tolerance is a Gate A failure.

## Frozen Effective Capacity Budget

The primary fairness contract is **equal active reconstruction capacity**, not
equal total parameters including the causal gate under test:

```text
B_recon = shared trunk + two active terminal heads
        = 89,728 + 2 * 111,235
        = 312,198 trainable parameters
```

Every learned control owns exactly `B_recon=312,198` active reconstruction
parameters. No dummy parameter, disconnected padding tensor, zero-multiplied
capacity, or optimizer-owned parameter outside the forward graph is permitted.

This explicitly refines the original design's ambiguous "same total parameter
budget" wording: single-head, uniform two-expert, and spatial mixture all share
equal reconstruction capacity; the total differs only by the causal spatial-gate
mechanism under test.

Control definitions:

1. `baseline`: current-primary only; 0 trainable parameters.
2. `single_head`: shared trunk plus two canonical correction heads whose
   corrections are averaged into one output added to `y0`. One learned RGB
   reconstruction output with the same active reconstruction budget; no
   expert-specific losses or telemetry.
3. `uniform_two_expert`: shared trunk plus E1/E2; fixed weights
   `(w0,w1,w2)=(0,0.5,0.5)`.
4. `spatial_mixture`: shared trunk plus E1/E2 plus the 28,723-parameter spatial
   gate. Active reconstruction budget is 312,198; total is **340,921**. The only
   excess over the uniform control is the causal routing mechanism being tested.

Gate A must report both `active_reconstruction_params` and
`total_trainable_params`. Controls 2/3/4 must have exactly 312,198 active
reconstruction parameters. The spatial control must differ from the uniform
control by exactly 28,723 gate parameters and no other tensors. This replaces any
invalid dummy-padding interpretation of matched budget.

## Code And File Boundaries

### New files

- `networks/spatial_reconstruction_mixture.py`
  - `SharedReconstructionTrunk`
  - `TerminalReconstructionHead`
  - `SpatialSoftGate`
  - `SpatialContinuousReconstructionMixture`
  - control-mode assembly and numeric telemetry
- `losses/spatial_mixture_losses.py`
  - deterministic supervision-region construction
  - frozen Phase 0 loss terms
  - collapse regularizers
- `scripts/analysis/materialize_spatial_mixture_phase0_folds.py`
  - exact train-role identity normalization, fold assignment, hashes, custody
- `scripts/analysis/validate_spatial_mixture_preflight.py`
  - current-primary hashes, frozen base, budget, feature-path, split, MPS checks
- `scripts/train/train_spatial_mixture_probe.py`
  - bounded control/fold/seed runner; no validation or checkpoint selection
- `scripts/eval/evaluate_spatial_mixture_phase0.py`
  - frozen full-page predictions, common scoring, control comparisons, PASS/KILL
- `configs/local/spatial-mixture-phase0/`
  - `baseline.yaml`
  - `single-head.yaml`
  - `uniform-two-expert.yaml`
  - `spatial-mixture.yaml`

Tests:

- `tests/test_spatial_reconstruction_mixture.py`
- `tests/test_spatial_mixture_parameter_budget.py`
- `tests/test_spatial_mixture_losses.py`
- `tests/test_materialize_spatial_mixture_phase0_folds.py`
- `tests/test_validate_spatial_mixture_preflight.py`
- `tests/test_train_spatial_mixture_probe.py`
- `tests/test_evaluate_spatial_mixture_phase0.py`

### Existing files that may change

- `networks/generator.py`: only expose the frozen feature bundle required for the
  disabled-by-default mixture host; default outputs and state-dict surface remain
  unchanged when disabled.
- `train.py`: add mixture-only checkpoint missing-key authorization and fail-closed
  config validation; no broad training-loop rewrite.
- Reporter state: record only that code/preflight or Phase 0 remains closed with a
  PASS/KILL result; never change product default.

No existing D-series sidecar class or config may be reused as the candidate
implementation.

## Frozen Phase 0 Data Pool And Folds

### Eligible Pool

Use only the effective 383 identities in:

```text
hardcase_lists/mixed_scut130_hw5k260_20260729.txt
```

Effective domain counts are SCUT 130 and HW5K 253, consistent with
`docs/sign-separated-residual-data-roles.json`. Source and target are read from
`data-links/samples/SCUT-HW5K-mixed-20260729/train`.

The following are prohibited for Phase 0 input, loss, sampling, selection,
thresholding, or debugging:

- `hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt`;
- Dev40;
- `docs/scut-test115-relative.txt` / SCUT115;
- `hardcase_lists/scut_val_holdout_40.txt` / holdout40;
- HW5K dev232;
- HW5K official-test reserved blind 525;
- visual review and promotion artifacts.

### Canonical Identity And Deterministic Assignment

Canonical identity is `<domain>/train/<basename>` with lower-case domain and
normalized POSIX separators. Materialization must:

1. validate every source/target pair and record SHA-256 of both files;
2. validate no identity or content hash overlaps any prohibited manifest;
3. split identities by domain;
4. compute `SHA256("spatial-mixture-phase0-v1\0" + canonical_identity)`;
5. sort each domain by the digest then canonical identity;
6. assign round-robin to folds `0..5` using the frozen per-domain fold counts;
7. emit immutable fold manifests plus a master JSON with the command, source
   manifest hash, source/target hashes, counts, and output hashes.

Frozen fold counts:

| fold | SCUT | HW5K | total |
| ---: | ---: | ---: | ---: |
| 0 | 22 | 42 | 64 |
| 1 | 22 | 42 | 64 |
| 2 | 22 | 42 | 64 |
| 3 | 22 | 42 | 64 |
| 4 | 21 | 43 | 64 |
| 5 | 21 | 42 | 63 |

Each Phase 0 run trains on five folds and evaluates on the held-out sixth fold.
Crops and tiles inherit their page fold and may never cross folds.

## Frozen Supervision Regions

All training-only regions are computed locally and deterministically from the
source/target pair. Routine visual-AI classification is prohibited.

Use 8-bit RGB/luminance quantities before tensor normalization:

- `changed`: mean absolute RGB source-target delta `>=12`, followed by the same
  3x3 opening and one-pixel 3x3 dilation used by
  `build_changed_mask(..., threshold=12)`.
- `target_lighter`: changed pixels with target mean luminance minus source mean
  luminance `>2` gray.
- `target_darker_or_ambiguous`: changed minus target-lighter.
- `unchanged_print_preserve`: outside changed mask, source local Sobel magnitude
  at or above the train-pool 75th percentile, frozen during materialization.
- `collision_boundary`: a 2-pixel morphological band around target-lighter that
  intersects `unchanged_print_preserve`.
- `paper`: outside changed mask and below the train-pool 25th percentile of source
  Sobel magnitude.
- `page_edge`: 16-pixel page border outside changed.
- `small_component_hard_negative`: outside changed connected components in the
  source dark map, area 4..64 pixels, using the frozen source-dark threshold
  derived from the train pool.

The 75th/25th Sobel thresholds and source-dark threshold are computed once from
the eligible train pool before any run, written into the master manifest, and
never changed after predictions exist. Class-filtered `Mb_gt` remains disabled
(`box_class_mode=all`) in Phase 0.

## Frozen Objective

There is one objective and one attempt per learned control. GAN, discriminator,
VGG perceptual loss, style loss, current mask losses, and broad loss sweeps are
disabled for Phase 0. Every term is computed on the final candidate unless
explicitly stated.

```text
L_total =
  1.00 * L_pair
+ 2.15 * L_residual12
+ 4.75 * L_overerase12
+ 2.00 * L_print_preserve
+ 1.00 * L_collision_grad
+ 0.50 * L_paper
+ 0.05 * L_expert_diversity
+ 0.02 * L_gate_usage
+ 0.05 * L_gate_TV
```

Definitions:

- `L_pair`: region-balanced Charbonnier loss to target over target-lighter,
  target-darker/ambiguous, unchanged-print, paper, and remaining pixels;
  epsilon `1e-3`, equal mean weight per non-empty region.
- `L_residual12`: sigmoid event surrogate on target-lighter support using
  candidate-target mean RGB absolute delta, threshold 12 gray, temperature
  0.25 gray; top 25% per-page tail mean then batch mean.
- `L_overerase12`: sigmoid event surrogate on outside-changed support using
  candidate-source mean RGB absolute delta, threshold 12 gray, temperature
  0.25 gray; top 25% per-page tail mean then batch mean.
- `L_print_preserve`: normalized Charbonnier candidate-source loss on
  unchanged-print preserve support.
- `L_collision_grad`: L1 difference between Sobel gradients of candidate and
  target on collision boundary.
- `L_paper`: normalized candidate-target Charbonnier loss on paper support.
- `L_expert_diversity`: uniform/spatial only, a bounded anti-collapse hinge on
  target-lighter support, `relu(1/255 - mean_abs(y1-y2))`; zero for baseline and
  single-head.
- `L_gate_usage`: spatial only,
  `relu(0.10 - mean(w0)) + sum_i relu(mean(w_i)-0.80)`; zero otherwise.
- `L_gate_TV`: spatial only, anisotropic total variation of the three weights;
  zero otherwise.

The diversity and gate terms are collapse guards, not specialization labels.
Dataset identity never enters the loss. The existing `lambda_eval_*` config keys
were never wired into `train.py`/`losses.py`, so the new loss module must consume
and assert these frozen weights explicitly rather than rely on any legacy wiring.

## Frozen Optimizer And Execution Budget

```text
optimizer = AdamW
lr = 5e-5
betas = (0.3037, 0.9)
weight_decay = 0
batch_size = 4 physical
num_workers = 0
image/tile size = 256
augmentation = disabled
precision = float32
scheduler = disabled
early stopping = disabled
checkpoint selection = disabled
max_steps = 640
seeds = [42, 31415, 27182]
```

Every learned control/fold/seed run executes exactly 640 optimizer steps and
saves only the final checkpoint plus the step trace. Baseline replay has no
optimizer. Run order is control-major then fold then seed, with the entire matrix
sealed before the first run. A failed unit may not be retried with a changed
seed, batch order, threshold, loss, step count, or checkpoint.

The Phase 0 learned matrix is **3 learned controls x 6 folds x 3 seeds = 54
training runs**, plus frozen current-primary repeatability replays. The four
control config identities still include the zero-trainable baseline for reporting.

## Repeatability Calibration

Before learned runs, replay current-primary predictions on every held-out fold
three times using seeds 42, 31415, and 27182 and the exact full-page inference
protocol. Compute the pooled standard deviation of per-page residual ratio across
replays:

```text
noise_sd = pooled within-page residual-ratio SD
materiality_floor = max(0.0005, 2 * noise_sd)
```

The calibration may only raise the 0.0005 floor. It may not change any loss,
architecture, fold, threshold, or regression budget. Non-deterministic baseline
output above the CPU/MPS custody tolerance is an environment/preflight failure,
not a model result.

## Frozen Phase 0 PASS/KILL Gate

The spatial candidate passes only if all checks below pass on frozen held-out
predictions. Metrics use 12-gray changed/evaluation thresholds and the same
morphology as `compute_residual_metrics`.

### Material Lift

- Pooled held-out residual gain versus current-primary is at least
  `materiality_floor`.
- For every fold and every seed, the within-fold page-median residual gain is
  strictly positive.
- No aggregation weights a page by pixel count; pooled means are page means unless
  a metric explicitly says otherwise.

### Matched-Control Superiority

- Spatial pooled residual advantage over `single_head` is at least 0.0005.
- Spatial pooled residual advantage over `uniform_two_expert` is at least 0.0005.
- For each comparison, spatial wins at least 14 of the 18 paired fold x seed units
  by mean page residual ratio.
- Spatial must not have worse pooled overerase than either control by more than
  0.0005.

### Balanced Research Regression Budget

Per frozen user decision:

- on each fold, no more than 2 pages may have residual regression versus
  current-primary;
- every residual-regressing page has absolute residual-ratio increase `<=0.025`;
- pooled aggregate overerase increase versus current-primary `<=0.002`;
- every page overerase-ratio increase `<=0.010`;
- any non-finite metric or missing prediction is an immediate KILL.

These research allowances apply only to Phase 0. They do not weaken later gates or
promotion.

### Gate Non-Collapse

For every page and seed:

- mean anchor share `mean(w0) >= 0.10`;
- per-expert pixel-share 99th percentile `<=0.98`;
- mean spatial standard deviation across `w0,w1,w2` `>=0.02`;
- a post-hoc logistic probe trained only on page-level gate summaries must have
  5-fold balanced accuracy `<=0.65` for SCUT-vs-HW5K identity. This probe is an
  audit only and never affects the model.

### Expert Non-Collapse

On target-lighter changed support, aggregated over each fold/seed:

- E1/E2 correction cosine similarity `<=0.95`;
- mean absolute E1/E2 output disagreement `>=1/255`;
- each expert receives a finite, nonzero gradient on its corresponding synthetic
  erase/repair Gate A fixture.

### Stop Conditions

Any failure below terminates the exact Phase 0 family and keeps `inner_val15` and
all later surfaces closed:

- manifest/hash/isolation failure;
- base parameter or BatchNorm buffer movement;
- initialization equivalence failure;
- dead expert/gate gradients;
- parameter-budget mismatch;
- edit-range reachability failure;
- non-finite output, simplex violation, or fallback during scoring;
- failure to beat either matched control;
- failure of any numerical PASS condition;
- any need for threshold, fold, loss, sample, seed, step, checkpoint, or
  page-specific rescue after scoring.

## Gate A Structural Preflight

Gate A must finish before any real Phase 0 training:

1. Verify current-primary config/checkpoint hashes.
2. Verify the disabled default path preserves legacy state-dict keys and outputs.
3. Verify all base parameters have `requires_grad=false` and all current-primary
   BatchNorm modules remain eval with immutable buffers.
4. Verify optimizer ownership exactly matches the approved active modules.
5. Verify the reconstruction/gate parameter counts above.
6. Verify zero-init equivalence on CPU and MPS.
7. Verify gate finite/simplex invariants and output range.
8. Verify a forced final-head raw value can generate at least 24 gray of observed
   pre-clamp correction, proving margin above the 12-gray event.
9. Verify live gradients on synthetic erase and repair fixtures.
10. Verify public signatures and config keys contain no domain/source/caller/path
    routing input.
11. Verify fold materialization is deterministic and prohibited surfaces are absent
    by identity and content hash.
12. Verify no output directories from the sealed matrix already exist.

Runtime structural fallback returns same-call `y0`, records a numeric reason code,
and kills the research candidate. It is not a PASS.

## MPS Preflight And Commands

Use exactly:

```text
/Volumes/Tool/source/clean-doc/.venv-torch310-mps-stable/bin/python
```

Before meaningful training, print and verify:

```python
import sys, torch
print(sys.executable)
print(torch.__version__)
print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())
print(torch.ones(1, device="mps"))
```

If MPS is unavailable or the tensor allocation fails, stop with
`PREREQUISITE_NEEDED`. Do not silently use CPU. CPU is allowed only for unit
tests, manifest checks, and a one-step control-flow smoke whose output directory
contains `cpu-smoke-not-a-model-result`.

Representative implementation verification commands:

```bash
PY=/Volumes/Tool/source/clean-doc/.venv-torch310-mps-stable/bin/python

$PY -m pytest \
  tests/test_spatial_reconstruction_mixture.py \
  tests/test_spatial_mixture_parameter_budget.py \
  tests/test_spatial_mixture_losses.py \
  tests/test_materialize_spatial_mixture_phase0_folds.py \
  tests/test_validate_spatial_mixture_preflight.py \
  tests/test_train_spatial_mixture_probe.py \
  tests/test_evaluate_spatial_mixture_phase0.py

$PY scripts/analysis/materialize_spatial_mixture_phase0_folds.py \
  --source-manifest hardcase_lists/mixed_scut130_hw5k260_20260729.txt \
  --data-root data-links/samples/SCUT-HW5K-mixed-20260729/train \
  --salt spatial-mixture-phase0-v1 \
  --fold-count 6 \
  --output-dir hardcase_lists/spatial-mixture-phase0-v1

$PY scripts/analysis/validate_spatial_mixture_preflight.py \
  --plan docs/plans/2026-08-16-spatial-continuous-reconstruction-mixture-implementation.md \
  --fold-root hardcase_lists/spatial-mixture-phase0-v1 \
  --output-dir outputs/spatial-mixture-phase0-preflight-v1
```

The final Phase 0 runner must read a sealed matrix JSON rather than construct
commands ad hoc. The matrix records config/fold/seed/output paths and SHA-256 for
every input.

## Implementation Sequence

1. Add failing unit tests for the disabled default, exact feature bundle, zero-init
   equivalence, simplex, edit range, gradients, and parameter counts.
2. Implement `networks/spatial_reconstruction_mixture.py` and the minimal
   disabled-by-default `Generator` integration.
3. Implement config/checkpoint validation and base/BN immutability checks.
4. Add deterministic region builders and the frozen loss objective with tests.
5. Implement fold materialization and prohibited-surface/content-hash isolation.
6. Implement the bounded training runner and CPU one-step smoke only.
7. Implement common prediction freezing, Phase 0 evaluator, collapse audits, and
   stop-on-first-failure reporter.
8. Run the focused unit/integration test matrix and `git diff --check`.
9. Run MPS preflight and Gate A only.
10. Write an implementation-complete Gate A decision. Only a PASS may enable the
    already frozen Phase 0 matrix; it must not change any number in this plan.
11. After separate authorization, run repeatability calibration, then the sealed
    Phase 0 matrix, freeze predictions before scoring, and emit a PASS/KILL
    decision.
12. Only a Phase 0 PASS may create a separate `inner_val15` research-admission
    plan. No quality split is opened by this document.

## Required Artifacts

Implementation/Gate A:

- code and tests listed above;
- `hardcase_lists/spatial-mixture-phase0-v1/master.json` and six fold manifests;
- `docs/spatial-mixture-phase0-matrix-v1.json`;
- `outputs/spatial-mixture-phase0-preflight-v1/audit.json`;
- implementation-complete decision under `docs/decisions/`.

Phase 0, only after authorization:

- final checkpoints and step traces for all 54 learned runs;
- immutable prediction directories for every held-out unit;
- per-page metrics CSVs;
- repeatability report;
- parameter/optimizer/base-buffer audit;
- gate and expert collapse reports;
- one terminal Phase 0 PASS/KILL decision;
- reporter state proving all quality and blind surfaces remained closed.

## Explicit Non-Goals

This implementation may not:

- train or mutate current-primary;
- open `inner_val15`, Dev40, SCUT115, holdout40, HW5K dev/test, reserved blind,
  visual review, or promotion;
- use dataset identity as a feature or loss target;
- use OCR/text-layout support;
- reuse the D3/D4/D5 global-gate/global-scale family;
- conduct an architecture, width, loss, edit-bound, seed, step, or threshold sweep;
- select intermediate checkpoints;
- treat CPU smoke output as a model result;
- replace `artifacts/current-primary`.

## Terminal Semantics

```text
IMPLEMENTATION_PASS = code/tests/manifests/Gate_A_pass; Phase_0 may be separately authorized
IMPLEMENTATION_KILL = architecture or safety contract cannot be implemented as frozen
PREREQUISITE_NEEDED = MPS/environment/custody infrastructure prevents Gate_A
PHASE0_PASS = all frozen numerical and collapse gates pass
PHASE0_KILL = any frozen Phase_0 gate fails; exact family closes without rescue
```

Confidence: medium-high
Scope-risk: moderate
Reversibility: clean
Directive: implement code, tests, manifests, and Gate A exactly as frozen; do not
train Phase 0 or access any quality split until a separate Gate A PASS record
authorizes the sealed matrix.
