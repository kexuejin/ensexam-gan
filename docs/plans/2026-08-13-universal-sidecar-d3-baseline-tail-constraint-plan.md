# Universal Sidecar D3: Cached Baseline-Tail Non-Regression Constraint

Date: 2026-08-13

## Goal

Open the single admissible causal change for the active bucket
`cross_domain_residual_headroom_vs_source_solved_pixel_regression`: add a
train-only current-primary baseline-tail non-regression constraint to the
sidecar training objective, holding everything else at the last safe
configuration (D2D). The candidate must earn HW5K development scoring by first
passing structure/gradient checks and the SCUT inner-val15
zero-page-regression gate.

## One-Change Hypothesis

The first measurable sidecar movement (U4B step20) regressed SCUT inner-val15
p95 residual and overerase because nothing in the objective anchored pages and
pixels that current-primary already solves. The verified cache
(`artifacts/caches/baseline-tail-universal-sidecar-d3-mixed-scut130-hw5k260-20260807`)
marks exactly those baseline-solved regions for every one of the 383 train
pages. Adding `lambda_cached_baseline_tail_nonregress` should let the sidecar
move on cross-domain residual headroom while suppressing the source-solved
pixel regressions that killed U4C.

Expected positive signal: zero page-level residual or overerase regression on
inner-val15 with at least one measurable page delta (distinguishing D3 from
D2D's safe-no-lift terminal).

Safety risk: the prior inner130 baseline-tail candidate
(`docs/decisions/2026-07-26-current-primary-trainproxy-inner130-baseline-tail-cache-step160-bs2-lambda05-inner-val15-decision.md`)
regressed overerase on every holdout page. That ran with a trainable base
model; D3 trains sidecar parameters only from a zero-init identity, so the
constraint binds movement away from the baseline instead of competing with a
free base. Overerase stability is still the first-class kill condition.

Falsified when: any inner-val15 page regresses on residual or overerase, or
the constraint cannot produce finite loss and sidecar-isolated gradients. A
second safe-no-lift terminal (zero measurable page delta) also ends the
family: it would prove the step20 surface cannot express a safe lift under
this constraint, and the next attempt must change representation or mechanism,
not steps, learning rate, or lambda.

## Frozen Training Surface

Everything below is fixed to the D2D-passing configuration; the cache
constraint is the only change:

```text
base config          = configs/local/config.local-universal-sidecar-u4a-mixed-scut130-hw5k260-step20-mps.yaml
learning rate        = 1.25e-05  (D2D halved LR; U4A 2.5e-05 is prohibited)
step budget          = 20 (max_steps_per_epoch), epochs 1
batch_size           = 4, gradient_accumulation_steps 1, num_workers 0
seed                 = 42, reproducibility_mode strict
init_checkpoint      = artifacts/current-primary/micro_region_probe_step0001.pth
trainable patterns   = ^universal_residual_adapter_sidecar\. only
BatchNorm            = frozen running stats
sidecar              = enabled, adapter_count 3, hidden 16, residual_bound 12/255
data                 = data-links/samples/SCUT-HW5K-mixed-20260729,
                       hardcase_lists/mixed_scut130_hw5k260_20260729.txt,
                       img 256, overlap 96, page_balanced sampling
inference            = matched-copy full-page protocol, page_overlap 32,
                       identical to the U4C/D2D evaluation commands
```

The one change:

```text
data.cached_baseline_tail_dir =
  artifacts/caches/baseline-tail-universal-sidecar-d3-mixed-scut130-hw5k260-20260807
loss.lambda_cached_baseline_tail_nonregress = 0.5
```

`lambda = 0.5` is preregistered to match the prior family's value so a failure
is directly comparable and terminal. No lambda sweep, step sweep, or
learning-rate sweep is authorized regardless of outcome.

## Stages And Kill Gates

### D3a: structure/gradient preflight (no dataset training)

- `validate_universal_sidecar_config` accepts the D3 config; every fail-closed
  rule from U3 still holds.
- `scripts/analysis/audit_universal_sidecar_structure.py` passes on the D3
  config.
- A bounded synthetic/dry check proves: gradients flow only to
  `universal_residual_adapter_sidecar.*` parameters; the cached baseline-tail
  loss term is finite and nonzero on at least one real cached batch; zero-init
  sidecar output remains exactly the baseline output.
- Kill: any structural violation terminates D3 with `PREREQUISITE_FAILED`
  before a single training step.

### D3b: step20 bounded training + smoke

- Train exactly the frozen surface for 20 steps into a fresh
  `artifacts/trials/universal-sidecar-d3-*` directory.
- Smoke: finite losses at every step, CPU-loadable checkpoint, sidecar
  parameters changed, base parameters byte-identical.
- Kill: nonfinite loss, unchanged sidecar, or any base-parameter drift.

### D3c: SCUT inner-val15 zero-page-regression gate

- Frozen matched-copy predictions into a fresh output directory, then CSV
  scoring against `outputs/scut_innerval15_current_primary_20260802`
  equivalents re-derived from committed anchors if that directory is absent.
- Gate (all required):
  - zero pages with residual delta > 0 vs current-primary;
  - zero pages with overerase delta > 0 vs current-primary;
  - aggregate mean/p95/max residual and overerase not worse than the U4C
    baseline table in the ledger.
- Kill: any page-level regression → `KILL`, evaluated no further.
- Pass with zero measurable delta on every page → `SAFE_NO_LIFT` terminal for
  the family (see falsification).
- Pass with measurable safe movement → D3 closes `PASS`; HW5K dev232 scoring
  requires a new bounded D4 plan.

## Authorization Boundary

This plan authorizes D3a, D3b, and D3c only: bounded step20 sidecar-only
training on the mixed train manifest and label-free inner-val15 matched-copy
evaluation with CSV scoring. It does not authorize HW5K development scoring,
SCUT115, holdout40, fresh or reserved blind evidence, target-image review,
threshold or postprocess rescue, routing, domain labels, base unfreezing,
lambda/step/LR sweeps, checkpoint promotion, or any mutation of
`artifacts/current-primary`. Every stage writes decision evidence into fresh
output directories and a terminal decision record into `docs/decisions/`, and
the ledger's active bucket must be updated in the same change.
