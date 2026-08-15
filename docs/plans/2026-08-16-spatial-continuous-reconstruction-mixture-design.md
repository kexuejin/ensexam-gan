# Spatial Continuous Reconstruction Mixture Design

## Status

```text
design_status = USER_APPROVED
program_scope = NEW_USER_AUTHORIZED_QUALITY_PROGRAM_OUTSIDE_EXHAUSTED_LEDGER
implementation_handoff = disabled
training_authorized = false
quality_gates_authorized = false
product_default = artifacts/current-primary
```

This document defines a materially new model-quality program after the current
`current-primary` successor ledger closed by durable exhaustion. It authorizes
neither implementation nor training. The next allowed artifact is a separate
implementation plan.

## Decision

Adopt a two-stage quality program:

1. a leakage-safe, page-grouped, train-only spatial-capacity probe that compares
   matched-budget reconstruction controls; then
2. only if that probe passes, a full spatial continuous reconstruction mixture
   anchored by the immutable `current-primary` output.

The program uses layered gates. Architecture research may use a small,
predeclared and bounded page-regression budget so that promotion-grade safety
does not force every early candidate into a no-op. Final promotion retains the
existing zero-page residual-regression and zero-page overerase-regression rules.

## Why A New Program Is Required

The current ledger terminal is durable exhaustion, not model improvement:

- no candidate achieved promotion-safe generalizing lift;
- source-only selector/postprocess families are exhausted;
- external text-layout support is predictive but direct edit application is
  unsafe;
- shared-weight SCUT/HW5K adaptation shows cross-domain interference;
- the residual sidecar family either regressed a source page or collapsed into
  sub-threshold/no-op behavior.

The existing residual sidecar is not a sufficient test of independent
reconstruction capacity. It trains only a small residual branch, uses one global
three-expert weight vector per image, and applies a single global residual scale.
The quality failures are spatial: handwriting residual, printed strokes, paper,
and collision regions coexist within a page. The next mechanism must therefore
make continuous local decisions and must have enough reconstruction capacity to
both remove residual and restore protected content.

## Product Contract

The product interface remains:

```text
clean(image) -> cleaned_image
```

Required constraints:

- no caller-provided domain label, source hint, file path, or checkpoint choice;
- no hard expert routing, `argmax`, top-k dispatch, or threshold-selected model;
- only inference-available image, baseline output, mask, and reconstruction
  features may condition the model;
- `artifacts/current-primary` remains immutable and independently runnable;
- any structural failure returns the same-call `current-primary` output;
- no result from this program changes the product default without the complete
  gate chain and a separate promotion decision.

Training-only dataset identity may be used to create page-balanced batches and
post-hoc audit reports. It must never enter the model input, gate input, loss as
a routing target, runtime telemetry decision, or inference branch.

## Causal Hypothesis

```text
Local residual-versus-print ambiguity and SCUT/HW5K shared-weight interference
can be reduced by anchoring one expert to current-primary, adding separate
trainable reconstruction capacity for erase and repair behavior, and mixing the
experts with a source-conditioned spatial soft gate.
```

The hypothesis is rejected if the spatial mixture fails to beat both an
equal-parameter single reconstruction head and a uniform mixture under matched
budgets, if the learned experts collapse to equivalent outputs, if the gate
acts only as a page/domain classifier, or if the mechanism cannot produce
meaningful residual lift without unacceptable preserve damage.

## Architecture

### Shared Frozen Path

```text
source image x
  -> frozen current-primary
       -> baseline output y0
       -> existing reconstruction features f
       -> existing masks and source-to-baseline features
```

All current-primary parameters and BatchNorm buffers remain immutable.

### Reconstruction Experts

The mixture contains three output experts:

1. **Anchor expert E0**: exact `current-primary` output `y0`; never trainable.
2. **Erase expert E1**: trainable full-resolution reconstruction decoder with
   capacity to remove target-lighter handwriting residual.
3. **Repair expert E2**: trainable full-resolution reconstruction decoder with
   capacity to preserve or restore unchanged print, collision edges, and paper.

E1 and E2 are not tiny per-pixel scalar heads. They are decoder-level RGB
reconstruction branches fed by frozen reconstruction features plus approved
inference-available source/baseline features. Their exact parameter count and
channel plan must be frozen in the implementation plan and must be identical
across matched-budget controls.

At initialization, E1 and E2 must reproduce `y0` exactly within fixed numerical
tolerances. A valid construction may use zero-initialized final correction
projections over copied baseline reconstruction features. The architecture must
not rely on a learned global residual scale whose sign can disable the entire
branch.

### Spatial Soft Gate

```text
G(x, y0, f, approved source/baseline features)
  -> logits shaped [B, 3, H, W]
  -> spatial softmax
  -> w0, w1, w2
```

For every pixel:

```text
w0 >= 0, w1 >= 0, w2 >= 0
w0 + w1 + w2 = 1
```

The final output is:

```text
y = clamp(w0 * y0 + w1 * y1 + w2 * y2)
```

The gate may use multi-scale features but must emit continuous spatial weights.
A single image-level weight vector is insufficient and is prohibited for the
primary experiment. Hard routing and page-specific threshold decisions are
prohibited.

### Effective Edit Range

The implementation preflight must analytically and synthetically prove that the
trainable experts can cross the fixed 12-gray evaluation event with a margin.
A maximum edit bound equal to the evaluation threshold is invalid because it can
make measurable lift unreachable by construction. The edit range must be
selected from train-only target-difference statistics and frozen before real
training; it must not be tuned after validation results.

## Phase 0: Train-Only Spatial-Capacity Probe

Phase 0 is the sole entry gate to full implementation training. It performs real
reconstruction learning but uses only registered train-role pages and
page-grouped held-out folds. It does not open `inner_val15`, Dev40, SCUT115,
holdout40, visual review, reserved blind, or promotion.

### Matched Controls

Train under identical data, seed family, optimizer, schedule, parameter budget,
step budget, and evaluation procedure:

1. `current-primary` frozen baseline;
2. one equal-parameter trainable reconstruction head;
3. two trainable reconstruction experts with fixed uniform mixture;
4. the spatial continuous mixture with the same total trainable parameter
   budget as controls 2 and 3.

The implementation plan must define how parameter equality is checked. Adding
capacity only to the spatial candidate is prohibited.

### Data Split

- folds are grouped by page, never by pixel or crop;
- only registered train-role SCUT and HW5K pairs are eligible;
- no validation, held-out quality, official consumed test, or reserved-blind
  page may enter Phase 0;
- source/target difference is computed locally and deterministically;
- class-filtered `Mb_gt` behavior stays behind an explicit experiment switch;
- dataset identity may balance batches but may not be a model feature;
- manifests, source hashes, target hashes, patch hashes, fold membership, and
  commands are sealed before training.

### Train-Only Supervision Regions

Target-difference analysis defines at least these regions:

- target-lighter residual/erase support;
- unchanged printed-content preserve support;
- target-darker or ambiguous support;
- handwriting/print collision boundaries;
- paper/background support;
- page-edge and small-component hard negatives.

Routine classification uses local pixel-difference statistics. Visual AI is not
used as the normal decision loop.

### Phase 0 Pass Contract

The spatial candidate passes only if all are true on page-held-out train folds:

- residual lift exceeds the predeclared repeatability/noise floor;
- it beats the equal-parameter single head;
- it beats the uniform mixture;
- overerase remains inside the predeclared research budget;
- worst-page regression remains inside the predeclared research budget;
- gate weights vary spatially and are not explained primarily by page identity
  or dataset identity;
- E1 and E2 show distinct, stable output behavior;
- the anchor contribution remains available on every page;
- no fold requires post-result threshold, loss, sampling, or checkpoint rescue.

Budgets and materiality floors must be derived from train-only repeatability and
train-fold distributions, then frozen before the first real Phase 0 run. Phase 0
failure closes the exact mixture family without opening quality splits.

## Training Objective

The first probe changes one primary causal axis: spatial independent
reconstruction capacity. It must not simultaneously introduce a broad objective
search.

The frozen objective should combine established paired reconstruction terms with
region-specific event-aligned terms:

- full-image paired reconstruction;
- target-lighter residual removal;
- unchanged-print preservation;
- collision/edge gradient preservation;
- paper/background fidelity;
- differentiable event surrogates aligned with the fixed 12-gray residual and
  overerase evaluation thresholds;
- bounded expert-diversity and gate-usage terms sufficient to detect collapse,
  not to force arbitrary specialization.

The implementation plan must preregister all weights. There is one bounded
attempt per control. Loss-weight, threshold, expert-count, edit-bound, and
schedule sweeps after observing a terminal result are prohibited.

External PP-OCR text-layout support is excluded from the first probe. It may
enter a later separately preregistered ablation only after detector-corpus
provenance and overlap are audited. Its existing support PASS does not authorize
direct edit application.

## Layered Evaluation Gates

### Gate A: Structural Preflight

Before decoding real training pairs:

- current-primary hashes match the registry;
- all base parameters are frozen;
- BatchNorm buffers do not mutate;
- zero-init mixture output matches current-primary;
- optimizer references only approved trainable parameters;
- spatial weights are finite, nonnegative, and sum to one;
- edit range can exceed the 12-gray event with margin;
- both expert branches receive live gradients on synthetic erase and repair
  fixtures;
- no metadata or domain label reaches the model.

### Gate B: Phase 0 Page-Grouped Probe

Uses train-role folds only and the matched-control pass contract above.

### Gate C: `inner_val15` Research Admission

Only a Phase 0 PASS may open this one-shot gate. The research gate requires:

- material aggregate residual improvement;
- strict aggregate and tail overerase limits;
- a small predeclared page-regression count and magnitude budget;
- no catastrophic page regression;
- predictions frozen before target scoring;
- no post-result parameter, threshold, sample-order, or checkpoint choice.

This gate is not a promotion decision. Its bounded regression allowance exists
only to test whether the new architecture generalizes without forcing a no-op.

### Gate D: Development And Source Generalization

If Gate C passes, evaluate in fixed order:

```text
Dev40 -> SCUT115 -> holdout40
```

The allowed regression budget narrows at each stage. Each surface is one-shot
after predictions and configuration hashes are frozen. Failure stops the
program; failed pages may be analyzed but cannot be used for rescue within the
same run family.

### Gate E: Promotion

Promotion remains unchanged and requires a later separate decision:

- reserved-blind authorization and PASS;
- visual review PASS;
- zero page-level residual regressions;
- zero page-level overerase regressions;
- aggregate and tail materiality;
- reproducible artifact and config custody;
- explicit replacement authorization.

## Telemetry And Audits

Record without using telemetry for runtime decisions:

- model/config/data/fold hashes;
- current-primary parameter and buffer immutability;
- expert output deltas and norms;
- spatial gate entropy, anchor share, maximum expert share, and spatial
  variation;
- expert-pair output disagreement;
- per-region residual and preserve events;
- per-page residual, overerase, tail, and maximum deltas;
- non-finite, range, simplex, saturation, and fallback counts;
- parameter counts and optimizer ownership for all controls.

Offline reports may group metrics by registered dataset for audit. Grouping must
not affect inference or checkpoint selection.

## Fail-Closed Conditions

Stop before expensive work or later gates when any of these occurs:

- preferred MPS environment is unavailable; no silent CPU fallback for
  meaningful training;
- manifests or content hashes are missing or inconsistent;
- current-primary parameters or BatchNorm state move;
- zero-init equivalence fails;
- effective edit range cannot cross the evaluation event;
- a trainable expert has dead gradients or non-finite output;
- the gate collapses globally to one expert or behaves as a page/domain router;
- E1 and E2 collapse to equivalent outputs;
- spatial mixture does not beat both matched controls;
- a result requires threshold, loss, edit-bound, fold, sample, seed, or
  checkpoint rescue after scoring;
- any prohibited validation or blind surface is accessed early.

Structural failure returns current-primary at runtime but terminates the research
candidate; fallback is not evidence of quality improvement.

## Verification Requirements

The implementation plan must include tests for:

- exact initialization equivalence on CPU and MPS;
- base-weight, optimizer, and BatchNorm immutability;
- spatial simplex and finite output invariants;
- synthetic erase and repair gradient liveness;
- edit-range reachability above the fixed evaluation event;
- no metadata/domain-label feature path;
- page-grouped split isolation and manifest custody;
- matched parameter budgets across controls;
- deterministic fold materialization;
- expert and gate collapse audits;
- stop-on-first-failure lifecycle;
- reporter output showing which surfaces remain closed.

Before MPS execution, print and verify `sys.executable`, Torch version,
`torch.backends.mps.is_built()`, `torch.backends.mps.is_available()`, and a tiny
MPS tensor allocation using the project-preferred environment. A failed MPS
preflight is an environment failure, not a model result.

## Explicit Non-Goals

This design does not authorize:

- restoring historical selector PNGs as the primary quality strategy;
- reopening source-edge, local-paper, chroma, achroma, target-dark, Delta-Trust,
  safe-metric, D3, D4, or D5 families;
- scalar rescue of the D5 global scale;
- direct PP-OCR support-to-edit projection;
- domain-routed deployment;
- hard expert routing;
- a broad hyperparameter sweep;
- validation-guided architecture debugging;
- training, inference, reserved-blind use, promotion, or current-primary
  replacement.

Restoring exact historical selector-replay PNG custody may proceed separately as
an evidence-repair lane, but it is not evidence for this architecture and must
not consume or alter this program's gates.

## Alternatives Considered

### Fidelity-First New Single Backbone

A NAFNet/Restormer-style restoration challenger with adversarial loss disabled
or reduced is materially different and remains a valid later program. It has a
higher data and training cost and lacks the exact current-primary anchor, so it
is not the first recommendation.

### Pareto-Aware Shared Generator

Gradient-conflict handling or group-robust optimization may improve a single
shared generator, but it remains exposed to the shared-weight interference
already observed between SCUT and HW5K. It is lower-cost but lower-confidence.

### Old-Surface Oracle Or Selector Probe

A target-aware selector over current-primary and old candidate surfaces tests
only the exhausted edit surface. It cannot determine whether new reconstruction
capacity has headroom and is therefore not the admission probe for this program.

## Successor Boundary

The only authorized successor to this design is a separate implementation plan
that freezes:

- code components and file boundaries;
- exact expert and gate parameter budgets;
- train-role manifests and page-grouped folds;
- control configurations;
- loss weights and edit range;
- Phase 0 materiality and research-regression budgets;
- MPS execution commands;
- test matrix, artifact paths, reporter updates, and stop conditions.

No implementation or training may start directly from this design document.
