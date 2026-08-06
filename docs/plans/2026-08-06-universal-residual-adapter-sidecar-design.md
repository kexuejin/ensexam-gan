# Universal Residual Adapter Sidecar Design

## Status

```text
design_status = USER_APPROVED_SECTIONS_1_TO_4
mechanism_status = DESIGN_APPROVED_NOT_ADMITTED
implementation_handoff = disabled
product_default = artifacts/current-primary
product_owner_universal_requirement = satisfied_by_user_approval
```

This document records the approved design boundary for a materially-new
universal handwriting-removal mechanism. It does not admit the mechanism,
authorize implementation, train a checkpoint, run inference, open a fresh blind
set, or change `artifacts/current-primary`.

The current M1 fresh-source lane remains externally blocked:

- no new HW5K-like blind source/custody root is admitted;
- no consumed blind set may be reused for selection or promotion;
- no router, broad retrain, scalar threshold rescue, or second-stage repair is
  opened by this design.

## Approved Product Contract

The universal path must satisfy these product constraints:

- one external interface: `clean(image) -> cleaned_image`;
- no external domain label, caller hint, path-derived source, or checkpoint
  selector;
- no hard routing, `argmax`, top-k dispatch, threshold branch, or sparse expert
  selection;
- internal continuous soft expert mixing is allowed only when conditioned on
  source-image features;
- failure fallback returns the same-call `current-primary` baseline output.

The first release claim, if the later product path ever reaches promotion, is
bounded to SCUT and HW5K development domains plus one fresh unseen source. It
does not claim arbitrary unknown-domain generalization. Development evidence
alone never supports a release claim; fresh unseen validation and final
promotion remain separate milestones.

The product-owner prerequisite from
`docs/decisions/2026-08-05-materially-new-universal-preflight.md` is now treated
as satisfied for design/admission purposes by the user's explicit approval that
the product must support universal capability with no domain label, one
external interface, continuous internal soft mixing allowed, and hard routing
plus caller-provided domain labels forbidden.

## Causal Hypothesis

The falsifiable hypothesis is:

```text
shared_weight_cross_domain_interference can be reduced by freezing the proven
current-primary reconstruction path and adding small residual capacity whose
continuous mixture weights are conditioned only on current image reconstruction
features.
```

This hypothesis is killed if:

- gains are explained by an equal-parameter single adapter;
- gains are explained by a uniform three-adapter mixture;
- the continuous gate becomes a near-hard source selector;
- source/default quality regresses;
- success requires external labels, path metadata, trunk unfreezing, threshold
  rescue, broad retrain, consumed blind reuse, or another repair stage.

## Approaches Considered

| Approach | Summary | Fit | Rejection / Risk |
| --- | --- | --- | --- |
| Frozen primary plus continuous residual adapter bank | Keep `current-primary` as the base path and add zero-output residual adapters mixed by an image-conditioned continuous gate. | Recommended | Capacity may be too weak to clear the HW5K development materiality gate. |
| Pareto-aware shared-generator continuation | Continue one shared generator with stronger multi-objective constraints. | Not first choice | Too close to rejected shared-weight scalar/objective continuation unless separately proven materially different. |
| Full soft expert generator mixture | Mix multiple large generator branches continuously. | Not first choice | Higher attribution risk and easier to drift into implicit routing or default-quality regression. |

The approved candidate is the frozen-primary residual adapter sidecar because it
isolates new capacity from the product default path and creates a direct
equivalence/fallback proof obligation.

## Architecture

```text
input image
  -> frozen current-primary Generator
       -> baseline clean output y0
       -> RefineNet reconstruction_feature f

f -> source-only conditioner C(f)
       -> continuous simplex weights g1, g2, g3

f -> residual adapters A1, A2, A3
       -> residuals r1, r2, r3

mixed residual r = alpha * sum(gi * ri)
candidate output y = clamp(y0 + bounded(r))

if structural or safety invariant fails:
    return y0
```

`RefineNet.forward(..., return_reconstruction_feature=True)` is the preferred
feature tap because it already exists in `networks/generator.py` and sits inside
the current reconstruction path. The sidecar may read this feature tensor, but
it must not read dataset names, file paths, caller manifests, labels, or
external source tags.

The base `current-primary` path stays independently runnable and remains the
fallback target. Adapter final projections and the global residual scale
initialize to zero, so the initial output must match `current-primary`.

## Data Flow And Interface

The runtime interface remains a single universal call:

```text
clean(image) -> cleaned_image
```

There is no product API for domain selection. Any evaluator-side source grouping
is offline reporting only and must not enter the model input, inference
contract, gate loss, telemetry trigger, or fallback policy.

The conditioner emits continuous simplex weights. Allowed constraints include
temperature floor, minimum entropy, maximum single-expert share, and residual
norm ceilings when they are preregistered as pass/kill evidence. Disallowed
behavior includes discrete selection, routing thresholds, per-domain losses,
domain-balanced sampling keyed by source labels, or telemetry-driven runtime
branching.

## Development-Only Gate Contract

The first executable successor, if later admitted, may only decide whether this
mechanism is `MECHANISM_ADMISSIBLE`. It cannot promote a model or replace the
default product path.

Gate order is fixed:

1. **Structure gate.** The external interface is single-call universal; no
   domain/caller/path label exists; no hard routing exists; zero-init output is
   equivalent to `current-primary`. Required proof:
   - base checkpoint SHA and config SHA match the model registry;
   - base parameters have `requires_grad = false`;
   - base BatchNorm modules run in eval mode and their buffers do not mutate;
   - the optimizer has no references to base parameters;
   - original `current-primary`, sidecar zero-init, and fallback `y0` outputs
     are `allclose(atol=1e-6, rtol=0)` on a fixed CPU smoke fixture and
     `allclose(atol=1e-5, rtol=0)` on the configured accelerator fixture.
2. **Mechanism attribution gate.** Compare four controls:
   `current-primary`, equal-parameter single residual adapter, three-adapter
   uniform mixture, and three-adapter image-conditioned continuous mixture.
   The image-conditioned mixture must beat both adapter controls, not just the
   baseline. Parameter budget, train split, training steps, optimizer, learning
   rate schedule, alpha schedule, early-stop policy, and evaluation commands
   must match across the three adapter controls.
3. **HW5K development gate.** On development-only HW5K evidence, mean residual
   must materially improve while overerase and tail metrics do not regress.
   Frozen thresholds:
   - mean residual ratio must be at least 20.0% lower than `current-primary`;
   - mean, p95, and max overerase ratios must be `<= current-primary + 1e-6`;
   - p95 and max residual ratios must be `<= current-primary + 1e-6`;
   - no post-freeze threshold, selector, or sample-order change is allowed.
4. **SCUT/source guard.** SCUT inner validation, Dev40, SCUT115, and holdout40
   residual, overerase, p95, max, and page-regression checks must tie or improve
   versus `current-primary`. Frozen thresholds:
   - mean, p95, and max residual ratios must be `<= current-primary + 1e-6`;
   - mean, p95, and max overerase ratios must be `<= current-primary + 1e-6`;
   - page-level residual regressions count must be zero;
   - page-level overerase regressions count must be zero;
   - Dev40 must pass before SCUT115 or holdout40 is evaluated.
5. **Non-degenerate mixing gate.** Gate entropy, max expert share, residual
   norm, fallback rate, and saturation must stay inside these frozen ranges:
   - normalized gate entropy mean `>= 0.60`;
   - per-source normalized gate entropy mean `>= 0.50`;
   - max expert weight p95 `<= 0.80`;
   - per-source single-expert assignment share `<= 70.0%`;
   - gate-only source classifier balanced accuracy `<= 70.0%` on frozen
     development features;
   - mixed residual L-infinity edit bound `<= 12/255` before output clamp;
   - fallback rate `<= 0.5%`, and every fallback reason must be structural.
6. **Stop boundary.** Passing development gates can only produce
   `MECHANISM_ADMISSIBLE`; fresh blind, artifact/custody, system validation,
   and promotion require later bounded Goals.

Kill immediately if initialization is not equivalent, any domain label enters
the model, the gate collapses into near-hard routing, the gain is explained by
single-adapter or uniform-mixture controls, SCUT/default quality regresses, or
the path needs broad retrain, threshold rescue, consumed blind reuse, extra
experts, hard routing, trunk unfreezing, or second-stage cleanup rescue.

## Evidence Split Ledger

| Evidence input | Status | Allowed use | Prohibited use |
| --- | --- | --- | --- |
| SCUT inner train / train-proxy manifests | development / train-safe only when already registered | Adapter training and train-only support caches | Promotion, final blind claim, or iterative guard rescue |
| SCUT inner validation | development guard | First source guard after freeze | Training, repeated threshold tuning, or promotion |
| SCUT Dev40 | development guard | One-shot post-freeze guard after inner validation passes | Training, threshold tuning, or promotion |
| SCUT115 | source guard | One-shot guard only after Dev40 passes | Training, tuning, or early exploratory selection |
| holdout40 | source guard | One-shot guard only after Dev40 passes | Training, tuning, or promotion |
| HW5K train/dev development manifests | development only | Adapter training/dev materiality gate if already isolated from consumed official test | Fresh-blind claim or release claim |
| HW5K official consumed test | consumed blind | Historical context only | Training, tuning, selection, promotion, claim expansion, or rescue |
| Future fresh unseen source | unavailable prerequisite | Later registration/isolation and blind evaluation only after separate Goal | Any use inside U1 |

Every U1 admission decision must include a split manifest ledger with SHA-256
hashes, allowed use, consumed/development/fresh status, and freeze order. Guard
splits are one-shot post-freeze checks, not iterative tuning surfaces.

## Anti-Router Audit

The sidecar is killed if any of these are observed:

- gate weights are near-deterministic per source family;
- a gate-only classifier exceeds the frozen balanced-accuracy limit;
- source grouping, file path, manifest metadata, or caller information enters
  training, loss, inference, telemetry-trigger logic, or fallback;
- telemetry is used to select a checkpoint, expert, threshold, or fallback path;
- any branch emits discrete source labels or hard expert decisions.

Offline source grouping is allowed only for audit reports that run after frozen
predictions and telemetry are already sealed.

## Telemetry

Runtime telemetry records structured numbers only:

- model and config hash;
- gate weights, normalized entropy, and max expert weight;
- per-adapter residual norm and mixed residual norm;
- final edit delta and output range status;
- fallback flag and fixed reason code;
- non-finite, simplex, residual-bound, shape, and range violation counts.

Telemetry must not save images, intermediate visual content, labels, source
tags, caller hints, paths, or domain guesses. Offline reports may group metrics
by registered evidence source, but grouping is evaluator-side only.

## Artifact And Custody

Every candidate must freeze and report:

- base `current-primary` checkpoint SHA;
- adapter checkpoint SHA;
- config SHA;
- code commit;
- train and development split manifest SHA;
- evaluation command and output SHA;
- consumed/development/fresh-unseen status for every evidence input.

The consumed HW5K official test cannot be used for model selection, gate tuning,
promotion, claim expansion, or rescue. Fresh unseen blind evidence remains a
separate prerequisite with formal custody, registration, isolation validation,
and final decision records.

## Failure And Fallback Policy

Allowed fallback reasons are structural failures only:

- NaN or Inf;
- invalid simplex weights;
- residual bound violation;
- shape mismatch;
- output range violation.

Fallback returns `y0`, the same-call `current-primary` baseline output. It must
not switch checkpoints or trigger because an image looks like a source family,
because a quality score is low, because gate confidence is low, or because an
operator wants specialist behavior.

## Promotion Boundary

Development success is not product success. A promoted universal product path
still requires separate bounded milestones for:

1. fresh unseen source/custody admission;
2. formal registration and isolation validation;
3. frozen blind evaluation;
4. SCUT/source guard;
5. artifact/custody audit;
6. end-to-end system validation;
7. final promotion decision.

Before that final decision, `artifacts/current-primary` remains the default
product path and Candidate 5 remains `research_only/gate_qualified_nonpromotion`.

## Next Bounded Goal Draft

The next runnable stage, if approved by Sol xhigh architecture admission, should
be:

```text
goal_name = U1 universal residual adapter sidecar architecture admission
decision_question =
  Does the approved frozen-primary continuous residual adapter sidecar have a
  causal hypothesis, development-safe evidence bundle, and pass/kill contract
  strong enough to become MECHANISM_ADMISSIBLE?

allowed_terminals =
  MECHANISM_ADMISSIBLE
  MECHANISM_REJECTED
  PREREQUISITE_NEEDED

authorized_actions =
  read code/docs/configs
  write architecture admission decision

prohibited_actions =
  training
  inference
  target-image viewing
  dataset payload download
  checkpoint mutation
  consumed-blind reuse
  router work
  broad retrain
  threshold rescue
  implementation handoff before admission

attempt_bound =
  one Sol xhigh admission pass plus one document-repair pass if Critic finds
  blocking defects

decision_lane = Sol xhigh
execution_lane = none in U1

successor_policy =
  MECHANISM_ADMISSIBLE -> close U1; open a separate U2 implementation-plan Goal
  MECHANISM_REJECTED -> close the sidecar path
  PREREQUISITE_NEEDED -> name the prerequisite and leave no implementation handoff
```

If Sol rejects admission, the universal sidecar path closes unless a future
bounded preregistration names a materially different mechanism.

Confidence: medium
Scope-risk: moderate
Reversibility: clean
Directive: Do not treat this approved design as implementation authority; U1 can
  only write an admission decision, and U2 must be separately opened before any
  Luna execution or code implementation.
Tested: Read existing program, preflight, Candidate 5 decision, model registry,
  and RefineNet feature-return surface; no code execution beyond read-only
  inspection.
Not-tested: No training, inference, image review, dataset download, checkpoint
  mutation, mechanism admission, or product promotion.
