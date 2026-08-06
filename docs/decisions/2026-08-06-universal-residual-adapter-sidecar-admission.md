# Universal Residual Adapter Sidecar Admission

## Terminal

```text
u1_terminal = MECHANISM_ADMISSIBLE
implementation_handoff = disabled
next_goal_required = U2 implementation-plan Goal
product_default = artifacts/current-primary
```

The frozen-primary continuous residual adapter sidecar is admitted as a
mechanism candidate for a later bounded implementation-planning Goal. This
decision does not authorize code changes, training, inference, target-image
access, dataset payload download, checkpoint mutation, fresh-blind use, router
work, broad retraining, threshold rescue, or product promotion.

U1 closes with this decision. Any implementation, even a small preflight
implementation, requires a separate U2 Goal.

## Admission Question

Does the approved frozen-primary continuous residual adapter sidecar have a
causal hypothesis, development-safe evidence bundle, and pass/kill contract
strong enough to become `MECHANISM_ADMISSIBLE`?

Answer: yes, for architecture admission only.

## Decisive Evidence

- Design artifact: `docs/plans/2026-08-06-universal-residual-adapter-sidecar-design.md`.
- Design commit: `b8c85893b92d`.
- Product-owner universal requirement: satisfied by user approval of a single
  `clean(image)` external interface, no domain label, no caller hint, no hard
  routing, internal continuous soft residual adapter mixing only, and fallback
  to `current-primary`.
- Release claim boundary: limited to SCUT and HW5K development domains plus one
  fresh unseen source if later promoted; no arbitrary unknown-domain
  generalization claim.
- Critic review: prior blockers resolved after the design added U1 scope
  narrowing, executable thresholds, split ledger, anti-router audit,
  `current-primary` protection, and a falsifiable hypothesis.
- Sol architecture verdict: `MECHANISM_ADMISSIBLE`, with decision artifact
  writeable and no implementation handoff.
- Runtime verification: the design commit only added the sidecar design doc; no
  training, inference, artifact mutation, or checkpoint creation accompanied it.

## Why This Is Materially New

The mechanism is distinct from prior rejected families because it does not
continue a shared generator, tune scalar losses, rescue thresholds, add an
automatic router, or append a second-stage cleanup model. It freezes the proven
`current-primary` path, adds residual-only sidecar capacity, and conditions a
continuous adapter mixture only on current-image reconstruction features.

The causal claim is falsifiable: if gains are explained by an equal-parameter
single adapter, a uniform adapter mixture, near-hard source routing, trunk
unfreezing, labels, threshold rescue, or broad retraining, the mechanism is
killed.

## Admission Basis

The design now satisfies the U1 admission requirements:

1. **Product authority.** Universal capability is explicitly required, with no
   caller-provided domain label and no hard routing.
2. **Single interface.** The only external product call is
   `clean(image) -> cleaned_image`.
3. **Current-primary preservation.** The base checkpoint/config remains the
   product default; base weights and BN state must remain immutable; zero-init
   and fallback outputs must match `current-primary` within fixed tolerances.
4. **Mechanism attribution.** The sidecar must beat `current-primary`, an
   equal-parameter single adapter, and a uniform three-adapter mixture under
   matched budgets.
5. **Development-safe gates.** HW5K development materiality, SCUT/source guards,
   non-degenerate mixing, and stop boundaries are frozen with numeric
   thresholds.
6. **Evidence custody.** The split ledger separates train-safe development
   inputs, one-shot development/source guards, consumed HW5K official test
   context, and unavailable future fresh unseen evidence.
7. **Anti-router controls.** Gate determinism, source predictability, metadata
   leakage, telemetry-driven branching, and hard decisions are explicit kill
   conditions.

## Required U2 Boundary

The only authorized successor is a new U2 implementation-plan Goal. U2 must be
created separately and must not inherit broad authority from this decision.

Minimum U2 shape:

```text
goal_name = U2 universal residual adapter sidecar implementation plan
entry_prerequisite = U1 terminal MECHANISM_ADMISSIBLE
allowed_terminal =
  IMPLEMENTATION_PLAN_READY
  IMPLEMENTATION_PLAN_REJECTED
  PREREQUISITE_NEEDED
authorized_actions =
  read code/docs/configs
  write implementation plan
  write test plan
prohibited_actions =
  code implementation
  training
  inference
  target-image viewing
  dataset payload download
  checkpoint mutation
  consumed-blind reuse
  router work
  broad retrain
  threshold rescue
```

Only after U2 closes as `IMPLEMENTATION_PLAN_READY` may a separate implementation
Goal assign Luna execution or subagents to code.

## Non-Claims

This decision does not claim:

- a working sidecar implementation exists;
- any development metric has improved;
- any fresh unseen source is admitted;
- any blind evaluation is available;
- any system is release-eligible;
- `artifacts/current-primary` can be replaced;
- Candidate 5 is promoted;
- arbitrary unknown-domain generalization is supported.

## Verification

- `git diff --check` passed on the design document before commit.
- Commit `b8c85893b92d` contains only
  `docs/plans/2026-08-06-universal-residual-adapter-sidecar-design.md`.
- Key model artifacts remain at their prior mtimes; no new model checkpoint was
  created during U1.
- Process audit found no active repository training or inference job.

Confidence: medium
Scope-risk: moderate
Reversibility: clean
Directive: Do not start implementation from U1; create and close U2 as an
  implementation-plan Goal before assigning Luna execution or writing code.
Tested: Sol architecture admission, Critic re-review, runtime verification, and
  `git diff --check` on the design.
Not-tested: No code implementation, training, inference, image review, dataset
  download, checkpoint mutation, fresh-blind evaluation, system validation, or
  product promotion.
Related: docs/plans/2026-08-06-universal-residual-adapter-sidecar-design.md
