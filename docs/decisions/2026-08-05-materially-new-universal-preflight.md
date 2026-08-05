# Materially-New Universal Mechanism Preflight

## Decision Status

```text
artifact_status = INTENDED_TERMINAL_PENDING_RUNTIME_RECONCILIATION
preflight_terminal = PREREQUISITE_NEEDED
operational_terminal_finalized = false
implementation_handoff = disabled
```

The preflight selects `PREREQUISITE_NEEDED` as its evidence-based terminal.
Operational finalization is pending because the Goal control plane still reports
a stale active P1/M1 handle even though the durable program closed as
`closed_no_viable_path`. The stale handle is non-authorizing: it does not reopen
P1/M1, admit another model mechanism, or permit implementation.

## Decision Question

After G1 closed as `PROTOCOL_REJECTED`, does the bounded repository evidence
admit a materially-new universal mechanism, prove that no such mechanism
exists, or identify one prerequisite that must be satisfied before another
architecture decision?

The allowed preflight terminals were:

- `MECHANISM_ADMISSIBLE`;
- `PREREQUISITE_NEEDED`;
- `NO_MATERIALLY_NEW_MECHANISM`.

The bounded evidence supports only `PREREQUISITE_NEEDED`.

## Sole External Prerequisite

```text
product_owner_universal_requirement =
  explicit product-owner decision that caller-known routing is insufficient
  and universal capability is required
```

The repository currently defines a caller-known product boundary, keeps
`artifacts/current-primary` as the default for unknown inputs, and does not
contain that product-owner decision. Technical novelty cannot substitute for
product authority.

A causal hypothesis and development-safe pass/kill contract are also absent,
so no mechanism is currently admissible. Those are not additional external
prerequisites. After the product decision, they are the required deliverables
of a new Sol xhigh architecture-admission stage.

## Mechanism Classification

```text
continuous_three_expert_reconstruction_mixture =
  distinct_untried_not_admitted
```

The shared-weight adaptation record explicitly leaves domain-conditioned
capacity and Pareto-aware multi-domain objectives as materially different
directions. A continuous three-expert reconstruction mixture is therefore a
genuine untried gap, not another scalar continuation, selector, second-stage
repair, automatic router, or paired-protocol repair.

That distinction is not an admission recommendation. The repository contains
no frozen causal hypothesis, development-only evidence bundle, attempt bound,
or pass/kill contract for the mixture. Consequently:

- `NO_MATERIALLY_NEW_MECHANISM` is unsupported because a distinct gap exists;
- `MECHANISM_ADMISSIBLE` is unsupported because product authority and the
  Sol-owned admission contract do not exist;
- `PREREQUISITE_NEEDED` preserves the option without creating runnable work.

## Product-First Versus Feasibility-First

The strongest case for immediate admission is that expert-specific continuous
capacity may avoid the cross-domain interference observed in shared-weight
adaptation, and a bounded development-only experiment could reduce product
uncertainty without making a promotion claim.

The countervailing constraint is lifecycle and product authority: an untried
technical direction does not justify opening a universal program when the
declared product accepts caller-known dispatch and no owner has required a
universal capability.

The synthesis is to retain the mixture classification verbatim, perform no
experiment now, and make the product decision the only external entry event.
If that event occurs, Sol xhigh must still decide whether one named mechanism
has a causal hypothesis and a bounded, development-safe pass/kill contract.

## Durable State And Runtime Mismatch

The authoritative durable state remains:

```text
durable_program_state = closed_no_viable_path
executable_runtime_goal = none
implementation_handoff = disabled
```

The control-plane observation on this preflight turn is recorded separately:

```text
observed_runtime_handle = stale_active_P1_M1
observed_runtime_status = active
runtime_handle_authorizes_execution = false
lifecycle_reconciliation = pending
resumed_blocked_audit_turn = 1
runtime_mutation_this_turn = none
finalized_waiting_invariant = not_yet_assertable
```

This is an internal lifecycle mismatch, not a second product prerequisite and
not an architecture deliverable. The unqualified statement that the runtime
service reports no active handle would be false. The correct operational claim
is narrower: no runtime Goal is executable or authorized by the durable
program, and the stale handle remains unchanged on resumed audit turn 1.

## Runtime Finalization Protocol

1. On each later resumed audit turn, read the Goal state before acting.
2. Count a recurrence only when the same stale-active P1/M1 conflict remains.
3. Before three consecutive resumed recurrences, keep
   `operational_terminal_finalized = false` and do not mutate the Goal.
4. At recurrence 3, if the same blocker remains and no meaningful recovery path
   exists, mark the stale Goal `blocked` through the Goal lifecycle surface.
5. Immediately read the Goal state back.
6. Assert the finalized waiting invariant only when readback proves a blocked,
   non-executable Goal. Otherwise retain
   `INTENDED_TERMINAL_PENDING_RUNTIME_RECONCILIATION`.

At every step, `implementation_handoff = disabled`.

## Entry Predicate For Any Future Universal Work

Universal implementation remains disabled unless all of the following are
true:

1. runtime lifecycle reconciliation is verified;
2. `product_owner_universal_requirement` is explicitly satisfied;
3. a new Sol xhigh architecture decision names and admits one mechanism with a
   causal hypothesis and bounded development-safe pass/kill contract;
4. a new program charter and one bounded Goal are approved.

Fresh data alone, Candidate 5 failure, G1 closure, a renamed old mechanism, or
the stale runtime handle satisfies none of these conditions.

## Prohibited Actions

This preflight authorizes no training, inference, target-image access, image
review, dataset download, checkpoint mutation, automatic routing, broad
retraining, scalar threshold rescue, consumed-blind reuse, Candidate 5 protocol
repair, inventory expansion, or Ralph/Team execution handoff.

`artifacts/current-primary` remains the product default. Candidate 5 remains
`research_only/gate_qualified_nonpromotion`.

## Evidence And Consensus

The bounded inventory found six rejected model families, the closed specialist
and paired-protocol surfaces, and the distinct-but-unadmitted capacity/objective
gap. Two inventory lanes that did not produce bounded final artifacts were
excluded from the evidence count.

The decision passed the sequential `$ralplan` review chain:

1. Sol xhigh Planner selected `PREREQUISITE_NEEDED`;
2. Sol xhigh Architect required exact external-prerequisite and runtime-state
   semantics, then approved the reconciled plan;
3. Sol xhigh Critic approved terminal accuracy, option fairness, testability,
   prohibited-action coverage, and handoff safety.

Durable supporting records:

- `docs/plans/2026-08-03-sustainable-multidomain-product-program.md`;
- `docs/decisions/2026-08-04-paired-blind-protocol-rejection.md`;
- `docs/decisions/2026-08-03-bounded-goal-lifecycle-rebase.md`;
- `docs/decisions/2026-08-02-hw5k-mixed-candidate5-gate-b-rejection.md`.

## ADR

**Decision:** Record `PREREQUISITE_NEEDED` as the intended preflight terminal,
pending runtime reconciliation, with no implementation handoff.

**Drivers:** The universal product requirement is absent; a distinct mechanism
gap exists but is not admitted; the durable program is closed; and the runtime
still exposes a stale non-authorizing handle.

**Alternatives considered:** Immediately admit the continuous three-expert
mixture; declare that no materially-new mechanism exists; treat the stale Goal
as executable; or block it prematurely on resumed audit turn 1.

**Why chosen:** This is the only state that preserves the technical option,
respects product authority and bounded-Goal lifecycle, and reports the current
control-plane mismatch truthfully.

**Consequences:** No code or model work starts. The product owner can create the
external entry event, but a later Sol admission decision and a new bounded
program are still mandatory. Runtime reconciliation proceeds independently
under its three-recurrence rule.

Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not interpret the stale P1/M1 handle or the untried mixture as execution authority; preserve the pending runtime-reconciliation status until Goal readback proves a blocked, non-executable state.
Tested: Bounded Luna inventory, sequential Sol xhigh Planner-Architect-Critic consensus, durable-state comparison, and read-only Goal inspection
Not-tested: No training, inference, image review, dataset download, checkpoint change, model admission, or product promotion was authorized
Related: docs/decisions/2026-08-04-paired-blind-protocol-rejection.md
