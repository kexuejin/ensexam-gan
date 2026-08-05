# Materially-New Universal Mechanism Preflight

## Decision Status

```text
artifact_status = OPERATIONALLY_FINALIZED
preflight_terminal = PREREQUISITE_NEEDED
operational_terminal_finalized = true
implementation_handoff = disabled
```

The preflight selects `PREREQUISITE_NEEDED` as its evidence-based terminal.
Operational finalization is verified. After the same stale active P1/M1 handle
conflicted with the durable `closed_no_viable_path` state on three consecutive
resumed audit turns, the Goal was marked `blocked` and immediate control-plane
readback confirmed that status. The blocked handle is non-authorizing: it does
not reopen P1/M1, admit another model mechanism, or permit implementation.

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

## Durable State And Runtime Reconciliation

The authoritative durable state remains:

```text
durable_program_state = closed_no_viable_path
executable_runtime_goal = none
implementation_handoff = disabled
```

The finalized control-plane state is recorded separately:

```text
observed_runtime_handle = stale_P1_M1
observed_runtime_status = blocked
runtime_handle_authorizes_execution = false
lifecycle_reconciliation = verified
consecutive_resumed_recurrences = 3
runtime_mutation_on_finalization = blocked
goal_readback_status = blocked
finalized_waiting_invariant = asserted
```

This is an internal lifecycle mismatch, not a second product prerequisite and
not an architecture deliverable. The runtime handle still exists as a blocked
record, but no runtime Goal is executable or authorized by the durable program.

## Runtime Finalization Evidence

1. Resumed audit turn 1 observed the stale active P1/M1 conflict and made no
   runtime mutation.
2. Resumed audit turn 2 reproduced the identical conflict, persisted the local
   audit count, and made no runtime mutation.
3. Resumed audit turn 3 reproduced the identical conflict. M1 was still durably
   complete as `NOT_FOUND_WITHIN_BOUND`, G1 still had no automatic successor,
   and no meaningful locally authorized recovery path existed.
4. The third recurrence satisfied the blocked-transition threshold. The Goal
   lifecycle surface accepted `status = blocked`.
5. Immediate `get_goal` readback returned the same parent thread and
   `status = blocked`, proving the Goal is non-executable.

The finalized waiting invariant is therefore asserted, and
`implementation_handoff = disabled` remains in force.

## Entry Predicate For Any Future Universal Work

Universal implementation remains disabled unless all of the following are
true:

1. runtime lifecycle reconciliation is verified;
2. `product_owner_universal_requirement` is explicitly satisfied;
3. a new Sol xhigh architecture decision names and admits one mechanism with a
   causal hypothesis and bounded development-safe pass/kill contract;
4. a new program charter and one bounded Goal are approved.

Fresh data alone, Candidate 5 failure, G1 closure, a renamed old mechanism, or
the blocked runtime handle satisfies none of these conditions.

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

**Decision:** Finalize `PREREQUISITE_NEEDED` as the preflight terminal after
verified runtime reconciliation, with no implementation handoff.

**Drivers:** The universal product requirement is absent; a distinct mechanism
gap exists but is not admitted; the durable program is closed; and the stale
runtime handle is now verified blocked and non-authorizing.

**Alternatives considered:** Immediately admit the continuous three-expert
mixture; declare that no materially-new mechanism exists; treat the stale Goal
as executable; or block it before the third identical resumed recurrence.

**Why chosen:** This is the only state that preserves the technical option,
respects product authority and bounded-Goal lifecycle, and reconciles the
control plane only after the required repeated evidence.

**Consequences:** No code or model work starts. The product owner can create the
external entry event, but a later Sol admission decision and a new bounded
program are still mandatory. The old P1/M1 Goal cannot execute while blocked.

Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not interpret the blocked P1/M1 handle or the untried mixture as execution authority; a future universal program still requires the explicit product-owner entry event and a new Sol admission decision.
Tested: Bounded Luna inventory, sequential Sol xhigh Planner-Architect-Critic consensus, three consecutive parent-thread Goal audits, blocked transition, and immediate blocked-status readback
Not-tested: No training, inference, image review, dataset download, checkpoint change, model admission, or product promotion was authorized
Related: docs/decisions/2026-08-04-paired-blind-protocol-rejection.md
