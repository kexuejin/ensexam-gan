# Sustainable Multi-Domain Product Program

## Status And Purpose

This document is the durable north star and transition registry for multi-domain
product work. It is not an always-active runtime Goal. A runtime Goal contains
exactly one bounded executable stage and closes when that stage reaches a
declared terminal.

There is currently **no active runtime Goal**. M1 and M2 completed, and G1 / M2I
closed on 2026-08-04 as `PROTOCOL_REJECTED`. Candidate 5's specialist-promotion
line is closed and the program is `closed_no_viable_path`. A materially new
universal mechanism would require a new Sol xhigh architecture decision and a
new bounded program; it is not a successor hidden inside this closed Goal.

Program states are:

| State | Meaning |
| --- | --- |
| `active_stage` | Exactly one bounded Goal has runnable, locally authorized work. |
| `parked_external_prerequisite` | No bounded Goal is active; a declared external event may reopen the program. |
| `complete` | A release-eligible path passed its final promotion decision. |
| `closed_no_viable_path` | Every authorized path has reached a kill terminal. |

`parked_external_prerequisite` is visible program state, not an instruction to
keep a runtime Goal alive and not authorization to invent another experiment.

## Requirements Summary

Deliver a release-eligible handwriting-removal system for caller-known document
domains while preserving the verified default behavior for unknown inputs:

- keep `artifacts/current-primary` as the default until another product path
  passes all preregistered gates;
- do not reuse a consumed blind set for model selection or promotion;
- keep Candidate 5 `research_only/gate_qualified_nonpromotion` until a fresh
  blind comparison and end-to-end validation both pass;
- prefer the implemented explicit caller-domain boundary over automatic routing;
- do not start broad retraining, scalar threshold rescue, or router work because
  a stage fails or an external prerequisite is absent;
- require every experiment to reduce one named uncertainty or kill one named
  path.

The explicit-domain research harness proves caller-provided dispatch and
artifact integrity, not product safety
(`docs/decisions/2026-08-03-explicit-domain-dual-checkpoint-research-harness.md:99`).
Candidate 5 is not a universal replacement because it failed both SCUT checks
(`docs/decisions/2026-08-02-hw5k-mixed-candidate5-gate-b-rejection.md:3`). The
official HW5K test cannot support a new promotion claim because its single blind
use is complete
(`docs/decisions/2026-07-26-hw5k-final-blind-current-primary.md:56`).

## Program Completion And Closure

The program becomes `complete` only when one product path is release-eligible:

1. **Explicit specialist path:** caller-domain contract, domain-development
   gate, source-risk report, contamination audit, fresh unseen HW5K-like blind
   evaluation, and end-to-end explicit-domain system validation all pass with
   frozen artifacts.
2. **Universal path:** a materially new, separately preregistered mechanism
   passes the HW5K-domain gate, all SCUT/source guards, a fresh unseen blind
   evaluation, and end-to-end system validation.

The final decision must identify exact code/config/model hashes, input custody,
fixed thresholds, metric outputs, failure policy, and rollback target.
`artifacts/current-primary` is not replaced before that decision.

If no runnable stage remains because fresh external evidence is unavailable,
the program becomes `parked_external_prerequisite`. If every authorized path is
killed, it becomes `closed_no_viable_path`. Neither state silently redirects to
automatic routing.

## Bounded Goal Contract

Every active Goal must declare:

1. one decision question;
2. frozen eligible and prohibited inputs;
3. one reproducible evidence bundle;
4. testable pass, kill, and prerequisite-needed outcomes where applicable;
5. the only authorized successor for every terminal;
6. a time, compute, or attempt bound;
7. the Sol decision lane and Luna execution lane.

An active Goal cannot absorb its successor, wait indefinitely for an entry
event, or remain open to try variants. A different mechanism requires a new
preregistration and a new Goal.

## Fork / Kill Graph

```text
COMPLETED HISTORY
M0 evidence admission
  -> prerequisite_needed: official HW5K test already consumed

M1 fresh source/custody admission
  -> NOT_FOUND_WITHIN_BOUND
  -> data lane parked as external prerequisite

M2 specialist product-contract freeze
  -> CONTRACT_PASS

COMPLETED HISTORY
G1 / M2I paired blind protocol closeout
  -> PROTOCOL_REJECTED on 2026-08-04
  -> close Candidate 5 specialist-promotion line
  -> program closed_no_viable_path; no active Goal
  -> no router, retraining, threshold rescue, blind reuse, or second repair

UNREACHABLE AFTER THE G1 KILL TERMINAL
G2 one-shot frozen specialist comparison
  entry requires:
    G1 == PROTOCOL_READY
    fresh paired source >= 200 pages formally admitted
    caller-known product-owner need still explicit
  ├─ specialist_promotable
  │    -> explicit-domain end-to-end system validation may open
  └─ specialist_not_promotable_line_closed
       -> stop; do not tune on blind outputs

G3 explicit-domain end-to-end system validation
  ├─ PASS -> product promotion decision
  └─ FAIL -> one separately bounded implementation correction only when model
             evidence remains valid; otherwise close the explicit path

SEPARATE CONDITIONAL BRANCH, NEVER AN AUTOMATIC SUCCESSOR
R1 automatic-router feasibility
  entry requires a separate Sol xhigh decision proving explicit caller routing
  is insufficient, plus independent routing data, unknown-domain policy,
  false-route cost, and rejection behavior
  ├─ router_feasible -> separately preregister routed-system validation
  └─ router_not_feasible -> keep explicit-only or close router branch
```

The previous universal path is outside this closed program. It can be proposed
only through a separate Sol xhigh architecture decision naming a materially new
mechanism, followed by a new charter and bounded Goal. The G1 kill terminal does
not open router or universal work.

## Completed History: M1 Fresh Blind Source And Custody Admission

M1 completed on 2026-08-03 as `NOT_FOUND_WITHIN_BOUND`. The bounded audit found
the locally known same-task sources consumed, development-only, task-mismatched,
or missing clean targets. It downloaded no payload, ran no inference, and opened
no target images. The data lane is parked as an external prerequisite. See
`docs/decisions/2026-08-03-fresh-blind-source-custody-admission.md`.

The data lane may reopen only from either:

1. a user-controlled external root with paired inputs/clean targets and truthful
   no-training/no-selection/no-review provenance; or
2. a new public-source shortlist after official metadata access is restored.

Admission authorizes registration and isolation checks, not inference.

## Completed History: M2 Specialist Product Contract Freeze

M2 completed on 2026-08-03 as `CONTRACT_PASS`. The frozen contract is
`docs/plans/2026-08-03-candidate5-specialist-product-contract.md`; its decision
record is
`docs/decisions/2026-08-03-candidate5-specialist-product-contract-freeze.md`.

The contract fixes the caller-known claim, artifacts, wrong-route policy,
minimum fresh-set size, comparison order, `>=20%` mean-residual requirement, and
five aggregate/tail non-regression checks. It does not promote Candidate 5.

## Completed History: G1 / M2I Paired Blind Protocol Closeout

### Decision Question

Can the existing blind registration, inference, scoring, gate, audit, and
completion surfaces be composed into one fail-closed paired-checkpoint protocol
that proves both prediction sets were frozen before either label score?

### Frozen Scope

Deliver and test only:

- `scripts/analysis/manage_paired_blind_comparison.py` with `prepare`,
  `seal-inference`, and `verify` stages;
- `tests/test_manage_paired_blind_comparison.py`;
- the already frozen `>=20%` mean-residual option in
  `scripts/analysis/gate_dev_candidate_metrics.py` and its focused tests;
- a compact terminal decision mapping every G1 acceptance criterion to evidence.

The implementation must compose existing blind-evaluation surfaces rather than
create a parallel framework. Synthetic fixtures are allowed. Dataset download,
training, model inference, image review, checkpoint changes, threshold sweeps,
automatic routing, and consumed blind data are prohibited.

### Bound And Evidence

Use one implementation design and one focused defect-repair pass. A repair may
fix code against the same frozen contract; it cannot change the protocol,
thresholds, candidate, or evidence requirements. If the contract remains
unexecutable after that pass, terminate as `PROTOCOL_REJECTED`.

The evidence bundle is:

- exact diffs for the owned script and tests;
- focused unit-test results for the paired protocol and gate;
- relevant blind preparation/completion regression tests;
- Python compile/static-import checks and scoped `git diff --check`;
- a Sol xhigh acceptance audit. No quality or promotion claim is produced.

### Actual Terminal

G1 closed on 2026-08-04 as `PROTOCOL_REJECTED`. The first Sol xhigh audit found
four fail-closed gaps. The one authorized repair pass fixed minimum sample
count, the exact 20% threshold, tolerance laundering, and stale inference-output
directories. Focused and blind regression suites passed after the repair.

The independent Critic then reproduced deeper trust-boundary bypasses: mutable
parent gate configuration before sealing, self-asserted completion and gate
JSON, incomplete binding of the frozen checkpoint/threshold/custody contract,
and prediction paths not proven disjoint from labels. A forged worse candidate
could therefore verify as passing. The bounded contract prohibits a second
repair pass, so Candidate 5's specialist-promotion line is closed. See
`docs/decisions/2026-08-04-paired-blind-protocol-rejection.md`.

## Model Routing

- `gpt-5.6-sol` with `xhigh`: Goal design, claim boundaries, preregistration,
  architecture, tradeoffs, promotion decisions, and final acceptance.
- `gpt-5.6-luna` with `max`: bounded inventories, implementation, tests,
  deterministic evaluation, MPS smoke, and exact-path Git operations.
- Execution subagents use Luna max. If native child routing cannot select Luna,
  use a bounded `codex exec --model gpt-5.6-luna -c
  'model_reasoning_effort="max"'` task.
- Luna returns structured evidence to Sol and cannot broaden the claim, change
  the Goal, or select the next product branch.

## Transition Verification

At every Goal terminal:

1. verify the named evidence bundle exists and referenced hashes resolve;
2. verify prohibited data and actions were not used;
3. run scoped tests and static checks for changed tooling;
4. have Sol xhigh map every acceptance criterion to evidence;
5. record the terminal and its single authorized successor or stop state;
6. close the runtime Goal before creating another one.

The lifecycle rebase is recorded in
`docs/decisions/2026-08-03-bounded-goal-lifecycle-rebase.md`.
