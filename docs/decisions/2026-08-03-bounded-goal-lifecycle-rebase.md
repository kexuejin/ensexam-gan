# Bounded Goal Lifecycle Rebase

## Decision

Separate the sustainable multi-domain program charter from the active runtime
Goal. Keep the charter as a durable transition registry, but permit exactly one
bounded executable Goal at a time. The sole current Goal is G1 / M2I Paired
Blind Protocol Closeout.

When G1 reaches `PROTOCOL_READY` or `PROTOCOL_REJECTED`, close it. If it is ready
but no fresh paired source has been admitted, set the program to
`parked_external_prerequisite` and leave no Goal active. Do not represent an
external wait as continuous execution.

## Why The Previous Shape Was Not Sustainable

The earlier P1 objective combined three different lifetimes:

- a long-lived product north star;
- completed M1 and M2 milestones;
- the executable M2I implementation stage.

Although the text required finite milestones, it still labeled all three as
current and kept the program itself active. That made an external-data wait look
like unfinished agent work and allowed future stages to read as a linear
roadmap. The new lifecycle makes active work, parked prerequisites, completed
history, and killed paths distinct states.

## Decision Drivers

1. Every active Goal needs an auditable terminal reachable with currently
   authorized work.
2. Missing fresh blind data is an external prerequisite, not a model failure and
   not permission to search indefinitely.
3. Explicit caller-domain product validation and automatic routing are separate
   claim surfaces with different evidence requirements.
4. `artifacts/current-primary`, consumed-blind custody, and Candidate 5's
   research-only state must remain invariant while evidence is incomplete.

## Transition Table

| Current stage | Terminal | Program effect | Only authorized next action |
| --- | --- | --- | --- |
| G1 / M2I | `PROTOCOL_READY` | Close G1 as `research_ready` | If no fresh source exists, park with no active Goal; otherwise evaluate G2 entry gates. |
| G1 / M2I | `PROTOCOL_REJECTED` | Close Candidate 5 specialist line | Stop this branch; no router, retraining, threshold rescue, or blind reuse. |
| Parked data lane | source admitted and caller-known need confirmed | Program may become `active_stage` | Create a new bounded G2 one-shot comparison Goal. |
| G2 comparison | `specialist_promotable` | Candidate remains pending system validation | Create bounded explicit-domain end-to-end validation; do not open router work. |
| G2 comparison | `specialist_not_promotable_line_closed` | Close Candidate 5 specialist line | Stop this branch. |
| Separate router preflight | entry contract absent | Router remains unauthorized | Keep explicit-only or stay parked/closed. |
| Separate router preflight | entry contract satisfied by Sol xhigh decision | Independent branch may open | Create a bounded router-feasibility Goal with its own pass/kill terminals. |

## Entry Events

No new Goal is created merely because the previous one closed. A future G2
requires all of:

1. G1 terminal `PROTOCOL_READY`;
2. a formally admitted, unseen, paired HW5K-like source with at least 200 pages;
3. an explicit product-owner need for authoritative caller-provided domain
   labels.

Automatic router feasibility is not G2's successor. It additionally requires a
separate Sol xhigh decision proving the caller-known contract is insufficient,
plus independent routing data, an unknown-domain policy, a false-route cost, and
rejection behavior.

## Alternatives Rejected

- **One permanent active Goal:** rejected because it conflates a north star with
  runnable work, cannot finish during an external-data wait, and encourages
  research drift.
- **Linear G2 to router to routed system:** rejected because successful explicit
  caller routing can proceed directly to explicit-system validation, while a
  router needs an independent product need and evidence contract.
- **Close Candidate 5 now because no fresh source exists:** rejected because no
  fresh-blind quality measurement has failed; the honest state is parked after
  the locally executable protocol closes.
- **Reuse the consumed official HW5K test:** rejected because its single blind
  use is complete and reuse would invalidate a promotion claim.

## Consequences

- Program continuity lives in the charter and decision log, not in an immortal
  runtime Goal.
- Every Goal can be completed or killed using evidence available within its
  declared boundary.
- External data can stop active execution without losing the product north star.
- Router work cannot appear as an automatic fallback from specialist failure or
  missing data.
- `artifacts/current-primary` remains the default and Candidate 5 remains
  `research_only/gate_qualified_nonpromotion` until later gates pass.

## Verification

This decision changes planning and lifecycle semantics only. It does not claim
new implementation, inference, training, dataset, or quality evidence. Verify
the rebase by checking that:

- only G1 / M2I is labeled active in the program charter;
- M1 and M2 are labeled completed history;
- every G1 terminal has one successor or explicit stop state;
- G2 entry requires fresh data and a caller-known product need;
- automatic routing is a separate conditional branch;
- no current-primary, checkpoint, threshold, blind-set, or promotion state is
  changed.

The governing charter is
`docs/plans/2026-08-03-sustainable-multidomain-product-program.md`.

## Outcome On 2026-08-04

G1 reached `PROTOCOL_REJECTED` after the one authorized repair pass left
independently reproduced fail-open trust-boundary gaps. Candidate 5's
specialist-promotion line is therefore closed, no successor Goal is active, and
the program is `closed_no_viable_path`. This is the intended lifecycle behavior:
the kill terminal stops execution instead of converting a failed protocol into
router work, retraining, threshold rescue, blind reuse, or an unbounded second
repair. The terminal evidence is recorded in
`docs/decisions/2026-08-04-paired-blind-protocol-rejection.md`.
