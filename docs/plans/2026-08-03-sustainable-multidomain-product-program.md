# Sustainable Multi-Domain Product Program

## Status

Active program charter. This document is the durable north star and transition
contract; it is not itself a never-ending experiment. Exactly one bounded
milestone is active at a time.

## Requirements Summary

Deliver a release-eligible handwriting-removal system for caller-known document
domains while preserving the verified default behavior for unknown inputs.
Progress must be evidence-driven and reproducible:

- keep `artifacts/current-primary` as the default until another product path
  passes all of its preregistered gates;
- do not reuse a consumed blind set for later model selection or promotion;
- prefer the already implemented explicit caller-domain boundary over automatic
  routing;
- do not start broad retraining, scalar threshold rescue, or router work merely
  because a milestone fails;
- every experiment must either reduce a named uncertainty or kill a named path.

The current research harness already proves strict caller-provided routing and
artifact integrity, but it does not prove product safety
(`docs/decisions/2026-08-03-explicit-domain-dual-checkpoint-research-harness.md:99`).
Candidate 5 is not a universal replacement because it failed both SCUT checks
(`docs/decisions/2026-08-02-hw5k-mixed-candidate5-gate-b-rejection.md:3`).
The official HW5K test cannot supply a new promotion claim because its one blind
use is complete (`docs/decisions/2026-07-26-hw5k-final-blind-current-primary.md:56`).

## Program Definition Of Done

The program completes only when one product path is release-eligible:

1. **Explicit specialist path:** caller-domain contract, HW5K-domain development
   gate, source-risk report, contamination audit, fresh unseen HW5K-like blind
   evaluation, and end-to-end system validation all pass with frozen artifacts.
2. **Universal path:** a materially new, preregistered mechanism passes the
   HW5K-domain gate, all SCUT/source guards, a fresh unseen blind evaluation, and
   end-to-end system validation.

For either path, the final decision record must identify exact code/config/model
hashes, input custody, fixed thresholds, metric outputs, failure policy, and
rollback target. `artifacts/current-primary` is not replaced until that record is
approved.

If neither path can proceed because an external prerequisite is unavailable,
the program is `blocked_external_prerequisite`, not complete and not silently
redirected to an automatic router.

## Rolling Milestone Contract

Every active milestone must declare:

1. one decision question;
2. frozen eligible inputs and prohibited inputs;
3. one reproducible evidence artifact;
4. testable pass, kill, and prerequisite-needed outcomes;
5. the only authorized successor for each outcome;
6. a time, compute, or attempt bound;
7. the model-routing lane used for decision versus execution.

Milestone statuses are:

- `pass`: its claim is proved and the declared successor may start;
- `kill`: the tested path is closed by evidence;
- `prerequisite_needed`: the claim is not testable yet; this is not a model
  failure and may only transition to a prerequisite-acquisition milestone.

No milestone may remain active merely to keep trying variants. A failed gate
closes the declared attempt; a new attempt requires a different mechanism and a
new preregistration.

## Fork / Kill Graph

```text
M0 evidence admission (complete: prerequisite_needed)
  official HW5K test already consumed
  ├─> M1 fresh blind source + custody admission
  └─> M2 freeze Candidate 5 specialist promotion protocol

M1 ADMITTED ───────────────────────────────┐
M1 NOT_FOUND_WITHIN_BOUND                  │
  -> park data lane as external prerequisite
M1 INVALID_SOURCE                         │
  -> try next preregistered source within │
     the same bounded shortlist           │
                                          ├─> M3 one-shot frozen specialist eval
M2 CONTRACT_PASS ─────────────────────────┘    (requires both M1 + M2)
M2 CONTRACT_FAIL
  -> kill Candidate 5 promotion line; do not infer

M3 METRIC_AND_RISK_PASS
  -> M4 end-to-end explicit-domain system validation
M3 FAIL
  -> kill Candidate 5 promotion line; do not tune on blind outputs

M4 PASS
  -> product promotion decision
M4 FAIL
  -> bounded implementation correction only if model evidence remains valid;
     otherwise kill the path
```

The scheduler is work-conserving without being scope-expanding: when one lane
ends in `prerequisite_needed`, park it and select the highest-value independent
milestone whose prerequisites are already satisfied. After all independent
milestones finish, an unresolved external prerequisite blocks the program. It
does not authorize a new model or router branch.

The universal path remains parked. It may be activated only by a separate Sol
xhigh architecture decision naming a materially new decoupling mechanism and
showing why the explicit path cannot satisfy the product need. Automatic routing
is outside this graph and requires its own independent routing data, unknown
policy, false-route cost, and rejection contract
(`docs/plans/2026-08-03-explicit-domain-dual-checkpoint-design.md:204`).

## Current Milestone: M1 Fresh Blind Source And Custody Admission

Decision question: does a task-compatible, legally usable, genuinely unseen
HW5K-like evaluation source exist that can be reserved before Candidate 5 sees
any input or label?

Bound:

- inspect at most five credible public or user-controlled sources;
- download no multi-gigabyte payload during source discovery;
- run no model inference and inspect no target images;
- stop when one source is admissible or the five-source shortlist is exhausted.

Admission criteria:

- paired contaminated-document input and clean target, or an upstream protocol
  that can produce an equivalent objective reference without project tuning;
- document-domain relevance to real handwriting/mark removal;
- license and provenance recorded;
- no prior appearance in this repository's training, development, outputs, or
  target-aware review history;
- a split/custody rule that can be frozen before inference;
- sufficient sample count and diversity for a promotion claim, with the exact
  minimum preregistered before download or scoring.

Output: a source-admission decision record with `ADMITTED`, `INVALID_SOURCE`, or
`NOT_FOUND_WITHIN_BOUND`, plus URLs, licenses, split candidates, contamination
queries, and the sole next action. `ADMITTED` authorizes registration tooling,
not Candidate 5 inference.

M1 completed on 2026-08-03 as `NOT_FOUND_WITHIN_BOUND`. All locally known
same-task sources are consumed, development-only, task-mismatched, or lack clean
targets, and no new official source could be admitted from verifiable metadata.
The data lane is parked as `external_data_prerequisite`; M2 is the only eligible
independent successor. See
`docs/decisions/2026-08-03-fresh-blind-source-custody-admission.md`.

## Current Milestone: M2 Specialist Product Contract Freeze

Decision question: can Candidate 5's caller-known specialist claim, failure
policy, source-risk boundary, fixed metrics, minimum fresh-set requirements, and
one-shot evaluation order be frozen without using a new blind input or label?

Bound:

- one contract draft and one Sol xhigh review pass;
- no training, inference, image review, threshold sweep, or checkpoint change;
- reuse existing evaluation tooling and fixed thresholds unless a documented
  incompatibility makes the contract unexecutable;
- terminal is `CONTRACT_PASS` or `CONTRACT_FAIL`.

`CONTRACT_PASS` does not promote Candidate 5. It makes the parked fresh-data lane
the sole remaining prerequisite before M3. `CONTRACT_FAIL` closes Candidate 5's
specialist-promotion line without activating a router.

## Model Routing

- `gpt-5.6-sol` with `xhigh`: program/milestone design, claim boundaries,
  preregistration, architecture, tradeoffs, promotion decisions, and final
  verification.
- `gpt-5.6-luna` with `max`: bounded repository inventories, source metadata
  collection, implementation, tests, deterministic evaluation, MPS smoke, and
  exact-path Git operations.
- Execution subagents also use Luna max. When native child-agent model routing
  cannot select Luna, launch a bounded `codex exec --model gpt-5.6-luna -c
  'model_reasoning_effort="max"'` task instead.
- Every Luna execution returns structured evidence to Sol; it does not broaden
  the claim, change the milestone, or choose the next product path.

## Risks And Mitigations

- **Research drift:** one-question milestones, bounded attempts, and kill states.
- **Blind-set laundering:** content/custody audit plus explicit consumed-set
  registry before inference.
- **Missing data misreported as model failure:** `prerequisite_needed` is a
  first-class terminal state.
- **Premature router complexity:** router is outside the graph and requires a
  new authorization decision.
- **Default regression:** current-primary remains the fallback and promotion
  gates include source-domain evidence.
- **Execution/decision role leakage:** Luna gathers or changes bounded artifacts;
  Sol freezes claims and makes decisions.

## Verification Steps

At each milestone transition:

1. verify the evidence artifact exists and hashes or links resolve;
2. verify prohibited datasets and actions were not used;
3. run scoped tests/static checks for changed tooling;
4. have Sol xhigh map every acceptance criterion to evidence;
5. record the terminal status and exactly one authorized successor;
6. update this graph only through a decision record, never through an ad hoc
   execution fallback.

## ADR

### Decision

Use a durable program charter with a rolling queue of finite, preregistered
milestones. Treat missing evidence as a prerequisite outcome rather than forcing
a pass/fail model verdict.

### Drivers

1. Product progress must survive individual experiment failures.
2. Promotion claims require honest unseen evidence.
3. The project must resist automatic scope expansion into training or routing.

### Alternatives Considered

- **One permanent active research Goal:** rejected because it has no auditable
  terminal point and encourages indefinite variant search.
- **Keep G2 and force a binary promotable/not-promotable result:** rejected
  because the current blocker is missing admissible data, not a Candidate 5
  measurement.
- **Start automatic-router work now:** rejected because routing data and the
  unknown-domain failure contract do not exist.

### Consequences

Progress is continuous at the program level but interruptible and verifiable at
the milestone level. Some milestones can end without a model verdict. External
data can legitimately block the program, and that state remains visible instead
of being hidden by another experiment.

### Follow-Ups

Execute M2 while the M1 data lane is parked. If M2 passes, keep the frozen
contract unchanged until a new source is admitted; if M2 fails, close Candidate
5's promotion line. In neither case may the program open the router branch.
