# Paired Blind Protocol Rejection

## Decision

Close G1 / M2I as `PROTOCOL_REJECTED`, close Candidate 5's
specialist-promotion line, and set the sustainable multi-domain program to
`closed_no_viable_path` with no active runtime Goal.

Do not promote or ship the paired manager draft. Do not rescue this line with a
second repair pass, a different threshold, a second candidate, broad retraining,
automatic routing, the consumed HW5K official test, or inference on an
unadmitted blind source. `artifacts/current-primary` remains the product default
and Candidate 5 remains `research_only/gate_qualified_nonpromotion`.

## Decision Question

Can the existing blind registration, inference, scoring, gate, audit, and
completion surfaces be composed in one bounded implementation plus one repair
pass into a fail-closed paired-checkpoint protocol that proves both prediction
sets were frozen before either label score and that the frozen six-metric gate
was actually satisfied?

The answer within the preregistered bound is **no**.

## Evidence Sequence

The initial implementation and gate extension were exercised only with
synthetic temporary fixtures. No dataset was downloaded, no model inference or
training ran, no target image was opened, and no consumed blind set was reused.

The first focused run passed 21 tests, and four existing blind regression
modules passed 51 tests. A Sol xhigh Architect review nevertheless found four
fail-closed gaps:

1. `--min-sample-count` could be lowered below the frozen 200-page minimum;
2. the paired threshold could differ from the frozen 20% improvement rule;
3. gate tolerance was not bound to `1e-12`, enabling tolerance laundering;
4. an existing empty or partial `frozen_predictions` directory was not rejected.

The single authorized repair pass fixed those four items. After repair:

- `python -m py_compile` passed for the paired manager and its focused test;
- `tests.test_manage_paired_blind_comparison` plus
  `tests.test_gate_dev_candidate_metrics` passed 22 tests in 29.506 seconds;
- `tests.test_prepare_blind_generalization_eval`,
  `tests.test_verify_blind_eval_completion`,
  `tests.test_evaluate_frozen_blind_predictions`, and
  `tests.test_build_frozen_blind_audit_pack` passed 51 tests in 15.510 seconds;
- scoped `git diff --check` passed.

The independent Sol xhigh Critic then reproduced deeper bypasses that the green
tests did not cover:

- the parent `gate_config` can be changed after prepare and before seal, after
  which seal and verify trust the changed 20%/tolerance values;
- completion verification accepts a hand-written object containing only the
  expected protocol name and `gate_pass=true`, without proving the existing
  completion verifier's checks ran;
- paired gate verification trusts self-asserted `decision=accept` and six
  `passed=true` records instead of recomputing the frozen gate from the bound
  post-score CSV files;
- the exact baseline and Candidate 5 checkpoint hashes, complete frozen
  inference/scoring thresholds, and custody schema are not independently bound
  to the specialist product contract at every phase;
- a prediction path is not proven to live under its branch's frozen output
  directory or to be disjoint in path and SHA from every registered clean label.

The strongest counterexample uses a worse Candidate post-score CSV, updates its
CSV hash, supplies six self-asserted passing gate checks and two minimal passing
completion objects, and can still obtain paired verification success. A
Candidate prediction can also point at a clean target. These are protocol claim
failures, not cosmetic validation gaps.

## Why The Line Closes

The G1 contract authorized one implementation design and one focused repair
pass. It explicitly required `PROTOCOL_REJECTED` if the frozen contract remained
unexecutable after that pass. Adding gate recomputation, completion replay,
checkpoint/threshold/custody rebinding, and label-disjoint path guards would be
a second architecture and repair cycle, not completion of the authorized pass.

Stopping is therefore the evidence-correct result. Keeping G1 active would turn
a bounded, falsifiable Goal into an immortal research loop—the exact lifecycle
failure that the program rebase was designed to prevent.

## Artifact Disposition

- `scripts/analysis/gate_dev_candidate_metrics.py` and its focused test add a
  reusable, independently tested `>=20%` six-metric development-gate option and
  may be retained as analysis infrastructure. This is not a promotion claim.
- `scripts/analysis/manage_paired_blind_comparison.py` and its focused test are a
  rejected local draft with known fail-open behavior. They must not be staged,
  promoted, or used for blind evaluation.
- No checkpoint, prediction, generated evaluation output, or dataset payload is
  part of this decision.

## Successor Policy

There is no automatic successor Goal.

A future materially new universal mechanism requires a new Sol xhigh
architecture decision, new program charter, and new bounded Goal. Automatic
routing remains a separate unauthorized claim surface and requires independent
routing data, an unknown-domain policy, false-route cost, and rejection
behavior. Neither is a continuation of G1.

Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not reopen Candidate 5 specialist promotion by repairing or reusing the rejected paired manager under a renamed Goal.
Tested: 22 focused tests, 51 blind regression tests, py_compile, scoped git diff --check, Sol xhigh Architect and Critic audits
Not-tested: No real blind inference, label scoring, visual review, training, or product promotion was authorized or run
Related: docs/plans/2026-08-03-sustainable-multidomain-product-program.md
