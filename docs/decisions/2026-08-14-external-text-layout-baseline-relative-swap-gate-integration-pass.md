# External Text Layout Baseline-Relative Swap Gate Integration Pass

## Decision

`PASS` for probe integration only. The one-page wrapper now captures absolute
swap inside the exclusive host-user lock, requires the preregistered `60s`
zero-growth launch window, and supplies the unchanged child monitor with
nonnegative swap growth from that baseline. The existing `512 MiB` numeric
limit, `45%` runtime free-memory floor, `8 GiB` process-tree RSS cap, spawned
process group, termination path, thread caps, timeout, and Simulator checks
remain active.

Readiness failures occur before `RUNNING` and leave the unique result path
absent with `attempt_count=0`. A stable synthetic `2 GiB` historical baseline
reached the page runner, while a one-byte launch-window increase failed before
the runner. A synthetic runtime increase above `512 MiB` invoked the existing
monitor termination path. PASS evidence records
`launch_swap_baseline_bytes`, `peak_swap_growth_bytes`, and renamed page and
failure health fields rather than presenting relative values as absolute swap.

## Verification

The focused probe suite passed `12/12` under Python `3.13.1` and `3.10.11`.
The complete `test_external_text_layout*.py` suite passed `70/70` under both
environments. Compile checks and `git diff --check` also passed.

The shared runtime and materializer retain their preregistered hashes. No
detector, MPS task, cache reconstruction, formal materialization, quality
evaluation, visual review, candidate inference, promotion, or reserved-blind
access occurred. The unique result path remains absent and its model attempt
is unconsumed.

## Next Boundary

Run a point-in-time static host check without detector execution. If the `70%`
free-memory, thread-cap, result-absence, zero-Simulator, and zero-conflicting-
process gates pass, the unique probe may start. Its own locked stability window
must still pass before it writes `RUNNING` or creates the detector child.

Cache reconstruction remains closed even after a probe PASS until its existing
monitor is adapted and verified against the same baseline-relative semantics.

Intent: Reopen bounded detector progress while preserving incremental memory-pressure protection and one-attempt authority.
Constraint: Persistent macOS swap is historical state, but new swap growth and low free memory remain unsafe.
Rejected: Change the shared runtime API | the probe-local process wrapper supplies the relative reader without widening shared behavior.
Rejected: Write a readiness failure result | would consume or permanently close the one-shot path before model execution.
Confidence: high for integration behavior and test evidence; medium for real detector completion until the unique page runs.
Scope-risk: moderate
Reversibility: clean
Directive: Keep all swap values passed to the shared monitor relative to `launch_swap_baseline_bytes`; adapt cache reconstruction separately only after probe PASS.
Tested: 12 focused and 70 complete external-text-layout tests in Python 3.13.1 and 3.10.11, compilation, source hashes, result absence, and diff checks.
Not-tested: Real detector execution, real 60-second readiness behavior, cache reconstruction, formal materialization, quality gates, visual review, promotion, or reserved-blind access.
Related: docs/external-text-layout-baseline-relative-swap-gate-integration-verification-20260814.json
Related: docs/external-text-layout-tiled-9x9-one-page-safety-probe-v2.json

Verification: docs/external-text-layout-baseline-relative-swap-gate-integration-verification-20260814.json
Verification-SHA256: 6177e4e6e8642cc0c4abdf86f9fd93adff0b6b4eea8b35937fa3135b435b02a8
