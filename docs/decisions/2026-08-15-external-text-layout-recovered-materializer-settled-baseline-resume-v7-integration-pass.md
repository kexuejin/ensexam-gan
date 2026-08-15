# External Text Layout Recovered Materializer Settled-Baseline Resume V7 Integration Pass

## Decision

`PASS`. The unchanged launcher already implements the v7 policy. Its first
locked swap reading becomes the baseline, the 60-second readiness window
requires zero growth and at least `70%` free memory, and only growth from that
baseline is exposed to the materializer. Focused tests prove that a high stable
absolute value passes, launch-window growth rejects before materialization, and
runtime growth above `512 MiB` terminates the batch process group.

Focused tests pass `23/23` and complete external-layout tests pass `132/132`
under Python 3.13.1 and 3.10.11. Both runtimes compile the four v6 files. A
real model-free readiness window also passed from a `1,494,745,088`-byte
absolute swap baseline: all 61 samples reported zero growth, minimum free
memory was `83%`, post-window free memory was `85%`, and no Booted Simulator
or conflicting model process existed.

No production source changed. The retained eight NPZs and eight records, their
aggregate content and filename hashes, progress file, next source page, and
the `39d5...` derived plan remain exact. The existing schema-v6 launcher result
therefore remains the correct execution record; schema v7 governs only whether
a stable current host baseline may proceed.

Formal CPU-only materialization resume from `hw5k_1214.jpg` is now authorized
after this PASS is pushed. It must acquire and pass a new live readiness window
and keep every v6 batching, RSS, free-memory, swap-growth, timeout, ordering,
resume, and output gate active. No restart or absolute swap reclamation is
required. Diagnostic, quality, and training surfaces remain closed.

Intent: Restore continuous resumable execution using the safety semantics the launcher already enforces.
Constraint: Absolute swap can remain high after process cleanup without indicating new pressure.
Rejected: Change production code or result schema | existing implementation and schema-v6 evidence already express current-baseline-relative execution exactly.
Rejected: Skip the live readiness window | stability and free memory still prove that the current baseline is settled before each run.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Resume from a fresh stable current baseline; never wait for historical swap reclamation or require a host restart.
Tested: Dual-runtime 23-test focused and 132-test complete suites, dual-runtime compile, real 60-second readiness window, three named swap semantics, and exact retained state hashes.
Not-tested: Resumed page nine, remaining 267 pages, final materialization, diagnostic, or quality gates.
Related: docs/external-text-layout-recovered-materializer-settled-baseline-resume-v7-integration-verification-20260815.json
Related: docs/external-text-layout-recovered-materializer-settled-baseline-resume-v7.json
