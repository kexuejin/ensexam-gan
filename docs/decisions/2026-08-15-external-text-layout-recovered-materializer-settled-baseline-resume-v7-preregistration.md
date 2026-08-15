# External Text Layout Recovered Materializer Settled-Baseline Resume V7 Preregistration

## Decision

`PREREQUISITE_NEEDED`. Continuous development must not depend on macOS
returning swap to an earlier historical value or on restarting the host. Swap
is a lagging host-level value that may remain allocated after the responsible
process exits; requiring historical attribution or reclamation does not prove
that a new run is unsafe. It instead creates an unbounded external prerequisite
that the materializer cannot control.

V7 makes the first locked absolute swap reading at the start of the existing
readiness window authoritative, regardless of its absolute value or history.
The launcher must still observe `60s` with at least `70%` free memory and zero
swap growth before model execution. During execution it must terminate the
single process group if swap grows more than `512 MiB` from that locked
baseline, process-tree RSS exceeds `10 GiB`, or free memory falls below `35%`.
Fixed CPU-only batches of at most eight serial pages, the `0.25s` monitor,
`900s` batch timeout, exact source order, atomic page progress, and all retained
resume hashes remain unchanged.

The current launcher already appears to implement this settled-baseline
policy. V7 therefore authorizes policy verification and evidence recording,
not a production source change. Real resume remains closed until the existing
high-stable-swap test and all focused and complete external-layout suites pass
under both registered Python runtimes and a separate v7 integration PASS is
pushed.

Intent: Remove an unbounded host-restart dependency from the resumable quality loop without weakening run-local memory protection.
Constraint: macOS can retain swap after model processes exit even when free memory is healthy and swap is no longer growing.
Rejected: Require swap to return to the pre-v5 value | the historical value is not controllable and does not measure risk added by the next run.
Rejected: Require a host restart | restarting is operational recovery, not a reproducible quality or safety gate.
Rejected: Remove the swap guard | run-local growth remains a useful pressure signal and stays capped at 512 MiB.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Never block this materializer on absolute or historically attributed swap; gate on a stable locked launch baseline and bounded run-local growth.
Tested: V6 integration already includes high-stable-swap and run-local growth paths; v7-specific dual-runtime verification is pending.
Not-tested: V7 evidence record, resumed page nine, remaining pages, final materialization, diagnostic, or quality gates.
Related: docs/external-text-layout-recovered-materializer-settled-baseline-resume-v7.json
