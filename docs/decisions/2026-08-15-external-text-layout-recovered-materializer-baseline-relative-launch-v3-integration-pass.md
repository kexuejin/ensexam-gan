# External Text Layout Recovered Materializer Baseline-Relative Launcher V3 Integration Pass

## Decision

`PASS`. The launcher now reuses the exact proven tiled-probe readiness and
baseline-relative reader while leaving the shared materializer, runtime,
probe, and shared test byte-for-byte unchanged. Focused tests pass `13/13` and
the complete external-layout suite passes `122/122` under both registered
Python runtimes.

The launcher owns the existing exclusive host-user lock from the first swap
baseline sample through final post-run health validation. It requires 60
seconds of zero swap growth at at least `70%` free memory, then adapts only
process-local shared call surfaces inside `try/finally`. The unchanged page
monitor receives swap growth, retains the `512 MiB` termination cap, and now
uses the stronger proven `8 GiB` RSS and `45%` runtime free-memory limits. Each
isolated child rechecks that no iOS Simulator is Booted before detector import.
All adapted call surfaces restore after success and resource failure.

The correct runtime is the existing project Python 3.13 environment, which
matches every registered package including OpenCV `4.13.0`. Its directory name
contains `mps`, but the frozen detector plan is explicitly `device=cpu`; this
PASS does not authorize MPS. The bare pyenv Python remains rejected because it
contains OpenCV `4.10.0`.

A real pre-integration command returned `PREREQUISITE_NEEDED` because v3
authority was still closed. It created no derived plan, detector output,
temporary state, marker, cleanup state, or launcher result. One real resumable
CPU materialization is authorized only after this PASS is committed and
pushed.

Intent: Resume safe materialization on a healthy host without treating retained macOS swap history as new detector pressure.
Constraint: Prior probes bind shared source hashes and require CPU-only, one-page isolation with process-group termination.
Rejected: Increase the old absolute swap threshold | any absolute threshold still confuses historical occupancy with current growth.
Rejected: Disable swap checks | growth above 512 MiB remains a hard runtime failure.
Rejected: Use bare pyenv Python | its OpenCV identity does not match the frozen plan.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Run only with the exact project Python 3.13 executable; preserve temporary page records after any interruption and never convert this into MPS or multi-worker execution.
Tested: Dual-runtime 13-test focused and 122-test complete suites, compile and diff checks, high stable swap, launch growth rejection, runtime growth termination, child Simulator flag, process-local restoration, malformed evidence rejection, exact shared hashes, and a closed-authority real probe.
Not-tested: Real 60-second readiness, detector import, 275-page materialization, downstream audit, quality evaluation, visual review, promotion, or reserved-blind access.
Related: docs/external-text-layout-recovered-materializer-baseline-relative-launch-v3-integration-verification-20260815.json
Related: docs/external-text-layout-recovered-materializer-baseline-relative-launch-v3.json
Related: docs/external-text-layout-tiled-9x9-one-page-safety-probe-v2.json
