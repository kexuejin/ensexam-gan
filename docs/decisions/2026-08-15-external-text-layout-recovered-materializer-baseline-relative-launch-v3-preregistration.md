# External Text Layout Recovered Materializer Baseline-Relative Launcher V3 Preregistration

## Decision

`PREREQUISITE_NEEDED`. The v2 launcher integration is valid, committed, and
pushed, but its first live read-only preflight exposed two runtime facts. The
bare pyenv Python has OpenCV `4.10.0` instead of the registered `4.13.0`, while
the existing project Python 3.13 environment matches every registered package.
The matching host is otherwise idle and healthy at `84%` free memory, zero
Booted iOS Simulators, and zero conflicting model processes, but macOS retains
about `970 MiB` of historical swap. The unchanged shared materializer still
applies its old absolute `512 MiB` gate and therefore cannot start.

V3 may adapt only the separate launcher. It must reuse the already proven
baseline-relative reader and launch-stability semantics from the exact tiled
probe while preserving the shared materializer, shared runtime, shared test,
and probe source hashes. No site-package change or MPS execution is allowed.

## Locked Runtime Boundary

The launcher must hold the existing host-user lock before it captures the
absolute swap baseline. It must then require a 60-second zero-growth window at
at least `70%` free memory, with no conflicting model process and no Booted
iOS Simulator before and after the window. During materialization it must
interpret the unchanged shared monitor's swap field as growth from that
baseline, retain the `512 MiB` growth cap, tighten runtime free memory to
`45%`, tighten detector process-tree RSS to `8 GiB`, and retain process-group
termination and the 15-minute page timeout.

The launcher may temporarily adapt only process-local shared call surfaces
while it holds the lock. Every surface must be restored in `finally` on both
success and failure. Each isolated CPU page child must recheck the Simulator
gate before detector import. A crash or resource rejection must retain exact
completed page records for the existing resume path.

## Execution Boundary

This preregistration authorizes changes only to the launcher and its focused
test. Real detector creation remains closed until dual-runtime focused and
complete external-layout suites pass, exact shared hashes are reverified, an
integration PASS is recorded, and that PASS is pushed. The subsequent real
run must use:

```text
/Volumes/Tool/source/ensexam-gan/.venv-py313-mps/bin/python3.13
```

The environment name does not authorize MPS; the frozen plan remains CPU-only
and one page at a time.

Intent: Allow a healthy host to make progress without misclassifying retained macOS swap as current detector pressure.
Constraint: Prior safety evidence binds the shared materializer, runtime, probe, and test hashes, so the adapter must remain launcher-local and reversible.
Rejected: Run v2 despite the gate | it would write the derived plan and fail before useful work.
Rejected: Modify or reinstall the bare pyenv environment | the exact registered project environment already exists and no site-package mutation is authorized.
Rejected: Remove swap monitoring | runtime growth still directly indicates pressure and must terminate the page process above 512 MiB.
Rejected: Modify the shared materializer or runtime | both are bound into prior probe and integration evidence.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Treat only growth after the locked baseline as task-attributable swap; preserve the 512 MiB runtime cap and all stronger probe limits.
Tested: Live read-only package identity, derived-plan hash, archive identities, free memory, process RSS, Simulator absence, conflict-process absence, and current absolute swap reading.
Not-tested: V3 implementation, in-memory adapter restoration, 60-second locked readiness, detector creation, 275-page materialization, diagnostic, quality evaluation, or promotion.
Related: docs/external-text-layout-recovered-materializer-baseline-relative-launch-v3.json
Related: docs/external-text-layout-recovered-materializer-launch-v2.json
Related: docs/external-text-layout-tiled-9x9-one-page-safety-probe-v2.json
