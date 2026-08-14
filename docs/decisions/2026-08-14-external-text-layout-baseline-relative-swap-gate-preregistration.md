# External Text Layout Baseline-Relative Swap Gate Preregistration

## Decision

`PREREQUISITE_NEEDED`. Replace the unconsumed tiled probe's absolute swap-used
launch limit with a hash-bound baseline-relative limit. This is a materially
new safety contract, not an exception to the existing protection: the probe
must still keep system free memory above `70%` before launch and `45%` during
execution, keep detector process-tree RSS at or below `8 GiB`, and terminate
the existing process group when swap grows by more than `512 MiB` after the
locked launch baseline.

macOS retained `1,362.62 MiB` of old swap while reporting `87%` free memory,
zero Booted Simulators, and zero conflicting model processes. Safe cleanup did
not materially reduce that value, and the one probe attempt remains
unconsumed. Absolute swap occupancy therefore does not identify incremental
pressure caused by this task and creates a permanent gate even on an otherwise
idle host.

## Frozen Readiness Window

Inside the existing exclusive host-user lock, readiness must proceed in this
order:

1. Verify thread caps, result absence, zero conflicting model processes, and
   zero Booted iOS Simulators.
2. Capture absolute swap usage as `launch_swap_baseline_bytes` while free
   memory is at least `70%`.
3. Sample health every `1s` for `60s`. Every sample must retain at least `70%`
   free memory and must not exceed the first sample's absolute swap usage.
4. Recheck result absence, model-process absence, and Simulator absence.
5. Only then write `RUNNING` with `attempt_count=1` and start the page process.

Any failure before step 5 returns `PREREQUISITE_NEEDED` without creating the
unique result path. Such readiness checks may be repeated because they neither
import the detector execution path nor consume the model attempt.

## Runtime Interpretation

The probe wrapper captures absolute swap once and adapts health samples passed
to the existing monitor:

```text
swap_used_bytes = max(0, current_absolute_swap - launch_swap_baseline_bytes)
```

The unchanged shared monitor can then enforce its registered `512 MiB` numeric
limit as a swap-growth cap. Probe evidence must name both the absolute launch
baseline and peak growth explicitly; an absolute post-run swap value must not
be presented as probe-attributable growth. Shared runtime, shared materializer,
process-group termination, detector identity, source page, output geometry,
and temporary-output cleanup remain frozen.

## Execution Boundary

This preregistration authorizes edits and synthetic verification only. It does
not authorize detector execution, MPS use, cache reconstruction, formal
materialization, quality evaluation, visual review, candidate inference,
promotion, or reserved-blind access. The unique detector probe may execute only
after the v2 implementation and both registered Python environments pass their
focused and external-text-layout test suites, followed by a passing static
readiness check.

On probe `PASS`, cache reconstruction still remains closed until its monitor is
adapted and separately verified against the same baseline-relative semantics.

## Registered Surface

```text
contract:
  docs/external-text-layout-tiled-9x9-one-page-safety-probe-v2.json
  sha256=2fb92aa625e0409fd7ed9db301d854333ca0852d714a8ed5fa8dcfc20e3527f6
probe before integration:
  scripts/analysis/probe_external_text_layout_runtime_safety.py
  sha256=8b6c563a7e9f5d879cc962b35fcc6ede15ce341a5f790fc49233cda9d235b43e
probe test before integration:
  tests/test_external_text_layout_runtime_safety_probe.py
  sha256=9f075754889dd637432f9beac93c4fe68dd3d96f049b2497dec454f8f4573545
shared materializer, unchanged:
  scripts/analysis/materialize_external_text_layout_support_train_only.py
  sha256=2a87d2a21b9141c9ca16e5f11f7ab1f523d59ffacf6f55759517fd0db26aafcf
shared runtime, unchanged:
  scripts/analysis/external_text_layout_materialization_runtime.py
  sha256=47d3bda97e0c6f100ed556d7260b1467fc0a236e6391e7414cb6aa932dd9d0d4
planned result:
  outputs/external-text-layout-runtime-safety-probe-tiled-9x9-20260814/result.json
```

Intent: Measure incremental probe pressure without treating persistent macOS swap history as current task load.
Constraint: The host has high free memory but retained old swap after safe cleanup, and the unique model attempt is still unconsumed.
Rejected: Remove swap protection | would lose direct protection against runtime paging growth.
Rejected: Use free memory alone | does not detect rapid swap growth while the detector runs.
Rejected: Require repeated restarts until absolute swap is below 512 MiB | macOS may retain swap independently of current pressure and keep the task permanently blocked.
Rejected: Change the shared runtime first | the probe wrapper can supply relative samples without widening the initial implementation surface.
Confidence: high for the measurement correction and preserved termination path; medium for host and detector behavior until the one page runs.
Scope-risk: moderate
Reversibility: clean
Directive: Treat the 512 MiB field as growth from the captured baseline only; do not remove it or silently return to absolute swap occupancy.
Tested: Historical host readings, process and Simulator absence, unchanged result-path absence, and frozen source hashes.
Not-tested: v2 implementation, stability-window rejection, runtime termination, detector execution, cache reconstruction, formal materialization, quality gates, or promotion.
Related: docs/external-text-layout-tiled-9x9-one-page-safety-probe-v2.json
Related: docs/external-text-layout-host-readiness-recovery-20260814.json
