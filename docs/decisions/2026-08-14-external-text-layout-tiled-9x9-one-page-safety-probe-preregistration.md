# External Text Layout Tiled 9x9 One-Page Safety Probe Preregistration

## Decision

`PREREQUISITE_NEEDED`. Freeze exactly one target-free `hw5k_1011.jpg` runtime
safety probe for the bitwise-verified four-row tiled `9x9` successor. This
record authorizes bounded repository integration now, but model execution only
after every host launch gate passes.

At registration time the host has `86%` free memory but `1,810.69 MiB` swap
used, above the unchanged `512 MiB` launch maximum. Execution is therefore
disabled. No detector import or page run may occur while this condition holds.

## Frozen Integration

Integration may only:

1. Replace the duplicate-upsample-only repair import and call with the combined
   tiled `9x9` repair before `TextDetection` construction.
2. Add probe-specific limits of at least `45%` free memory and at most `8 GiB`
   process-tree RSS without weakening shared defaults.
3. Reject any Booted iOS Simulator before detector import.
4. Bind the new contract hash and result path.

The detector model, weights, CPU Transformers engine, float32 dtype, source
page, preprocessing geometry, thresholds, batch size, output ordering,
postprocessing, and materialization semantics remain unchanged. No global
site-package file may be modified.

## Launch And Runtime Gates

```text
launch free memory minimum                 70%
launch and runtime swap maximum            512 MiB
Booted iOS Simulator count                 0
conflicting model process count            0
result path                                absent
runtime free memory minimum                45%
runtime process-tree RSS maximum           8 GiB
thread caps                                OMP/OpenBLAS/MKL/vecLib = 1
page timeout                               15 minutes
attempt count                              exactly 1
```

The existing host-user lock, spawned process group, one-second monitoring,
active termination, temporary-output cleanup, and zero-residual-process checks
remain mandatory. These limits strengthen the earlier runtime floor and RSS
cap; they do not relax any safeguard.

## Terminal Rules

`PASS` requires a completed page, reason code `runtime_safety_probe_passed`, no
formal output, no target/label/recognition/routing access, no retained temporary
page output, and zero residual model processes. PASS may authorize only serial
frozen-cache reconstruction under its existing hash-bound contract.

Any `PREREQUISITE_NEEDED`, resource breach, timeout, crash, output drift, or
cleanup failure closes this exact tiled probe path. Clearing swap and repeating
the same run is prohibited.

## Registered Surface

```text
contract:
  docs/external-text-layout-tiled-9x9-one-page-safety-probe-v1.json
  sha256=1fd02d49250150f85ce190601b21b36d60a308ef92b07e564c8a21575124aee4
implementation commit:
  c8c228d6adb6edf6fe11228be2daca632fac4bbd
repair:
  scripts/analysis/external_text_layout_tiled_9x9_runtime_repair.py
  sha256=07b894a5a43b5331c5ba866dd2a714e6346723c8112f0b23bebfb56852f839ac
repair test:
  tests/test_external_text_layout_tiled_9x9_runtime_repair.py
  sha256=01ef4e03ff41bbb440a7a028e05e07a4c6f2aa1dde049f124a362129b0b37b39
planned result:
  outputs/external-text-layout-runtime-safety-probe-tiled-9x9-20260814/result.json
```

Intent: Measure the only materially lower-memory equivalent detector path under stricter host safeguards.
Constraint: The prior clean-baseline path crossed memory limits, while the tiled successor has only synthetic bitwise evidence and the current host swap gate fails.
Rejected: Run while swap is high because current free memory is 86% | historical swap and launch reproducibility remain outside the frozen gate.
Rejected: Reuse the prior failed result path | would blur distinct repair identities and attempt counts.
Rejected: Increase tile rows to reduce the 406-call overhead | changes the verified memory bound and fake-feature surface.
Confidence: high for integration boundaries and safeguards; medium for full-map value identity; low for completion time and host peak memory until the one page runs.
Scope-risk: moderate
Reversibility: clean
Directive: Integrate only the exact tiled repair now; do not import or run the detector until swap is at most 512 MiB and every launch gate passes.
Tested: Static feasibility, four bitwise CPU float32 fake cases, exact AST replacement, drift rejection, and idempotence without real detector import.
Not-tested: Integrated detector construction, full-map tiled convolution, one-page completion, timeout, peak memory, cache reconstruction, formal materialization, training, quality gates, or promotion.
Related: docs/decisions/2026-08-14-external-text-layout-tiled-9x9-runtime-repair-verification-pass.md
