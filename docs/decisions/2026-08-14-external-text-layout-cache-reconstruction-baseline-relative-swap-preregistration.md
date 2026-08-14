# External Text Layout Cache Reconstruction Baseline-Relative Swap Preregistration

## Decision

`PREREQUISITE_NEEDED`. Freeze a narrow adapter over the existing v2 cache
handoff and v1 process monitor. The adapter binds the exact successful v2 probe
result and changes only the interpretation of swap during each locked cache
stage: historical absolute usage becomes a recorded launch baseline, while the
unchanged `512 MiB` numeric limit applies to growth after that baseline.

The one-page probe proved that this host can complete the tiled detector with
`6.92 GiB` peak process-tree RSS, `63%` minimum free memory, and zero swap
growth even when absolute launch swap is about `1.03 GiB`. Cache reconstruction
must not return to the same permanently closed absolute-swap gate, but it also
must not run without swap-growth protection.

## Frozen Stage Semantics

For each of the serial `primary` and `second_stage` stages, inside the existing
exclusive host-user lock:

1. Validate the exact v2 probe result and all frozen historical inputs before
   importing the historical helper.
2. Check zero conflicting model processes, at least `35%` free memory, and at
   most `10 GiB` process-tree RSS.
3. Capture absolute swap as `launch_swap_baseline_bytes`; do not apply an
   absolute maximum to this historical value.
4. Wrap every child health sample as
   `max(0, absolute_swap - launch_swap_baseline_bytes)`.
5. Terminate the existing child process group if growth exceeds `512 MiB`,
   free memory falls below `35%`, RSS exceeds `10 GiB`, or health reading
   fails.
6. Record explicit baseline, peak-growth, and post-run growth evidence.

The existing one-second monitor, `start_new_session=True`, SIGTERM/SIGKILL
termination, one-model-process-at-a-time ordering, exclusive logs, temporary
cache suffix, zero-exit requirement, metrics path rewrite, exact expected
hashes, and atomic publication remain unchanged.

## Integration Boundary

Only the reconstruction script and its existing test file may change. Shared
runtime, historical helper, base reconstruction contract, v1 monitor contract,
commands, paths, output hashes, and dependencies are frozen. Integration and
synthetic tests are authorized; primary/second-stage cache execution, formal
materialization, quality evaluation, visual review, candidate inference,
promotion, and reserved-blind access remain closed.

After both registered Python environments and the complete external-text-
layout suite pass, a separate static readiness check may authorize serial
cache execution. No stage may be retried after creating its exclusive log or a
terminal non-PASS record without new authority.

Intent: Let cache stages prove incremental memory pressure without removing swap-growth termination.
Constraint: The exact probe PASS supersedes the old probe schema, while base cache paths, outputs, and historical runtime must remain unchanged.
Rejected: Edit the frozen v2 handoff and v1 monitor contracts in place | would erase the historical decision boundary.
Rejected: Remove swap validation from cache stages | would lose protection during two long-running model processes.
Rejected: Reuse the probe's 70%/8 GiB limits | the already frozen cache stage contract requires 35%/10 GiB and exact historical commands.
Confidence: high for the adapter boundary; medium for real cache resource behavior until serial stages run.
Scope-risk: moderate
Reversibility: clean before cache execution.
Directive: Use one absolute baseline per locked stage and pass only nonnegative growth to the existing monitor; never reinterpret the exact probe result as an old v1 schema.
Tested: Exact probe result/hash, existing monitor termination coverage, frozen source hashes, and absent reconstructed cache destinations.
Not-tested: Adapter implementation, real cache stages, output hash reproduction, publication, formal materialization, quality gates, or promotion.
Related: docs/external-text-layout-cache-reconstruction-baseline-relative-swap-v1.json
Related: outputs/external-text-layout-runtime-safety-probe-tiled-9x9-20260814/result.json

Contract: docs/external-text-layout-cache-reconstruction-baseline-relative-swap-v1.json
Contract-SHA256: a7cafb5358370585da926a99ad7d844a2cb9b5ec676dbead4803f53e35a09b12
