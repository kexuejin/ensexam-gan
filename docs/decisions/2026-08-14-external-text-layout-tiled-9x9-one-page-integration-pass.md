# External Text Layout Tiled 9x9 One-Page Integration PASS

## Decision

`PASS` for repository integration only. The materializer now applies the
combined duplicate-upsample and four-row tiled `9x9` repair before PaddleOCR
construction. The one-page probe is bound to the exact preregistered contract,
support plan, `hw5k_1011.jpg` source, strict result path, and one-attempt
sentinel.

This PASS does not authorize detector execution under the current host state.
Free memory is `86%` and no iOS Simulator is Booted, but swap remains
`1,778.69 MiB`, above the frozen `512 MiB` launch maximum. The planned result
path remains absent.

## Safety Integration

The shared materialization defaults remain `10 GiB` process-tree RSS, `35%`
free memory, and `512 MiB` swap. Probe-specific overrides may only strengthen
those defaults and are fixed to:

```text
launch free memory minimum       70%
runtime free memory minimum      45%
process-tree RSS maximum         8 GiB
swap maximum                     512 MiB
page timeout                     15 minutes
Booted iOS Simulator count       0
thread caps                      OMP/OpenBLAS/MKL/vecLib = 1
```

The parent checks launch gates under the existing host-user lock. The spawned
page child checks Booted Simulator state again before detector import. A
`RUNNING` result is atomically persisted immediately before the page starts, so
a crash or host reboot cannot silently create a second attempt. Existing
results are never overwritten, and temporary-output cleanup failures are
terminal rather than ignored.

## Verification

The complete external-text-layout test surface passed `61/61` under both the
registered Python `3.13.1` detector environment and historical Python
`3.10.11` environment. Tests cover strict-versus-shared threshold behavior,
structured Simulator parsing, parent and child import ordering, high-swap
launch rejection, contract/source/result binding, `RUNNING` persistence,
non-overwrite behavior, resource termination, cleanup, and synthetic PASS
acceptance fields. No detector or model executed.

Full repository discovery also found five unrelated environment/artifact
modules that cannot pass in this worktree: two require the absent gitignored
monotonic checkpoint, one requires absent historical train275 archive metrics
and symlinks, and two require `torchvision`, which is not installed in the
Python 3.13 detector environment. None imports or exercises the modified
external-text-layout runtime path.

```text
verification:
  docs/external-text-layout-tiled-9x9-one-page-integration-verification-20260814.json
next result:
  outputs/external-text-layout-runtime-safety-probe-tiled-9x9-20260814/result.json
current execution state:
  disabled_swap_1778.69_MiB_exceeds_512_MiB
```

Intent: Make the only authorized tiled detector attempt fail closed before model import and non-repeatable after page start.
Constraint: The host swap gate remains unsatisfied, so integration can be verified only with static and synthetic execution.
Rejected: Reuse shared 10 GiB/35% limits for the probe | weaker than the preregistered one-page contract.
Rejected: Check Simulator state only in the parent | a Booted device could appear before child import.
Rejected: Write the result only after page completion | a host reboot could erase the attempt record and permit an unsafe retry.
Confidence: high for contract binding, launch gating, process ordering, and synthetic terminal behavior; medium for cleanup under host-level termination; low for real page completion and peak memory until the single attempt runs.
Scope-risk: moderate
Reversibility: clean
Directive: Do not invoke the probe until swap is at most 512 MiB and all other launch gates pass; after the result path exists, never rerun this exact tiled path.
Tested: 61 external-text-layout tests under Python 3.13.1 and 61 under Python 3.10.11; live Booted Simulator parser returned zero; Python compile and git diff checks passed.
Not-tested: Real detector import, full-map tiled convolution, one-page completion, host peak memory, frozen-cache reconstruction, formal materialization, training, quality gates, visual review, or promotion.
Related: docs/decisions/2026-08-14-external-text-layout-tiled-9x9-one-page-safety-probe-preregistration.md
