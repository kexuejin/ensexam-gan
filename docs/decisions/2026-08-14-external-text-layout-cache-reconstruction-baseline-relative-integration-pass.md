# External Text Layout Cache Reconstruction Baseline-Relative Integration Pass

## Decision

`PASS` for cache-monitor integration. The reconstruction path now validates the
exact v2 detector result before helper import, captures one absolute swap
baseline per locked stage, and requires callers to pass an explicit relative
health reader into the existing one-second monitor. Absolute historical swap
is recorded but not capped; only nonnegative growth is supplied to the
unchanged resource enforcement and process-group termination path.

The `35%` free-memory floor, `10 GiB` process-tree RSS cap, `512 MiB` growth
cap, serial primary/second-stage ordering, host-user lock, exclusive log,
`start_new_session=True`, SIGTERM/SIGKILL escalation, historical commands,
temporary cache directory, metrics rewrite, expected output hashes, and atomic
publication all remain unchanged. Runtime and post-run evidence now names
`launch_swap_baseline_bytes`, `peak_swap_growth_bytes`, and
`swap_growth_bytes` explicitly.

## Verification

The focused cache suite passed `14/14` under Python `3.13.1` and `3.10.11`.
The complete `test_external_text_layout*.py` suite passed `71/71` under both
environments. Coverage includes high stable absolute swap, zero clamping,
growth above `512 MiB`, resource and health-reader failures, process-group
termination, SIGKILL escalation, nonzero child exit, atomic success, exact v2
probe schema/hash rejection, publication conflicts, and helper-import ordering.

The real static preflight passed under the frozen Python `3.10.11` runtime:
all `275` historical sources were present, the exact probe SHA matched, runtime
versions matched, and both build and archive destinations were absent. The
preflight correctly kept `execution_authorized=false`; no cache model process,
MPS task, cache output, formal materialization, quality evaluation, visual
review, candidate inference, promotion, or reserved-blind access occurred.

## Next Boundary

Run a point-in-time host readiness check under Python `3.10.11`. A primary
stage may execute only if free memory is at least `35%`, no conflicting model
process or cache/log path exists, and its locked baseline snapshot passes. The
second stage remains dependent on a hash-exact primary cache and must run
serially under a fresh stage baseline.

Intent: Reopen serial historical cache reconstruction without confusing retained macOS swap with task-added pressure.
Constraint: Cache outputs must reproduce exact historical hashes, and every failure must leave final cache publication absent.
Rejected: Keep an optional absolute-reader default | future callers could silently bypass the adapter.
Rejected: Run both cache stages as one unbounded subprocess | loses per-stage baseline, validation, and failure isolation.
Confidence: high for integration and static prerequisites; medium for real stage resource behavior and exact output reproduction.
Scope-risk: moderate
Reversibility: clean before cache execution.
Directive: Capture a fresh baseline for each stage, never reuse the detector baseline, and do not publish or continue after any resource or hash failure.
Tested: 14 focused and 71 complete tests in both Python environments, exact real probe validation, frozen runtime preflight, hashes, compilation, and diff checks.
Not-tested: Real primary/second-stage reconstruction, output hash reproduction, archive publication, formal materialization, quality gates, or promotion.
Related: docs/external-text-layout-cache-reconstruction-baseline-relative-swap-v1.json
Related: outputs/external-text-layout-runtime-safety-probe-tiled-9x9-20260814/result.json

Verification: docs/external-text-layout-cache-reconstruction-baseline-relative-integration-verification-20260814.json
Verification-SHA256: ce86ec854606592ec315e5177286bddbf8d1365f3ce243d1e7e134ce951183ac
