# External Text Layout Cache Reconstruction Runtime Monitor Preregistration

## Decision

`PREREQUISITE_NEEDED`. Freeze a bounded runtime-monitor integration before any
historical primary or second-stage cache reconstruction. The current v2
reconstruction path checks resource health immediately before and after each
stage, but delegates the full child lifetime to an unbounded
`subprocess.run`. A one-page detector PASS does not prove that either 275-page
historical cache stage will remain below the reconstruction RSS, free-memory,
and swap limits.

This record authorizes implementation and synthetic verification only. It does
not authorize the tiled detector probe, cache reconstruction, formal external
layout materialization, training, candidate inference, quality evaluation,
visual review, promotion, or reserved-blind access.

## Frozen Monitor

The reconstruction wrapper will preserve the historical command builders and
helper hash while replacing only its unmonitored atomic-command execution:

```text
child launch                    subprocess.Popen(start_new_session=True)
monitor root                    child pid and all descendants
monitor interval                1.0 second
maximum process-tree RSS        10 GiB
minimum system free memory      35%
maximum swap used               512 MiB
resource/reader failure         terminate child process group
termination                     SIGTERM, wait 5 seconds, then SIGKILL
successful publication          zero exit + expected temporary directory only
metrics path rewrite            existing historical helper function
```

The host-user lock remains outside the complete monitored stage. The final
cache directory must remain absent until a zero-exit child has produced the
registered temporary directory. Nonzero exit, resource-limit failure,
health-reader failure, missing temporary output, or cleanup ambiguity must not
publish a final cache.

## Preserved Boundaries

The historical helper, v2 contract, cache paths, exact metrics and prediction
hashes, Python 3.10.11 runtime identity, probe PASS requirement, forbidden data
access, and relative archive publication rules remain unchanged. No new
dependency or site-packages mutation is allowed. Synthetic tests must cover a
successful monitored command, nonzero exit, resource-limit termination,
health-reader failure termination, process-group escalation, and absence of
partial final publication.

## Current Host Gate

At registration, swap remains about `1,410.62 MiB`, above the unchanged
`512 MiB` gate; free memory is `86%`, no tiled result exists, and no model or
cache process has run. Runtime-monitor integration can proceed without model
execution, but the single detector attempt remains closed.

Intent: Prevent a long historical cache stage from escaping the resource limits after a safe launch check.
Constraint: Preserve the hash-bound historical helper and all cache content/publication semantics.
Rejected: Continue using subprocess.run with pre/post checks | cannot stop a child at the point a runtime limit is crossed.
Rejected: Modify the historical helper | would invalidate the frozen helper artifact and blur historical reconstruction provenance.
Rejected: Treat the one-page detector probe as cache-stage safety evidence | it covers a different model, process lifetime, and workload.
Confidence: high for the monitoring gap and bounded wrapper design; medium for live stage behavior until the registered caches are actually reconstructed.
Scope-risk: moderate
Reversibility: clean
Directive: Do not execute primary or second-stage cache reconstruction until the monitored wrapper passes synthetic tests and is hash-registered.
Tested: Read-only audit of v2 reconstruction launch/post checks, the historical helper subprocess.run path, shared process-tree health reader, resource-limit enforcement, and process-group termination helpers.
Not-tested: Runtime-monitor implementation, child termination behavior, tiled detector execution, historical cache reconstruction, quality evaluation, visual review, or promotion.
Related: docs/external-text-layout-tiled-probe-cache-reconstruction-v2.json
Related: scripts/analysis/reconstruct_external_text_layout_frozen_caches.py
