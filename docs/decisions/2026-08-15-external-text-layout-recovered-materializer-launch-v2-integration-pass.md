# External Text Layout Recovered Materializer Launcher V2 Integration Pass

## Decision

`PASS`. The separate v2 launcher adapts recovered cache provenance without
changing the probe-bound shared materializer or shared test. The rejected v1
in-place implementation was never committed and never started a detector.

The launcher-focused suite passed `9/9` and the complete external-text-layout
suite passed `118/118` under both Python 3.13.1 and Python 3.10.11. Shared hashes
remain exactly `2a87d2a...` and `79e64906...`, preserving every earlier safety
probe contract.

V2 validates the archive PASS, both canonical links, both exact cache
identities, original plan, v1 provenance overlay, and ledger before any write
or detector call. It deterministically builds the registered `39d5d801...`
derived plan, writes or reuses it atomically, and calls the unchanged
materializer only with that plan and one worker. Resume and final manifest
provenance naturally bind the derived plan hash. Final validation rehashes all
275 NPZ payloads before launcher PASS.

The real launcher probe before this ledger update returned
`PREREQUISITE_NEEDED`. Derived plan, final output, temporary output, marker,
cleanup state, and launcher result all remain absent. This PASS authorizes only
a subsequent host-health preflight and one resumable target-free CPU
materialization under the plan's Python 3.13.1 runtime. It does not authorize
the train-target diagnostic or any quality surface.

Intent: Enable recovered-input materialization while retaining the exact shared implementation already proven safe.
Constraint: Real execution uses Python 3.13.1, one isolated CPU detector page at a time, and all existing health gates.
Rejected: Change shared materializer source | complete regression proved its SHA is part of prior safety evidence.
Rejected: Run under the cache-reconstruction Python 3.10.11 environment | the materialization plan requires the registered Python 3.13.1 package identity.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Push this PASS, then perform host-health preflight before launcher execution; preserve resume state on any health rejection.
Tested: Dual-runtime 9-test focused and 118-test complete suites, exact shared hashes, deterministic plan derivation, archive/cache gates, atomic plan write, closed-authority rejection, unchanged materializer invocation, final page-hash validation, terminal idempotency, and result-gap recovery.
Not-tested: Real derived-plan write, detector creation, 275-page materialization, train275 diagnostic, quality evaluation, visual review, promotion, or reserved-blind access.
Related: docs/external-text-layout-recovered-materializer-launch-v2-integration-verification-20260815.json
Related: docs/external-text-layout-recovered-materializer-launch-v2.json
Related: docs/external-text-layout-tiled-9x9-one-page-safety-probe-v2.json
