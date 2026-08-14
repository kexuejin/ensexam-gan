# External Text Layout Recovered Materializer Launcher V2 Preregistration

## Decision

`PREREQUISITE_NEEDED`. Supersede the unintegrated v1 in-place adapter with a
separate launcher and persisted derived plan. Only launcher implementation and
synthetic verification are authorized. Real detector materialization remains
closed until a separate v2 integration PASS is committed and pushed.

The v1 design was rejected by the complete external-layout suite before commit:
changing the shared materializer changed SHA-256 `2a87d2a...` and correctly
failed the already-passed tiled safety-probe contract. Both shared source and
test files were restored exactly; no v1 implementation was committed or used
for detector execution.

V2 never changes the shared materializer. After all evidence and execution
gates pass, a separate launcher constructs a derived plan in memory, requires
candidate SHA-256 `39d5d801...`, and atomically persists it at:

```text
outputs/external-text-layout-recovered-materializer-input-20260815/effective-plan.json
```

The derived plan changes the second-stage metrics evidence hash from historical
`b800fdf3...` to audited recovered `79fd6127...` and adds explicit provenance
for the original plan, v1 overlay, archive publication result, historical hash,
and exact `2ffa40fc...` 275-file prediction identity. All original plan fields
used by the materializer remain unchanged. The existing materializer naturally
binds the derived plan hash in both resume `progress.json` and final
`manifest.json`.

The launcher must validate the v2 contract, archive PASS, canonical links,
recovered cache, ledger, and derived candidate before writing the plan or
calling the materializer. It invokes the unchanged materializer with exactly
the derived plan and one worker. An exact existing derived plan may resume; an
unknown plan fails without detector or materialization mutation.

Intent: Adapt recovered input provenance without invalidating the shared materializer hash already proven by the runtime safety chain.
Constraint: Shared materializer and tests are immutable; only a hash-bound derived plan and separate launcher may be added.
Rejected: Update every prior probe contract to the modified materializer hash | would rewrite validated safety evidence for an input-only change.
Rejected: Continue the v1 in-memory patch | complete regression already proved it violates shared source identity.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Keep shared source hashes exact; commit and push v2 integration PASS before launcher execution creates the derived plan or starts detector work.
Tested: Complete-suite v1 rejection, exact shared-file restoration, deterministic derived-plan candidate hash, archive PASS, canonical links, and absent materialization state.
Not-tested: V2 launcher implementation, derived-plan write, detector execution, resume, 275-page materialization, train275 diagnostic, or quality surfaces.
Related: docs/external-text-layout-recovered-materializer-launch-v2.json
Related: docs/external-text-layout-recovered-materializer-input-v1.json
Related: docs/external-text-layout-tiled-9x9-one-page-safety-probe-v2.json
