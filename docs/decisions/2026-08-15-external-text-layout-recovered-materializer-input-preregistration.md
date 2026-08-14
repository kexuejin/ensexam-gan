# External Text Layout Recovered Materializer Input Preregistration

## Decision

`PREREQUISITE_NEEDED`. Freeze one recovered-input adapter for the existing
external text-layout materializer. Only adapter implementation and synthetic
verification are authorized. Real 275-page detector materialization remains
closed until a separate integration PASS is committed and pushed.

The original plan remains byte-for-byte unchanged at SHA-256
`bf7b9c7d...`. Its historical second-stage metrics evidence remains
`b800fdf3...`, explicitly unavailable and unreproduced. The adapter may create
an in-memory effective plan with exactly one changed value:

```text
evidence.second_stage_metrics.sha256
  b800fdf385075bac46cc50db08a726dc2b9a6201b11a1229a164738b595a708d
  ->
  79fd61278e689a0003e37a5bdf20f856184b49c8fdb3af8ad9af03a3a13c451b
```

The evidence path remains the canonical recovered archive link. The adapter
must also validate the exact `2ffa40fc...` 275-file prediction population, the
archive publication PASS, and both canonical relative links before detector
creation. The primary `efd58814...` / `6400c941...` input remains unchanged.

Every detector file, package/runtime identity, threshold, tile repair,
train275 source and manifest, output transaction, health limit, single-process
rule, feature, ablation, target-access prohibition, and acceptance gate remains
the original plan value. Materialized manifests must bind both the original
plan hash and recovered overlay hash so resume and published validation cannot
silently switch provenance.

Programmatic calls without an overlay retain legacy behavior for existing
synthetic tests. The production CLI defaults to the exact registered overlay.
Any wrong overlay, plan, archive result, link, cache identity, ledger status,
or output state must reject before detector creation or output mutation.

Intent: Let the frozen materializer consume the audited recovered cache without rewriting the historical plan or widening any model or quality contract.
Constraint: Only one in-memory evidence hash may differ; every computational, data, runtime, output, and acceptance field remains frozen.
Rejected: Edit the original plan in place | would erase the b800 historical record and make prior decisions ambiguous.
Rejected: Skip second-stage metrics validation | would weaken provenance instead of substituting one audited exact identity.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Commit and push adapter integration PASS before real materialization; require manifest provenance for both original plan and overlay.
Tested: Original plan and recovered archive evidence hashes, exact canonical links, absent materialization state, and one-field overlay design.
Not-tested: Adapter implementation, detector creation, 275-page materialization, resume, train275 diagnostic, quality evaluation, visual review, promotion, or reserved-blind access.
Related: docs/external-text-layout-recovered-materializer-input-v1.json
Related: docs/external-text-layout-support-prerequisite-v1.json
Related: docs/external-text-layout-recovered-archive-publication-verification-20260815.json
