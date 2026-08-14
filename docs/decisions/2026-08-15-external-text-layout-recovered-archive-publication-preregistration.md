# External Text Layout Recovered Archive Publication Preregistration

## Decision

`PREREQUISITE_NEEDED`. Freeze a separate relative-link publication path for the
exact primary cache and the semantically audited recovered second-stage cache.
Only implementation and synthetic verification are authorized. Real archive
link creation remains closed until a separate integration PASS is committed
and pushed.

The primary identity remains metrics `efd58814...`, prediction content
`6400c941...`, 275 files, and filename aggregate `8c75e1db...`. The recovered
second-stage identity remains metrics `79fd6127...`, prediction content
`2ffa40fc...`, 275 files, and the same filename aggregate. The unavailable
historical second-stage metrics identity `b800fdf3...` remains recorded as
`not_reproduced` and is not changed in the original reconstruction contract.

Publication may create only these relative symlinks:

```text
outputs/archive/sign-separated-residual-repair-20260810/train275-primary
  -> outputs/sign-separated-residual-repair-train275-primary-v1
outputs/archive/sign-separated-residual-repair-20260810/train275-frozen-pipeline
  -> outputs/sign-separated-residual-repair-train275-frozen-pipeline-v1
```

Both real cache directories and the exact 275-name manifest must pass before
any link mutation. Both final and `.publishing` paths are preflighted before
creation. Correct partial relative links or exact temporary links may resume;
absolute, broken, stale, or conflicting paths fail without mutation. Failure
to create the second link or write the result removes only links created by the
current attempt and never changes either cache directory.

The implementation must not call the original `publish_caches()` route because
that route remains correctly bound to historical `b800...`. It may reuse the
read-only cache validators. Model execution, historical helper loading, cache
mutation, formal materialization, train275 diagnostic, and every quality or
promotion surface remain closed.

Intent: Restore the frozen archive interface while preserving the explicit boundary between historical and recovered second-stage metadata identities.
Constraint: Archive publication may create only two relative symlinks after exact validation of both immutable cache directories.
Rejected: Change the original controller expected hash | would rewrite historical provenance and silently broaden an existing execution route.
Rejected: Copy cache directories into the archive | would duplicate large payloads and create divergent ownership.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Commit and push integration PASS before real link creation; never mutate either cache or accept an absolute, broken, or wrong-target link.
Tested: Exact cache/result evidence, absent archive paths, original relative-link semantics, and preregistered crash-recovery design.
Not-tested: Publisher implementation, synthetic failures, real archive links, formal materialization, train275 diagnostic, quality evaluation, visual review, promotion, or reserved-blind access.
Related: docs/external-text-layout-recovered-archive-publication-v1.json
Related: docs/external-text-layout-second-stage-recovered-cache-publication-verification-20260815.json
Related: docs/external-text-layout-frozen-cache-reconstruction-v1.json
