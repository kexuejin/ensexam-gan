# External Text Layout Recovered Archive Publication Pass

## Decision

`PASS`. The single authorized archive publisher created exactly two canonical
relative symlinks and one atomic terminal result. It did not run a model or
change either source cache.

The primary link target is
`../../sign-separated-residual-repair-train275-primary-v1`; the second-stage
target is
`../../sign-separated-residual-repair-train275-frozen-pipeline-v1`. Both links
resolve to their registered real directories. Independent validation over the
exact 275-name manifest proves linked identities equal source identities:
primary metrics `efd58814...` with prediction content `6400c941...`, and
recovered second-stage metrics `79fd6127...` with prediction content
`2ffa40fc...`. No `.publishing` link or conflicting model process remains.

Historical `b800fdf3...` remains explicitly unreproduced. The original external
text-layout materialization plan still binds that historical hash and will
correctly reject the recovered archive. It must not be edited in place or run
yet.

## Next Action

Preregister a narrow recovered-input adapter for the existing materializer.
The adapter must preserve the original plan, detector, runtime, source,
manifest, feature, output, and acceptance contracts; bind this archive PASS;
and replace only second-stage metrics evidence validation with recovered
`79fd...` plus the exact `2ffa...` prediction population. Materialization
execution remains closed until adapter integration PASS is committed and
pushed.

Intent: Restore the archive interface while retaining an explicit recovered-input boundary for downstream materialization.
Constraint: Two canonical relative links reference immutable exact caches; historical b800 metadata remains unavailable.
Rejected: Edit the original materialization plan to expect 79fd | would overwrite historical provenance instead of registering a recovered overlay.
Rejected: Run the materializer against the recovered archive now | its current exact b800 evidence gate should and would reject.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Add only a hash-bound recovered-input adapter before materialization; keep every detector, source, feature, runtime, and diagnostic gate unchanged.
Tested: Single archive publication, canonical targets, exact source and linked cache identities over 275 names, atomic result, temporary absence, and model-process absence.
Not-tested: Recovered-input adapter, formal 275-page detector materialization, train275 diagnostic, quality evaluation, visual review, promotion, or reserved-blind access.
Related: outputs/external-text-layout-recovered-archive-publication-20260815/result.json
Related: docs/external-text-layout-recovered-archive-publication-verification-20260815.json
Related: docs/external-text-layout-support-prerequisite-v1.json
