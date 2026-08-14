# External Text Layout Second-Stage Cache Reconstruction Prerequisite

## Decision

`PREREQUISITE_NEEDED`. The frozen second-stage command completed all 275 pages
without a resource-limit termination, but the temporary cache did not pass the
opaque historical metrics hash. The corrected publication order kept the final
second-stage directory absent and retained the complete temporary cache for
bounded diagnosis.

The prediction payload is exact: count `275`, filename SHA-256
`8c75e1dbebc162f316c24137540add99e51877e07aedc6abb419de872c58b5de`,
and content SHA-256
`2ffa40fc0c9b2a7e721d560f6f12edfe2ccdc1c1988582fa7a8104665cdc088a`.
The monitor reached cache preparation, so model execution and monitored child
completion passed. No conflicting model process remains.

After the historical helper rewrote the temporary path to the registered final
path, metrics SHA-256 was
`4870821aa8672e228a23e375ec68597ac4176d4b7ef4f6ed5a2b527defb22439`.
Replacing the 275 current repository roots with the frozen historical root in
candidate bytes produced
`79fd61278e689a0003e37a5bdf20f856184b49c8fdb3af8ad9af03a3a13c451b`,
not the frozen
`b800fdf385075bac46cc50db08a726dc2b9a6201b11a1229a164738b595a708d`.
Candidate validation therefore stopped before writing or directory publication.

The model gate is not the source of drift: all 275 `gate_ratio` values exactly
match the retained historical dual-input audit. The current 275 labels also
reproduce the historical label-content SHA-256 exactly. The historical
`b800...` metrics payload itself is no longer present locally, so its remaining
field-level difference cannot yet be inspected directly.

## Next Action

Do not rerun second stage. Preregister a temporary-cache salvage audit that
recomputes every non-path metric from the frozen sources, labels, and exact
prediction PNGs; binds all 275 historical gate ratios; and determines whether
the missing `b800...` identity can be reconstructed from retained evidence. If
the opaque payload cannot be recovered, require a separate decision before
registering a semantic-equivalence cache identity. Keep final and archive paths
absent until that decision passes.

Intent: Preserve exact second-stage model work while refusing to publish a cache whose historical metadata identity is unproven.
Constraint: Predictions, gate ratios, labels, resource gates, and the original expected hash remain frozen; the historical metrics payload is absent.
Rejected: Rerun second-stage MPS inference | the prediction payload is already exact and rerun cannot recover missing historical metadata provenance.
Rejected: Accept the new candidate hash immediately | exact predictions alone do not explain the opaque metrics difference.
Rejected: Rename the temporary directory manually | bypasses the fixed validation-before-publication contract.
Confidence: high for payload identity and publication safety; medium for the remaining metrics provenance cause.
Scope-risk: narrow
Reversibility: clean
Directive: Audit and preregister salvage from the retained temporary cache; never rerun the model or publish final/archive paths to bypass metrics evidence.
Tested: 275 log completions, process/resource terminal, exact prediction hashes, source and canonical candidate metrics hashes, final/temporary/archive path state, 275 historical gate-ratio comparisons, 275-label content hash, and conflicting model process absence.
Not-tested: Full field-by-field deterministic metrics recomputation, historical payload recovery, temporary-cache salvage, archive publication, formal materialization, train275 diagnostic, quality evaluation, visual review, promotion, or reserved-blind access.
Related: docs/external-text-layout-second-stage-cache-reconstruction-result-20260814.json
Related: docs/external-text-layout-primary-cache-recovery-result-20260814.json
Related: docs/external-text-layout-cache-metrics-canonicalization-integration-verification-20260814.json
