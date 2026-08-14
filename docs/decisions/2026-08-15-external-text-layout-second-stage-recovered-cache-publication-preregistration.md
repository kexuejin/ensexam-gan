# External Text Layout Recovered Second-Stage Cache Publication Preregistration

## Decision

`PREREQUISITE_NEEDED`. Freeze one no-model publication path for the semantically
audited second-stage cache. Implementation and synthetic verification are
authorized; real metrics mutation and final-directory publication remain
closed until a separate integration PASS is committed and pushed.

The recovered identity is metrics SHA-256 `79fd6127...` plus the exact frozen
275-file prediction set. Its provenance is the 4,400-field semantic audit, not
reproduction of the absent historical `b800fdf3...` payload. Both identities
remain recorded separately.

Execution may atomically replace temporary `metrics.csv` only when its source
SHA is exactly `4870821a...`, all 275 current-root occurrences are confined to
`pred_path`, and candidate bytes are exactly `79fd6127...`. The complete
temporary cache must pass before an atomic rename to the absent final path, and
the complete final cache must pass again afterward. Archive paths remain
absent.

Crash retry is bounded: an already-canonical `79fd...` temporary cache may
continue to validation and rename, but any unknown metrics or directory state
is `PREREQUISITE_NEEDED` without mutation. An existing final is accepted only
with exact recovered identity, absent temporary state, and an existing PASS
result.

Intent: Publish the fully audited recovered identity atomically without rewriting or pretending to reproduce unavailable historical metadata.
Constraint: Only 275 pred_path root substitutions and one temporary-to-final rename are allowed; no model/helper/archive/quality path is open.
Rejected: Modify the original reconstruction contract expected hash | would erase historical provenance instead of adding a recovered overlay.
Rejected: Rename before complete recovered validation | would reintroduce the unverified-final failure already fixed.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Commit and push integration PASS before real publication; preserve archive absence until a separate dual-cache publication decision.
Tested: Preregistration evidence, exact temporary and recovered identities, semantic audit PASS, path state, and crash-retry design.
Not-tested: Publication implementation, synthetic failures, real metrics mutation, final rename, archive publication, formal materialization, or quality evaluation.
Related: docs/external-text-layout-second-stage-recovered-cache-publication-v1.json
Related: docs/external-text-layout-second-stage-cache-salvage-audit-verification-20260815.json
Related: outputs/external-text-layout-second-stage-cache-salvage-audit-20260815/audit.json
