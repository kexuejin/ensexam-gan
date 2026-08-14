# External Text Layout Recovered Second-Stage Cache Publication Integration Pass

## Decision

`PASS`. The recovered-cache publisher implements the preregistered no-model
publication path without changing the historical `b800fdf3...` identity or
opening archive, quality, promotion, or reserved-blind authority.

The focused publisher suite passed `14/14` and the complete external-text-layout
suite passed `95/95` under both Python 3.13.1 and the frozen Python 3.10.11
runtime. The implementation validates the exact prediction set before metrics
mutation, permits only the registered `4870821a...` to `79fd6127...`
canonicalization, validates the complete temporary cache before rename, and
validates the complete final cache afterward.

Synthetic failures prove that wrong candidate bytes, predictions, replacement
counts, archive state, temporary identity, final identity, and result writes do
not expose a final cache. A canonicalized temporary cache can resume after an
interruption. A completed final cache is accepted only with absent temporary
and archive paths, exact recovered identity, and an existing terminal PASS
result; this terminal revalidation remains available after execution authority
is closed.

The real publisher invocation before this ledger update returned
`PREREQUISITE_NEEDED`. The retained temporary metrics remained exactly
`4870821a...`, while final, archive, and result paths remained absent. This PASS
authorizes one frozen-runtime publication command after the commit is pushed.
It does not authorize model execution or claim reproduction of the unavailable
historical metrics payload.

Intent: Publish the semantically audited second-stage cache without rerunning the model or conflating recovered and historical identities.
Constraint: Only the exact 275-row root canonicalization and one temporary-to-final rename are authorized; archive paths must remain absent.
Rejected: Change the original reconstruction expected hash | would erase the distinction between unavailable historical bytes and audited recovered bytes.
Rejected: Write the final directory before complete cache validation | would expose an unverified cache after interruption.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Push this PASS before the single real publication command; require exact final identity and result evidence before closing reconstruction.
Tested: Dual-runtime 14-test focused and 95-test complete suites, compile and diff checks, pre-ledger execution rejection, mutation-before-publication failures, crash retry, final rollback, terminal idempotency, and archive/model-route closure.
Not-tested: Real 487082-to-79fd mutation and final rename, archive symlink publication, formal materialization, quality evaluation, visual review, promotion, or reserved-blind access.
Related: docs/external-text-layout-second-stage-recovered-cache-publication-integration-verification-20260815.json
Related: docs/external-text-layout-second-stage-recovered-cache-publication-v1.json
Related: docs/external-text-layout-second-stage-cache-salvage-audit-verification-20260815.json
