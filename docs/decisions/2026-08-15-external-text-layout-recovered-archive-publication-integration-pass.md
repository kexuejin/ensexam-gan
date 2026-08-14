# External Text Layout Recovered Archive Publication Integration Pass

## Decision

`PASS`. The separate recovered-identity archive publisher implements the
preregistered two-link path without calling the original historical
`publish_caches()` route, loading a historical helper, starting a model, or
changing either source cache.

The focused archive suite passed `14/14` and the complete external-text-layout
suite passed `109/109` under both Python 3.13.1 and frozen Python 3.10.11. Both
source caches and all 275 expected prediction names validate before any link
mutation. Both final and `.publishing` destinations preflight before link
creation, and linked cache identities validate again afterward.

Synthetic coverage proves that absolute, broken, wrong-target, stale, and
conflicting paths fail without publication. Correct partial final links and
temporary links resume. A second-link promotion failure or result-write
failure restores the exact prior link state and removes only current-attempt
links. A terminal exact PASS remains idempotently readable after execution
authority closes.

The real publisher invocation before this ledger update returned
`PREREQUISITE_NEEDED`. Both archive links, both temporary links, and the result
remain absent. Primary metrics remain `efd58814...`; recovered second-stage
metrics remain `79fd6127...`. This integration PASS authorizes one real
relative-link publication after the commit is pushed.

Intent: Publish the recovered archive interface without weakening historical provenance or cache validation.
Constraint: Only two exact relative symlinks and one atomic result may be created; source cache bytes are immutable.
Rejected: Reuse the historical publish_caches route | it remains correctly bound to unreproduced b800 metadata.
Rejected: Accept any equivalent relative target spelling | one canonical relative target keeps terminal identity deterministic.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Push this PASS before the single real publication command; verify both links and both linked caches independently before closing archive publication.
Tested: Dual-runtime 14-test focused and 109-test complete suites, dual-runtime compilation, diff checks, destination conflicts, partial crash recovery, promotion/result rollback, terminal idempotency, and pre-ledger execution rejection.
Not-tested: Real archive link creation, formal external-layout materialization, train275 diagnostic, quality evaluation, visual review, promotion, or reserved-blind access.
Related: docs/external-text-layout-recovered-archive-publication-integration-verification-20260815.json
Related: docs/external-text-layout-recovered-archive-publication-v1.json
Related: docs/external-text-layout-second-stage-recovered-cache-publication-verification-20260815.json
