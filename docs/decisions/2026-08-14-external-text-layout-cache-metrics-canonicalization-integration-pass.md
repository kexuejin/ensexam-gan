# External Text Layout Cache Metrics Canonicalization Integration Pass

## Decision

`PASS`. The preregistered metrics-only canonicalizer and cache controller
ordering are implemented without changing model commands, device selection,
resource limits, expected hashes, prediction population, or downstream quality
authority.

The focused canonicalization/controller suite passed `21/21` and the complete
external-text-layout suite passed `78/78` under both Python 3.13.1 and the
frozen Python 3.10.11 runtime. The production canonicalizer also transformed a
temporary copy of the real 275-row metrics file from SHA-256
`81b75410da6f0c63397348a788f369a41e2782c9871566beb12b38fe8f9325d0`
to the exact frozen SHA-256
`efd58814583089e888482a7e1604efc1d19ee5f514085cbef0e0c6cabf479846`
with exactly 275 field-checked substitutions.

Future reconstruction now validates prediction identity, rewrites temporary
paths, canonicalizes only `pred_path`, and validates the complete temporary
cache before publishing the final directory. Synthetic hash failure leaves the
final directory absent. Existing-primary recovery has no model-command or
historical-helper route and is blocked until this integration PASS is present
in the ledger.

The retained primary cache was not changed during integration. A real recovery
invocation before this ledger update returned `PREREQUISITE_NEEDED`, and its
metrics SHA remains `81b754...`. This PASS authorizes only the exact
`recover-primary` metrics mutation after the commit is pushed. It does not
authorize second stage, archive publication, formal materialization, quality
evaluation, visual review, promotion, or reserved-blind access.

Intent: Make exact cache validation precede publication and prove the retained metadata can be recovered without rerunning the model.
Constraint: Recovery is exactly 275 pred_path-only repository-root substitutions with fixed before/after hashes and unchanged predictions.
Rejected: Keep publication ordering and repair after failure | would continue exposing unverified final directories.
Rejected: Couple recovery to the historical helper | would create an unnecessary route back to model materialization.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Push this PASS before running recover-primary; require exact cache validation afterward and keep second stage closed until the primary recovery result is recorded.
Tested: Dual-runtime focused and complete suites, compile checks, static preflight, pre-ledger recovery rejection, synthetic failure-before-publication, idempotent synthetic recovery, prediction identity, and real-metrics temporary-copy canonicalization.
Not-tested: On-disk retained-primary recovery, second-stage reconstruction, archive publication, formal materialization, quality evaluation, visual review, promotion, or reserved-blind access.
Related: docs/external-text-layout-cache-metrics-canonicalization-integration-verification-20260814.json
Related: docs/external-text-layout-cache-metrics-canonicalization-recovery-v1.json
Related: docs/external-text-layout-primary-cache-reconstruction-result-20260814.json
