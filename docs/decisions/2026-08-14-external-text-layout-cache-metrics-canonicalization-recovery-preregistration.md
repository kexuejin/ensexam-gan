# External Text Layout Cache Metrics Canonicalization Recovery Preregistration

## Decision

`PREREQUISITE_NEEDED`. Freeze one implementation-only recovery for the exact
primary cache metadata failure. No model, second-stage, archive publication,
formal materialization, quality, visual, promotion, or reserved-blind command
is authorized by this record.

The retained primary prediction set already matches all frozen hashes. The
only admissible data change is a literal replacement of exactly 275 occurrences
of `/Volumes/Tool/source/ensexam-gan-h0-monotonic-safe` with
`/private/tmp/ensexam-gan-h0-P0vNwp`, each occurring once in the `pred_path`
field of one of the 275 metrics rows. The existing metrics SHA-256 must be
`81b75410da6f0c63397348a788f369a41e2782c9871566beb12b38fe8f9325d0`
before recovery, and candidate bytes must have SHA-256
`efd58814583089e888482a7e1604efc1d19ee5f514085cbef0e0c6cabf479846`
before an atomic file replacement is allowed.

The same canonicalization is required for future reconstruction publication.
The controller must first rewrite the temporary output path to the registered
final cache path, canonicalize the repository root, and validate the complete
temporary cache. Only then may it atomically replace the absent final
directory. Any canonicalization, prediction, metrics, or cache-surface failure
must leave the final directory absent.

## Verification Gate

Integration must prove successful 275-row canonicalization, prediction identity,
pre-mutation rejection for wrong source hash/count/field/candidate hash, and
future failure-before-publication behavior. It must also prove that explicit
recovery cannot invoke a model or historical helper. Run focused and complete
external-text-layout tests under both registered Python environments. Only a
separate integration PASS record may authorize the exact existing-primary
recovery command.

Intent: Recover portable historical cache identity without repeating model work or weakening exact cache validation.
Constraint: Only repository-root metadata drift is admissible; all prediction bytes, names, non-pred_path fields, row ordering, resource limits, and expected hashes are frozen.
Rejected: Change expected metrics hashes to current-worktree values | would make historical cache identity host-dependent.
Rejected: Rewrite CSV through a serializer | could alter quoting, ordering, or line endings outside the single literal root substitution.
Rejected: Publish and validate afterward | repeats the failure mode that left an unverified final directory present.
Rejected: General path normalization | broader mutation would not be justified by the exact observed failure.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Validate candidate bytes and the full temporary cache before any atomic replacement; never start second stage from a merely present primary directory.
Tested: Preregistration inputs, exact current and expected metrics SHA-256 values, 275 current-root occurrences, zero historical-root occurrences, exact prediction set, and in-memory candidate hash.
Not-tested: Canonicalization implementation, on-disk recovery, failure-before-publication regression, second-stage reconstruction, archive publication, formal materialization, quality evaluation, visual review, promotion, or reserved-blind access.
Related: docs/external-text-layout-cache-metrics-canonicalization-recovery-v1.json
Related: docs/external-text-layout-primary-cache-reconstruction-result-20260814.json
Related: docs/external-text-layout-tiled-probe-cache-reconstruction-v2.json
