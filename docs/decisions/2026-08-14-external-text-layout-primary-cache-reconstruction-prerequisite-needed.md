# External Text Layout Primary Cache Reconstruction Prerequisite

## Decision

`PREREQUISITE_NEEDED`. The frozen primary reconstruction completed model
inference for all 275 pages under the registered Python 3.10.11 MPS runtime,
but the exact cache gate rejected `metrics.csv`. The second-stage command was
not started and neither archive symlink was published.

The prediction set is hash-exact: count `275`, filename SHA-256
`8c75e1dbebc162f316c24137540add99e51877e07aedc6abb419de872c58b5de`,
and content SHA-256
`6400c9413af963e3de280e348bd635cd962e5387c2e975e930036d320214274a`.
Only the metrics hash differs: actual
`81b75410da6f0c63397348a788f369a41e2782c9871566beb12b38fe8f9325d0`
versus frozen
`efd58814583089e888482a7e1604efc1d19ee5f514085cbef0e0c6cabf479846`.

All 275 metrics rows contain the current repository root in `pred_path`.
Replacing only `/Volumes/Tool/source/ensexam-gan-h0-monotonic-safe` with the
frozen historical root `/private/tmp/ensexam-gan-h0-P0vNwp` in memory produces
the exact frozen metrics hash. No cache file has been changed.

The monitor reached atomic publication without a resource-limit termination,
so the model command and monitor completed. The controller then renamed the
temporary directory before rewriting and validating `metrics.csv`; therefore
the failed gate incorrectly left the unverified final primary directory
present. The temporary primary directory and final second-stage directory are
absent.

## Next Action

Before any cache mutation, preregister one deterministic recovery that rewrites
exactly 275 current-root occurrences to the frozen historical root, preserves
the exact prediction set, and requires the frozen metrics hash. Fix future
ordering so canonicalization and exact validation happen in the temporary
directory before publication and every hash failure leaves the final path
absent. Recover the existing primary cache only after those bounds are frozen
and verified. Do not rerun primary inference.

Intent: Preserve the completed primary predictions while preventing an unverified cache from being mistaken for a published result.
Constraint: The frozen cache contract requires exact historical prediction and metrics hashes, and primary MPS inference must not be repeated.
Rejected: Rerun primary inference in another worktree | predictions are already exact and the failure is deterministic metadata drift.
Rejected: Accept repository-root-dependent metrics | that would weaken the frozen hash contract and make cache identity host-dependent.
Rejected: Start second stage from the current directory | the primary cache has not passed exact validation.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Preregister and test canonicalization before touching the existing primary cache; publication must occur only after exact validation.
Tested: 275 log completions, final/temporary/archive path state, exact prediction count/name/content hashes, metrics row count and SHA-256, current-root occurrence count, and in-memory historical-root canonicalization SHA-256.
Not-tested: On-disk metrics recovery, future failure-before-publication behavior, second-stage reconstruction, archive publication, formal materialization, quality evaluation, visual review, promotion, or reserved-blind access.
Related: docs/external-text-layout-primary-cache-reconstruction-result-20260814.json
Related: docs/external-text-layout-tiled-probe-cache-reconstruction-v2.json
Related: docs/external-text-layout-cache-reconstruction-baseline-relative-integration-verification-20260814.json
