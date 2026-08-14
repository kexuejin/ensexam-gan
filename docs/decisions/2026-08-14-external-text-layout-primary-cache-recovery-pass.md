# External Text Layout Primary Cache Recovery Pass

## Decision

`PASS`. The retained primary cache was recovered under the pushed integration
contract without starting a model command or loading the historical
materialization helper. The command returned `status=recovered` after exactly
275 `pred_path` root substitutions and one atomic `metrics.csv` replacement.

The metrics SHA-256 changed only from the registered root-drift value
`81b75410da6f0c63397348a788f369a41e2782c9871566beb12b38fe8f9325d0`
to the frozen value
`efd58814583089e888482a7e1604efc1d19ee5f514085cbef0e0c6cabf479846`.
All 275 rows now contain the frozen historical root and none contain the
current worktree root.

Independent validation confirmed the unchanged prediction population: count
`275`, filename SHA-256
`8c75e1dbebc162f316c24137540add99e51877e07aedc6abb419de872c58b5de`,
and content SHA-256
`6400c9413af963e3de280e348bd635cd962e5387c2e975e930036d320214274a`.
No canonicalization candidate, primary materialization temporary directory,
second-stage cache, archive symlink, or conflicting model process remains.

## Next Action

Commit and push this result before downstream execution. Then take a fresh
point-in-time stage baseline and run only the frozen second-stage reconstruction
under Python 3.10.11. The controller must validate and canonicalize the complete
temporary cache before publication. Any resource or hash failure remains
`PREREQUISITE_NEEDED` and must leave the final second-stage path absent. Do not
publish archive symlinks until both exact caches pass.

Intent: Convert the retained exact predictions into a portable, fully verified primary cache without repeating inference.
Constraint: Only metrics path metadata could change; predictions, expected hashes, model commands, and downstream authority remained frozen.
Rejected: Reinvoke recover-primary to demonstrate idempotence on the real cache | synthetic idempotence is already covered and the registered mutation was single-purpose.
Rejected: Start second stage before recording recovery | downstream execution must consume a durable exact-cache proof.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Run only second_stage next under a fresh baseline; do not publish archive links until its exact cache also passes.
Tested: Recovery command result, exact complete-cache validator, metrics SHA and row count, current/historical root counts, prediction count/name/content hashes, temporary/downstream/archive path absence, and conflicting model process absence.
Not-tested: Second-stage reconstruction, archive publication, formal external-layout materialization, train275 diagnostic, quality evaluation, visual review, promotion, or reserved-blind access.
Related: docs/external-text-layout-primary-cache-recovery-result-20260814.json
Related: docs/external-text-layout-cache-metrics-canonicalization-integration-verification-20260814.json
Related: docs/external-text-layout-primary-cache-reconstruction-result-20260814.json
