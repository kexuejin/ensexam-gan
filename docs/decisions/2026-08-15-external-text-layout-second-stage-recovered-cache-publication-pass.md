# External Text Layout Recovered Second-Stage Cache Publication Pass

## Decision

`PASS`. The single authorized frozen-runtime publisher completed without model
execution. It validated the exact 275-file second-stage prediction set,
atomically canonicalized 275 `pred_path` roots, validated the complete
temporary cache, renamed it to the registered final path, validated the final
cache, and atomically wrote a terminal PASS result.

The metrics SHA-256 changed only from the registered source identity
`4870821a...` to the semantically audited recovered identity `79fd6127...`.
Independent read-only validation confirmed prediction count `275`, filename
SHA-256 `8c75e1db...`, and content SHA-256 `2ffa40fc...`. The temporary path and
both archive paths are absent, and no conflicting model process remains.

The unavailable historical metrics identity `b800fdf3...` remains explicitly
`not_reproduced`. This PASS does not replace that identity in the original
reconstruction contract. It closes the retained second-stage reconstruction
through the separately audited recovered identity.

## Next Action

Commit and push this result before archive work. Then preregister a narrow
recovered-identity archive publisher. It must independently validate the exact
primary and recovered second-stage caches, create only the two frozen relative
symlinks, and avoid the original controller path that still correctly expects
historical `b800...`. Archive execution remains closed until a separate
integration PASS is committed and pushed.

Intent: Preserve the completed exact second-stage predictions as a validated portable cache without claiming unavailable historical bytes were reproduced.
Constraint: The only mutation was the registered 275-root canonicalization followed by one validated final rename and atomic result write.
Rejected: Change the original controller expected hash to 79fd | would erase the historical-versus-recovered provenance boundary.
Rejected: Use the historical archive publisher now | it remains correctly bound to the unreproduced b800 identity.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Bind archive publication to both exact cache identities in a separate contract; do not alter either cache or the original b800 expectation.
Tested: Single real publisher execution, exact final metrics SHA, exact 275-file prediction aggregate, terminal result read-back, temporary/archive absence, and conflicting model-process absence.
Not-tested: Archive symlink publication, formal external-layout materialization, train275 diagnostic, quality evaluation, visual review, promotion, or reserved-blind access.
Related: outputs/external-text-layout-second-stage-recovered-publication-20260815/result.json
Related: docs/external-text-layout-second-stage-recovered-cache-publication-verification-20260815.json
Related: docs/external-text-layout-second-stage-recovered-cache-publication-integration-verification-20260815.json
