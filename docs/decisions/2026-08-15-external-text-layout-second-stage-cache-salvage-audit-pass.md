# External Text Layout Second-Stage Cache Salvage Audit Pass

## Decision

`PASS`. The retained temporary cache is semantically complete and remained
byte-unchanged during audit. All 275 prediction files retain the frozen count,
filename, and content hashes.

Every metrics field now has closed evidence:

- `1,100` command-constant fields matched.
- `825` manifest/path fields matched.
- `275` `gate_ratio` values matched the historical dual-input audit exactly.
- `2,200` residual/overerase fields matched recomputation from frozen source,
  label, and exact prediction PNGs.

The 275-label content SHA-256 remains
`dfd459f552bd0828221c90258f33f4eacc54220494c7e02b21a179894853e99e`.
The in-memory historical-root candidate remains exactly `79fd6127...` after 275
substitutions, while the retained temporary metrics file remains exactly
`4870821a...`.

The original `b800fdf3...` CSV payload is still absent and is explicitly not
reproduced. This PASS does not rewrite that history. It authorizes only a
separate preregistration for a recovered identity whose evidence chain is the
exact prediction set, `79fd...` canonical metrics, and this semantic audit.

Intent: Prove every downstream-consumed field before allowing a recovered cache identity to replace unavailable opaque metadata provenance.
Constraint: Semantic equivalence and original byte identity remain distinct claims; only the former passed.
Rejected: Declare b800 reproduced from field equivalence | the original payload bytes are unavailable and the hashes differ.
Rejected: Publish the temporary directory as-is | its pred_path fields still contain the current worktree root.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Preregister one hash-bound 487-to-79fd mutation and validate the complete cache before final-directory rename; keep archive and downstream work closed until publication PASS.
Tested: Dual-runtime 24 focused and 81 complete tests, compile and diff checks, exact predictions, source and label identities, 4,400 total field matches, canonical candidate hash, and temporary cache non-mutation.
Not-tested: Recovered metrics mutation, final cache publication, archive symlinks, formal materialization, train275 diagnostic, quality evaluation, visual review, promotion, or reserved-blind access.
Related: docs/external-text-layout-second-stage-cache-salvage-audit-verification-20260815.json
Related: outputs/external-text-layout-second-stage-cache-salvage-audit-20260815/audit.json
Related: docs/external-text-layout-second-stage-cache-salvage-audit-v1.json
