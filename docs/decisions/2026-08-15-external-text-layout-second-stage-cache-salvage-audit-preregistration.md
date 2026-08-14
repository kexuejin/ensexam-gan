# External Text Layout Second-Stage Cache Salvage Audit Preregistration

## Decision

`PREREQUISITE_NEEDED`. Freeze one read-only, no-model audit of the retained
second-stage temporary cache. This record does not authorize metrics mutation,
final-directory publication, archive publication, formal materialization,
quality evaluation, visual review, training, promotion, or reserved-blind
access.

The audit must validate the exact 275-file prediction payload first. It then
recomputes the eight residual/overerase fields from each frozen source, label,
and prediction PNG through the hash-bound historical
`compute_residual_metrics` function. It binds four command constants, three
manifest/path fields, and all 275 `gate_ratio` values from the retained
historical dual-input audit. Every field in the 16-column metrics CSV therefore
has one explicit evidence source.

The audit must also construct the historical-root candidate in memory and
require SHA-256
`79fd61278e689a0003e37a5bdf20f856184b49c8fdb3af8ad9af03a3a13c451b`
without modifying the temporary file. The original `b800...` payload remains
the frozen historical identity and must not be relabeled as reproduced.

`PASS` means semantic equivalence is complete and may authorize only a separate
preregistration for a new hash-bound recovered identity plus atomic temporary
cache publication. `KILL` closes salvage on any field mismatch. Missing or
changed frozen evidence is `PREREQUISITE_NEEDED`.

Intent: Determine whether exact model output and fully recomputable metadata are sufficient to salvage the retained cache despite a missing opaque historical CSV payload.
Constraint: The temporary cache is the only second-stage payload; it must remain byte-unchanged during audit and the model must not rerun.
Rejected: Treat prediction hashes as the whole cache contract | ignores metrics fields consumed by later diagnostics.
Rejected: Change expected b800 to 79fd inside the historical contract | destroys the distinction between original and recovered provenance.
Rejected: Recompute gate ratios from a new model pass | all 275 historical values are already retained and exact.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Produce only the registered audit JSON; PASS still requires a separate rebaseline/publication preregistration.
Tested: Preregistration evidence hashes, exact temporary metrics and prediction hashes, 275 historical gate-ratio agreement, label-content identity, canonical candidate hash, and final/archive absence.
Not-tested: Audit implementation, full field recomputation, recovered identity registration, temporary-cache mutation, final/archive publication, formal materialization, train275 diagnostic, or quality evaluation.
Related: docs/external-text-layout-second-stage-cache-salvage-audit-v1.json
Related: docs/external-text-layout-second-stage-cache-reconstruction-result-20260814.json
Related: outputs/dual-input-support-separation-prerequisite-20260811/audit.json
