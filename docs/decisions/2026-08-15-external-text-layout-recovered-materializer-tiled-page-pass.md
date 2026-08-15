# External Text Layout Recovered Materializer Tiled-Page Pass

## Decision

`PASS`. The allocator-relief batch-reuse design was killed by a renewed
resource gate at page 24: the detector crossed the `35%` runtime free-memory
floor before any new page record could be committed. The successor keeps the
same schema-v10 authority and resource limits, but changes execution topology:
each page now runs in its own spawned detector child, and pages above the
memory-safe image budget are detected as overlapping height tiles whose
quadrilaterals are translated back into full-page coordinates.

The final real materialization completed all 275 train pages with no target,
label, candidate, or quality access. The final manifest is
`outputs/external-text-layout-support-materialization-20260813/manifest.json`
with SHA256 `2578fb78cc3188776d97fff2e1efd2fc8f91d224b4efb680ffa83e1847c96579`.
The materialized page content SHA256 is
`8d4753dbd160c85043b1b65c1161e4cdc00059a07216bcc3d63b879754896750`.
The launcher result is schema `10` and terminal `PASS`.

Runtime safety remained inside the existing gates: peak process-tree RSS was
`10942496768` bytes, minimum runtime free memory was `38.0%`, and run-local
swap growth was `0` bytes. Launch health passed the `70%` free-memory stable
window, post-run health was `70%` free memory, and no materialization
transaction residue remained. A second launcher invocation validated the
existing terminal result instead of rerunning model inference.

The code change is intentionally narrow. The shared materializer and shared
runtime are unchanged. `batch_size=8` remains a progress/reporting window, not
a detector lifetime. The detector model, weights, CPU transformers engine,
thresholds, source order, atomic NPZ/record semantics, result schema, and final
RSS/free-memory/swap limits are unchanged.

Intent: Finish recovered external-layout support without weakening host safety gates.
Constraint: Full-page detector inference on larger pages can cross the free-memory gate even when a smaller tiled 9x9 probe passes.
Rejected: Raise RSS, free-memory, or swap limits | those limits were frozen as the final host-safety contract.
Rejected: Reuse one detector across an eight-page child | renewed gate failure showed batch reuse is not a sustainable execution unit.
Rejected: Downscale the registered source pages globally | that would change detector input semantics more broadly than page tiling.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Keep page-level process isolation and tiled large-page inference for this recovered materializer unless a stronger memory proof replaces them.
Tested: Real CPU-only materialization PASS over 275 pages; repeated launcher terminal-result validation; Python 3.13.1 and 3.10.11 complete external-layout suites, 137/137 each; diff whitespace check.
Not-tested: Downstream SCUT115/holdout40 quality gates using the newly materialized support cache.
Related: outputs/external-text-layout-recovered-materializer-input-20260815/result.json
Related: outputs/external-text-layout-support-materialization-20260813/manifest.json
