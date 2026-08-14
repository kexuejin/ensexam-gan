# External Text Layout Recovered Materializer Formal RSS Launcher V4 Integration Pass

## Decision

`PASS`. V4 separates the unique probe's `8 GiB` launch-parent limit from the
unchanged formal materializer's `10 GiB` detector runtime limit. The observed
page-two trigger at `10,334,617,600` bytes is now accepted below the formal
cap, while `10 GiB + 1` still terminates through the unchanged process-group
path. The stronger `70%` launch and `45%` runtime free-memory floors and
`512 MiB` swap-growth cap remain unchanged.

The page monitor interval is `0.25s` only while the host-user lock and
process-local launcher adapters are active. Tests prove the interval and all
other shared call surfaces restore after both PASS and resource failure.
Shared materializer, runtime, tiled probe, and shared test hashes remain exact.

Focused tests pass `14/14` and the complete external-layout suite passes
`123/123` under both Python runtimes. A real closed-authority probe returned
schema-v4 `PREREQUISITE_NEEDED` without changing the exact retained progress,
NPZ, or record hashes. Resume must therefore skip `hw5k_1011.jpg` and begin at
`hw5k_1029.jpg` after this PASS is committed and pushed.

Intent: Apply the repository's formal RSS boundary without sacrificing faster monitoring or resumable safety.
Constraint: Formal execution already has a frozen 10 GiB cap, while the completed first page must remain byte-exact and must not rerun.
Rejected: Exceed 10 GiB | no probe or formal contract authorizes it.
Rejected: Return to one-second monitoring | a shorter interval reduces overshoot at the corrected cap.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Resume from page two with CPU one-page isolation; preserve completed records after any stop and never use MPS or multiple workers.
Tested: Dual-runtime 14-test focused and 123-test complete suites, formal RSS pass/fail boundary, monitor restoration, shared-surface restoration, compile/diff checks, exact shared hashes, retained-state hashes, and a closed-authority live probe.
Not-tested: Page-two completion under 10 GiB, remaining 274 pages, final publication, diagnostic, quality evaluation, or promotion.
Related: docs/external-text-layout-recovered-materializer-formal-rss-launch-v4-integration-verification-20260815.json
Related: docs/external-text-layout-recovered-materializer-formal-rss-launch-v4.json
