# External Text Layout Recovered Materializer Formal RSS Launcher V4 Preregistration

## Decision

`PREREQUISITE_NEEDED`. The pushed v3 launcher passed its locked readiness
window and completed `hw5k_1011.jpg` with `7,741,816,832` bytes peak RSS,
`63%` minimum free memory, and zero swap growth. The next same-size page reached
`10,334,617,600` bytes (`9.63 GiB`) and was terminated by v3's probe-specific
`8 GiB` cap. The launcher exited cleanly with one exact page record retained,
no model process, unchanged absolute swap, and `72%` free memory.

The `8 GiB` value belongs to the unique one-page safety probe. The unchanged
shared formal materializer has always registered `10 GiB` process-tree RSS and
the historical cache stages also use that formal limit. V3 incorrectly
carried the stronger probe cap into a 275-page formal run. V4 may restore only
the runtime RSS cap to the existing `10 GiB` formal value. It must keep the
stronger `70%` launch and `45%` runtime free-memory floors, `512 MiB` swap
growth cap, CPU one-page isolation, Simulator checks, and process-group
termination.

To reduce allocation overshoot, V4 must temporarily shorten the existing
one-second page monitor interval to `0.25s` while the host-user lock is held.
That value and every other process-local adapter must restore in `finally`.
The launch parent remains under the probe's `8 GiB` check during the 60-second
stability window. Any page exceeding the formal `10 GiB` cap must still be
terminated and retained resume state must remain authoritative.

Implementation and synthetic verification are authorized only in the separate
launcher and its test. Detector resume remains closed until a v4 integration
PASS is committed and pushed. The existing completed page must not be deleted
or recomputed.

Intent: Correct the accidental use of a one-page probe cap for formal serial materialization while preserving hard host protection.
Constraint: Page two empirically needs 9.63 GiB, below the already registered 10 GiB formal cap, and the run retained exact resumable state.
Rejected: Remove or exceed the shared 10 GiB cap | no evidence authorizes a weaker upper bound.
Rejected: Keep retrying under 8 GiB | the same deterministic page already crossed it and retries would not create progress.
Rejected: Change tiling or detector geometry | that would alter the frozen feature producer and invalidate output equivalence.
Rejected: Delete the completed page and restart | the transaction is explicitly resumable and its exact hashes are valid.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Resume from `hw5k_1029.jpg`; never rerun `hw5k_1011.jpg`, exceed 10 GiB, use MPS, or widen worker count.
Tested: Real locked v3 readiness, exact first-page completion, page-two RSS termination, unchanged swap, post-stop host recovery, no conflicting model process, and exact retained progress/NPZ/record hashes.
Not-tested: V4 implementation, 0.25-second monitor restoration, page-two completion under 10 GiB, remaining 274 pages, final publication, diagnostic, or quality gates.
Related: docs/external-text-layout-recovered-materializer-formal-rss-launch-v4.json
Related: docs/external-text-layout-recovered-materializer-baseline-relative-launch-v3-integration-verification-20260815.json
Related: docs/decisions/2026-08-14-external-text-layout-tiled-9x9-one-page-integration-pass.md
