# External Text Layout Recovered Materializer Batched Launcher V6 Preregistration

## Decision

`PREREQUISITE_NEEDED`. V5 resumed at page 9 with every formal limit active, but
the new page child crossed the `512 MiB` swap-growth cap while loading the
detector: growth reached `746,460,283` bytes from the locked
`1,016,720,261`-byte launch baseline. The child was terminated before another
page completed. Free memory recovered to `72%`, no conflicting or residual
model process remained, and the same eight NPZ/record pairs remain exact.

The safety limit is valid and must not be weakened. The avoidable pressure is
the v5 lifecycle: each page starts a fresh child and reloads the same detector.
V6 may instead start one CPU child for each fixed batch of at most eight source
pages, create the detector once, process those pages serially in exact manifest
order, and preserve each page with the unchanged atomic NPZ-then-record path.
This reduces the remaining detector loads from about 267 to 34 without adding
parallelism or changing the detector, plan, source order, page schema, resume
validation, final manifest, or any quality surface.

The parent must monitor the entire batch child every `0.25s` under the same
`10 GiB` process-tree RSS, `35%` free-memory, and `512 MiB` launch-relative
swap-growth limits. The total batch timeout is `900s`. Limit, timeout, or child
failure must terminate the process group while retaining already completed
page records; existing `prepare_resume_state` behavior remains responsible for
removing an orphan NPZ that lacks a record. Launch readiness remains a locked
60-second window with at least `70%` free memory and zero swap growth.

Only the new cohesive batch-runtime module, launcher integration, focused
tests, and synthetic verification are authorized. The shared materializer and
runtime must remain byte-exact. Real resume from `hw5k_1214.jpg` remains closed
until v6 integration PASS is pushed and a clean host state prevents the prior
`746 MiB` task-created swap from being absorbed into a new baseline.

Intent: Amortize detector startup across fixed serial batches while preserving every formal safety and output boundary.
Constraint: V5 crossed the formal swap-growth cap during its ninth detector load before completing page nine.
Rejected: Raise or remove the 512 MiB swap-growth limit | the limit correctly prevented additional host pressure.
Rejected: Treat the current 1.76 GiB absolute swap as historical baseline | 746 MiB was created by this task and must first be cleared by host recovery or restart.
Rejected: Process pages concurrently | detector concurrency previously caused memory-pressure failure and remains fixed at one.
Rejected: Unbounded detector lifetime | fixed eight-page batches bound leak accumulation and preserve resumable progress.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Do not resume model execution until v6 integration PASS is pushed and host swap no longer includes the v5 task-created growth.
Tested: V5 formal swap termination, eight retained page/record identities, 72% post-stop free memory, and zero residual/conflicting model processes.
Not-tested: V6 implementation, batched detector lifetime, page-nine completion, remaining pages, final publication, diagnostic, or quality gates.
Related: docs/external-text-layout-recovered-materializer-batched-launch-v6.json
Related: docs/external-text-layout-recovered-materializer-formal-memory-launch-v5-integration-verification-20260815.json
