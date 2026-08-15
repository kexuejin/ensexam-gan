# External Text Layout Recovered Materializer Batched Launcher V6 Integration Pass

## Decision

`PASS`. V6 replaces per-page detector reloads with one spawned CPU child per
fixed batch of at most eight serial source pages. The child creates the frozen
detector exactly once, preserves registered source order, and commits every
page through the unchanged atomic NPZ-then-record path. A partial batch failure
retains completed records; the unchanged `prepare_resume_state` removes an
orphan NPZ without a record. Final publication uses the unchanged shared
manifest builder and schema.

The launcher now passes the locked baseline-relative health reader and all
formal limits explicitly instead of monkey-patching shared module globals.
Every batch is monitored at `0.25s` under `10 GiB` RSS, `35%` free memory,
`512 MiB` swap growth, and a `900s` total timeout. Tests prove RSS, free-memory,
swap, and timeout failures all terminate the process group. Batch timeout and
monitor interval may only be strengthened, and the child Simulator recheck
cannot be disabled. Execution remains CPU-only with one model process.

Focused tests pass `23/23` and complete external-layout tests pass `132/132`
under Python 3.13.1 and 3.10.11. Both runtimes compile the four touched Python
files. Shared materializer, shared test, runtime, and safety-probe hashes remain
exact. A real schema-v6 closed-authority probe exited before detector creation
and left the exact eight-page transaction plus derived `39d5...` plan unchanged.

Real resume from `hw5k_1214.jpg` is authorized only after this PASS is pushed
and the v5-created `746,460,283` bytes of swap growth has cleared through host
recovery or restart. Quality evaluation and training remain closed.

Intent: Resume the frozen materialization with bounded detector reuse instead of repeated startup pressure.
Constraint: The existing 512 MiB swap-growth limit, 10 GiB RSS limit, and 35% runtime memory floor remain authoritative.
Rejected: Global monkey-patching of shared runtime functions | explicit dependency injection is narrower and removes restoration risk.
Rejected: Configurable or concurrent batches | fixed serial batches preserve the approved memory and ordering contract.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Clear the v5-created swap growth before launch; never treat it as historical baseline merely because v6 is ready.
Tested: Dual-runtime 23-test focused and 132-test complete suites, dual-runtime compile, closed-authority probe, exact retained hashes, process-group limit paths, atomic partial progress, orphan recovery, and shared manifest equality.
Not-tested: Real page-nine completion, remaining 267 pages, final publication, diagnostic, or quality gates.
Related: docs/external-text-layout-recovered-materializer-batched-launch-v6-integration-verification-20260815.json
Related: docs/external-text-layout-recovered-materializer-batched-launch-v6.json
