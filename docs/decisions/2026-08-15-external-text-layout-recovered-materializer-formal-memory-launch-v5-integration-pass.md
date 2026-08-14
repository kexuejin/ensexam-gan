# External Text Layout Recovered Materializer Formal Memory Launcher V5 Integration Pass

## Decision

`PASS`. V5 keeps the strengthened `70%` launch-readiness floor but applies the
unchanged shared formal `35%` floor while detector pages run. Tests prove the
observed `39%` sample passes and `34.9%` still terminates through the existing
process-group path. The formal `10 GiB` RSS cap, `0.25s` monitor, `512 MiB`
swap-growth cap, CPU isolation, Simulator check, timeout, and restoration
semantics are unchanged.

Focused tests pass `15/15` and complete external-layout tests pass `124/124`
under both Python runtimes. Shared hashes and all eight retained page/record
pairs remain exact after a schema-v5 closed-authority probe. Resume from
`hw5k_1214.jpg` is authorized only after this PASS is pushed.

Intent: Complete formal materialization under its frozen runtime memory floor while retaining stronger launch protection.
Constraint: Eight completed pages are authoritative resume state and must not rerun.
Rejected: Lower below 35% | no formal evidence authorizes it.
Confidence: high
Scope-risk: moderate
Directive: Resume CPU serial execution from page nine and preserve every completed record after any stop.
Tested: Dual-runtime 15-test focused and 124-test complete suites, 39%/34.9% boundary, compile/diff checks, shared hashes, retained aggregates, and closed-authority probe.
Not-tested: Page-nine completion, remaining pages, final publication, diagnostic, or quality gates.
Related: docs/external-text-layout-recovered-materializer-formal-memory-launch-v5-integration-verification-20260815.json
Related: docs/external-text-layout-recovered-materializer-formal-memory-launch-v5.json
