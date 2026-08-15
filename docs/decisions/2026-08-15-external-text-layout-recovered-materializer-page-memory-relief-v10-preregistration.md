# External Text Layout Recovered Materializer Page-Memory Relief V10 Preregistration

## Decision

`PREREQUISITE_NEEDED`. V9 advanced from 18 to 23 pages, then crossed its final
`13 GiB` boundary at `14,049,755,136` bytes. The next supervisor attempt was
cancelled during the readiness window before detector creation because the v9
contract requires an algorithmic change rather than another RSS increase. All
23 completed NPZ/record pairs remain exact and no model process remains.

The detector process retained increasing CPU RSS across completed pages even
though each page's result objects had left Python scope. V10 may add explicit
post-commit memory relief: after the page NPZ and JSON record are atomically
durable, run `gc.collect()` and macOS
`malloc_zone_pressure_relief(NULL, 0)` before decoding the next page. The
libSystem symbol is available in the exact Python 3.13 environment and a
model-free call passed. Both APIs are standard platform/runtime facilities, so
no dependency or site-package change is needed.

Relief must run once per completed page and only after its record commit. A
missing symbol or call failure is fatal to the child so execution cannot claim
bounded reuse without the registered release step. Output values, detector,
source order, fixed eight-page batches, and all runtime limits remain
unchanged. The final `13 GiB` cap is not raised.

Intent: Return completed-page CPU allocations to macOS while retaining one detector per bounded batch.
Constraint: The final RSS cap was crossed only after five pages accumulated in one child, while atomic progress remained valid.
Rejected: Raise RSS above 13 GiB | v9 explicitly closed threshold increases.
Rejected: Return to one detector load per page | repeated startup already caused avoidable swap growth and poor throughput.
Rejected: Best-effort relief | bounded reuse must fail closed if the registered platform release is unavailable.
Confidence: medium
Scope-risk: moderate
Reversibility: clean
Directive: Keep relief after atomic record commit and before the next page; never let it alter page output semantics.
Tested: V9 five-page progress before final RSS stop, exact 23-page hashes, cancelled pre-model retry, zero residual processes, and live libSystem symbol call.
Not-tested: V10 implementation, memory release under detector load, resumed page 24, remaining pages, final materialization, diagnostic, or quality gates.
Related: docs/external-text-layout-recovered-materializer-page-memory-relief-v10.json
