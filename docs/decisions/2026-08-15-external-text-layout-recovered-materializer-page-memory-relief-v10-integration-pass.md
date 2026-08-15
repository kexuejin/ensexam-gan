# External Text Layout Recovered Materializer Page-Memory Relief V10 Integration Pass

## Decision

`PASS`. Each page now reaches an atomic NPZ and JSON record commit before the
recovered CPU child runs `gc.collect()` and
`malloc_zone_pressure_relief(NULL, 0)`. Relief completes before the next page,
and one detector is still reused only within the existing fixed eight-page
child. The libSystem handle remains alive with its function pointer. Missing
platform support, relief-call failure, and injected post-commit failure all
close execution while retaining valid completed records for resume.

The final `13 GiB` process-tree RSS cap, `35%` runtime free-memory floor,
`512 MiB` run-local swap-growth cap, CPU-only execution, one model child,
monitor interval, timeout, source order, and output semantics are unchanged.
The shared materializer and runtime remain byte-exact with their `10 GiB`
default. Schema-v10 output and ledger authority are closed until this PASS is
pushed.

Focused tests pass `27/27` and complete external-layout tests pass `136/136`
under Python 3.13.1 and 3.10.11. Both runtimes compile all five touched files.
The real resumable population still contains exactly 23 validated NPZ/record
pairs, with unchanged aggregate hashes, progress file, derived plan, and next
manifest item `hw5k_1447.jpg`.

Formal CPU-only resume is authorized after this PASS is pushed. It gets one
bounded production conclusion: schema-v10 materialization PASS, or KILL of the
allocator-relief batch-reuse design at an existing resource gate. A failure
must lead to an algorithmic execution redesign, not another resource-threshold
version. Diagnostic, training, and quality gates remain closed until all 275
pages pass integrity validation and transaction residue is absent.

Intent: Bound cross-page allocator retention without weakening host limits or turning resource tuning into an open-ended development loop.
Constraint: V9 retained valid progress but crossed the final 13 GiB cap after five pages in one detector child.
Rejected: Raise any RSS, free-memory, or swap boundary | v9 made the host-capacity boundary final.
Rejected: Treat relief as best effort | claiming bounded batch reuse requires the registered release step to succeed.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Give v10 one bounded real conclusion; any repeated 13 GiB stop KILLs this reuse design and requires an algorithmic replacement.
Tested: Dual-runtime 27-test focused and 136-test complete suites, dual-runtime compile, exact relief ordering/count, symbol and call fail-closed paths, partial-progress recovery, shared hashes, and exact 23-page retained state.
Not-tested: Real v10 page-24 completion, remaining pages, final materialization, diagnostic, or quality gates.
Related: docs/external-text-layout-recovered-materializer-page-memory-relief-v10-integration-verification-20260815.json
Related: docs/external-text-layout-recovered-materializer-page-memory-relief-v10.json
