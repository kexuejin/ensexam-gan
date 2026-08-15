# External Text Layout Recovered Materializer Host-Capacity RSS Launcher V9 Integration Pass

## Decision

`PASS`. The recovered batch monitor now uses the final `13 GiB` process-tree
RSS containment cap. The observed zero-progress `11,992,580,096`-byte peak
passes, while `13 GiB + 1` terminates the process group. Independent `35%`
free-memory and `512 MiB` run-local swap-growth failures still terminate through
the same monitored path. The shared materializer and runtime remain byte-exact
with their `10 GiB` default.

The v9 contract validator is a cohesive 148-line module. The launcher only
adds a narrow validation adapter, schema-v9 result identity, and v9 ledger
requirements; it remains 1182 lines rather than crossing the project's 1200
line architecture signal. Execution authority stays closed until both v9
records are present.

Focused tests pass `24/24` and complete external-layout tests pass `133/133`
under Python 3.13.1 and 3.10.11. Both runtimes compile all five touched files.
The exact 18-page resume population, aggregate hashes, progress file, derived
plan, and next manifest item `hw5k_1376.jpg` remain unchanged.

Formal CPU-only resume is authorized after this PASS is pushed. The `13 GiB`
cap is final: crossing it closes this execution design for an algorithmic
change. Diagnostic, quality, and training surfaces remain closed until all 275
pages and terminal schema-v9 launcher evidence pass.

Intent: Make the recovered RSS policy stable, host-bound, and final while preserving stronger live pressure gates.
Constraint: Same-sized pages produced variable peaks, so the containment cap cannot be tuned from dimensions alone.
Rejected: Add v9 validation inline to the launcher | extraction keeps the large orchestration file below its architecture signal.
Rejected: Permit another automatic cap increase | 13 GiB is the terminal containment boundary for this design.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Any 13 GiB failure requires an algorithmic memory reduction; never weaken this bound.
Tested: Dual-runtime 24-test focused and 133-test complete suites, dual-runtime compile, v9 contract validation, 11.17 GiB observed acceptance, 13 GiB plus-one termination, independent pressure gates, shared hashes, and exact retained state.
Not-tested: Real v9 page-19 completion, remaining pages, final materialization, diagnostic, or quality gates.
Related: docs/external-text-layout-recovered-materializer-host-capacity-rss-launch-v9-integration-verification-20260815.json
Related: docs/external-text-layout-recovered-materializer-host-capacity-rss-launch-v9.json
