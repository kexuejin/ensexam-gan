# External Text Layout Recovered Materializer Bounded-RSS Launcher V8 Integration Pass

## Decision

`PASS`. The recovered batch runtime now owns a narrow `11 GiB` RSS maximum
while the shared materializer and runtime remain byte-exact at `10 GiB`.
`11,147,149,312` bytes, the observed `hw5k_121.jpg` peak, passes. `11 GiB + 1`
raises the same resource-limit error and terminates the process group. The
independent `35%` free-memory and `512 MiB` run-local swap-growth terminations
remain unchanged.

The launcher validates the hash-bound v8 preregistration, requires both v8
ledger prerequisites, passes `11 GiB` only to the recovered runtime, and emits
schema-v8 safety evidence. A real repository authority check remained closed
before this integration record, so implementation did not silently authorize
model execution.

Focused tests pass `24/24` and complete external-layout tests pass `133/133`
under Python 3.13.1 and 3.10.11. Both runtimes compile all four modified files.
The shared materializer and runtime hashes are unchanged. The retained eight
NPZs and records, progress hash, derived plan, and actual next manifest file
`hw5k_121.jpg` remain exact.

Formal CPU-only resume is authorized after this PASS is pushed. Any remaining
page that exceeds `11 GiB`, crosses a host-pressure gate, or times out must
terminate with completed records retained. No automatic cap increase follows
from this integration PASS. Diagnostic, quality, and training surfaces remain
closed until all 275 pages and terminal launcher evidence pass.

Intent: Resume the correct large page with the smallest empirically justified recovered-only RSS boundary.
Constraint: The observed page is 10.38 GiB, above the unchanged shared 10 GiB default but below 11 GiB.
Rejected: Mutate shared runtime limits | the exception belongs only to this hash-bound recovered materializer.
Rejected: Clamp or falsify RSS readings | the monitor records and enforces the actual process-tree value.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Stop and retain progress at any page above 11 GiB; do not raise the limit automatically.
Tested: Dual-runtime 24-test focused and 133-test complete suites, dual-runtime compile, exact contract validation, closed pre-integration authority, actual observed RSS acceptance, 11 GiB plus-one termination, independent memory/swap terminations, and retained hashes.
Not-tested: Real v8 page-nine completion, remaining pages, final materialization, diagnostic, or quality gates.
Related: docs/external-text-layout-recovered-materializer-bounded-rss-launch-v8-integration-verification-20260815.json
Related: docs/external-text-layout-recovered-materializer-bounded-rss-launch-v8.json
