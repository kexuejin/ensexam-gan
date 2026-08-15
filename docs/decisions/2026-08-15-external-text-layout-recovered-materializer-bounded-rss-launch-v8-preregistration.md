# External Text Layout Recovered Materializer Bounded-RSS Launcher V8 Preregistration

## Decision

`PREREQUISITE_NEEDED`. The first v7 resume passed its stable current-baseline
window and loaded the frozen CPU detector, then correctly terminated when the
next manifest page reached `11,147,149,312` bytes (`10.38 GiB`) of process-tree
RSS, above the shared `10 GiB` cap. The stop retained the exact eight completed
NPZ/record pairs and progress hash. Post-stop free memory was `74%`; no Booted
Simulator, conflicting model process, final output, or launcher result remains.

The actual next page is manifest index 9, `hw5k_121.jpg`, at `2202x3027`.
Earlier v6/v7 prose incorrectly named `hw5k_1214.jpg`, which is not in the
manifest. This was a documentation-only error: the launcher always derives
remaining work from the hash-bound manifest and attempted the correct page.

V8 may add a recovery-specific `11 GiB` process-tree RSS cap. The observed
`10.38 GiB` page is below that bound; `11 GiB + 1` must terminate the process
group. The shared materializer and runtime retain their byte-exact `10 GiB`
default. The launcher must continue enforcing the `35%` runtime free-memory
floor and `512 MiB` run-local swap-growth cap independently, so the narrow RSS
increase cannot hide system pressure. CPU-only execution, one child, fixed
eight-page batches, exact source order, atomic progress, `0.25s` monitoring,
and `900s` timeout remain unchanged.

The 275-page header audit found only 19 pages at least as large by pixel count
as the stopped page; the largest is 1.1653 times its pixels. V8 does not assume
that every remaining page will fit. Any page crossing `11 GiB` stops safely
with completed records retained and requires new evidence rather than another
automatic cap increase.

Intent: Admit the first empirically observed large page without weakening host-pressure or shared-runtime boundaries.
Constraint: The correct next manifest page needs 10.38 GiB RSS while free memory remains healthy and run-local swap growth is separately bounded.
Rejected: Raise the shared 10 GiB default | unrelated materializers must keep their existing contract and hashes.
Rejected: Remove RSS monitoring | a bounded recovered-specific cap remains useful even with free-memory and swap guards.
Rejected: Preemptively size the cap for every page | observed execution should determine whether any rarer page needs a different algorithm.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Use `hw5k_121.jpg` as the exact next page; never repeat the incorrect `hw5k_1214.jpg` governance label.
Tested: V7 fail-closed termination, exact retained hashes, 275-page dimension audit, 74% post-stop free memory, and zero residual model processes.
Not-tested: V8 implementation, 11 GiB boundary, resumed page nine, remaining pages, final materialization, diagnostic, or quality gates.
Related: docs/external-text-layout-recovered-materializer-bounded-rss-launch-v8.json
