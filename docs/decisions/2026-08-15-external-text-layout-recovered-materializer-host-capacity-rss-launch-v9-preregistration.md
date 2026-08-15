# External Text Layout Recovered Materializer Host-Capacity RSS Launcher V9 Preregistration

## Decision

`PREREQUISITE_NEEDED`. V8 safely advanced formal materialization from 8 to 18
pages. Its first complete batch peaked at `10,823,958,528` bytes RSS with `40%`
minimum free memory; a second child then crossed the run-local swap-growth cap
after atomically completing two more pages. A bounded supervisor correctly
started a new stable-baseline attempt, but that attempt made zero progress and
stopped at `11,992,580,096` bytes RSS, `173 MiB` above the `11 GiB` cap.

The zero-progress peak occurred on `hw5k_1376.jpg`, which has the same
`1719x2436` dimensions as several completed pages. Page dimensions therefore
do not provide a reliable fine-grained RSS predictor. Repeating 1 GiB or
sub-GiB adjustments would be threshold chasing rather than a sustainable
runtime policy.

V9 freezes a final recovered-only RSS cap from host capacity: `13 GiB` on this
`24 GiB` host. This is an abnormal-process containment bound, not permission to
consume the host. The existing `35%` runtime free-memory floor and `512 MiB`
run-local swap-growth cap remain independent and terminate first whenever
actual system pressure rises. The shared materializer and runtime remain
byte-exact at `10 GiB`. V9 never authorizes another automatic RSS increase.

The v9 contract validator must live in a cohesive new module instead of adding
another historical contract block to the 1162-line launcher. Model behavior,
CPU-only execution, fixed eight-page batches, ordering, atomic progress,
monitoring, timeout, and output semantics remain unchanged.

Intent: Replace page-by-page threshold chasing with one host-capacity-bound recovered limit and independent pressure gates.
Constraint: RSS varied by more than 1 GiB across same-sized pages while free-memory and swap checks remained separately observable.
Rejected: Continue raising by the latest miss | same-size runtime variation makes that policy unstable and non-predictive.
Rejected: Remove the RSS cap | 13 GiB remains a hard abnormal-process containment boundary.
Rejected: Raise the shared 10 GiB default | unrelated materializers keep their existing safety contract.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: The recovered RSS cap is final at 13 GiB; any 13 GiB rejection requires an algorithmic change, not another increase.
Tested: V8 advanced 10 pages, exact 18-page retained hashes, zero-progress 11 GiB rejection, 24 GiB host capacity, and zero residual model processes.
Not-tested: V9 implementation, 13 GiB boundary, resumed page 19, remaining pages, final materialization, diagnostic, or quality gates.
Related: docs/external-text-layout-recovered-materializer-host-capacity-rss-launch-v9.json
