# External Text Layout Runtime Equivalence Repair Probe Kill

## Decision

`KILL` the exact duplicate-upsample repair probe path. The one authorized
target-free `hw5k_1011.jpg` run started from a clean host gate, applied the
hash/version/AST-bound in-memory repair, and still crossed both the fixed free
memory and swap limits before completing the page. Do not repeat the same
detector, runtime, input geometry, or repair probe.

The broader external text-layout support family remains
`PREREQUISITE_NEEDED`. This result does not evaluate the train-only support
hypothesis because no formal layout output, target, label, quality metric,
candidate, or promotion surface was accessed.

## Clean Launch Evidence

Before the probe, all booted iOS Simulator devices and their runtime children
were stopped without deleting simulator data. The host and exact detector
runtime then passed every launch prerequisite:

```text
system free memory                 82%
swap used                          367.00 MiB
conflicting model processes        0
booted simulators                  0
Python                             3.13.1
OpenCV / NumPy                     4.13.0 / 2.3.5
Paddle / PaddleOCR / PaddleX       3.0.0 / 3.7.0 / 3.7.2
Torch / Transformers               2.12.0 / 5.12.1
detector artifact hashes           6/6 exact
page                               hw5k_1011.jpg
```

The command used one isolated spawned page process, the host-user lock, fixed
single-thread caps, one-second health monitoring, a 15-minute timeout, and the
unchanged registered safety limits. The repaired result is runtime-safety
evidence only.

## Result

The process loaded all 350 weight entries, but the page did not complete:

```text
raw probe terminal                 PREREQUISITE_NEEDED
reason                             system memory safety limit crossed
minimum free memory                26.0%
peak process-tree RSS              5,643,206,656 bytes
peak swap used                     2,973,562,306 bytes
formal outputs written             false
targets / labels / recognition     false / false / false
temporary page outputs retained    false
```

The monitor terminated and cleaned the isolated model process. Post-cleanup
inspection found zero residual model processes and no booted simulator. Free
memory recovered to `79%`; swap settled at `1,906.69 MiB`. Python's resource
tracker reported one leaked loky semaphore at shutdown, but no detector or
Python model process remained.

The earlier static equivalence proof remains valid: the removed first
`upsampled` construction is overwritten before use. This run proves only that
removing it is insufficient to make this detector path safe. It does not
support a causal comparison with the earlier unrepaired run because launch
state and incomplete measurements differ.

## Next Boundary

Do not rerun this probe after merely clearing swap again. The next bounded
uncertainty is read-only static analysis of whether the full-resolution `9x9`
path has a numerically equivalent tiled or streamed implementation with a
materially lower peak-memory bound. Any implementation or real model run
requires a separate preregistration, fake-feature equivalence tests, fixed
memory estimates, and a new one-page authorization.

Historical frozen-cache reconstruction remains blocked because its contract
requires a successful repaired safety probe. Training, candidate inference,
quality gates, visual review, promotion, and reserved blind remain closed.

## Registered Evidence

```text
structured result:
  docs/external-text-layout-runtime-equivalence-repair-probe-20260814.json
  sha256=d56d49f9ac37d127b0a17359af9f2b53a2fbcee0fcff3beabf638487a29ae5b9
raw local result:
  outputs/external-text-layout-runtime-safety-probe-repaired-20260814/result.json
  sha256=1f3d6d3ec1797b6a2495da87bde8e6f05f7b56b69b81c44f609f99b6e42ba52d
```

Intent: Close an empirically unsafe detector repair path without weakening the external-layout quality hypothesis or host safeguards.
Constraint: The only authorized repaired page crossed the fixed memory and swap limits from a clean launch baseline.
Rejected: Clear swap and repeat the same probe | the clean-baseline run already established that the repair is insufficient.
Rejected: Relax memory, swap, geometry, detector, or engine settings in place | changes frozen safeguards or producer identity without preregistration.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not repeat this detector/runtime/geometry/repair probe; require a materially lower-memory, separately preregistered equivalent path before another model run.
Tested: Exact runtime and six detector hashes; one isolated target-free repaired page; active health termination; result-schema checks; zero residual model processes; no booted simulator after cleanup.
Not-tested: Completed detector page, formal layout materialization, frozen cache reconstruction, train-only audit, training, candidate inference, quality gates, or promotion.
Related: docs/decisions/2026-08-14-external-text-layout-runtime-equivalence-repair-preregistration.md
