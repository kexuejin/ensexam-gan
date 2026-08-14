# External Text Layout Tiled 9x9 Runtime Repair Preregistration

## Decision

`PREREQUISITE_NEEDED`. Freeze one implementation-only experiment that replaces
the two frozen neck `9x9` call sites with four-output-row spatial tiles while
retaining the same convolution weights, bias, stride, padding, dilation,
groups, output coordinates, dtype, device, model, and detector parameters.

This record authorizes only repository-local implementation and CPU float32
fake-feature equivalence tests. It does not authorize importing or constructing
the real detector, running another page, reconstructing caches, materializing
layouts, training, evaluating quality, or promoting a candidate.

## Causal Basis

The only clean-baseline duplicate-upsample-repaired page fell from `82%` to
`26%` free memory and raised swap from `367.00 MiB` to a peak of
`2,973,562,306` bytes before active termination. That exact retry path is
closed.

Static source and shape analysis identifies two ordinary dense `Conv2d`
constructors in `PPOCRV6MediumDetNeck`. Both use `kernel_size=9`, `stride=1`,
`padding=4`, `dilation=1`, `groups=1`, zero padding, and bias. The highest
resolution map is `432x608`. A four-output-row tile with a four-row halo covers
every original output coordinate exactly once.

The explicit-unfold equivalents are comparison bounds, not measured PyTorch
allocations:

```text
projection 256->64 full spatial bound        21,785,739,264 bytes
projection 256->64 four-row tile bound          201,719,808 bytes
projection bound reduction                                108x
lateral 64->64 full spatial bound             5,446,434,816 bytes
lateral 64->64 four-row tile bound                50,429,952 bytes
all-stage tiled convolution calls per page                    406
```

The reduction is material enough to justify a bounded equivalence test. The
406-call count creates a real timeout/performance risk and is not treated as a
free improvement.

## Frozen Implementation

The implementation must remain source-hash, Transformers-version, and AST
bound. It may replace only `PPOCRV6MediumDetNeck.forward` in memory and inject
one private helper into the already validated model module. Global
site-packages must not be written.

The repaired forward must combine two changes:

1. Preserve the already-proven removal of the first overwritten `upsampled`
   construction.
2. Replace only the projection and lateral `9x9` module calls with the frozen
   row-tiled helper.

The helper must fail closed unless gradients are disabled and the input is a
contiguous CPU float32 rank-four tensor. It must validate every Conv2d
attribute, preallocate the full output, copy each four-row result exactly once,
and reject any output-shape drift.

## Verification Gate

The implementation passes only if every fake-feature case satisfies:

```text
torch.equal(full, tiled)                         true
maximum absolute difference                      0
shape / dtype / device / contiguity              identical
projection 256->64 and lateral 64->64            both covered
top / middle / bottom / short final tile         covered
source, version, AST, module, Conv2d drift        rejected
second identical repair application              already_applied
real detector import or execution                 false
```

Any nonzero difference is `KILL`; tolerance-based acceptance is prohibited.
Passing fake features authorizes only a separate decision about whether one
new safety probe is justified. It does not authorize that probe by itself.

## Registered Surface

```text
contract:
  docs/external-text-layout-tiled-9x9-runtime-repair-v1.json
  sha256=203e67ac2d12557034f546d9e25e475d083bb2086d2f0ed7bfc1a3244ba3b250
static report:
  docs/external-text-layout-tiled-9x9-feasibility-20260814.json
  sha256=5b4a2cf381b8c0046b68d3b002c5676cc845b0e3bdd1cd00b275a439d9b2a723
planned implementation:
  scripts/analysis/external_text_layout_tiled_9x9_runtime_repair.py
planned test:
  tests/test_external_text_layout_tiled_9x9_runtime_repair.py
```

Intent: Bound the dominant full-resolution convolution workspace without changing the frozen detector function.
Constraint: The exact duplicate-upsample repair remained unsafe from a clean launch baseline, and another real model run is prohibited without a materially lower static bound.
Rejected: Resize, tile, crop, or otherwise change detector input geometry | changes the frozen producer and boundary detections.
Rejected: Replace 9x9 weights with separable kernels | arbitrary trained dense kernels are not exactly separable.
Rejected: Accept tolerance-based fake-feature similarity | a changed detector map could alter thresholded polygons.
Rejected: Run the detector immediately after implementation | fake-feature equivalence and a separate one-page authorization are required first.
Confidence: high for coordinate coverage and static bound reduction; medium for bitwise CPU equivalence; low for eventual runtime performance until measured.
Scope-risk: moderate
Reversibility: clean
Directive: Kill this repair on any nonzero fake-feature difference, source drift, or convolution-surface drift; never write global site-packages.
Tested: Static AST validation, exact row coverage and halo arithmetic, fixed-shape memory bounds, and evidence-hash binding without Torch or model import.
Not-tested: Tiled Torch execution, fake-feature value equivalence, real detector import, one-page completion, cache reconstruction, formal materialization, training, quality gates, or promotion.
Related: docs/decisions/2026-08-14-external-text-layout-runtime-equivalence-repair-probe-kill.md
