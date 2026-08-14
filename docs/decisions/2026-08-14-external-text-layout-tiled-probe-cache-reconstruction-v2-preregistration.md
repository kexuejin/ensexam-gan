# External Text Layout Tiled Probe Cache Reconstruction v2 Preregistration

## Decision

`PREREQUISITE_NEEDED`. Freeze a v2 handoff for the missing historical
train275 primary and second-stage cache reconstruction so a future PASS from
the tiled one-page probe can be consumed without accepting the old
duplicate-upsample-only KILL result. This record authorizes bounded validator
integration only. It does not authorize OCR, MPS, cache reconstruction,
formal layout materialization, training, candidate inference, quality gates,
visual review, promotion, or reserved-blind access.

## Why v2 Is Required

The existing v1 reconstruction contract is hash-bound to
`outputs/external-text-layout-runtime-safety-probe-repaired-20260814/result.json`
and to the old single-page probe name and `10 GiB/35%` safety schema. The
registered tiled probe writes
`outputs/external-text-layout-runtime-safety-probe-tiled-9x9-20260814/result.json`,
uses the tiled probe identity, and reports stronger `8 GiB/70% launch/45%
runtime` limits. Reusing v1 after a tiled PASS would therefore fail closed for
the wrong reason and leave the long-lived quality loop unable to reach cache
reconstruction.

## Frozen Boundaries

The v2 contract preserves the v1 cache build paths, exact historical metrics
and prediction hashes, Python `3.10.11` / Torch `2.5.1` / NumPy `2.2.6` /
OpenCV `5.0.0` runtime, relative publication rules, helper ordering, and all
forbidden data/quality access. It separates the tiled probe gate from the
later reconstruction gate:

```text
tiled probe result                         new tiled result path only
probe terminal                             PASS
probe reason                               runtime_safety_probe_passed
probe page                                 hw5k_1011.jpg
probe attempt                              exactly 1
probe page_completed                       true
probe residual model processes             0
probe launch free memory                   >= 70%
probe runtime free memory                  >= 45%
probe process-tree RSS                     <= 8 GiB
probe swap                                 <= 512 MiB
reconstruction free memory                 >= 35%
reconstruction process-tree RSS            <= 10 GiB
reconstruction swap                        <= 512 MiB
historical runtime                         exact registered identity
```

The validator must verify the tiled probe contract hash, integration
verification hash, result path, result authority, safety-limit object, thread
caps, completion fields, and zero residual model processes before loading the
historical helper. Cache outputs and publication remain unchanged.

## Current Gate

At registration, the host has `86%` free memory and zero Booted iOS
Simulators, but swap is `1,690.69 MiB`, above the unchanged `512 MiB` gate.
The tiled result is absent. No model or cache process may start until the
tiled probe itself passes and the v2 reconstruction validator later sees that
terminal result.

Intent: Keep the evidence-gated quality loop continuous after the tiled probe without accepting a superseded detector result.
Constraint: The v1 reconstruction contract is immutable evidence for the old path, while the tiled probe has a distinct result identity and stricter safety schema.
Rejected: Change the v1 contract in place | would erase the old KILL-bound provenance and make historical reconstruction evidence ambiguous.
Rejected: Let the validator accept either old or tiled result | permits a rejected old path to authorize cache reconstruction.
Rejected: Relax tiled probe limits to match v1 | contradicts the separately frozen one-page safety contract.
Confidence: high for the handoff mismatch and v2 boundary; medium for future cache reconstruction completion until the tiled result exists.
Scope-risk: moderate
Reversibility: clean
Directive: Integrate only the v2 contract and result validator; never load the historical helper before the tiled result passes every frozen field and the historical runtime matches exactly.
Tested: Read-only comparison of v1 validator expectations, v1 contract, tiled probe contract, tiled integration verification, result paths, and current host gates.
Not-tested: v2 implementation, tiled probe execution, historical primary/second-stage reconstruction, cache publication, quality gates, visual review, or promotion.
Related: docs/external-text-layout-frozen-cache-reconstruction-v1.json
Related: docs/external-text-layout-tiled-9x9-one-page-safety-probe-v1.json
