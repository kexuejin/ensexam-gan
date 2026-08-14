# External Text Layout Tiled 9x9 Runtime Repair Verification Pass

## Decision

`PASS` the implementation and fake-feature equivalence gate for the frozen
four-output-row `9x9` neck repair. The implementation combines the previously
proven duplicate-upsample removal with row tiling at exactly the two registered
projection and lateral call sites. It writes no global site-packages and is not
integrated into the real detector path yet.

The broader external text-layout family remains `PREREQUISITE_NEEDED`. This
PASS does not authorize importing the real detector or running another page.
A separate one-page preregistration and a host that passes every launch gate
are still required.

## Verification

The exact registered Python `3.13.1` / Torch `2.12.0` CPU environment ran six
implementation tests. Four deterministic float32 cases cover the real
`256->64` projection and `64->64` lateral channel signatures, top and bottom
padding, a map shorter than one tile, multiple complete tiles, and a short
final tile.

```text
input 1x256x9x13  -> 64 channels    torch.equal=true  max_abs=0.0
input 1x64x10x11  -> 64 channels    torch.equal=true  max_abs=0.0
input 1x64x3x7    -> 64 channels    torch.equal=true  max_abs=0.0
input 1x64x17x19  -> 64 channels    torch.equal=true  max_abs=0.0
height-10 tile rows                  4, 4, 2
shape/dtype/device/contiguity        identical
gradient-enabled call                rejected
float64 call                         rejected
stride drift                         rejected
combined AST call replacements       2
duplicate upsample assignments left  1
second repair application            already_applied
partial marker drift                 rejected
```

The four static-feasibility tests also pass, for ten bounded tests total. No
PaddleOCR detector, Transformers model, weights, source image, target, label,
quality split, or model process was imported or executed. The test tensors are
small synthetic CPU tensors only.

## Interpretation

This establishes exact fake-feature values for the tested shapes and confirms
that the implementation enforces the registered surface. It does not prove
that Torch will choose the same backend algorithm at the full `432x608` map,
that 406 tiled calls will finish inside the page timeout, or that total host
memory will remain below the safety limits. Those are runtime uncertainties,
not reasons to weaken this gate.

Any one-page successor must bind this exact implementation and test hashes,
integrate only this repair before `TextDetection` construction, retain the
existing host lock and health monitor, start with swap at most `512 MiB`, and
remain runtime-safety evidence only. It must be separately preregistered before
the integration or execution occurs.

## Registered Evidence

```text
verification:
  docs/external-text-layout-tiled-9x9-runtime-repair-verification-20260814.json
  sha256=6961f9beff92626dcd65547687072473a4f230ab25a05e2746ef2a01dd6ad758
implementation:
  scripts/analysis/external_text_layout_tiled_9x9_runtime_repair.py
  sha256=07b894a5a43b5331c5ba866dd2a714e6346723c8112f0b23bebfb56852f839ac
test:
  tests/test_external_text_layout_tiled_9x9_runtime_repair.py
  sha256=01ef4e03ff41bbb440a7a028e05e07a4c6f2aa1dde049f124a362129b0b37b39
```

Intent: Prove a materially lower-bound convolution path without spending another unsafe detector run.
Constraint: The prior clean-baseline detector probe crossed both host safeguards, so only synthetic CPU verification was authorized.
Rejected: Treat static bounds or small fake tensors as runtime-safety proof | full-map backend selection, timeout, and total process memory remain unknown.
Rejected: Accept nonzero numerical tolerance | detector threshold and polygon output could change.
Confidence: high for the tested implementation surface and fake-feature equality; medium for full-map numerical identity; low for runtime completion until measured.
Scope-risk: moderate
Reversibility: clean
Directive: Do not integrate or execute the real detector until a separate one-page contract binds these exact hashes and all host gates pass.
Tested: Six implementation tests plus four static-feasibility tests under the exact Python 3.13.1 / Torch 2.12.0 CPU environment.
Not-tested: Full 432x608 fake map, real detector import, one-page completion, timeout, peak RSS, cache reconstruction, formal materialization, training, quality gates, or promotion.
Related: docs/decisions/2026-08-14-external-text-layout-tiled-9x9-runtime-repair-preregistration.md
