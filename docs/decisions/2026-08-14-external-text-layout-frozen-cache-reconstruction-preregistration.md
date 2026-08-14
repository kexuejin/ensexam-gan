# External Text Layout Frozen Cache Reconstruction Preregistration

## Decision

`PREREQUISITE_NEEDED`. Register one recovery-only path for the missing frozen
train275 primary and second-stage caches required by the external text-layout
diagnostic. Reconstruct each cache at its original historical build path,
verify exact metrics and prediction hashes, and only then publish the archive
paths as relative symlinks.

This path does not authorize external-layout materialization, target or label
access, quality evaluation, training, candidate inference, promotion, or
reserved-blind access. The current product default remains
`artifacts/current-primary`.

## Frozen Identity

The archived train275 manifest remains the only sample list:

```text
hardcase_lists/archive/sign-separated-residual-repair-20260810-train275-v1.txt
sha256=ba31900496161322f839f366fa40765d71182d99a59ddad2537786310aae432f
count=275
```

Expected cache identities are frozen from the historical materialization
audit:

```text
primary metrics
  efd58814583089e888482a7e1604efc1d19ee5f514085cbef0e0c6cabf479846
primary prediction content
  6400c9413af963e3de280e348bd635cd962e5387c2e975e930036d320214274a
second-stage metrics
  b800fdf385075bac46cc50db08a726dc2b9a6201b11a1229a164738b595a708d
second-stage prediction content
  2ffa40fc0c9b2a7e721d560f6f12edfe2ccdc1c1988582fa7a8104665cdc088a
prediction filenames, both stages
  8c75e1dbebc162f316c24137540add99e51877e07aedc6abb419de872c58b5de
prediction count, both stages
  275
```

The commands retain the original build paths so path-bearing `metrics.csv`
rows can reproduce the historical hashes. Only the missing manifest path is
replaced by the identical archived manifest. The fixed historical helper is
loaded lazily after the runtime gate and remains bound to SHA-256
`cbc3d107f9410f83d1698b17cd56c3b6121b3b698cdff7f66914ff960bcf19ec`.
The historical execution environment is also frozen before helper loading:

```text
Python  3.10.11
Torch   2.5.1
NumPy   2.2.6
OpenCV  5.0.0.93
```

## Execution Gates

The static `preflight` stage reads registered files and hashes only. It never
loads the historical helper, imports PaddleOCR, constructs a detector, or runs
primary/second-stage inference. Its current result is:

```text
static_terminal=PASS
terminal=PREREQUISITE_NEEDED
historical_manifest_count=275
historical_runtime_ready=false
observed_runtime=Python 3.13.1 / Torch 2.12.0 / NumPy 2.3.5 / OpenCV 4.13.0
build_paths_absent=true
archive_paths_absent=true
probe_result_present=false
execution_authorized=false
```

Every primary or second-stage reconstruction requires all of the following
before the helper or model command is loaded:

```text
repaired one-page probe terminal                    PASS
probe page                                          hw5k_1011.jpg
system free memory                                  >= 35%
process-tree RSS                                    <= 10 GiB
swap used                                           <= 512 MiB
conflicting model processes                         none
host-wide external-layout lock                      acquired
historical Python/Torch/NumPy/OpenCV identity       exact
```

The probe evidence must include complete, finite, nonnegative initial,
per-page peak, and post-run health fields under the same frozen limits.
Missing or malformed values fail closed. Second-stage reconstruction also
requires the exact primary cache. Existing build paths are validated and are
never overwritten, including broken symlinks.

Publication begins only after both real build directories pass exact metrics,
filename, count, and prediction-content validation. Both archive destinations
are preflighted before either link is created. Existing links are accepted
only when they are relative and resolve to the registered build directory;
conflicting paths and absolute links fail closed.

## Current Host Boundary

On 2026-08-14 the host reports `74%` system-wide free memory but
`997.75 MiB` swap used. Swap therefore still exceeds the unchanged
`512 MiB` launch gate. The currently executable repository environment also
does not match the frozen historical cache runtime. No OCR, MPS, primary
inference, second-stage inference, training, or repaired one-page probe was
run for this preregistration.

After swap falls below the gate, the next action remains exactly one isolated,
target-free repaired `hw5k_1011.jpg` probe. Only a complete safe result can
authorize serial primary and second-stage cache reconstruction. Exact cache
reconstruction additionally requires restoration of the frozen historical
runtime. Verification can then authorize relative-link publication; it still
cannot authorize quality, optimizer, candidate, or promotion surfaces.

## Registered Surface

```text
contract:
  docs/external-text-layout-frozen-cache-reconstruction-v1.json
  sha256=e23d13c7cd93346153940bd42198f74f4114dd9ad6221a246cbe9893cc7b70a8
reconstruction:
  scripts/analysis/reconstruct_external_text_layout_frozen_caches.py
  sha256=ad718ce8327fe116636b340fb16fc25d6fdee3aa5526c7b198ed0fb7256639c6
test:
  tests/test_external_text_layout_frozen_cache_reconstruction.py
  sha256=b1827310aaf9a7a723f2bb3b781a7cb18cbce5d0dac8168d61e16dabddb94039
```

Intent: Restore exact historical support inputs without opening any quality or optimization surface.
Constraint: The frozen caches and repaired probe are absent, host swap remains above the fixed launch gate, and the current executable environment differs from the historical cache runtime.
Rejected: Rebuild directly into archive paths | historical metrics contain original build-path strings and would not reproduce their registered hashes.
Rejected: Publish copied cache directories | duplicates large payloads and weakens one-source provenance.
Rejected: Run reconstruction before a safe repaired probe | bypasses the detector-runtime prerequisite and risks another memory-pressure incident.
Confidence: high for the static reconstruction and verification contract; low for future runtime success until the repaired probe passes.
Scope-risk: narrow
Reversibility: clean
Directive: Never relax runtime identity, hash, memory, swap, process-isolation, or relative-publication gates to make reconstruction pass.
Tested: Python compilation; JSON parsing; focused tests covering runtime identity, probe health, cache hashes, command paths, helper ordering, and publication; real static preflight with 275 frozen sources and artifacts.
Not-tested: Repaired detector probe, primary reconstruction, second-stage reconstruction, cache publication, formal layout materialization, train-only audit, training, candidate inference, quality gates, or promotion.
Related: docs/decisions/2026-08-14-external-text-layout-runtime-equivalence-repair-preregistration.md
