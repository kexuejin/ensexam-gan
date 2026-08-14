# External Text Layout Historical Runtime Restoration

## Decision

`PREREQUISITE_NEEDED`. Restore the missing pyenv Python `3.10.11` base
interpreter so the existing frozen cache-reconstruction virtual environment is
executable again. Do not reinstall or upgrade its packages, and do not use the
current Python `3.13.1` / Torch `2.12.0` environment for historical cache
reconstruction.

This closes only the historical-runtime prerequisite. Model execution remains
disabled because the repaired one-page detector probe is absent and host swap
remains above the fixed `512 MiB` launch gate.

## Recovery Evidence

The virtual environment at
`/Volumes/Tool/source/clean-doc/.venv-torch310-mps-stable` retained its package
payload, but its `python` symlink targeted the missing
`/Users/kexuejin/.pyenv/versions/3.10.11/bin/python`. The original repository
environment contract and the historical materialization commit both identify
that Python `3.10.11` environment.

Python was rebuilt from the official `3.10.11` source payload. The source was
downloaded through a mirror only after its advertised SHA-256 matched the
pyenv definition's official checksum:

```text
Python-3.10.11.tar.xz
sha256=3c3bc3048303721c904a03eb8326b631e921f11cc3be2988456a42f115daf04c

restored executable
/Users/kexuejin/.pyenv/versions/3.10.11/bin/python3.10
sha256=d4b4cc767bb187aaa228ebf4706413d33ee827c782fe872b92ff791c617142dc
```

The build used existing local OpenSSL 3, readline, ncurses, and Xcode zlib.
No global Homebrew formula, repository dependency, or frozen virtual-
environment package was installed or upgraded.

## Exact Runtime Match

The reconstruction preflight now observes the full frozen identity:

```text
Python distribution              3.10.11
Torch distribution               2.5.1
NumPy runtime                    2.2.6
OpenCV runtime                   5.0.0
OpenCV wheel distribution        5.0.0.93
historical_runtime_ready         true
```

OpenCV's runtime and wheel versions are intentionally separate. The wheel is
`5.0.0.93`, while `cv2.__version__` is `5.0.0`; both must match before helper
loading.

The static preflight validates all 275 historical source paths and frozen
artifacts, imports no PaddleOCR or Torch model, starts no model process, and
returns:

```text
static_terminal=PASS
terminal=PREREQUISITE_NEEDED
execution_authorized=false
historical_runtime_ready=true
probe_result_present=false
build_paths_absent=true
archive_paths_absent=true
```

All 42 related standard-library unit tests pass in the restored environment.
No `pytest` dependency was added because the frozen environment does not need
it and the relevant suites use `unittest`.

## Remaining Boundary

After restoration, the host reports `75%` system-wide free memory and
`989.75 MiB` swap used. Swap still exceeds `512 MiB`; therefore no OCR, MPS,
primary inference, second-stage inference, training, or repaired detector
probe was run.

The next action remains: wait for swap below the unchanged gate, then run
exactly one isolated target-free repaired `hw5k_1011.jpg` safety probe. Only a
complete safe probe can authorize serial primary and second-stage cache
reconstruction under this restored runtime.

## Registered Surface

```text
restoration report:
  docs/external-text-layout-historical-runtime-restoration-20260814.json
  sha256=13d09cfbc3876947522c54ab34878cd67be710df1bc7df9bb8a999258aeb2c95
reconstruction contract:
  docs/external-text-layout-frozen-cache-reconstruction-v1.json
  sha256=6e03547f445523a5e89f226039b8d4a6583cca338429691345ed0d822bbbfd2f
reconstruction implementation:
  scripts/analysis/reconstruct_external_text_layout_frozen_caches.py
  sha256=fda812eeabb09ee1bf773ba81a4a73bc01303dec01d4cf6e5392cdec8de73502
test:
  tests/test_external_text_layout_frozen_cache_reconstruction.py
  sha256=b1827310aaf9a7a723f2bb3b781a7cb18cbce5d0dac8168d61e16dabddb94039
```

Intent: Restore exact historical execution provenance before any frozen cache reconstruction.
Constraint: The frozen virtual environment packages remained intact but their pyenv base interpreter had been deleted.
Rejected: Reconstruct with Python 3.13.1 and Torch 2.12.0 | post-run hashes would fail closed only after an expensive and unsafe MPS execution.
Rejected: Reinstall the frozen virtual-environment packages | risks changing compiled wheels and historical output identity.
Rejected: Run the repaired probe immediately after runtime restoration | swap remains above the fixed launch gate.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Preserve both OpenCV runtime and wheel identities, and do not upgrade the restored cache-reconstruction environment.
Tested: Python version and executable identity; package metadata; OpenCV runtime version; 42 standard-library unit tests; static 275-source reconstruction preflight.
Not-tested: OCR detector probe, MPS primary reconstruction, MPS second-stage reconstruction, cache publication, formal layout materialization, quality gates, or promotion.
Related: docs/decisions/2026-08-14-external-text-layout-frozen-cache-reconstruction-preregistration.md
