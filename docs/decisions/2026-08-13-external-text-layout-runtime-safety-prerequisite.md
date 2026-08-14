# External Text Layout Runtime Safety Prerequisite

## Decision

`PREREQUISITE_NEEDED`. One nonformal, target-free CPU Transformers detector
page was attempted before the registered 275-page materialization. The existing
host lock, spawn isolation, process-group termination, and one-second health
monitor stopped detector initialization when system free memory fell to
`31.0%`, below the fixed `35%` safety floor. After termination, memory recovered
to an observed `67-74%`, no OCR or model process remained, and swap use was
`1,269.75 MiB`, above the fixed `512 MiB` launch gate.

The result is runtime-safety evidence only. It is not a materialized layout
page, not a train275 diagnostic, and not quality evidence. The registered
external text-layout family remains open, while
`external_text_layout_support_train_only_diagnostic` remains pending. The
frozen preregistration plan and its hash are unchanged.

No OCR recognition, recognized text, target, label, route, domain, quality
split, optimizer, checkpoint, candidate inference, SCUT115, holdout40, visual
review, reserved blind, promotion, or current-primary replacement occurred.

## Runtime Observation

~~~text
probe time                                  2026-08-13 23:26 +0800
device / engine                             CPU / Transformers
page                                        hw5k_1011.jpg
thread caps                                 OMP/OpenBLAS/MKL/vecLib = 1
minimum system free memory                  31.0%
required minimum free memory                35.0%
post-termination free-memory range          67-74%
post-termination swap                       1,269.75 MiB
required maximum swap                       512 MiB
residual OCR/model processes                0
formal page/audit output                    none
terminal                                    PREREQUISITE_NEEDED
~~~

The detector process-tree peak RSS is intentionally recorded as unknown. The
health monitor raised on the failing system-memory sample before the prototype
persisted its aggregate peak, so inferring a value would overstate the
evidence. A leaked `loky` semaphore warning was observed, but static inspection
did not establish that PaddleX explicitly requested joblib parallel workers;
internal parallelism is therefore not assigned as the root cause.

Static inspection confirms that this path uses Torch/Transformers on CPU, not
MPS. The frozen weights occupy `88,020,412` bytes (`83.94 MiB`), and neither
the frozen model config nor the registered engine parameters request a lower
precision dtype. The earlier claim that the first page was resized to about
`1056x736` was incorrect: `limit_type=min` does not shrink the
`2436x1719` source. It only rounds the dimensions to multiples of 32, producing
an exact processed size of `2432x1728`.

## Static Memory-Risk Evidence

The reproducible static report reads the frozen JSON/configuration and Python
source as text and performs shape arithmetic only. It does not import or run
the detector. The large PPLCNetV4 stem and registered stage strides produce
the following backbone outputs for the exact processed page:

~~~text
stage 1                                    128 x 432 x 608
stage 2                                    256 x 216 x 304
stage 3                                    512 x 108 x 152
stage 4                                    896 x  54 x  76
highest-resolution 256-channel neck map    256.50 MiB (float32 payload)
one four-level upsampled list               256.50 MiB (float32 payload)
duplicate additional interpolated outputs  192.375 MiB (float32 payload)
old + replacement distinct list storage    448.875 MiB (float32 payload)
one list + torch.cat output                 513.00 MiB (float32 payload)
9x9 projection explicit-unfold equivalent   20.29 GiB (not measured allocation)
9x9 lateral explicit-unfold equivalent       5.07 GiB (not measured allocation)
~~~

Transformers `5.12.1` contains two high-resolution `9x9` neck convolutions and
constructs `upsampled` twice at source lines `306` and `314`. Python evaluates
the replacement list before dropping the old list, so the duplicate
construction can add three interpolated high-resolution outputs totaling
`192.375 MiB`; the `scale=1` entry reuses its existing feature. Separately,
`torch.cat` allocates a `256.50 MiB` output while its `256.50 MiB` input list is
live. The explicit unfold equivalents quantify the input shape presented to a
possible lowering implementation; they do not assert that PyTorch materialized
those buffers. These findings make full-resolution activation/workspace
pressure a concrete risk, but they remain static evidence and do not prove the
observed runtime failure's root cause or establish a safe retry configuration.

## Restart Separation

The abnormal `22:16:48` system restart occurred before this `23:26` probe.
There is no new panic, SOCD, or normal shutdown record for that restart, and no
model process was running at the restart time. The restart therefore cannot be
attributed to MPS or to this detector probe from current evidence. The earlier
`19:09` incident has separate watchdog/memory-pressure evidence and must not be
conflated with the later reset.

## Next Boundary

Do not retry OCR, MPS, primary inference, second-stage inference, or training
while swap exceeds `512 MiB`. Use the static full-resolution evidence to
establish a separately justified runtime that meets the existing limits
without changing registered detector parameters or relaxing safety gates.
Runtime parameter tuning is not authorized by the preregistration.

Only after swap falls below the gate may one target-free raw-source page be
retried. It must complete with safe peak RSS, free memory, and swap, leave zero
residual processes, and remain outside formal evidence paths. Formal
materialization remains additionally blocked by missing frozen primary and
second-stage prediction evidence; reconstructed caches must exactly match the
registered historical content hashes before publication.

## Evidence Hashes

~~~text
docs/external-text-layout-runtime-safety-probe-20260813.json
sha256 = 43423dec234c904d2117ae8dff8634d04b056109f635b2b2780d6f53b74b4f83

docs/external-text-layout-static-memory-risk-20260814.json
sha256 = 144bf4b1c55cfa8b4b19ec125b603dfb5daaf0fa02d0f92a9725adf60ab1651f

scripts/analysis/external_text_layout_materialization_runtime.py
sha256 = c40c4f771998e537baa7fa8914a7c3f84ee70efb052aabfa96d39314f11d2ecf

scripts/analysis/probe_external_text_layout_runtime_safety.py
sha256 = c8059c18d8e59a382da8a6de888dcd04a59d1cb90924d6155523e29512f086d7

scripts/analysis/report_external_text_layout_static_memory_risk.py
sha256 = c919f88b82c6d368d6868e8aa5d9d27ae9d2387ebab8ff40a22aeb544e6d3a71

tests/test_external_text_layout_runtime_safety_probe.py
sha256 = 680b0994c2139b0eb6dbe04d4cd001d6780e42a85d7feddcd9327e0c6e05626c

tests/test_external_text_layout_static_memory_risk.py
sha256 = 87473757551f021b2e9f5179347961e2e75cc1720f2c090a5e5359b6ebd8fd6e

tests/test_external_text_layout_support_prerequisite.py
sha256 = 201ab7be4c931014d1e52433c38f941ee70d101860556f63c3ed09518a3ac131

docs/external-text-layout-support-prerequisite-v1.json
sha256 = bf7b9c7d74adabb9a0d5b0f35f83d015642952c5d5b0e56389b1b05508d7d3c7
~~~

Intent: Preserve the external-layout hypothesis without exposing the host to another unsafe detector run.
Constraint: Transformers detector initialization crossed the fixed system-memory floor on a 24 GB M4 Pro, and swap remains above the launch gate.
Rejected: Relax memory or swap limits | repeats the unsafe condition instead of repairing it.
Rejected: Change detector thresholds, model, engine, geometry, or runtime parameters | post-registration runtime rescue is prohibited.
Rejected: Attribute the 22:16 restart to MPS or this probe | no model process or corresponding panic/SOCD evidence supports that claim.
Confidence: high for the runtime prerequisite; low for the cause of the 22:16 reset.
Scope-risk: narrow
Reversibility: clean
Directive: Do not run any model process while swap exceeds 512 MiB. Keep the family pending and require a safe one-page runtime result before formal cache reconstruction or materialization.
Tested: One target-free raw-source CPU detector page under spawn/process-group isolation, host lock, conflicting-process check, one-second health monitoring, fixed thread caps, cleanup, and zero residual model processes; static exact-shape/source analysis without model import or execution; focused fake-detector and static-report tests.
Not-tested: A safe detector runtime, formal 275-page materialization, reconstructed prediction-cache equality, train-only layout audit, optimizer, candidate inference, or any quality/promotion surface.
Related: docs/decisions/2026-08-13-external-text-layout-support-preregistration.md
