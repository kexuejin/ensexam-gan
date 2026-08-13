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
MPS. The frozen weights are about `84 MiB`, and PaddleX does not pass an
explicit low-precision dtype to `from_pretrained`. The first page is resized
from `2436x1719` to approximately `1056x736`; the medium-det neck then retains
and upsamples several feature maps, uses high-resolution `9x9` convolutions,
and contains a duplicate construction of its upsampled list in Transformers
`5.12.1`. These are plausible contributors to temporary activation/workspace
pressure, but their exact contribution has not been measured and they are not
promoted to a root-cause claim.

## Restart Separation

The abnormal `22:16:48` system restart occurred before this `23:26` probe.
There is no new panic, SOCD, or normal shutdown record for that restart, and no
model process was running at the restart time. The restart therefore cannot be
attributed to MPS or to this detector probe from current evidence. The earlier
`19:09` incident has separate watchdog/memory-pressure evidence and must not be
conflated with the later reset.

## Next Boundary

Do not retry OCR, MPS, primary inference, second-stage inference, or training
while swap exceeds `512 MiB`. Continue static investigation of Transformers
detector initialization and establish a separately justified runtime that
meets the existing limits without changing registered detector parameters or
relaxing safety gates. Runtime parameter tuning is not authorized by the
preregistration.

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

scripts/analysis/probe_external_text_layout_runtime_safety.py
sha256 = 1dafff0f6a2d50ebe4ddc087502408c008e0ddd68990bc243f4d12fe3768749c

tests/test_external_text_layout_runtime_safety_probe.py
sha256 = 1efce15af7b9e3f31e159f3a6fb7904d62cd091c330030d263572789b0870c3d

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
Tested: One target-free raw-source CPU detector page under spawn/process-group isolation, host lock, conflicting-process check, one-second health monitoring, fixed thread caps, cleanup, and zero residual model processes; 24 focused fake-detector tests.
Not-tested: A safe detector runtime, formal 275-page materialization, reconstructed prediction-cache equality, train-only layout audit, optimizer, candidate inference, or any quality/promotion surface.
Related: docs/decisions/2026-08-13-external-text-layout-support-preregistration.md
