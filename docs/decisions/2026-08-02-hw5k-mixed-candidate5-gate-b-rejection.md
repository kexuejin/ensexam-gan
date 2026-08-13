# HW5K Mixed Candidate 5: Gate A Passed, Gate B Rejected

## Decision

Reject candidate 5 as a universal-checkpoint replacement. It passes the
HW5K-domain Gate A, but fails both SCUT source-domain checks in Gate B. Per the
predeclared stop-on-first-failure protocol, do not run Gate C, SCUT115, or
holdout40, and do not change `artifacts/current-primary`.

Candidate 5 is still a useful result: a 50/50 SCUT/HW5K mix plus train-only
baseline-tail protection retained and slightly improved candidate 4's HW5K
gain while recovering most of candidate 4's SCUT residual regression. It did
not recover the final source-domain margin, and overerase remained a systematic
failure. The current shared-weight objective therefore does not yet support a
universal checkpoint across these domains.

## Candidate And Controls

- Training config:
  `configs/local/config.local-hw5k-mixed-scut130-hw5k130-50pct-guard-step6400-respress-bs4-mps.yaml`
- Frozen run config:
  `artifacts/trials/hw5k-mixed-scut130-hw5k130-50pct-guard-jointtail-lite-step6400-respress-bs4-20260730/ensexam/20260801_183409/config.yaml`
- Checkpoint:
  `artifacts/trials/hw5k-mixed-scut130-hw5k130-50pct-guard-jointtail-lite-step6400-respress-bs4-20260730/ensexam/20260801_183409/epoch_1.pth`
- Checkpoint SHA-256:
  `8da25117dd883f95059b6d7067e3dc3580da11339de365ef904f711db4a1f490`
- Frozen config SHA-256:
  `c0ab5cc2a96dcaffa86dc75754c2a9bb9bfdc741c8ff7319e93bf8e2abc8adf8`
- Training completed all 6,400 steps with finite final losses:
  `G=18.3065`, `D=1.5840`, `baseline_tail=0.1898`.
- Data: 130 SCUT `inner_train_130` pages plus 130 disjoint HW5K-train
  pages, page-balanced sampling, physical batch 4, MPS, no AMP.
- Protection cache: 260 residual-safe masks plus 260 outside-safe masks.
- No HW5K-test, Dev40, SCUT115, holdout40, selector, page-specific rule, or
  post-freeze threshold choice was used for training.

## Gate A: HW5K Dev 232 — Pass

Evidence:

- `outputs/hw5k_dev_candidate5_full232_20260801/post_freeze_metrics.csv`
  (`sha256=35a499413df7e7cf1e164d6bd7bb3caa726ef8da89e771285f95c5786517574f`)
- `outputs/hw5k_dev_candidate5_full232_20260801/post_freeze_summary.json`
  (`sha256=fe2d5ede3620f6f081a71679a134590cdddb6a2d4c2b8e1b9a8b21a58f49114b`)
- `outputs/hw5k_train_intake_20260729/gate_a_result_candidate5.json`

| Metric | Current-primary | Candidate 5 | Result |
| --- | ---: | ---: | --- |
| Mean residual ratio | 0.723852852720 | 0.562573946165 | pass (-22.281%) |
| Mean overerase ratio | 0.064444891090 | 0.027853222229 | pass (-56.780%) |
| Max residual ratio | 0.954781812446 | 0.930664938705 | pass |
| Max overerase ratio | 0.455976452017 | 0.446714811668 | reported |

Candidate 5 does not merely preserve candidate 4's HW5K result. Its mean
residual is better than candidate 4 (`0.562574` versus `0.575718`), although
its mean overerase is worse (`0.027853` versus `0.018721`). All predeclared
Gate A checks still pass.

## Gate B: SCUT Inner-Val15 — Fail

Both current-primary and candidate 5 were rerun source-only with identical
settings: MB copy-mask, `mb_cov8_step` automatic threshold, page overlap 32,
batch 8, and fixed post-freeze thresholds 12/12.

Evidence:

- Current-primary metrics:
  `outputs/scut_innerval15_current_primary_20260802/post_freeze_metrics.csv`
  (`sha256=4e039dc36967e4fa5f5c762cb0230a5f1c61ebe331d220c34b4573d74ab7bfe2`)
- Current-primary summary:
  `outputs/scut_innerval15_current_primary_20260802/post_freeze_summary.json`
  (`sha256=dad177e3868722c5eaaec26409f99fe441ce8d5e71702f84ce60973ba599c80a`)
- Candidate 5 metrics:
  `outputs/scut_innerval15_candidate5_20260802/post_freeze_metrics.csv`
  (`sha256=a209e042170913a2ce5b9b213c6ec6e95ce18ffa7fd45f4269aa3fa21e2d0eed`)
- Candidate 5 summary:
  `outputs/scut_innerval15_candidate5_20260802/post_freeze_summary.json`
  (`sha256=0e2af61b10c78a040f4a28671ea024f463388eb04b0105b30364710954f65146`)
- Compact gate result:
  `outputs/hw5k_train_intake_20260729/gate_b_result_candidate5.json`

| Metric | Current-primary | Candidate 5 | Result |
| --- | ---: | ---: | --- |
| Mean residual ratio | 0.176948604902 | 0.181196254772 | fail (+2.400%) |
| Mean overerase ratio | 0.002324671360 | 0.003718713621 | fail (+59.967%) |
| Max residual ratio | 0.386987730998 | 0.421656569770 | regresses |
| Max overerase ratio | 0.006105781457 | 0.011770824440 | regresses |

Page regressions versus current-primary:

- Residual: 10/15 pages.
- Overerase: 12/15 pages.

This is not a one-page tail miss. Candidate 5 nearly closes the aggregate
residual gap left by candidate 4, but it remains a systematic source-domain
regression and fails the exact gate that was fixed before training.

## Interpretation And Next Boundary

The experiment supports the mechanism but rejects the promotion:

1. Increasing SCUT share and applying baseline-aware support is compatible
   with strong HW5K adaptation; it improved HW5K mean residual beyond
   candidate 4.
2. The same controls recover most of candidate 4's SCUT residual damage
   (`0.254132` to `0.181196`), showing that source retention is steerable.
3. They do not eliminate source-domain regression, especially overerase. The
   remaining miss is too large and too broad to justify threshold rescue.

Do not continue scalar tuning of selector or evaluation thresholds. A further
universal-checkpoint attempt needs a materially different mechanism with a
predeclared reason to decouple the domains, such as domain-conditioned model
capacity or a Pareto-aware multi-domain objective. The lower-risk product path
is a separately approved domain-routed checkpoint design. Until one of those
paths passes the same full gate chain, keep `artifacts/current-primary` as the
default.

## Verification

- 260/260 train pages had both required cache masks.
- 24 cached-support/runtime tests passed before training.
- Training completed 6,400/6,400 steps and wrote a loadable checkpoint.
- HW5K Gate A used 232/232 frozen predictions.
- SCUT Gate B used 15/15 frozen predictions for both checkpoints.
- Gate C, SCUT115, holdout40, and a new final blind set were not run because
  Gate B failed.
