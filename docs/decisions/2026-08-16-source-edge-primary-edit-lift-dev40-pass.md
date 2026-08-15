# Source-Edge Primary-Edit Lift Dev40 Pass

## Decision

`PASS`. The fixed `source_edge_primary_edit_lift_v1` source candidate passed the deterministic Dev40 metric gate after passing `inner_val15`.

This is not a whole-checkpoint `blind_eval_manifest` gate. The candidate is a deterministic source/baseline postprocess, so the applicable evidence is a fixed 40-page candidate metric CSV compared against the frozen current-primary Dev40 metrics CSV with no page-specific selection.

## Evidence

- Dev40 result JSON:
  `docs/source-edge-primary-edit-lift-dev40-gate-v1.json`
- Dev40 result JSON SHA256:
  `de34c11435838b2880e63305f149b48f4eb1898d856f8a61f22fd5e4e63f5945`
- Dev40 input rows:
  `outputs/source-edge-primary-edit-dev40-gate-20260816-v1/dev40_review_rows.csv`
- Dev40 input rows SHA256:
  `f9c521138fb9d1edc6f528f3d1d0464c45aac346ec3ceefb714bd38e7549c752`
- Dev40 source summary:
  `outputs/source-edge-primary-edit-dev40-gate-20260816-v1/source_candidate/summary.json`
- Dev40 source summary SHA256:
  `81bec1302a07acc57c303d12b6bd6da30fcaea56d3eb89a28ab960635709a5b7`
- Dev40 source diagnostics:
  `outputs/source-edge-primary-edit-dev40-gate-20260816-v1/source_candidate/diagnostics.csv`
- Dev40 source diagnostics SHA256:
  `7a15a6c7447034346308b097fda4a5dfac9b39ccce0d7380fa957be68413d4aa`
- Dev40 source review rows:
  `outputs/source-edge-primary-edit-dev40-gate-20260816-v1/source_candidate/review_rows.csv`
- Dev40 source review rows SHA256:
  `67b8cc98c384e14ae6658ce40e90e714b6ad9c63266593d0f3e6d831821ffeed`
- Dev40 candidate metrics:
  `outputs/source-edge-primary-edit-dev40-gate-20260816-v1/source_candidate/post_freeze_metrics.csv`
- Dev40 candidate metrics SHA256:
  `092089d5a70cca5b5f6842a699a43fde8e4258d6fe8beb2636ae81c54d80282b`
- Dev40 candidate gate JSON:
  `outputs/source-edge-primary-edit-dev40-gate-20260816-v1/dev_candidate_gate.json`
- Dev40 candidate gate JSON SHA256:
  `2221061e09bbc28807455fdc7717672ca2d692c3c765f04136c8d1f7a7db6a81`

Frozen baseline CSV:
`/Volumes/Kapp/tmp/scut_dev40_current_primary_baseline_20260726_primaryonly/post_freeze_metrics.csv`
with SHA256 `d409591570434507a401b801361d7156e2039b4eb7a2203a41f258fd1c1aebe7`.

## Gate Result

| Metric | Baseline | Candidate | Result |
| --- | ---: | ---: | --- |
| Mean residual ratio | 0.13591130004908952 | 0.13470177127700783 | improves |
| Mean overerase ratio | 0.002511352723878475 | 0.002511352723878475 | does not regress |
| P95 residual ratio | 0.3049251144096621 | 0.30289272749294965 | does not regress |
| P95 overerase ratio | 0.0064474865233278816 | 0.0064474865233278816 | does not regress |
| Max residual ratio | 0.46329031644365526 | 0.4619902974189866 | does not regress |
| Max overerase ratio | 0.010905079318222868 | 0.010905079318222868 | does not regress |

Page-level checks also passed: zero residual-regression pages and zero overerase-regression pages across all 40 Dev40 samples.

## Boundary

The Dev40 gate compared one fixed candidate surface against one frozen baseline surface. It did not train, generate a checkpoint, tune thresholds, choose page-specific outputs, run SCUT115, run holdout40, open reserved blind, perform visual review, promote, or replace `artifacts/current-primary`.

Intent: Advance a fixed source-edge primary-edit candidate after it passed the development metric gate.
Constraint: The source candidate is deterministic postprocess output, not a checkpoint-backed blind_eval_manifest candidate.
Rejected: Treat this as a whole-checkpoint fixed-regression gate | no candidate checkpoint exists for this deterministic postprocess route.
Rejected: Tune thresholds, alpha, lift cap, component bounds, kernel, or page choices from Dev40 | Dev40 is a gate, not a selector.
Confidence: medium
Scope-risk: moderate
Directive: Evaluate SCUT115 and holdout40 only for the same fixed source candidate; do not use those surfaces for tuning or substitution.
Tested: deterministic Dev40 materialization over 40 frozen samples.
Tested: generic dev candidate metric gate accepted all aggregate, tail, and page-level checks.
Not-tested: SCUT115, holdout40, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-16-source-edge-primary-edit-lift-inner-val15-pass.md
