# External Text Layout Conditioned Monotonic Patch Materialization Pass

## Decision

`PASS`. The conditioned monotonic train patch index is now materialized and
audited from train275 only. The index reuses the monotonic target-lighter support
selection, adds frozen PP-OCRv6 text occupancy/confidence metrics, and remains
bound to recovered second-stage RGB, train labels, and frozen external
text-layout NPZ files.

## Evidence

- Patch index: `hardcase_lists/external-text-layout-conditioned-monotonic-train-patches-v1.csv`
- Patch index SHA256: `68d3099cfc1023593ef9558e8c356ba5eb6100ac34964767cbdd605325f5f10a`
- Summary: `outputs/external-text-layout-conditioned-monotonic-train-patches-v1/summary.json`
- Summary SHA256: `9e5ac380c4c9e83938f312e295402279f6b2427f71cfe84e342d870c5b917434`
- Audit: `outputs/external-text-layout-conditioned-monotonic-train-patches-v1/audit.json`
- Audit SHA256: `066612a6cb95a600a2d840e4d6a71756321ab412febc41b793eb87d5db436417`
- Builder: `scripts/analysis/build_external_text_layout_conditioned_monotonic_patch_index.py`
- Builder SHA256: `8df16d7237fdb8d4fc9156e53efa7e5979178d71422d6bdb3395f02d239624a1`
- Audit script: `scripts/analysis/audit_external_text_layout_conditioned_patch_materialization.py`
- Audit script SHA256: `417c57540d0bdbaf4dacceb6312503a43c715bbc84bd1a0f526890f7e5caa546`

## Results

| Check | Result |
| --- | ---: |
| Train role count | `275` |
| Candidate patches recomputed | `52,645` |
| Selected patches | `256` |
| Selected pages | `24` |
| Min target-lighter ratio | `0.3106536865234375` |
| Max target-lighter ratio | `0.9453582763671875` |
| Min preserve-negative ratio | `0.0546417236328125` |
| Avg text occupancy ratio | `0.40631022326928534` |
| Avg text confidence | `0.32617453634611593` |
| Input content SHA256 | `90e0ec3b7f5e676801d2cab2f9f0a7af9d1aff053ca01154d810258b3692004e` |
| Label content SHA256 | `dfd459f552bd0828221c90258f33f4eacc54220494c7e02b21a179894853e99e` |
| Layout content SHA256 | `8d4753dbd160c85043b1b65c1161e4cdc00059a07216bcc3d63b879754896750` |

## Boundary

This materialization used train labels only for patch-support construction. It
did not start real training, checkpoint generation, candidate inference,
inner-val15, SCUT115, holdout40, visual review, reserved blind access,
promotion, current-primary replacement, detector-threshold tuning, or resource
threshold tuning.

Intent: Bind the conditioned trainer to an exact train275 patch index before real training.
Constraint: The conditioned family may consume train labels only for registered target-lighter patch support; external text-layout grids remain target-free frozen NPZ inputs.
Rejected: Reuse the old monotonic patch CSV without layout evidence | it would not bind the conditioned causal change to layout input hashes or coverage metrics.
Rejected: Start training directly after surface integration | the quality loop requires a materialized and audited train patch index first.
Confidence: high
Scope-risk: narrow
Directive: Run only the registered 80-step CPU conditioned training command next; do not alter patch selection, detector/layout transforms, thresholds, or training schedule.
Tested: py313 and py310 patch-index tests 4/4; py313 and py310 materialization-audit tests 4/4; py313 and py310 preflight tests 7/7; py313 materialization audit PASS; py313 preflight replay PASS with patch_index_materialized=true; py313 and py310 py_compile; JSON validation; git diff --check.
Not-tested: real conditioned training, checkpoint audit, candidate inference, inner-val15, SCUT115, holdout40, visual review, reserved blind, promotion.
