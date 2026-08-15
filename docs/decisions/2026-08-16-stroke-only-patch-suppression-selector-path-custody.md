# Stroke-Only Patch Suppression Selector Path Custody

## Decision

`PREREQUISITE_NEEDED` remains in force, with a more precise artifact path. The
registered train160 relaxed-interval source-candidate rows are now reconciled
against the selector replay CSV, and `docs/product-quality-review-pages.csv`
uses the selector replay's authoritative `candidate/*.png` paths instead of the
stale `pred/*.png` paths.

The three source-candidate PNGs are still absent locally, so the stroke-only
preflight remains closed. This result narrows the remaining work to restoring
the exact historical `candidate/166.png`, `candidate/190.png`, and
`candidate/192.png` artifacts or preregistering a materially different
available train-only candidate.

## Evidence

- Path-reconciled audit:
  `outputs/stroke-only-patch-suppression-input-custody-audit-path-reconciled-20260816/audit.json`
- Path-reconciled audit SHA256:
  `ea6c7c89852f62ed7ceb94f2eecab2e80f2194bd9e55d7d1edc72a560de05ff9`
- Selector replay:
  `outputs/selector_replay_exact129_outside_edit_lam16_union_train160_20260706/page_choices.csv`
- Selector replay SHA256:
  `24c35c8d4eb73ee1e0ed4a60752e1b517e1cc94ff5e4575ce525664364b75517`
- Review CSV:
  `docs/product-quality-review-pages.csv`
- Review CSV SHA256:
  `cd187a148cd2ecf53f64a250fbdef44917eb026f3dde46e94e96dd74f8a4c64c`
- Audit script:
  `scripts/analysis/audit_stroke_only_patch_suppression_inputs.py`
- Audit script SHA256:
  `5c1eb7b46b61d31f888d75ab2b8838af5a686e02364535ca3214921a26983b18`
- Audit test:
  `tests/test_stroke_only_patch_suppression_input_audit.py`
- Audit test SHA256:
  `553b1e9bb1333c22f258062ebe3470a6e53be5023bef84850859e5d724dada82`

## Reconciled Missing Paths

- `outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_interval_relaxed_gate_20260706/candidate/166.png`
- `outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_interval_relaxed_gate_20260706/candidate/190.png`
- `outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_interval_relaxed_gate_20260706/candidate/192.png`

## Selector Metrics

| Sample | Candidate Residual | Candidate Overerase | Residual Gain | Overerase Regret |
| --- | ---: | ---: | ---: | ---: |
| `train160/166.jpg` | 0.07994737750172533 | 0.0034830767347813004 | -0.0009273636991028317 | 0.0007477395698935697 |
| `train160/190.jpg` | 0.23782447740019064 | 0.003084212157415808 | 0.0019174924076167665 | 0.00020744814638484418 |
| `train160/192.jpg` | 0.40817832317191566 | 0.002329736409215052 | -0.0004951264028581481 | 0.00028278412496022436 |

## Boundary

The audit reads CSV metadata and file custody only. It does not decode candidate
pixels, generate candidate pixels, train a model, run a quality gate, inspect
validation/blind rows, or promote a checkpoint. The selector replay alignment
guard is a custody invariant, not a candidate inference authorization.

Intent: Prevent the stroke-only successor from chasing stale `pred/` paths when the selector replay records the source-candidate outputs under `candidate/`.
Constraint: The selector replay CSV is the authoritative record for the rejected relaxed-interval train160 source-candidate paths and metrics.
Rejected: Keep auditing stale `pred/*.png` paths | that masks the actual missing source-candidate artifacts and repeats wasted recovery work.
Rejected: Treat path reconciliation as source-candidate restoration | the `candidate/*.png` files are still absent.
Confidence: high
Scope-risk: narrow
Directive: Restore the exact selector-replay `candidate/*.png` files or preregister a materially different available train-only candidate before running stroke-only materialization.
Tested: py313 focused input-audit tests 7/7 with warnings as errors.
Tested: py313 py_compile for audit script and test.
Tested: live path-reconciled audit returned PREREQUISITE_NEEDED with 9 present paths, 3 missing `candidate/*.png` paths, and no training/candidate/quality/blind/promotion flags opened.
Not-tested: stroke-only materialization, inner-val15, SCUT115, holdout40, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-15-stroke-only-patch-suppression-baseline-restore-custody.md
