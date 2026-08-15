# Stroke-Only Patch Suppression Input Custody Audit

## Decision

`PREREQUISITE_NEEDED`. The registered stroke-only patch suppression preflight
cannot run on the current worktree because the only authorized train-only source
rows are present, but their required baseline and source-candidate prediction
PNGs are absent.

This result does not KILL `stroke_only_patch_suppression_v1`: it proves an input
custody gap before image generation. The family remains pending until the exact
historical train160 prediction artifacts are restored or a materially different
train-only source-candidate input is preregistered. Do not recreate the missing
source candidate by training, fabricate predictions from metrics, or substitute
validation/blind rows.

## Evidence

- Audit: `outputs/stroke-only-patch-suppression-input-custody-audit-20260815/audit.json`
- Audit SHA256: `0fab10c783a89703c4342039df14d0a5372c68e3fab86ab94978e8073264427c`
- Audit script: `scripts/analysis/audit_stroke_only_patch_suppression_inputs.py`
- Audit script SHA256: `d43f98b44efe16d617d26428175e7b985457e5a21313333c2169ee91438bf2d9`
- Audit test: `tests/test_stroke_only_patch_suppression_input_audit.py`
- Audit test SHA256: `a42dd86d880519c9fec9fc323bf3b441c6c24bb4699ef1c300dee528d00a0713`
- Review CSV: `docs/product-quality-review-pages.csv`
- Review CSV SHA256: `5c0182d739a41db3962f1ee51b760cd93ce5bc2f68fd96979e81cd410cf52a52`
- Train160 selector replay rows: `outputs/selector_replay_exact129_outside_edit_lam16_union_train160_20260706/page_choices.csv`
- Train160 selector replay rows SHA256: `24c35c8d4eb73ee1e0ed4a60752e1b517e1cc94ff5e4575ce525664364b75517`

## Audit Result

- Selected rows: `train160/166.jpg`, `train160/190.jpg`, `train160/192.jpg`
- Present required paths: 6 source/target paths
- Missing required paths: 6 baseline/source-candidate prediction paths
- Missing baseline predictions:
  - `outputs/scut_train160_nonholdout_second_stage_baseline_20260706/pred/166.png`
  - `outputs/scut_train160_nonholdout_second_stage_baseline_20260706/pred/190.png`
  - `outputs/scut_train160_nonholdout_second_stage_baseline_20260706/pred/192.png`
- Missing source-candidate predictions:
  - `outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_interval_relaxed_gate_20260706/pred/166.png`
  - `outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_interval_relaxed_gate_20260706/pred/190.png`
  - `outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_interval_relaxed_gate_20260706/pred/192.png`

The audit verified that planned validation/blind outputs remain absent:
`inner_val15`, SCUT115, holdout40, reserved blind, and promotion surfaces are
still closed.

## Boundary

The audit reads CSV and file metadata only. It does not decode source, target,
baseline, or candidate pixels; it does not run model training; it does not
generate a checkpoint; it does not run candidate inference; and it does not run
any quality gate.

Intent: Keep the stroke-only successor evidence-gated by turning missing train-only source pixels into a formal prerequisite result.
Constraint: The registered preflight forbids model training, checkpoint generation, candidate inference, validation gates, reserved blind, promotion, and current-primary replacement.
Rejected: Recreate the missing relaxed-interval candidate by one-step training | preflight authority treats source-candidate predictions as inputs, not new checkpoint work.
Rejected: Fabricate candidate pixels from selector metrics | metrics contain page-level scores and paths, not reversible pixel outputs.
Rejected: Substitute SCUT115 or holdout40 rows | validation and blind surfaces remain closed before the train-only preflight passes.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Restore exact historical train160 baseline and relaxed-interval prediction PNGs, or preregister a different available train-only source candidate before running the stroke-only generator.
Tested: py313 focused input-audit tests 6/6; live audit returned PREREQUISITE_NEEDED with no training, candidate inference, target decode, quality gate, reserved blind, or promotion.
Not-tested: real stroke-only preflight materialization, inner-val15, SCUT115, holdout40, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-15-stroke-only-patch-suppression-preregistration.md
