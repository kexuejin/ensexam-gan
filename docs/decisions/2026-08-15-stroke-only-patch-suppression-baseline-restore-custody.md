# Stroke-Only Patch Suppression Baseline Restore Custody

## Decision

`PREREQUISITE_NEEDED` remains in force, but the input custody gap is now
narrowed. The three registered train160 baseline predictions for
`train160/166.jpg`, `train160/190.jpg`, and `train160/192.jpg` were restored
from `artifacts/current-primary` with the checked-in residual-repair runner, and
the expected baseline prediction paths now resolve through local symlinks.

The stroke-only preflight still cannot run because the registered relaxed-
interval source-candidate predictions for the same three rows remain absent.
No candidate surface, validation surface, blind surface, promotion path, or
`artifacts/current-primary` replacement is open.

## Evidence

- Restored baseline metrics:
  `outputs/stroke-only-patch-suppression-baseline-restore-20260815/metrics.csv`
- Restored baseline metrics SHA256:
  `d70683dbd58d301504336fa8387babf732af59431ff2908c60aeb47b21ffa447`
- Post-restore custody audit:
  `outputs/stroke-only-patch-suppression-input-custody-audit-after-baseline-restore-20260815/audit.json`
- Post-restore custody audit SHA256:
  `069c10cbafa01b4168606e1d8dce07ab57453e9c247d894542408c40998561d4`
- Restored baseline prediction SHA256s:
  - `166.png`: `f547b6df01cf0756c5cdad9242f92ac509e6e646141127909b60d810b1007806`
  - `190.png`: `05604489541b7b43e8f967850aec363e83216bbe9047e0aeaa1cf54f60f57ea3`
  - `192.png`: `a85ca68dee08e8f8439bf5fba21981d551c821f40379506ed61bd4d9fcb0d64e`

## Restored Baseline Metrics

| Sample | Residual | Overerase |
| --- | ---: | ---: |
| `train160/166.jpg` | 0.0790200138026225 | 0.0027353371648877307 |
| `train160/190.jpg` | 0.2397419698078074 | 0.002876764011030964 |
| `train160/192.jpg` | 0.4076831967690575 | 0.0020469522842548275 |

## Post-Restore Audit Result

- Selected rows: 3 train160 rows.
- Present required paths: 9.
- Missing required paths: 3.
- Remaining missing paths:
  - `outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_interval_relaxed_gate_20260706/pred/166.png`
  - `outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_interval_relaxed_gate_20260706/pred/190.png`
  - `outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_interval_relaxed_gate_20260706/pred/192.png`
- Planned outputs remain absent:
  - `outputs/stroke-only-patch-suppression-preflight-20260815`
  - `outputs/stroke-only-patch-suppression-inner-val15-candidate`
  - `outputs/stroke-only-patch-suppression-scut115-candidate`
  - `outputs/stroke-only-patch-suppression-holdout40-candidate`
  - `outputs/stroke-only-patch-suppression-reserved-blind-candidate`

## Boundary

The restore only reconstructs current-primary baseline predictions for the
authorized train-only rows. It does not recreate the missing relaxed-interval
source candidate, train a model, generate a checkpoint, run candidate inference,
decode targets during generation, run quality gates, inspect validation/blind
rows, or promote any model.

Intent: Narrow the stroke-only successor custody blocker with exact baseline evidence before any pixel-generation preflight.
Constraint: Source-candidate predictions are still preregistered inputs, not artifacts that this preflight may regenerate.
Rejected: Treat restored baselines as a stroke-only preflight PASS | the source-candidate side is still missing.
Rejected: Use baseline predictions as the source candidate | that would be a no-op, not a materially different registered candidate.
Rejected: Substitute validation, SCUT115, holdout40, or reserved-blind rows | every non-train surface remains closed before the preflight passes.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Restore the exact relaxed-interval train160 source-candidate PNGs or preregister a materially different available train-only candidate before running `generate_stroke_only_patch_suppression.py`.
Tested: Post-restore custody audit returned 9 present and 3 missing required paths, with training/candidate/quality/blind/promotion flags all closed.
Not-tested: stroke-only preflight materialization, inner-val15, SCUT115, holdout40, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-15-stroke-only-patch-suppression-input-custody-audit.md
