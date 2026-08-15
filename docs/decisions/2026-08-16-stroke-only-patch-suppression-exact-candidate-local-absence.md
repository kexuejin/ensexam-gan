# Stroke-Only Patch Suppression Exact Candidate Local Absence

## Decision

`PREREQUISITE_NEEDED`. The exact historical selector-replay source-candidate
PNGs for train160 rows `166`, `190`, and `192` are still absent in the current
local workspace after `c8f471d`.

Do not substitute stale `pred/*.png` paths, review-pack montage pixels, or the
KILLed source-dark local-paper candidate. The stroke-only patch suppression
preflight remains closed until either the exact historical
`candidate/*.png` files are restored and hash-audited, or a materially
different train-only source candidate is preregistered.

## Evidence

- Audit output:
  `outputs/stroke-only-patch-suppression-exact-candidate-local-absence-20260816/audit.json`
- Audit output SHA256:
  `ea6c7c89852f62ed7ceb94f2eecab2e80f2194bd9e55d7d1edc72a560de05ff9`
- Audit script:
  `scripts/analysis/audit_stroke_only_patch_suppression_inputs.py`
- Audit script SHA256:
  `adfb9899ff4d3b575d9e1b9eefe30629f83e2d6912dc1b60219b3c8b2f022f42`
- Review CSV:
  `docs/product-quality-review-pages.csv`
- Review CSV SHA256:
  `cd187a148cd2ecf53f64a250fbdef44917eb026f3dde46e94e96dd74f8a4c64c`
- Selector replay:
  `outputs/selector_replay_exact129_outside_edit_lam16_union_train160_20260706/page_choices.csv`
- Selector replay SHA256:
  `24c35c8d4eb73ee1e0ed4a60752e1b517e1cc94ff5e4575ce525664364b75517`

## Result

The audit selected exactly three authorized `train160` rows and found:

| Check | Count |
| --- | ---: |
| present source/baseline/target paths | 9 |
| missing source-candidate paths | 3 |
| model training started | 0 |
| candidate inference started | 0 |
| quality gate started | 0 |

Missing paths:

- `outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_interval_relaxed_gate_20260706/candidate/166.png`
- `outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_interval_relaxed_gate_20260706/candidate/190.png`
- `outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_interval_relaxed_gate_20260706/candidate/192.png`

A local candidate-file search for those sample IDs under `*/candidate/`
returns only the already KILLed
`outputs/source-dark-local-paper-candidate-train160-20260816/candidate/*.png`
files, so they are not valid substitutes.

## Boundary

No target pixels were decoded by the audit. No model training, checkpoint
generation, candidate inference, inner-val15, SCUT115, holdout40, reserved
blind, visual review, promotion, or `artifacts/current-primary` replacement
occurred.

Intent: Stop spending iterations on unavailable historical PNGs while preserving fail-closed custody.
Constraint: Exact selector-replay candidate pixels cannot be reconstructed from metrics, stale pred paths, or review-pack composites.
Rejected: Use `pred/166.png`, `pred/190.png`, or `pred/192.png` | selector replay proves the authoritative paths are `candidate/*.png`.
Rejected: Use source-dark local-paper candidate PNGs | that family was KILLed for overerase before validation.
Confidence: high
Scope-risk: narrow
Directive: Next work should preregister a materially different train-only source candidate unless exact historical candidate PNGs are externally restored and hash-audited.
Tested: py313 custody audit returned PREREQUISITE_NEEDED with 9 present paths, 3 missing candidate paths, and zero training/inference/gate starts.
Not-tested: inner-val15, SCUT115, holdout40, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-16-stroke-only-patch-suppression-selector-path-custody.md
Related: docs/decisions/2026-08-16-source-dark-local-paper-lift-source-candidate-kill.md
