# External Text Layout Binary Mask Residual Reachability KILL

## Decision

`KILL`. The train-only binary external-text occupancy mask diagnostic completed
on the registered 256 train patches using the frozen recovered external-layout
cache. The detector occupancy mask is reachable on target-lighter pixels, but it
is not preserve-safe: preserve pixels are inside the same occupancy mask at a
high rate.

This closes `external_text_layout_binary_mask_residual_v1`. Do not rescue it by
changing detector thresholds, adding confidence-threshold sweeps, changing layout
transforms, lowering/raising gates, or opening candidate/validation surfaces.

## Evidence

- Diagnostic output: `outputs/external-text-layout-binary-mask-residual-diagnostic-20260815/audit.json`
- Diagnostic output SHA256: `b78d48013cbb86915a2556bced41753d15ca2f5c6335ad632607b21748f69155`
- Diagnostic script: `scripts/analysis/audit_external_text_layout_binary_mask_residual_reachability.py`
- Diagnostic script SHA256: `bf7b6c754f7a4910fb0c2e666bc2902a4a671789b538ee68f46060ec2f8a92e2`
- Diagnostic test: `tests/test_external_text_layout_binary_mask_residual_reachability.py`
- Diagnostic test SHA256: `b94a5afc2c555dd84eb9bbb568a8fc8c9459ccf73ffa015019887c3e717641bc`
- Preflight output: `outputs/external-text-layout-binary-mask-residual-preflight-20260815/preflight.json`
- Preflight output SHA256: `b2da9a2c0f76074261a8d9b80fb02b3d2eec4d99d29fcc85a7cf59e626850501`
- Preflight test: `tests/test_external_text_layout_binary_mask_residual_preflight.py`
- Preflight test SHA256: `56ec17c068a2f97ee2d8565b7bcdf83c4b34fa9750817d0e83dfd69499f4768f`

## Results

| Metric | Result | Gate |
| --- | ---: | ---: |
| Patch count | `256` | `256` |
| Positive gate ratio | `0.4167254554223624` | `>= 0.05` |
| Preserve gate ratio | `0.38511495786256056` | `<= 0.005` |
| Positive-over-preserve gate margin | `0.03161049755980183` | `>= 0.04` |
| Reachable patch ratio | `0.8203125` | `>= 0.1` |
| Positive delta mean | `8.501199290616192` gray | reference |
| Preserve delta mean | `7.856345140396235` gray | reference |

The failure is preserve safety, not reachability. The occupancy mask covers
target-lighter pixels on many patches, but it also covers preserve pixels at
`77.0x` the allowed rate (`0.38511495786256056 / 0.005`). The positive-over-
preserve margin is below the frozen minimum as well.

## Boundary

No model training, checkpoint generation, candidate inference, inner-val15,
SCUT115, holdout40, visual review, reserved blind access, promotion, or
current-primary replacement was started. The next quality attempt should stop
treating external text layout alone as an edit support signal and should
preregister a materially different preserve-separation mechanism.

Intent: Close the raw external text occupancy mask as a direct residual support path before it reaches candidate inference.
Constraint: The diagnostic used only registered 256 train patches and the frozen recovered external-layout cache; preserve gate must stay at or below 0.005.
Rejected: Tune detector confidence or occupancy thresholds | preserve leakage is structural at the frozen detector mask and threshold sweeps would be post-result rescue.
Rejected: Open inner-val15 because reachability is high | preserve gate ratio is 0.385115, so page-level candidate inference would be unsafe.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not continue `external_text_layout_binary_mask_residual_v1`; external layout alone has now failed as full score, incremental score, and binary support.
Tested: py313 and py310 binary-mask focused tests 9/9 with warnings as errors; py313 and py310 py_compile; py313 live `run_diagnostic` recompute returned the persisted KILL metrics; jq validation of plan, preflight, and diagnostic JSON.
Not-tested: model training, checkpoint generation, candidate inference, inner-val15, SCUT115, holdout40, visual review, reserved blind, promotion.
