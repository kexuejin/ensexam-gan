# External Text Layout Incremental Support Residual Reachability KILL

## Decision

`KILL`. The train-only incremental support residual diagnostic completed on the
registered 256 train patches using only the frozen external-layout full and
RGB-ablation fold fits. The incremental score did not provide reliable preserve
separation: only one of five page-folded calibration folds had positive-center
score above preserve-center score.

This closes `external_text_layout_incremental_support_residual_v1`. Do not
rescue it by changing score normalization, flipping score direction,
retuning detector thresholds or layout transforms, lowering/raising the gate,
or opening candidate/validation surfaces.

## Evidence

- Diagnostic output: `outputs/external-text-layout-incremental-support-residual-diagnostic-20260815/audit.json`
- Diagnostic output SHA256: `80c15c71e0e878a081a65c2fbcb81c51319b05857758c0788e4e9491dd8194c1`
- Diagnostic script: `scripts/analysis/audit_external_text_layout_incremental_support_residual_reachability.py`
- Diagnostic script SHA256: `0703ef29fb32412a9b5d864c6a58cff3673906335ad1c5a752fc6bd521808e51`
- Diagnostic test: `tests/test_external_text_layout_incremental_support_residual_reachability.py`
- Diagnostic test SHA256: `74884ffa6bff7f24e7f594138537af9a0cfdd6bb1cc4ce049e4a764d03612867`
- Preflight output: `outputs/external-text-layout-incremental-support-residual-preflight-20260815/preflight.json`
- Preflight output SHA256: `cdb48622584d501cc82e8783d690171ac9fa24b9bed1a1d0273358eed42dd9bf`
- Preflight test: `tests/test_external_text_layout_incremental_support_residual_preflight.py`
- Preflight test SHA256: `d385763f48e632db0d2bea07014aa9db5ce892e07824888768049b4355ea1395`

## Results

| Metric | Result | Gate |
| --- | ---: | ---: |
| Patch count | `256` | `256` |
| Ordered center folds | `1` | `>= 5` |
| Positive gate ratio | `0.06302786269657285` | `>= 0.05` |
| Preserve gate ratio | `0.06281005370228102` | `<= 0.005` |
| Positive-over-preserve gate margin | `0.00021780899429182932` | `>= 0.04` |
| Reachable patch ratio | `0.125` | `>= 0.1` |
| Positive delta mean | `1.285768399010086` gray | reference |
| Preserve delta mean | `1.2813250955265332` gray | reference |

The failure is not merely low reachability: the positive gate ratio and reachable
patch ratio clear their minimums. The blocker is that the incremental score
does not rank positive pixels above preserve pixels in a stable way. Positive
and preserve gate rates are effectively identical, and preserve leakage remains
`12.6x` the allowed rate (`0.06281005370228102 / 0.005`).

## Boundary

No model training, checkpoint generation, candidate inference, inner-val15,
SCUT115, holdout40, visual review, reserved blind access, promotion, or
current-primary replacement was started. The next quality attempt needs a
materially different preserve-separation mechanism, not a rescue of this
incremental score family.

Intent: Close a layout-incremental projection path whose held-out train-fold score centers do not separate preserve from target-lighter pixels.
Constraint: The diagnostic reused the frozen direct-support gates and the registered 256 train patches; no candidate, quality, or validation surface was allowed.
Rejected: Flip the incremental score direction | only one fold orders correctly and flipping would be post-result score rescue, not a preregistered causal mechanism.
Rejected: Lower the gate to capture weak deltas | positive and preserve gate ratios are nearly identical, so a lower gate would increase leakage rather than establish safety.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not continue `external_text_layout_incremental_support_residual_v1`; preregister a different preserve-separation mechanism before any candidate surface opens.
Tested: py313 and py310 incremental support residual focused tests 12/12 with warnings as errors; py313 and py310 py_compile; py313 live `run_diagnostic` recompute returned the persisted KILL metrics; jq validation of plan, preflight, and diagnostic JSON.
Not-tested: model training, checkpoint generation, candidate inference, inner-val15, SCUT115, holdout40, visual review, reserved blind, promotion.
