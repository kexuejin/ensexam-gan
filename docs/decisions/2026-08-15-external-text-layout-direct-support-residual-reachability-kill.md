# External Text Layout Direct Support Residual Reachability KILL

## Decision

`KILL`. The train-only direct support-score residual diagnostic completed on the
registered 256 train patches using only the existing external text-layout support
fold fits. It proved the closed-form projection is reachable on target-lighter
pixels, but it is not safe: preserve pixels also cross the frozen 12-gray gate at
a very high rate.

This closes the direct-support residual v1 family. Do not rescue it by lowering
the gate, changing score normalization, tuning detector thresholds or layout
transforms, adjusting support-score weights, or opening candidate/validation
surfaces.

## Evidence

- Diagnostic output: `outputs/external-text-layout-direct-support-residual-proposal-diagnostic-20260815/audit.json`
- Diagnostic output SHA256: `3a5ecf3b8acec9eea35524d4d1e7e3789303bd0ba69954197e1ee2a3d09f2d69`
- Diagnostic script: `scripts/analysis/audit_external_text_layout_direct_support_residual_reachability.py`
- Diagnostic script SHA256: `a1894f6727e7cc49243f274b76d291f33cd518eac37558222b046c7f78793c3d`
- Diagnostic test: `tests/test_external_text_layout_direct_support_residual_reachability.py`
- Diagnostic test SHA256: `f366d5c9fc762cfd8cf96452888bca03ee04062d6901c3c047126e7542a58010`
- Preflight lifecycle test: `tests/test_external_text_layout_direct_support_residual_preflight.py`
- Preflight lifecycle test SHA256: `0d021ffaf97b58178a6f086a5e5de9839eae91c1dbadb894b6e14e6c7b4b3c6f`

## Results

| Metric | Result | Gate |
| --- | ---: | ---: |
| Patch count | `256` | `256` |
| Positive gate ratio | `0.8899201285493683` | `>= 0.05` |
| Preserve gate ratio | `0.4356632845245391` | `<= 0.005` |
| Positive-over-preserve gate margin | `0.4542568440248292` | `>= 0.04` |
| Reachable patch ratio | `1.0` | `>= 0.1` |
| Positive delta mean | `18.297960147062593` gray | reference |
| Preserve delta mean | `9.047266582509286` gray | reference |

The failure is not lack of effect. All 256 patches had positive-gate support,
and positive pixels reached the registered gate at high coverage. The blocker is
that preserve pixels reached the same gate at `87.1x` the allowed rate
(`0.4356632845245391 / 0.005`), so page-level candidate inference would be an
overerase-risk path rather than a promotion-safe path.

## Boundary

No model training, checkpoint generation, candidate inference, inner-val15,
SCUT115, holdout40, visual review, reserved blind access, promotion, or
current-primary replacement was started. The next quality attempt needs a
materially different mechanism that separates preserve safety before any
candidate output is generated.

Intent: Close a reachable but unsafe closed-form residual path before it reaches candidate inference.
Constraint: The diagnostic used only registered 256 train patches and the previously PASSed external text-layout fold fits; preserve gate must stay at or below 0.005.
Rejected: Open inner-val15 because positive reachability is strong | preserve gate ratio is 0.435663, so the projection is structurally unsafe before validation.
Rejected: Tune score normalization or lower/raise gates | that would be post-result threshold rescue after a registered KILL.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not continue direct-support residual v1; preregister a different mechanism with explicit preserve separation before any candidate surface opens.
Tested: py313 and py310 direct-support residual preflight/reachability tests 11/11 with warnings as errors; py313 and py310 py_compile; py313 formal reachability diagnostic KILL; planned training/candidate/quality outputs stayed closed.
Not-tested: model training, checkpoint generation, candidate inference, inner-val15, SCUT115, holdout40, visual review, reserved blind, promotion.
