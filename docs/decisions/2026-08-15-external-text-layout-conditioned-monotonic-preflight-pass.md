# External Text Layout Conditioned Monotonic Preflight Pass

## Decision

`PASS`. The external text-layout conditioned monotonic family is now frozen at
preflight scope only: recovered second-stage RGB plus PP-OCRv6 text occupancy and
confidence may be used as five input channels, while the monotonic model still
emits RGB-only preserve-or-brighten candidates.

## Evidence

- Plan: `docs/external-text-layout-conditioned-monotonic-preflight-v1.json`
- Plan SHA256: `16671dc4d1afa6d1740702c709dad058a5bcc30e5a4282407c49c6fd979e9aae`
- Preflight output: `outputs/external-text-layout-conditioned-preflight-20260815/preflight.json`
- Preflight output SHA256: `dfe2ea2749d5439f6628a29c17669d4078c89ecd47126046fe30aa83f419a7c0`
- Validator: `scripts/analysis/validate_external_text_layout_conditioned_preflight.py`
- Validator SHA256: `80b43319004a58391026a31e5825b0192eb562e7bb0fc4871d2f506fbaed5475`
- Test: `tests/test_external_text_layout_conditioned_preflight.py`
- Test SHA256: `4ae0c52a40698e101ab0cbdaefa72c3a12dfee550f5525d022487e1216ac1034`

## Results

| Check | Result |
| --- | --- |
| Support diagnostic authority | `passed` |
| Model input channels | `5` |
| Model output channels | `3` |
| Identity initialization | exact |
| Layout encoder gradient | `6.388266774592921e-05` |
| Synthetic delta min | `1.1205673217773438e-05` |
| Candidate application | identity no-op, reachable brighten applied, darker candidate rejected |
| Training | not started |
| Checkpoint generation | not started |
| Candidate inference | not started |
| Quality gate | not started |
| Promotion | disabled |

## Boundary

This preflight does not authorize SCUT115, holdout40, visual review, reserved
blind access, promotion, current-primary replacement, threshold rescue, or
resource-threshold tuning. The next step is implementation of the exact
layout-conditioned trainer and application surface against the frozen plan, with
the first quality gate remaining `inner_val15`.

Intent: Freeze the leakage-aware text-layout-conditioned trainer/application contract before any quality surface opens.
Constraint: External text-layout support is train-only evidence; upstream detector corpus overlap with SCUT/HW5K remains unverified.
Rejected: Treat support PASS as direct SCUT115/holdout40 authority | the boundary requires a separate conditioned preflight first.
Rejected: Tune application thresholds or detector transforms | the preflight preserves the fixed 12-gray monotonic application gate and exact recovered layout channels.
Confidence: high
Scope-risk: moderate
Directive: Implement only the frozen five-channel trainer/application next; do not run candidate inference or broader quality gates before that implementation has its own focused verification.
Tested: py313 validator PASS; py313 7 conditioned preflight tests; py313 2 layout-conditioned monotonic tests; py310 validator PASS; py310 7 conditioned preflight tests; py310 2 layout-conditioned monotonic tests; git diff --check.
Not-tested: candidate training, candidate inference, SCUT115, holdout40, visual review, reserved blind, promotion.
