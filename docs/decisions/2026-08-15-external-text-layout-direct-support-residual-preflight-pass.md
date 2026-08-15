# External Text Layout Direct Support Residual Preflight PASS

## Decision

`PASS`. The external text-layout direct-support residual proposal is frozen as a
materially different successor to the killed conditioned monotonic U-Net path.
It may use the already PASSed closed-form external text-layout support score to
define a bounded, nonnegative RGB residual projection, but only for a train-role
reachability diagnostic.

This does not authorize model training, checkpoint generation, candidate
inference, inner-val15, SCUT115, holdout40, visual review, reserved blind access,
promotion, current-primary replacement, or any threshold/resource rescue.

## Evidence

- Proposal: `docs/external-text-layout-direct-support-residual-proposal-v1.json`
- Proposal SHA256: `fbdfc25d977238f33c7631a59b3ef76ac56b8a8c9f59384730f44a4555aab054`
- Preflight output: `outputs/external-text-layout-direct-support-residual-preflight-20260815/preflight.json`
- Preflight output SHA256: `5b2b753399e11e2a679e161cb389420e7332265184b02dda1dc4d59ed25363b5`
- Validator: `scripts/analysis/validate_external_text_layout_direct_support_residual_preflight.py`
- Validator SHA256: `410fba7e04c9e6b1b94bf7f7b1442dd5d2dd1ab7bd5bd0b0510582585139c5c5`
- Test: `tests/test_external_text_layout_direct_support_residual_preflight.py`
- Test SHA256: `08da07e58133bdbe9b1a0a666153eb792d65bf362916eb12ef1d538ec0bacc81`

## Results

| Check | Result |
| --- | --- |
| Support diagnostic authority | `PASS` |
| Conditioned monotonic checkpoint authority | `KILL` |
| Synthetic projection direction | nonnegative RGB only |
| Synthetic delta range | `0.0` to `20.4` gray |
| Synthetic gate support | `2 / 5` scores reach the frozen 12-gray gate |
| Training | not started |
| Checkpoint generation | not started |
| Candidate inference | not started |
| Quality gate | not started |
| Target decode | false |
| Promotion | disabled |

The upstream support diagnostic remains train-only evidence: full mean fold AUC
was `0.6879765165680307`, full-minus-RGB-ablation margin was
`0.04347041960931275`, and macro median page AUC was
`0.7221097946166992`. The prior conditioned monotonic checkpoint remains closed
because all 256 train patches stayed below the frozen 12-gray gate.

## Boundary

The next authorized step is only the train-role direct-support reachability
diagnostic registered in the proposal:

- required patch count: `256`
- minimum positive gate ratio: `0.05`
- maximum preserve gate ratio: `0.005`
- minimum positive-over-preserve gate margin: `0.04`
- minimum reachable patch ratio: `0.1`

If that diagnostic KILLs, close this family without detector-threshold tuning,
layout-transform tuning, gate lowering, repeated conditioned-monotonic training,
learning-rate or step sweeps, loss-weight sweeps, or early validation-gate access.

Intent: Preserve progress after the conditioned U-Net gate failure by freezing a smaller train-only closed-form reachability test.
Constraint: The support score is train-role evidence only, and the conditioned checkpoint already failed at zero reachable train patches.
Rejected: Run candidate inference from the support diagnostic | no target-free validation authority exists before a train-role reachability diagnostic passes.
Rejected: Rescue the conditioned monotonic checkpoint | changing thresholds, schedule, loss weights, or layout transforms after its KILL would violate the registered gate.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Run only the registered train-role direct-support reachability diagnostic next; keep all candidate and quality gates closed until it passes.
Tested: py313 direct-support preflight tests; py310 direct-support preflight tests; py313 validator PASS; synthetic nonnegative bounded projection; planned-output absence checks.
Not-tested: train-role reachability diagnostic, model training, checkpoint generation, candidate inference, inner-val15, SCUT115, holdout40, visual review, reserved blind, promotion.
