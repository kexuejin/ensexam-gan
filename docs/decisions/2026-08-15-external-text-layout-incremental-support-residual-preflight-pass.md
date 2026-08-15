# External Text Layout Incremental Support Residual Preflight PASS

## Decision

`PASS`. The incremental support residual proposal is preregistered as a
materially different follow-up to the killed raw direct-support projection. The
only causal change is to score each pixel by the fold-specific external-layout
incremental contribution: full support fit score minus RGB-ablation fit score.

This preflight authorizes only the registered train-role preserve-separation
diagnostic. It does not authorize model training, checkpoint generation,
candidate inference, inner-val15, SCUT115, holdout40, visual review, reserved
blind access, promotion, threshold rescue, score-normalization rescue, or
detector/layout tuning.

## Evidence

- Plan: `docs/external-text-layout-incremental-support-residual-proposal-v1.json`
- Plan SHA256: `4fb6c73717365dffc50596b20853147efd77e82582b2e1d3b216c5348b274514`
- Preflight output: `outputs/external-text-layout-incremental-support-residual-preflight-20260815/preflight.json`
- Preflight output SHA256: `cdb48622584d501cc82e8783d690171ac9fa24b9bed1a1d0273358eed42dd9bf`
- Validator: `scripts/analysis/validate_external_text_layout_incremental_support_residual_preflight.py`
- Validator SHA256: `1537854f94d3bcc5df62ae763d8837ff910debbf64c5bdbce179f1dd02eb316f`
- Preflight test: `tests/test_external_text_layout_incremental_support_residual_preflight.py`
- Preflight test SHA256: `d385763f48e632db0d2bea07014aa9db5ce892e07824888768049b4355ea1395`

## Boundary

The preflight verified that the upstream external-layout support diagnostic
still PASSes, the direct-support residual diagnostic is still a preserve-safety
KILL, the registered 256-patch train index hash is unchanged, and all
incremental-support candidate/quality output directories are absent.

The synthetic projection remained nonnegative and bounded at 20.4 gray with the
same frozen 12-gray gate used by the killed direct-support diagnostic. That gate
is intentionally reused rather than tuned from the new result.

Intent: Test whether layout-only incremental contribution separates preserve pixels better than the killed raw full support score.
Constraint: The direct-support residual v1 family is closed and cannot be rescued with score normalization, detector transforms, or gate changes.
Rejected: Reuse raw full support score | direct-support v1 already KILLed on preserve leakage before candidate inference.
Rejected: Tune a lower application gate | the next path must inherit the frozen 12-gray gate until a separate preregistered mechanism justifies otherwise.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: A PASS here authorizes only the train-only incremental preserve-separation diagnostic; candidate surfaces remain closed.
Tested: py313 and py310 incremental support residual focused tests 12/12 with warnings as errors; py313 and py310 py_compile; jq validation of plan and preflight JSON.
Not-tested: model training, checkpoint generation, candidate inference, inner-val15, SCUT115, holdout40, visual review, reserved blind, promotion.
