# External Text Layout Binary Mask Residual Preflight PASS

## Decision

`PASS`. The binary external-text occupancy mask residual proposal is
preregistered as a materially different follow-up to the killed score-based
external-layout projections. The only causal change is to use the frozen
external text occupancy bit as a deterministic support mask, with no fitted
score, score normalization, confidence threshold, model training, or candidate
surface.

This preflight authorizes only the registered train-role binary-mask
preserve-separation diagnostic. It does not authorize model training, checkpoint
generation, candidate inference, inner-val15, SCUT115, holdout40, visual review,
reserved blind access, promotion, confidence threshold sweeps, or detector/layout
tuning.

## Evidence

- Plan: `docs/external-text-layout-binary-mask-residual-proposal-v1.json`
- Plan SHA256: `0b1789a85ce4d2af428827717e8ed79107affa64965da098d942d8b401c4e4f9`
- Preflight output: `outputs/external-text-layout-binary-mask-residual-preflight-20260815/preflight.json`
- Preflight output SHA256: `b2da9a2c0f76074261a8d9b80fb02b3d2eec4d99d29fcc85a7cf59e626850501`
- Validator: `scripts/analysis/validate_external_text_layout_binary_mask_residual_preflight.py`
- Validator SHA256: `55cf0a4279697bf01d79fcd1e0fe91fb1c90360694fd4d7124cc91d0b756d868`
- Preflight test: `tests/test_external_text_layout_binary_mask_residual_preflight.py`
- Preflight test SHA256: `56ec17c068a2f97ee2d8565b7bcdf83c4b34fa9750817d0e83dfd69499f4768f`

## Boundary

The preflight verified that the upstream external-layout support diagnostic
still PASSes, the incremental support residual diagnostic is still a KILL, the
registered 256-patch train index hash is unchanged, and all binary-mask
candidate/quality output directories are absent.

The synthetic projection is exactly nonnegative: occupancy `0` maps to `0.0`
gray and occupancy `1` maps to `20.4` gray. The same frozen 12-gray gate is
retained rather than tuned.

Intent: Test whether the external text detector's raw occupancy bit separates target-lighter pixels from preserve pixels better than killed score projections.
Constraint: Score-based external-layout projections are closed; this preflight cannot reopen direct-support or incremental-score rescue paths.
Rejected: Confidence threshold sweep | threshold search would be detector tuning, not a single preregistered causal change.
Rejected: Candidate inference before train-only mask diagnostic | preserve safety must be proven before any page-level output.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: A PASS here authorizes only the train-only binary-mask diagnostic; candidate surfaces remain closed.
Tested: py313 and py310 binary-mask focused tests 9/9 with warnings as errors; py313 and py310 py_compile; jq validation of plan and preflight JSON.
Not-tested: model training, checkpoint generation, candidate inference, inner-val15, SCUT115, holdout40, visual review, reserved blind, promotion.
