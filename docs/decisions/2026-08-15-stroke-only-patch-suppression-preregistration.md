# Stroke-Only Patch Suppression Preregistration

## Decision

`PREREQUISITE_NEEDED`. The next active quality-loop successor is
`stroke_only_patch_suppression_v1`, a non-layout, target-free preflight that
constrains a separately supplied repair candidate to source-dark lifted stroke
pixels only. It is registered because external text layout has now failed as a
direct edit support signal in the conditioned monotonic, direct support-score,
incremental support-score, and binary occupancy paths.

This does not authorize model training, checkpoint generation, candidate
inference, inner-val15, development gates, SCUT115, holdout40, visual review,
reserved blind access, promotion, or current-primary replacement.

## Evidence

- Proposal: `docs/stroke-only-patch-suppression-preflight-v1.json`
- Proposal SHA256: `da5a0f6d5de6259049082a3bff769aa76b8ebd81139367ac004c842ae2fbc036`
- Implementation: `scripts/analysis/generate_stroke_only_patch_suppression.py`
- Implementation SHA256: `df4e621fc2779bed04c311c80f3da82d01be3e0953071c96714a32077faa5d6e`
- Test: `tests/test_generate_stroke_only_patch_suppression.py`
- Test SHA256: `3641a1085089b7b71807a77389dbbf726ea772c108cfb439cc929583cf2fb238`

## Boundary

The only authorized next step is a train-only preflight that proves:

- synthetic bright paper/background pixels remain baseline unchanged;
- synthetic source-dark lifted stroke pixels can receive the supplied repair;
- default split authority rejects validation and blind surfaces;
- planned validation/blind outputs remain absent.

The implementation defaults to `train` and `train160` only. Any use of
`inner_val15`, SCUT115, holdout40, visual review, reserved blind, or promotion
requires a separate PASS record after this preflight and the normal gate order.

## Stop Rules

KILL this family rather than rescue it if the preflight shows background edits,
split-authority leakage, or dependence on target pixels during generation. Do
not tune source-dark thresholds, baseline-dark thresholds, lift thresholds,
component-size caps, blend values, or split inclusion after seeing preflight or
candidate results.

Intent: Move past killed external-layout direct-support routes by freezing a non-layout, preserve-first successor before any candidate surface opens.
Constraint: Current-primary remains the product default, and validation/blind surfaces must remain closed until the registered preflight and gate order pass.
Rejected: Continue external layout as edit support | full score, incremental score, binary occupancy, and conditioned monotonic routes are already terminally unsafe or subthreshold.
Rejected: Run the stroke-only postprocessor directly on SCUT115 or holdout40 | that would skip the leakage-safe train-only preflight and gate sequence.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Run only the registered train/train160 preflight next; do not treat generated review rows as quality lift until inner-val15 and development gates pass.
Tested: py313 stroke-only focused tests 3/3 with warnings as errors; py313 py_compile.
Not-tested: real preflight materialization, model training, checkpoint generation, candidate inference, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-15-external-text-layout-binary-mask-residual-reachability-kill.md
Related: docs/decisions/2026-08-15-external-text-layout-direct-support-residual-reachability-kill.md
Related: docs/decisions/2026-08-15-external-text-layout-incremental-support-residual-reachability-kill.md
