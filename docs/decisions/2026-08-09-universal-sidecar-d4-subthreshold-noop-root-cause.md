# Universal Sidecar D4 Subthreshold No-Op Root Cause

## Result

`PASS`. The dominant D4 failure mode is raw direction-mode magnitude collapse,
coupled to a zero-positive-support dead zone. Matched-copy threshold
suppression is rejected as the primary explanation. The frozen D4 sidecar
replay reproduces the archived D4 PNGs exactly while contributing zero learned
residual on every page and patch. `artifacts/current-primary` remains the
product default.

## Frozen Evidence Boundary

This writeup is read-only and reuses the frozen D4 artifacts, metrics, and
checkpoint audit already admitted by the ledger. It also uses the frozen
in-memory `no_grad`/`eval` replay described below. No retraining, threshold
sweep, learning-rate sweep, new prediction artifact, or quality-gate evidence
was created.

The checkpoint-level sidecar scale is effectively zero:

~~~text
raw_global_residual_scale = 0.0010000000474974513
applied_tanh_scale = 0.0009999996982514858
residual_bound = 12 / 255
theoretical_max_sidecar_contribution_uint8 = 0.005999998189508915
~~~

That upper bound is already far below the matched-copy evaluation threshold of
`12` and, by itself, makes any learned sidecar movement operationally
sub-threshold unless the internal residual tensor produces meaningful support.

## Exact Replay Findings

Frozen in-memory `no_grad`/`eval` replay on MPS used the exact admitted
15-page manifest, `512` patches, overlap `32`, batch `8`, `mb`,
`mb_cov8_step`, and copy-mask dilation `0`. That replay reproduced all 15
frozen D4 PNGs exactly.

Across every replayed patch and page:

~~~text
mixed_residual positive_support_fraction = 0
bounded_magnitude = 0
scaled_residual = 0
pre_copy candidate_vs_internal_baseline pixels >= 1 = 0
pre_copy candidate_vs_internal_baseline pixels >= 12 = 0
matched_copy retained = 0
matched_copy removed = 0
~~~

This excludes matched-copy threshold suppression as the dominant cause. The
sidecar enters matched-copy with zero internal residual support, so the copy
threshold has nothing to suppress.

## Gate Analysis

The learned gate is not collapsed. Frozen replay statistics:

~~~text
gate_max_mean = 0.3904135089509095
gate_max_p95 = 0.4999946132302284
gate_max = 0.5460743308067322
gate_entropy_mean = 1.0841718127455893
gate_entropy_p95 = 1.0915001034736633
gate_entropy_max = 1.0921919345855713
three_way_maximum_entropy_ln3 = 1.0986122886681098
~~~

These values stay close to the three-way maximum-entropy regime rather than a
hard one-branch collapse. The sidecar therefore fails before gate selection
becomes decisive: the raw direction-mode branch does not generate positive
support that survives into bounded magnitude.

## External Baseline Comparison

The existing frozen baseline comparison remains true:

~~~text
final_png_hash_differences = 13 / 15 pages
changed_pixels_total = 2464
delta_mean = 0.37459415197372437
delta_p95 = 0.6666666865348816
delta_max = 1.0
delta_pixels_gte_12 = 0
~~~

Because the exact sidecar replay above contributes zero residual while still
reproducing the frozen D4 PNGs, the conservative inference is that those
byte-level differences reflect execution/replay/uint8 variance rather than
learned D4 sidecar movement. The available evidence does not isolate a more
specific variance source.

## Adapter Bias Context

The final checkpoint adapter final biases are all approximately
`-3.8112e-05`, which is consistent with the observed final non-positive mixed
support. This final-state evidence supports the zero-positive-support
diagnosis, but it does not establish a step-by-step training trajectory.

## Root-Cause Conclusion

The best-supported D4 root-cause statement is:

1. the primary-edit-direction sidecar kept the gate alive;
2. the residual-producing branch remained magnitude-collapsed, with
   non-positive mixed support on every replayed patch;
3. the bounded sidecar path therefore stayed exactly zero before matched-copy;
4. matched-copy threshold suppression is rejected as the dominant failure mode.

The admissible successor space narrows to one material causal fix: preserve the
same nonnegative direction semantics while folding negative raw magnitudes into
the allowed direction branch so that positive support is not annihilated before
bounded scaling.

Intent: Close the D4 causal investigation with a read-only root-cause finding that distinguishes magnitude collapse from gate collapse or threshold suppression.
Constraint: Only frozen D4 artifacts, audits, metrics, and the in-memory frozen replay are admissible in this closure; no new prediction artifacts or quality-gate evidence may be created.
Rejected: Matched-copy threshold rescue | frozen replay shows zero pre-copy residual support, so thresholding is downstream of the dominant failure.
Rejected: Gate-collapse diagnosis | gate maxima and entropy stay far from a hard collapsed branch while residual support remains zero.
Rejected: Scalar rescue by more scale, learning rate, or steps | that would not close the measured causal question for exact D4.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Treat D4 as causally closed; any successor must preserve D4 unchanged and prove the folded-magnitude contract in synthetic fail-closed preflight before real training is reconsidered.
Tested: Frozen checkpoint-audit reconciliation, frozen 15-page replay reproduction, replay-sidecar support accounting, existing D4 metric and source-guard evidence reconciliation.
Not-tested: Any new training, any new inference run outside the frozen replay package, SCUT115, holdout40, reserved blind, or successor implementation.
Related: docs/decisions/2026-08-09-universal-sidecar-d4-primary-edit-direction-inner-val15-kill.md
Related: docs/current-primary-quality-loop-ledger.json
