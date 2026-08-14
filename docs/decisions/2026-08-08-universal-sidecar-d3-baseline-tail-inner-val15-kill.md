# Universal Sidecar D3 Baseline-Tail Inner-Val15 Kill

## Result

`KILL`. D3 fails the first leakage-safe SCUT inner-val15 source guard. No
development split, SCUT115, holdout40, visual-review promotion shard, or
reserved-blind evidence was opened. `artifacts/current-primary` remains the
product default.

## Frozen Run

~~~text
code_commit = 067eb82189bc7cde8e51a1b715024850a53d92ca
checkpoint = artifacts/trials/universal-sidecar-d3-d1-mixed-scut130-hw5k260-step80-baseline-tail-20260808/ensexam/20260808_214648/epoch_1.pth
checkpoint_sha256 = 843c8a7b7d3d42f7535d6e459bde97d51c36058259cd6b938619b1380fd1de77
train_log_sha256 = d40348ad60bb35b5621ba3253a1e03feb0df9aad911c14f9a29ceed03d132f50
steps = 80
train_G = 31.38758239746094
train_D = 1.7910181045532227
cached_baseline_tail_loss = 0.027386
~~~

The runtime loaded current-primary with `17` expected sidecar keys missing and
`0` unexpected keys, trained only `7,437 / 24,690,655` generator parameters,
froze `40` BatchNorm modules, and skipped validation/final test. The checkpoint
audit passed:

~~~text
checkpoint_audit_sha256 = 07cc8e3f84a6e18e6c77dcd04d6a7b3a042be22c09a3ae01e0d0cd4398a50400
sidecar_key_count = 17
moved_final_projection_key_count = 8
base_changed_count_vs_current_primary = 0
unexpected_non_sidecar_key_count = 0
has_optimizer_state = false
has_scheduler_state = false
~~~

## Inner-Val15 Gate

The candidate used the frozen 15-page manifest and exact current-primary
matched-copy protocol (`mb`, `mb_cov8_step`, overlap `32`, batch `8`, thresholds
`12`). Predictions were frozen before label scoring.

| Metric | Baseline | D3 | D3 - Baseline |
| --- | ---: | ---: | ---: |
| mean residual ratio | 0.17694860490191527 | 0.17694920338356732 | +0.0000005984816520476777 |
| residual p95 delta | - | - | +0.00000269316743421454 |
| residual max delta | - | - | +0.000008977224780715165 |
| mean overerase ratio | 0.0023246713604556293 | 0.0023246713604556293 | 0.0 |
| overerase p95/max delta | - | - | 0.0 |

Only one page crossed the thresholded source guard:

~~~text
file = 301.jpg
candidate_minus_baseline_residual_ratio = +0.000008977224780715165
candidate_minus_baseline_overerase_ratio = 0.0
source_guard_status = fail
failure = residual_source_guard_regression
post_freeze_metrics_sha256 = 72ab5be6d9fc56913d31557fb1bfd007a2fe112d471e938fadcb0b15bdc8c69a
source_guard_summary_sha256 = 601290fcaa225a5bc760256d23a28925c1ee4c1a9bb6c9599b51ab16b8b17ccd
~~~

## Interpretation

D3 changed all 17 sidecar parameters relative to D2, and 14 of 15 frozen
prediction PNGs differ bytewise from D2, so the new loss was active and the
candidate is not a copied checkpoint. Nevertheless, thresholded behavior is
equivalent to D2: the same page `301.jpg` gains exactly one newly bad residual
event and no newly good event.

On that page D3 changes `146` pixels by at most one channel level. Among changed
pixels with a nonzero current-primary edit vector, `58` candidate changes align
with the current-primary source-to-clean direction, `53` oppose it, and `31`
are orthogonal. The binary baseline-safe support therefore changes the
optimization path but does not control the free RGB sidecar's edit direction.

Historical whole-model baseline-tail separate, relative-tail, max, CVaR, and
solved-pixel variants also failed inner-val15 tail/page gates. The baseline-safe
and baseline-relative guard-loss family is closed for this failure bucket. Do
not retry it by sweeping lambda, alpha, tail fraction, learning rate, or steps.

## Successor Boundary

The next admissible causal representation is a direction-restricted continuous
sidecar: predict only a nonnegative per-pixel magnitude along the same-call
current-primary source-to-clean edit vector. Its prerequisite is a synthetic
proof of exact zero-init equivalence, gradient liveness, bounded magnitude,
single-interface behavior, sidecar-only trainability, and no ability to create
an RGB direction opposed to current-primary. It must start from D2 controls,
not stack another cache loss on D3.

Intent: Kill the binary baseline-tail guard after it fails to change D2's thresholded source regression.
Constraint: Inner-val15 is a kill gate only; no page-specific rescue or later split was used to design D3.
Rejected: Increase baseline-tail weight or alter its tail fraction | this is a nearby scalar sweep in an historically exhausted guard-loss family.
Rejected: Add cached relative/max/CVaR/solved-pixel terms | those representations already have durable inner-val15 failures.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not evaluate D3 on any later split and do not repeat baseline-safe/relative guard-loss variants for this bucket.
Tested: 80-step MPS run, checkpoint isolation audit, 15-page frozen matched-copy inference, post-freeze scoring, strict source-guard summary, D2/D3 checkpoint and prediction comparison.
Not-tested: Development splits, SCUT115, holdout40, visual review, reserved blind, product replacement.
Related: docs/decisions/2026-08-08-universal-sidecar-d3-baseline-tail-preregistration.md
Related: docs/current-primary-quality-loop-ledger.json
