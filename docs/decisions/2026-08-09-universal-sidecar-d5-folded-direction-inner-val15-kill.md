# Universal Sidecar D5 Folded-Direction Inner-Val15 Kill

## Result

`KILL`. D5 passes both preflights and completes the one authorized `80`-step
run, but it produces zero measurable lift at the first leakage-safe
inner-val15 gate. The learned raw global scale crossed below zero, so the
registered nonnegative-scale clamp disables the entire folded residual before
quantization and matched-copy. No later split or visual-promotion gate was
opened. `artifacts/current-primary` remains the product default.

## Frozen Run

~~~text
code_commit = 29e6d68
checkpoint = artifacts/trials/universal-sidecar-d5-d1-mixed-scut130-hw5k260-step80-primary-edit-direction-folded-20260809/ensexam/20260809_112035/epoch_1.pth
checkpoint_sha256 = 875feb5b9f824b5075746b03415d07f1cfb9a1addb062d6d57181a3c9555c87b
train_log = outputs/universal-sidecar-d5-step80-20260809/train.log
train_log_sha256 = 2a13892194328df8d634b60081eb8f48c9c41bf8a464b37166f8047f6b4542dd
steps = 80
train_G = 30.858111572265624
train_D = 1.7540874481201172
~~~

The process exited `0`, saved only the epoch checkpoint, skipped validation and
final test, and did not rerun or extend the registered budget.

## Checkpoint Audit

The checkpoint scope audit passed:

~~~text
checkpoint_audit = outputs/universal_sidecar_d5_d1_step80_primary_edit_direction_folded_20260809/checkpoint_audit.json
checkpoint_audit_sha256 = b63ac6aee002386f9d2dea5c756a4780b11fc67aaeaab3440dafd1257b6b71ae
status = pass
sidecar_key_count = 17
moved_final_projection_key_count = 8
base_changed_count_vs_current_primary = 0
base_missing_count_vs_current_primary = 0
unexpected_non_sidecar_key_count = 0
has_optimizer_state = false
has_scheduler_state = false
~~~

The signed scalar evidence is decisive:

~~~text
initial_global_residual_scale = 0.0010000000474974513
final_raw_global_residual_scale = -0.000019280019841971807
nonnegative_scale = max(final_raw_global_residual_scale, 0) = 0
applied_scale = tanh(nonnegative_scale) = 0
scaled_residual = 0
~~~

The generic checkpoint audit now records signed scalar `value` in addition to
`norm` and `maxabs`; a regression test prevents a negative scale from being
misreported as positive movement evidence.

## Inner-Val15 Gate

The candidate used the frozen 15-page manifest and current-primary matched-copy
protocol (`mb`, `mb_cov8_step`, overlap `32`, batch `8`, dilation `0`, change
and evaluation thresholds `12`). Predictions were frozen before labels were
opened by the scorer.

| Metric | Baseline | D5 | D5 - Baseline |
| --- | ---: | ---: | ---: |
| mean residual ratio | 0.17694860490191527 | 0.17694860490191527 | 0.0 |
| residual p95 delta | - | - | 0.0 |
| residual max delta | - | - | 0.0 |
| mean overerase ratio | 0.0023246713604556293 | 0.0023246713604556293 | 0.0 |
| overerase p95 delta | - | - | 0.0 |
| overerase max delta | - | - | 0.0 |

Frozen gate evidence:

~~~text
frozen_metrics = outputs/universal_sidecar_d5_d1_inner_val15_step80_primary_edit_direction_folded_20260809/frozen_predictions/metrics.csv
frozen_metrics_sha256 = a13dafc92bf0accae8e901eb86c355d8a2f7e38ce714ea4b5c118a7384773654
post_freeze_metrics = outputs/universal_sidecar_d5_d1_inner_val15_step80_primary_edit_direction_folded_20260809/post_freeze_metrics.csv
post_freeze_metrics_sha256 = 13565b1dbd156b7dd797403a0d6fbd7132c6938313bd0a74b52b94a5e01bbe72
source_guard_summary = outputs/universal_sidecar_d5_d1_inner_val15_step80_primary_edit_direction_folded_20260809/source_guard_summary.json
source_guard_summary_sha256 = 4e17003441c3e5a68a9f5e944ff351505f2bb150d65246611649697f0edc37ce
source_guard_status = pass
pages = 15
measurable_page_delta = false
nonzero_delta_count_residual = 0
nonzero_delta_count_overerase = 0
~~~

Row-wise `pred_sha256` comparison shows all `15/15` D5 predictions are
identical to D4. They differ from the older current-primary replay on `13/15`
pages, exactly matching the previously closed D4 replay-variance pattern. D4's
frozen in-memory audit already proved its sidecar contribution was zero; D5's
negative scale independently proves the same before matched-copy.

## Decision Boundary

D5 fails the calibrated minimum residual gain of `0.0005` and has no measurable
page movement. Visual review is neither required nor admissible as promotion
evidence for an exact metric no-op. SCUT115, holdout40, reserved blind, and
current-primary replacement remain closed.

Do not rescue exact D5 with additional steps, a learning-rate sweep, a scale
floor, threshold tuning, or selector tuning. Folding fixed D4's raw-magnitude
dead zone, but the training objective shut the now-active branch off through
the global scale. A successor must be materially different and preregistered
against a different named failure bucket; scalar coercion of this branch would
override its measured non-regression preference rather than establish safe
generalizing lift.

Intent: Kill exact D5 after the folded branch learns parameters but its global scale crosses negative and clamps the entire residual to zero at the first quality gate.
Constraint: Inner-val15 was the first and only opened gate; no later split or visual review was used.
Constraint: The run completed exactly once at the registered 80-step budget.
Rejected: More steps or learning-rate sweep | scalar continuation cannot establish that forcing the disabled branch improves quality safely.
Rejected: Positive scale floor | it would override the measured negative scale direction and is not the preregistered D5 causal change.
Rejected: Matched-copy threshold rescue | applied sidecar scale is already exactly zero before matched-copy.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not repeat or scalar-rescue exact D5; select and preregister a materially different named failure bucket before any new training.
Tested: Exact 80-step MPS run, checkpoint scope and signed-scale audit, frozen 15-page matched-copy inference, post-freeze scoring, source guard, and row-wise prediction hash comparison.
Not-tested: SCUT115, holdout40, visual review, reserved blind, or any successor.
Related: docs/decisions/2026-08-09-universal-sidecar-d5-real-preflight-pass.md
Related: docs/decisions/2026-08-09-universal-sidecar-d4-subthreshold-noop-root-cause.md
