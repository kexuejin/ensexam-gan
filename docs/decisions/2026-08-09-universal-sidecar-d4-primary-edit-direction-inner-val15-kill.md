# Universal Sidecar D4 Primary-Edit-Direction Inner-Val15 Kill

## Result

`KILL`. D4 clears the source guard by removing D2/D3's positive page residual
regression, but it also collapses into sub-threshold/no-metric movement at the
first leakage-safe SCUT inner-val15 gate. No development split, SCUT115,
holdout40, visual-review promotion shard, or reserved-blind evidence was
opened. `artifacts/current-primary` remains the product default.

## Frozen Run

~~~text
code_commit = cce4a02288bd99407f17b4ea9204e8a82c8e37a8
checkpoint = artifacts/trials/universal-sidecar-d4-d1-mixed-scut130-hw5k260-step80-primary-edit-direction-20260809/ensexam/20260809_090610/epoch_1.pth
checkpoint_sha256 = 0bbb27ed795d674912a09a60bcdcbc1ce2d50517f2e8b968b2b9662ddf40f97c
train_log = outputs/universal-sidecar-d4-step80-20260809/train.log
train_log_sha256 = 715e8a1359a6984a4deab71634a203586637d1037a9cee2606c0ad43ceff6254
steps = 80
train_G = 30.738397216796876
train_D = 1.7996253967285156
~~~

The training command completed all 80 steps and saved the checkpoint. A later
outer shell bookkeeping wrapper returned exit code `1` because zsh reserves the
variable name `status`; record that as operational context only, not as a model
failure.

The checkpoint audit passed:

~~~text
checkpoint_audit = outputs/universal_sidecar_d4_d1_step80_primary_edit_direction_20260809/checkpoint_audit.json
checkpoint_audit_sha256 = 4006a5ba7fccaceee153ed0bd4c02ed67ad96c5d2be3c6da92745ea7332d7d78
status = pass
sidecar_key_count = 17
moved_final_projection_key_count = 8
base_changed_count_vs_current_primary = 0
unexpected_non_sidecar_key_count = 0
has_optimizer_state = false
has_scheduler_state = false
~~~

## Inner-Val15 Gate

The candidate used the frozen 15-page manifest and exact current-primary
matched-copy protocol (`mb`, `mb_cov8_step`, overlap `32`, batch `8`,
change threshold `12`, automatic per-page copy-mask thresholds observed at
`76` and `160`). Predictions were frozen before label scoring.

| Metric | Baseline | D4 | D4 - Baseline |
| --- | ---: | ---: | ---: |
| mean residual ratio | 0.17694860490191527 | 0.17694860490191527 | 0.0 |
| residual p95 delta | - | - | 0.0 |
| residual max delta | - | - | 0.0 |
| mean overerase ratio | 0.0023246713604556293 | 0.0023246713604556293 | 0.0 |
| overerase p95 delta | - | - | 0.0 |
| overerase max delta | - | - | 0.0 |

Frozen gate evidence:

~~~text
frozen_metrics = outputs/universal_sidecar_d4_d1_inner_val15_step80_primary_edit_direction_20260809/frozen_predictions/metrics.csv
frozen_metrics_sha256 = 8ddc27cecaabf9a4b7b855fc1fef30b003dee8677296e82c8123441100e9cfcb
post_freeze_metrics = outputs/universal_sidecar_d4_d1_inner_val15_step80_primary_edit_direction_20260809/post_freeze_metrics.csv
post_freeze_metrics_sha256 = 941d0b85449f80233af14c639ed6de6ba1a942e82071da6feb9660d9c62a9fab
source_guard_summary = outputs/universal_sidecar_d4_d1_inner_val15_step80_primary_edit_direction_20260809/source_guard_summary.json
source_guard_summary_sha256 = 4592ed71d9bcc1ec94cab7b51526238157649b174111152b9d20ee8b792d110d
source_guard_status = pass
pages = 15
measurable_page_delta = false
nonzero_delta_count_residual = 0
nonzero_delta_count_overerase = 0
~~~

The frozen evidence also records that 13 of 15 prediction PNG hashes differ
from the baseline-current-primary predictions even though all 15 metric deltas
are exactly zero. D4 therefore moved bytes without producing measurable
thresholded residual or overerase movement.

## Interpretation

Direction restriction fixed the specific failure mode that killed D2 and D3:
there is no positive page residual regression and no source-guard failure on
inner-val15. But the replacement representation did not clear the minimum gain
bar either. Aggregate residual gain is `0.0`, which is below the calibrated
`0.0005` minimum, overerase delta is `0.0`, and every page-level metric delta is
`0.0`.

This closes exact D4 as a product-quality branch. Do not try to rescue it by
sweeping steps, learning rate, or matched-copy thresholds, and do not open
later gates for this exact run family. The surviving question is causal, not
operational: why did a direction-restricted sidecar that changes prediction
bytes collapse into thresholded no-op behavior?

The active root-cause space remains limited to three hypotheses:

1. raw direction-mode magnitude collapse;
2. learned gate/support collapse;
3. matched-copy threshold suppression.

No successor architecture is selected here. The next admissible action is
read-only diagnostics that quantify changed-pixel magnitude, gate/support
activation, and threshold interaction, followed by preregistration of at most
one materially new successor if the evidence justifies it.

Intent: Kill exact D4 after direction restriction removes D2/D3's regression but still produces zero measurable inner-val15 lift.
Constraint: Inner-val15 remains the first and only opened gate for this run family; later splits stayed closed.
Constraint: The post-training wrapper exit 1 came from zsh reserving `status`, not from incomplete training or missing checkpoint output.
Rejected: Step sweep rescue | the exact D4 causal representation already completed its registered 80-step run and showed zero metric movement.
Rejected: Learning-rate rescue | nearby scalar rescue would not answer whether the direction-restricted sidecar collapsed in magnitude, support, or threshold interaction.
Rejected: Copy-threshold rescue | selector-only threshold search is analysis infrastructure, not an admissible product-quality path for exact D4.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not repeat exact D4, do not open later gates for it, and do not run step/LR/copy-threshold rescue before a read-only root-cause writeup preregisters a materially new successor.
Tested: Checkpoint SHA verification, train-log SHA verification, checkpoint audit reconciliation, 15-page frozen matched-copy inference, post-freeze scoring, source-guard summary reconciliation.
Not-tested: Development splits, SCUT115, holdout40, visual review, reserved blind, successor architecture.
Related: docs/decisions/2026-08-09-universal-sidecar-d4-primary-edit-direction-preregistration.md
Related: docs/decisions/2026-08-09-universal-sidecar-d4-real-preflight-pass.md
Related: docs/current-primary-quality-loop-ledger.json
