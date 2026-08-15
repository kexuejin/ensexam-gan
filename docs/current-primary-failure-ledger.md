# Current-Primary Failure-Bucket Ledger

This is the durable ledger for the active current-primary generalization
program defined by the sustainable generalization-safe quality loop. It exists
so that program status is reproducible from committed repository state alone,
without reading disposable `outputs/` experiment directories or untracked
runtime plans.

Role boundaries:

- This ledger tracks current-primary baseline anchors, calibrated evaluation
  noise, and named failure buckets with their terminal states.
- `docs/rejected-directions.md` keeps the detailed narrative of each rejected
  family. This ledger references, and never replaces, those records.
- `docs/quality-goal.md` tracks the legacy fixed-set selector backlog. Its
  status command is not a verifier for this program.

The machine-readable source of truth is the single `yaml ledger` block below.
`scripts/analysis/report_current_primary_program_status.py` parses exactly that
block, verifies every anchor hash and evidence path fail-closed, and refuses to
synthesize status when any input is missing. Update the YAML block first; prose
elsewhere in this file is commentary only.

```yaml ledger
schema_version: 1
updated: "2026-08-16"
program: sustainable-generalization-safe-quality-loop
product_default: artifacts/current-primary
program_state: all_exhausted

anchors:
  checkpoint:
    path: artifacts/current-primary/micro_region_probe_step0001.pth
    sha256: e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae
  config:
    path: artifacts/current-primary/config.yaml
    sha256: 8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e
  inner_val15_manifest:
    path: hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt
    sha256: fb25bb2aef2f9285403f908deb3da6d88b07b5d1c2c812965ce9e0636ddc172e
  readiness_smoke_expected:
    residual: 0.125765
    overerase: 0.002500

calibration:
  gate: scut_inner_val15_matched_copy
  scoring_thresholds:
    change_threshold: 12
    eval_threshold: 12
  replays: 3
  replay_result: byte_identical_metric_identical
  residual_lift_floor: 0.0005
  baseline_metrics:
    residual_mean: 0.1769486049
    residual_p95: 0.2850987046
    residual_max: 0.3869877310
    overerase_mean: 0.0023246714
    overerase_p95: 0.0035794460
    overerase_max: 0.0061057815
  metrics_csv_sha256: 4e039dc36967e4fa5f5c762cb0230a5f1c61ebe331d220c34b4573d74ab7bfe2
  note: >-
    Replay evidence directories under outputs/ are disposable; the calibrated
    floor and baseline metrics recorded here are the durable facts. The p95
    values match the frozen guard table in the U4C kill record.

development_reference:
  hw5k_dev232:
    rows: 232
    residual_mean: 0.7238528527
    overerase_mean: 0.0644489109
    residual_max: 0.9547818124
    note: >-
      Current-primary HW5K development headroom, frozen from the 2026-07-29
      baseline scoring run. Recorded here because the source outputs/ directory
      is disposable.

buckets:
  - name: cross_domain_residual_headroom_vs_source_solved_pixel_regression
    status: exhausted
    summary: >-
      Current-primary has large HW5K development headroom (dev232 residual mean
      0.7239 vs SCUT inner-val15 0.1769), but the first measurable universal
      sidecar movement regressed SCUT page 301.jpg and failed the inner-val15
      p95/overerase source guard. The later train-only baseline-tail
      non-regression sidecar consumed its cache prerequisite and exact D3 run,
      then KILLed on the same source residual regression class as D2; D4 and D5
      direction-repair variants also terminated as no-lift/noop KILLs.
    evidence:
      - docs/decisions/2026-08-06-universal-residual-adapter-sidecar-u4c-scut15-kill.md
      - docs/plans/2026-08-06-universal-residual-adapter-sidecar-u4-development-validation.md
      - docs/decisions/2026-08-13-universal-sidecar-d3-baseline-tail-cache-verification.md
      - docs/decisions/2026-08-08-universal-sidecar-d3-baseline-tail-inner-val15-kill.md
      - docs/decisions/2026-08-09-universal-sidecar-d4-primary-edit-direction-inner-val15-kill.md
      - docs/decisions/2026-08-09-universal-sidecar-d5-folded-direction-inner-val15-kill.md
      - docs/decisions/2026-08-16-current-primary-failure-ledger-successor-selection-reconciliation.md
    prerequisite_status: >-
      SATISFIED (2026-08-13): the exact train-manifest baseline-support cache
      exists at
      artifacts/caches/baseline-tail-universal-sidecar-d3-mixed-scut130-hw5k260-20260807
      with 383/383 manifest pages, zero missing safe-mask files, and
      primary/config/list/rows hashes verified against the ledger anchors
      (rows_csv_sha256
      592f6383164af92ec10008881a8b160cee6828132831ac66c4d3316d2742545a). See
      the 2026-08-13 cache-verification decision record. The prerequisite is no
      longer actionable because the exact registered D3 run reached terminal
      KILL.
    prohibited: >-
      Do not reopen D3/D4/D5 through threshold rescue, postprocess rescue, hard
      routing, domain labels, source selectors, base unfreezing, learning-rate
      or step-count sweeps, SCUT115, holdout40, reserved blind, or
      current-primary mutation.

  - name: successor_selection_outside_closed_buckets
    status: exhausted
    summary: >-
      The current loop is between candidate families after the cross-domain
      sidecar bucket, external text-layout support successors, stroke-only
      source-candidate bucket, and target-dark/overerase-risk bucket all reached
      terminal current-state records, while universal mechanism admission remains
      product-owner blocked. The next admissible move is successor selection,
      not training or validation. Broader durable exhaustion has now closed
      successor selection because no executable ledgered bucket remains.
    evidence:
      - docs/current-primary-quality-loop-ledger.json
      - docs/successor-selection-current-state-inventory-v1.json
      - docs/decisions/2026-08-16-successor-selection-current-state-inventory.md
      - docs/current-primary-broader-durable-exhaustion-v1.json
      - docs/decisions/2026-08-16-current-primary-broader-durable-exhaustion.md
      - docs/decisions/2026-08-16-stroke-only-source-candidate-bucket-exhaustion.md
      - docs/decisions/2026-08-16-target-dark-overerase-bucket-exhaustion.md
      - docs/decisions/2026-08-16-current-primary-failure-ledger-successor-selection-reconciliation.md
    terminal_basis: >-
      The broader durable exhaustion record found no remaining executable
      ledgered bucket. Future re-entry requires materially new leakage-safe
      evidence, exact selector-replay PNG restoration with hash custody, the
      universal-mechanism product-owner unblock, or a new user-authorized
      quality program outside this exhausted scope.
    prohibited: >-
      No validation, SCUT115, holdout40, visual review, reserved blind,
      promotion, current-primary replacement, threshold rescue, page-specific
      rescue, or reuse of a terminal family is authorized during successor
      selection.

  - name: external_text_layout_support_successors
    status: exhausted
    summary: >-
      External text-layout evidence added target-aligned train-role signal, but
      every registered route to use it for edits failed before candidate or
      quality admission: the conditioned monotonic checkpoint was subthreshold,
      direct support leaked preserve pixels, incremental support failed preserve
      separation, and binary occupancy masks also leaked preserve pixels.
    evidence:
      - docs/decisions/2026-08-15-external-text-layout-support-diagnostic-pass.md
      - docs/decisions/2026-08-15-external-text-layout-conditioned-monotonic-checkpoint-kill.md
      - docs/decisions/2026-08-15-external-text-layout-direct-support-residual-reachability-kill.md
      - docs/decisions/2026-08-15-external-text-layout-incremental-support-residual-reachability-kill.md
      - docs/decisions/2026-08-15-external-text-layout-binary-mask-residual-reachability-kill.md
    prohibited: >-
      Do not reuse external text layout alone as an edit-support signal through
      detector threshold, confidence, score normalization, direction, binary
      mask, application-gate, candidate, or validation rescue.

  - name: stroke_only_source_candidate_successors
    status: exhausted
    summary: >-
      Available source-only successor candidates are exhausted: exact historical
      selector-replay PNGs remain absent, local-paper and thin-component
      candidates failed train-only preserve-first screening, chroma and achroma
      variants failed inner-val15, and the source-edge route failed SCUT115
      page-level no-regression.
    evidence:
      - docs/decisions/2026-08-16-stroke-only-source-candidate-bucket-exhaustion.md
      - docs/decisions/2026-08-16-source-dark-local-paper-lift-source-candidate-kill.md
      - docs/decisions/2026-08-16-source-dark-thin-component-lift-source-candidate-kill.md
      - docs/decisions/2026-08-16-source-chroma-primary-edit-lift-inner-val15-kill.md
      - docs/decisions/2026-08-16-source-achroma-primary-edit-lift-inner-val15-kill.md
      - docs/decisions/2026-08-16-source-edge-primary-edit-lift-scut115-kill.md
    prohibited: >-
      Do not reopen source-only successors via threshold, alpha, component,
      kernel, dilation, stroke-only blend, page-specific, visual-review, or
      held-out rescue. Only exact selector-replay PNG restoration with hash
      custody can reopen that missing-asset branch.

  - name: target_dark_or_overerase_risk_successors
    status: exhausted
    summary: >-
      The current-state target-dark/overerase-risk successor bucket is
      exhausted: component-context, Delta-Trust oracle ceiling, and safe-metric
      fallback reconstruction routes all failed before candidate admission.
    evidence:
      - docs/decisions/2026-08-16-target-dark-overerase-bucket-exhaustion.md
      - docs/decisions/2026-08-16-target-dark-component-context-feature-recheck-kill.md
      - docs/decisions/2026-08-16-balanced007-delta-trust-oracle-ceiling-recheck-kill.md
      - docs/decisions/2026-08-16-safe-metric-fallback-reconstruction-no-headroom-kill.md
    prohibited: >-
      Do not repeat target-dark/overerase successors through component-ranker,
      Delta-Trust, safe-metric fallback, threshold, feature-subset, page, patch,
      validation, or blind rescue without materially new train-only evidence.

  - name: universal_sidecar_step_lr_nearby_sweep
    status: exhausted
    summary: >-
      U4B step20 sidecar was killed at the U4C SCUT inner-val15 source guard
      (residual p95 +0.0002424, overerase mean +0.0000543, overerase max
      +0.0002803). Follow-up D2/D2C runs reproduced the regression; D2D halved
      the learning rate, passed the guard, but produced no measurable page
      delta (safe-no-lift). Failed-trial artifacts were pruned per policy; this
      entry is the durable terminal record for the nearby step/LR family.
    evidence:
      - docs/decisions/2026-08-06-universal-residual-adapter-sidecar-admission.md
      - docs/decisions/2026-08-06-universal-residual-adapter-sidecar-u3-code.md
      - docs/decisions/2026-08-06-universal-residual-adapter-sidecar-u4a-static-readiness.md
      - docs/decisions/2026-08-06-universal-residual-adapter-sidecar-u4b-step20-smoke.md
      - docs/decisions/2026-08-06-universal-residual-adapter-sidecar-u4c-scut15-kill.md
    prohibited: >-
      No further nearby step or learning-rate sweeps for the sidecar. The D2D
      safe pass is safety evidence only and never justifies a lift sweep.

  - name: hw5k_mixed_shared_weight_adaptation
    status: killed
    summary: >-
      Shared-weight mixed SCUT+HW5K adaptation candidates 1-5 all terminated:
      candidates 1-3 failed gate A, candidate 4 passed gate A but failed gate
      B, candidate 5 failed gate B. The family is closed without a promotion.
    evidence:
      - docs/decisions/2026-07-30-hw5k-mixed-candidate1-step160-gate-a-rejection.md
      - docs/decisions/2026-07-30-hw5k-mixed-candidate2-step1600-gate-a-rejection.md
      - docs/decisions/2026-07-30-hw5k-mixed-candidate3-step1600-respress-gate-a-rejection.md
      - docs/decisions/2026-07-30-hw5k-mixed-candidate4-step6400-respress-gate-a-pass.md
      - docs/decisions/2026-07-30-hw5k-mixed-candidate4-gate-b-rejection.md
      - docs/decisions/2026-08-02-hw5k-mixed-candidate5-gate-b-rejection.md
    prohibited: >-
      No shared-weight mixed retraining without a materially new causal
      mechanism admitted through an explicit architecture decision.

  - name: target_dark_or_overerase_risk_component_representation
    status: killed
    summary: >-
      The component-representation family for the target-dark/overerase-risk
      bucket is exhausted: 18 scalar features plus the preregistered multiscale
      context and printed-line continuity extensions all worsened ROC-AUC, AP,
      and reject ratio on the exact train160 leave-one-page-out gate.
    evidence:
      - docs/quality-goal.md
      - docs/rejected-directions.md
    prohibited: >-
      Do not repeat or widen this representation family with more labels.

  - name: universal_mechanism_admission
    status: blocked
    summary: >-
      Universal-capability work is blocked on PREREQUISITE_NEEDED: an explicit
      product-owner decision that caller-known routing is insufficient. The
      continuous three-expert reconstruction mixture is classified
      distinct_untried_not_admitted and stays non-executable.
    evidence:
      - docs/decisions/2026-08-05-materially-new-universal-preflight.md
      - docs/decisions/2026-08-06-universal-residual-adapter-sidecar-admission.md
    next_allowed: >-
      Only the product-owner entry event can unblock this bucket, and a new
      bounded architecture admission is still required afterwards.

  - name: legacy_fixed_set_selector_backlog
    status: out_of_scope
    summary: >-
      The fixed-set selector backlog for the known 435 pages is tracked by
      docs/quality-goal.md. Its legacy status command depends on removed
      outputs/ inputs and must not be used as a verifier for this program.
    evidence:
      - docs/quality-goal.md
```

## Bucket States

Allowed `status` values:

```text
active         the single bucket the loop is currently working
exhausted      every admissible variant in the family was run and terminated
killed         the family was terminated by a failing gate before exhaustion
blocked        progress requires a named external prerequisite
out_of_scope   tracked elsewhere; recorded here only to prevent re-entry
```

The loop works exactly one `active` bucket at a time. Opening a new bucket or
reopening a terminated family requires a decision record that names the
materially new causal reason, and a ledger update in the same change.

## Reporter

```bash
source .env
$ENSEXAM_PYTHON scripts/analysis/report_current_primary_program_status.py \
  --output-dir outputs/current_primary_program_status_YYYYMMDD
```

The reporter is fail-closed: missing anchors, hash mismatches, unparseable
YAML, unknown bucket states, untracked or missing evidence paths, `outputs/`
paths listed as evidence, or a bucket-count violation (not exactly one active
bucket while `program_state: active`) each produce a non-zero exit and an
explicit failure list. It writes `status.json` and `status.md` into a fresh or
empty output directory and refuses to reuse a nonempty one. Generated status
files are local evidence and must not be committed.

## Update Discipline

- Update the YAML block in the same change that adds the decision record it
  cites; the reporter enforces that every evidence path exists and is tracked
  by git.
- Copy durable numbers (calibration floors, frozen baseline metrics) into this
  ledger instead of referencing disposable `outputs/` files.
- Never edit a terminated bucket's history; append a new decision record and
  flip the status field.
