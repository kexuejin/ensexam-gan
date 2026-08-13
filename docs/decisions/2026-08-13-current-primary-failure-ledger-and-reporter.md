# Current-Primary Failure Ledger And Fail-Closed Program Status Reporter

```text
subgoal_terminal = PASS
subgoal_kind = maintenance_hygiene
resolves = 2026-08-07 loop plan PREREQUISITE_NEEDED (legacy status command)
product_default = artifacts/current-primary
training_or_inference_authorized = none
```

## Scope

The sustainable generalization-safe quality loop recorded a
`PREREQUISITE_NEEDED` maintenance subgoal: the legacy selector status command
(`scripts/analysis/report_quality_goal_status.py`) depends on removed
`outputs/balanced007_ranker_expansion_source_eval_20260708/` inputs and reports
the fixed-set selector backlog, not the active current-primary generalization
program. Until replaced, the loop was not reproducible from committed
repository state.

This subgoal added exactly two durable artifacts and made the evidence they
cite durable:

```text
docs/current-primary-failure-ledger.md
scripts/analysis/report_current_primary_program_status.py
```

The ledger freezes the current-primary anchors (checkpoint, config, inner-val15
manifest hashes), the calibrated inner-val15 evaluation noise (three
byte-identical matched-copy replays, residual lift floor 0.0005, baseline
mean/p95/max residual and overerase), the HW5K dev232 headroom reference
(residual mean 0.7238528527), and six named failure buckets with terminal
states: one `active`, one `exhausted`, two `killed`, one `blocked`, one
`out_of_scope`.

## Durability Repair

Twelve terminal decision records existed only as untracked files and were
staged into git so the ledger's evidence check passes:

```text
docs/decisions/2026-07-30-hw5k-mixed-candidate1-step160-gate-a-rejection.md
docs/decisions/2026-07-30-hw5k-mixed-candidate2-step1600-gate-a-rejection.md
docs/decisions/2026-07-30-hw5k-mixed-candidate3-step1600-respress-gate-a-rejection.md
docs/decisions/2026-07-30-hw5k-mixed-candidate4-step6400-respress-gate-a-pass.md
docs/decisions/2026-07-30-hw5k-mixed-candidate4-gate-b-rejection.md
docs/decisions/2026-08-02-hw5k-mixed-candidate5-gate-b-rejection.md
docs/decisions/2026-08-06-universal-residual-adapter-sidecar-u3-code.md
docs/decisions/2026-08-06-universal-residual-adapter-sidecar-u4a-static-readiness.md
docs/decisions/2026-08-06-universal-residual-adapter-sidecar-u4b-step20-smoke.md
docs/decisions/2026-08-06-universal-residual-adapter-sidecar-u4c-scut15-kill.md
docs/plans/2026-08-06-universal-residual-adapter-sidecar-u4-development-validation.md
```

The sidecar D2/D2C/D2D follow-up results previously existed only in the
untracked runtime plan; their terminal state (D2/D2C reproduced the U4C
regression, D2D halved learning rate and passed the guard with no measurable
page delta, artifacts pruned) is now fixed in the ledger's
`universal_sidecar_step_lr_nearby_sweep` bucket as `exhausted`.

## Verification

Fail-closed behavior was exercised before the durability repair: with the
ledger and twelve evidence files untracked, the reporter exited nonzero with
thirteen explicit failures and synthesized no status. After staging:

```text
program_status ok=True state=active
active_bucket=cross_domain_residual_headroom_vs_source_solved_pixel_regression
buckets=6 failures=0  (exit 0)
```

Negative cases verified with nonzero exits: reusing a nonempty
`--output-dir` (exit 1) and a missing ledger path (exit 1). Anchor sha256
checks cover the 362 MB checkpoint, config, and inner-val15 manifest on every
run. Generated status files stay local under `outputs/` and are not committed.

## Follow-Up Boundary

This subgoal authorizes no training, inference, dataset access, checkpoint
mutation, or candidate evaluation. The active bucket and its prerequisite are
unchanged: the next admissible causal change remains the train-only
current-primary baseline-tail non-regression constraint for the sidecar, whose
prerequisite is an exact train-manifest baseline-support cache or an
equivalent validated online frozen-teacher signal, gated by structure/gradient
checks and the SCUT inner-val15 zero-page-regression gate before any HW5K
development scoring. SCUT115, holdout40, and reserved blind remain disabled at
this stage.

Every ledger update must ship in the same change as the decision record it
cites; the reporter enforces existence, git tracking, and non-`outputs/`
evidence paths.
