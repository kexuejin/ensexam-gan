# Explicit-Domain Dual-Checkpoint Research Harness Decision

## Decision

Close the current Candidate 5 universal-checkpoint research line and implement a
research-only dual-checkpoint inference harness whose domain is provided
explicitly by the caller.

The immutable dispatch contract is:

```text
default -> artifacts/current-primary/micro_region_probe_step0001.pth
unknown -> artifacts/current-primary/micro_region_probe_step0001.pth
hw5k    -> Candidate 5 epoch_1.pth (research acknowledgement required)
```

Do not run the previously planned mask/repair attribution, gradient-conflict
diagnostic, or final universal training attempt. Do not train a new specialist,
implement an automatic router, change `artifacts/current-primary`, or claim
product promotion from this harness.

## Evidence And Artifact Identity

Candidate 5 passed HW5K Gate A but failed both SCUT Gate B means:

- HW5K residual `0.723853 -> 0.562574` and overerase
  `0.064445 -> 0.027853`;
- SCUT residual `0.176949 -> 0.181196` (`+2.400%`) and overerase
  `0.002325 -> 0.003719` (`+59.967%`);
- SCUT residual regressed on `10/15` pages and overerase on `12/15` pages.

The complete rejection evidence remains in
`docs/decisions/2026-08-02-hw5k-mixed-candidate5-gate-b-rejection.md`.

Frozen artifacts:

| Role | Path | SHA-256 |
| --- | --- | --- |
| Default checkpoint | `artifacts/current-primary/micro_region_probe_step0001.pth` | `e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae` |
| Default config | `artifacts/current-primary/config.yaml` | `8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e` |
| HW5K research checkpoint | `artifacts/trials/hw5k-mixed-scut130-hw5k130-50pct-guard-jointtail-lite-step6400-respress-bs4-20260730/ensexam/20260801_183409/epoch_1.pth` | `8da25117dd883f95059b6d7067e3dc3580da11339de365ef904f711db4a1f490` |
| HW5K research config | `artifacts/trials/hw5k-mixed-scut130-hw5k130-50pct-guard-jointtail-lite-step6400-respress-bs4-20260730/ensexam/20260801_183409/config.yaml` | `c0ab5cc2a96dcaffa86dc75754c2a9bb9bfdc741c8ff7319e93bf8e2abc8adf8` |

## Why This Replaces The Universal Attempt

Five mixed-domain candidates failed to provide a shared checkpoint that both
materially improves HW5K and preserves the SCUT source guard. Candidate 5 shows
that the HW5K benefit is real and that source retention is partly steerable,
but its SCUT regression remains systematic. The remaining universal experiment
would require new instrumentation and one more full 6,400-step attempt with a
low expected product return.

Explicit caller selection removes the unvalidated domain-classification problem
and lets current-primary remain the safe branch for default and unknown inputs.
It is more auditable than the earlier content-feature router proposal: the
existing `dev232_content_stats.csv` statistics were computed inside a
source-versus-label change mask and are therefore not inference-time
source-only routing evidence.

## Harness Contract

- Input is a strict `image_path,domain` CSV; every row must declare exactly one
  of `default`, `unknown`, or `hw5k`.
- Missing/invalid domains, forbidden target paths, duplicates, basename
  collisions, artifact drift, branch failure, missing/extra predictions, or SHA
  mismatch fail the whole run.
- Candidate 5 use requires an explicit research acknowledgement.
- Both branches run the existing source-only primary inference script with the
  same frozen production parameters and are serialized under one MPS lock.
- The merged output must preserve branch prediction bytes and record complete
  per-page route/artifact provenance.
- No label or target is read during inference. Metrics remain a separate
  post-freeze workflow.

The approved design is
`docs/plans/2026-08-03-explicit-domain-dual-checkpoint-design.md`.

## State Transitions

- Candidate 5 universal candidate: `rejected`; `universal_line_closed=true`.
- Candidate 5 explicit HW5K branch:
  `research_only/gate_qualified_nonpromotion`.
- Dual-checkpoint harness: `research_only`.
- Automatic router: `not_authorized`.
- Current-primary: unchanged product default.

This decision supersedes the execution authorization in
`.omx/plans/candidate5-next-bounded-fork-20260802.md` while preserving that file
as historical planning evidence.

## Stop Conditions

Stop and record failure without fallback if the harness cannot prove strict
caller-domain validation, serialized branch execution, complete output
coverage, artifact identity, and merged prediction SHA equality. Do not rescue
the run through content thresholds, automatic rerouting, a second specialist,
or a new training configuration.

## Promotion Boundary

The harness does not prove that caller domain labels are correct. Candidate 5
also does not become a product HW5K specialist merely because the caller can
select it.

A future explicit HW5K-domain product checkpoint requires its own preregistered
domain-development gate, source-risk report, caller contract, contamination
audit, and fresh unseen HW5K-domain blind set. A future automatic router further
requires an independent routing set, unknown-domain policy, false-route cost
threshold, and rejection behavior. HW5K-test cannot be reused for these claims.

## Verification Status

Completed 2026-08-03. The recorded bounded smoke in
`outputs/explicit_domain_dual_checkpoint_smoke_20260803/smoke_verification.json`
passed with `status=complete`, two routed rows (`default=1`, `hw5k=1`), an exact
file set, label-free inference, unchanged research status, matching branch and
merged prediction hashes, and no blockers. This remains research-only evidence;
it does not promote the specialist or change the default branch.

The focused suites passed under `ENSEXAM_PYTHON` loaded from `.env` with
`-m unittest tests.test_run_explicit_domain_dual_checkpoint
tests.test_prepare_blind_generalization_eval tests.test_verify_blind_eval_completion
tests.test_run_second_stage_residual_repair`: 61 tests total (`13`, `16`, `30`,
and `2` in that order). `-m py_compile` passed for the harness and its focused
test, and scoped `git diff --check` was clean. The four frozen artifact SHA-256
values were recomputed with `sha256sum` and matched the values above:

- default checkpoint: `e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae`
- default config: `8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e`
- HW5K research checkpoint: `8da25117dd883f95059b6d7067e3dc3580da11339de365ef904f711db4a1f490`
- HW5K research config: `c0ab5cc2a96dcaffa86dc75754c2a9bb9bfdc741c8ff7319e93bf8e2abc8adf8`
