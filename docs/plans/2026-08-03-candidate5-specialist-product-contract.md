# Candidate 5 Caller-Known Specialist Product Contract

## Status

Frozen for implementation with terminal `CONTRACT_PASS`. This contract does not
promote Candidate 5. It defines the only claim and one-shot evidence path that
may later make the specialist eligible for system validation.

## Claim Boundary

The only candidate claim is:

> For inputs that an authoritative caller explicitly labels `hw5k`, the frozen
> Candidate 5 checkpoint materially reduces handwriting residual versus the
> frozen current-primary checkpoint without regressing aggregate or tail
> overerase on a separately reserved, unseen, HW5K-like paired evaluation set.

The claim does not cover unknown documents, inferred domains, SCUT inputs,
automatic routing accuracy, universal-checkpoint replacement, or general
handwriting-removal quality outside the admitted source distribution.

## Frozen Model Identities

```text
default config:
  artifacts/current-primary/config.yaml
  sha256=8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e
default checkpoint:
  artifacts/current-primary/micro_region_probe_step0001.pth
  sha256=e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae

hw5k config:
  artifacts/trials/hw5k-mixed-scut130-hw5k130-50pct-guard-jointtail-lite-step6400-respress-bs4-20260730/ensexam/20260801_183409/config.yaml
  sha256=c0ab5cc2a96dcaffa86dc75754c2a9bb9bfdc741c8ff7319e93bf8e2abc8adf8
hw5k checkpoint:
  artifacts/trials/hw5k-mixed-scut130-hw5k130-50pct-guard-jointtail-lite-step6400-respress-bs4-20260730/ensexam/20260801_183409/epoch_1.pth
  sha256=8da25117dd883f95059b6d7067e3dc3580da11339de365ef904f711db4a1f490
```

Any artifact drift invalidates the contract. A different checkpoint is a new
candidate and requires a new preregistration.

## Caller Domain Contract

- The caller supplies the exact manifest columns `image_path,domain`.
- Allowed labels remain `default`, `unknown`, and `hw5k`.
- `default` and `unknown` always select current-primary.
- Only literal `hw5k` selects Candidate 5; the harness never infers or repairs a
  domain label.
- Missing/invalid labels, duplicate images, artifact drift, branch failure,
  partial output, or prediction-hash mismatch fail the entire run.
- No branch silently falls back after inference starts.
- Until a promotion decision changes the registry status, an `hw5k` row still
  requires `--ack-research-specialist` and remains research-only.

The caller owns semantic correctness of the label. Because wrong routing is a
known quality risk, product integration must obtain the label from an
authoritative workflow attribute, not a filename, image heuristic, or user text
guess.

## Source-Risk Boundary

Candidate 5 is unsafe as the default or universal checkpoint. On frozen SCUT
inner-val15 it regressed mean residual by `+2.400499%` and mean overerase by
`+59.967283%`, with residual regressions on 10/15 pages and overerase regressions
on 12/15 pages. This is why `unknown -> current-primary` is an invariant rather
than a configurable preference.

No SCUT re-evaluation is required to prove the specialist claim because the
explicit branch leaves SCUT/default inference on current-primary. The known
wrong-route evidence must remain in the final risk report.

## Fresh Blind Admission

The comparison set must satisfy all of these before either checkpoint runs:

- a separately reserved source not used for training, development, model or
  threshold selection, target-aware review, or prior benchmark scoring;
- paired contaminated-document inputs and clean targets for the same-task
  handwriting/mark-removal objective;
- an entire upstream held-out split or a source-determined sample, never a
  model-outcome-selected subset;
- at least 200 distinct paired pages, so the linear p95 contains at least ten
  tail observations; the exact admitted count is then frozen;
- license, origin, split, custody, archive/content hashes, and download date are
  recorded;
- content-identity contamination audit, formal registration, and isolation
  validation pass against all required project sources.

The consumed HW5K official test, HW5K train/dev, SCUT, ExamInk-Seg, generated
outputs, and synthetic overlays are ineligible.

## Frozen Inference And Scoring

Both checkpoints use the same registered sample list and fixed primary-only
arguments:

```text
page_overlap=32
batch_size=8
copy_input_outside_mask=mb
copy_mask_threshold=70
copy_mask_threshold_auto=mb_cov8_step
copy_mask_dilate=0
device=auto
inference_reads_labels=false
post_freeze_change_threshold=12
post_freeze_eval_threshold=12
p95_convention=linear
```

Required order:

1. Freeze one parent comparison manifest, both model identities, both runbooks,
   source registration/isolation evidence, code hashes, commands, thresholds,
   sample count, and sample-list hash.
2. Run source-only current-primary inference for every sample.
3. Run source-only Candidate 5 inference for every sample.
4. Verify both prediction sets are complete and label-free before any scoring.
5. Score both frozen prediction sets against labels with thresholds `12/12`.
6. Apply the fixed comparison gate once.
7. Build local metric/audit evidence only after the gate result exists. Visual
   review is allowed only for a named metric disagreement or paper-tone risk and
   cannot overturn a failed metric gate.

If either inference run fails or is incomplete, the comparison is failed and
cannot be rescued by rerunning one branch under changed code or arguments.

## Promotion-Readiness Gate

Let `B` be current-primary and `C` be Candidate 5 on the exact same registered
fresh set. All six checks must pass with tolerance `1e-12`:

```text
G1 mean_residual(C) <= 0.80 * mean_residual(B)
G2 mean_overerase(C) <= mean_overerase(B)
G3 p95_residual(C) <= p95_residual(B)
G4 p95_overerase(C) <= p95_overerase(B)
G5 max_residual(C) <= max_residual(B)
G6 max_overerase(C) <= max_overerase(B)
```

G1 retains the preregistered `>=20%` material-improvement requirement from the
HW5K development program. G2-G6 prevent aggregate and tail safety regression.
No page-specific selection, threshold change, subset removal, second candidate,
or visual override is permitted.

Passing produces `specialist_promotable_pending_system_validation`; it does not
replace current-primary. Failing any check produces
`specialist_not_promotable_line_closed`, consumes the fresh set for Candidate 5,
and authorizes no tuning or router work.

## Development Evidence

The contract is executable against existing development evidence without
relaxing a metric. On all 232 HW5K development pages Candidate 5 passed the six
mean/p95/max comparisons:

```text
baseline:  mean residual=0.723852852720  mean overerase=0.064444891090
           p95 residual=0.891158374540  p95 overerase=0.202427577662
           max residual=0.954781812446  max overerase=0.455976452017
candidate: mean residual=0.562573946165  mean overerase=0.027853222229
           p95 residual=0.817533070216  p95 overerase=0.114394432559
           max residual=0.930664938705  max overerase=0.446714811668
```

Evidence:
`outputs/hw5k_train_intake_20260729/gate_a_result_candidate5_six_metric_contract_preflight.json`,
SHA-256 `708db2f562087357de86261566bfff855d7836b095d319edf00eab37e2d661e8`.

The development pass only supports freezing the contract. It is not fresh blind
evidence and cannot promote the checkpoint.

## Implementation Gap Before Blind Execution

Existing tools can prepare and verify one blind checkpoint and can compare two
development metric CSVs, but no parent protocol currently proves that both model
prediction sets were frozen before either set of labels was scored. Before M3,
implement and test a paired-checkpoint preparation/completion verifier plus the
exact `20%` G1 comparison. The implementation must compose existing registration,
isolation, freeze-evidence, inference, scoring, and completion helpers rather
than duplicate them.

## Prohibited Follow-Ons

- automatic router or inferred-domain classifier;
- reuse of consumed HW5K-test;
- any Candidate 5 retraining or checkpoint substitution under this contract;
- threshold, mask, page, or subset rescue after fresh-set exposure;
- promotion from development metrics alone.
