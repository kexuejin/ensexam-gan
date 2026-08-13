# Independent HW5K Expert Disagreement Support Preregistration

## Decision

`PREREQUISITE_NEEDED`. The next bounded uncertainty is whether a separately
trained frozen HW5K expert contributes target-aligned support evidence beyond
the frozen current-primary RGB surface on pages that expert never saw during
training.

This is not explicit-domain routing and does not reopen Candidate 5 product
promotion. Candidate 5 improved HW5K but regressed SCUT, so its routed product
path remains closed. The materially new causal source is paired same-page
output from two independently trained checkpoints: both checkpoints run on
every diagnostic source page, without routing, and the specialist RGB is
removed in the independent ablation. Repository history contains no prior
train-only test of that paired support source.

Only exact target-free paired materialization and one fixed train-label
separability diagnostic are authorized. No model training, checkpoint,
candidate inference, inner-val15, development gate, SCUT115, holdout40,
visual review, reserved blind, promotion, or current-primary replacement is
authorized.

## Leakage Boundary

The specialist was trained from 260 named SCUT/HW5K pages. A diagnostic over
all train275 pages would leak 152 seen pages into the support claim. Freeze the
diagnostic population as the train275 manifest in its original order after
excluding every basename in the specialist training manifest:

```text
train275 source pages:                275
specialist-training overlap:          152
overlap domains:             HW5K 130 / SCUT 22
eligible unseen pages:                123
eligible domains:            HW5K 123 / SCUT 0
```

The source manifest SHA is
`ba31900496161322f839f366fa40765d71182d99a59ddad2537786310aae432f`.
The external exclusion manifest SHA is
`12397a11b33f856e6db5c20447678e27428c6ee324175e8feca50a4ce565aefc`.
The derived basename-only newline SHA is
`e2921d717086c080606acd69dbec2de0e4a97281edc460aa6c0b74af41097698`;
the ordered full-path newline SHA is
`ad7c794706edb1b832cb30af978663853fb10646531bc4cf83011023338d81e2`.

Do not create a derived manifest in this commit. The materializer must first
reproduce every source, exclusion, count, domain, and content identity. Any
drift is `PREREQUISITE_NEEDED` before image decode. This HW5K-only diagnostic
can establish incremental support, but it cannot establish SCUT product
safety or authorize routing.

## Frozen Inputs And Representation

Current-primary remains frozen at:

```text
checkpoint: artifacts/current-primary/micro_region_probe_step0001.pth
sha256: e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae
config: artifacts/current-primary/config.yaml
sha256: 8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e
```

The research-only HW5K expert remains frozen at the registered
`20260801_183409` run:

```text
checkpoint sha256: 8da25117dd883f95059b6d7067e3dc3580da11339de365ef904f711db4a1f490
config sha256: c0ab5cc2a96dcaffa86dc75754c2a9bb9bfdc741c8ff7319e93bf8e2abc8adf8
```

Run exact source-only current-primary inference separately with each artifact
pair on all 123 pages. No domain field, caller label, route, or expert selector
is accepted. Both runs use `page_overlap=32`, `batch_size=8`,
`copy_input_outside_mask=mb`, `copy_mask_threshold_auto=mb_cov8_step`, fallback
threshold `70`, dilation `0`, `device=auto`, and `--skip-label-metrics`.
Version 1 full features are exactly current-primary RGB/255 plus
frozen-specialist RGB/255. The ablation is the identical probe and coordinates
with specialist RGB removed. Do not add an explicit difference channel,
source or second-stage RGB, masks, page scalars, thresholds, neighborhoods,
transforms, or alternative checkpoints.

## Frozen Diagnostic

After target-free paired outputs pass artifact, source, set, alignment, and
content validation, train-role targets may define positive pixels exactly as
`target_luma - current_primary_luma > 2` gray. All other pixels are preserve.
The audit must first validate the complete frozen train275 label-set identity:
275 files with content SHA
`dfd459f552bd0828221c90258f33f4eacc54220494c7e02b21a179894853e99e`.

Reuse the established five basename-hash page folds, deterministic SplitMix64
sampling, and at most 1024 pixels per class per page. Fit float64 closed-form
ridge with `lambda=1.0`, fitting-fold-only standardization, and an unpenalized
intercept. Compare the six-channel full representation with current-primary
RGB on identical coordinates and folds. No threshold or hyperparameter is
learned.

`PASS` requires every condition:

- exactly 123 unseen HW5K pages and five nonempty held-out folds;
- mean held-out fold AUC at least `0.65`;
- every held-out fold AUC at least `0.55`;
- macro median per-page AUC at least `0.60`;
- mean AUC at least `0.03` above current-primary RGB;
- positive mean score above preserve in at least four of five folds.

Any metric failure is `KILL`. Missing or drifting provenance is
`PREREQUISITE_NEEDED`. `PASS` authorizes only a separately preregistered
expert-conditioned data/training/application preflight.

## Terminal Successors

- `PASS`: freeze a separate expert-conditioned preflight.
- `KILL`: close paired independent-expert RGB without channel, difference,
  transform, neighborhood, threshold, checkpoint, probe, routing, or training
  rescue.
- `PREREQUISITE_NEEDED`: repair exact provenance or implementation only.

## Registered Surface

```text
plan:
  docs/independent-hw5k-expert-disagreement-support-prerequisite-v1.json
future materializer:
  scripts/analysis/materialize_independent_hw5k_expert_outputs_train_only.py
future audit:
  scripts/analysis/audit_independent_hw5k_expert_disagreement_support.py
future test:
  tests/test_independent_hw5k_expert_disagreement_support_prerequisite.py
future materialization:
  outputs/independent-hw5k-expert-support-materialization-20260813/
future audit:
  outputs/independent-hw5k-expert-support-prerequisite-20260813/audit.json
```

Intent: Test a genuinely independent frozen model producer after same-pipeline support surfaces failed their incremental-margin gates.
Constraint: Exclude every page seen by the specialist and keep all optimizer and quality surfaces closed.
Rejected: Reopen explicit-domain Candidate 5 routing | its HW5K gain did not preserve SCUT.
Rejected: Use all train275 pages | 152 pages overlap specialist training and would inflate the support claim.
Rejected: Add expert-difference transforms or a nonlinear probe | that would rescue the family before its independent raw-RGB ablation is measured.
Confidence: medium
Scope-risk: moderate
Reversibility: clean
Directive: A failed ablation margin closes paired specialist RGB; do not reinterpret this prerequisite as routing approval.
Tested: Frozen artifact hashes, specialist-training overlap, eligible count, eligible domains, and both derived content hashes were verified locally before registration.
Not-tested: Paired materialization, support separability, training, candidate inference, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-03-explicit-domain-dual-checkpoint-research-harness.md
Related: docs/decisions/2026-08-12-second-stage-alpha-support-diagnostic-kill.md
