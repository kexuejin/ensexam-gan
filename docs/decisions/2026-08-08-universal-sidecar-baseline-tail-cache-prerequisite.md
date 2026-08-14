# Universal Sidecar Baseline-Tail Cache Prerequisite

~~~text
cache_prerequisite_terminal = PASS
active_iteration = universal-sidecar-baseline-tail-nonregression
failure_bucket = sidecar_measurable_movement_source_residual_regression
causal_change = train_only_current_primary_baseline_tail_nonregression_constraint
product_default = artifacts/current-primary
promotion = disabled
reserved_blind = disabled
scut115 = disabled
holdout40 = disabled
~~~

## Scope

This prerequisite builds and validates the train-only baseline-support cache
needed before testing a single causal change for the universal residual adapter
sidecar. The cache is a static training support signal only. It is not an
inference selector, threshold rescue, domain route, product API change, model
candidate, promotion claim, or blind-evaluation artifact.

The cache was generated from the registered mixed training manifest and the
frozen current-primary checkpoint:

~~~text
cache_dir = artifacts/caches/baseline-tail-universal-sidecar-d3-mixed-scut130-hw5k260-20260807
protocol = train_only_cached_baseline_tail_support
train_file_list = hardcase_lists/mixed_scut130_hw5k260_20260729.txt
train_file_list_sha256 = 0385fb96aa7aee1812b95b90acd4198e2af39e96c895a7cd8cfb2681258470ca
sample_count = 383
device = mps
page_overlap = 32
batch_size = 8
changed_threshold_px = 12.0
residual_threshold_px = 12.0
edit_threshold_px = 12.0
~~~

Frozen baseline:

~~~text
primary_config = artifacts/current-primary/config.yaml
primary_config_sha256 = 8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e
primary_weights = artifacts/current-primary/micro_region_probe_step0001.pth
primary_weights_sha256 = e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae
~~~

Local generation tool:

~~~text
script = scripts/analysis/cache_train_baseline_tail_support.py
script_sha256 = 25d3e5a796c10677ddce99c4f1078cfe20e316abb9a5f48ceb3ec25ea6406595
~~~

The tool is local evidence in the current dirty main worktree. This decision
does not commit the generated cache payload or rely on it for promotion.

## Validation

The independent cache audit re-read the generated manifest and rows CSV,
compared row order against the registered train manifest, rejected duplicate
filenames, checked the frozen baseline hashes, confirmed no filename overlap
with the SCUT inner-val15 gate manifest, counted both mask directories, and
recomputed source, label, residual-mask, and outside-mask hashes for all rows.

~~~text
validation_status = passed
cache_manifest_sha256 = 92c78488cbc59e5b380fa0496f395dcfd69624b8aff58186e1559bcc66bfa21b
rows_csv_sha256 = 592f6383164af92ec10008881a8b160cee6828132831ac66c4d3316d2742545a
sample_count = 383
residual_mask_count = 383
outside_mask_count = 383
inner_val_name_overlap = 0
mean_residual_safe_ratio = 0.47513981613155154
mean_outside_safe_ratio = 0.9549883370803015
min_residual_safe_ratio = 0.04053494666454386
max_residual_safe_ratio = 0.9969635258136802
min_outside_safe_ratio = 0.5285174068504178
max_outside_safe_ratio = 0.9997060453752933
~~~

The generated local artifacts are intentionally not committed:

~~~text
artifacts/caches/baseline-tail-universal-sidecar-d3-mixed-scut130-hw5k260-20260807/
  manifest.json
  cache_rows.csv
  residual_safe/*.png
  outside_safe/*.png
~~~

## Decision

The exact train-manifest baseline-support prerequisite passes. The active
iteration may move from PREREQUISITE_NEEDED to PENDING for the next bounded
preflight: one universal-sidecar candidate that adds the train-only
current-primary baseline-tail non-regression constraint while holding the
architecture, data manifest, seed, step budget, optimizer, learning-rate
schedule, and matched-copy inference protocol fixed.

The cache does not authorize SCUT115, holdout40, reserved blind, broad
retraining, threshold rescue, learning-rate or step sweeps, hard routing, or
artifacts/current-primary replacement. The next admissible gate is still the
leakage-safe SCUT inner-val15 structure/source guard.

Intent: Satisfy the train-only support prerequisite for the single causal baseline-tail sidecar preflight.
Constraint: Cache generation uses only the registered training manifest plus frozen current-primary predictions; promotion and blind splits stay disabled.
Rejected: Treat partial cache output as usable support | every row, mask, and hash had to validate before unblocking the preflight.
Rejected: Use the cache as an inference selector or routing signal | it is a static train-only loss support artifact.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not run SCUT115, holdout40, reserved blind, or product replacement from this prerequisite; run only the registered inner-val15 preflight next.
Tested: 383-row manifest/order check, source/label/mask SHA-256 recomputation, frozen current-primary hash check, mask counts, and inner-val15 filename-overlap check.
Not-tested: Candidate training, inner-val15 candidate inference, development gates, promotion gates, visual review, or reserved-blind verification.
Related: docs/current-primary-quality-loop-ledger.json
Related: docs/decisions/2026-08-06-universal-residual-adapter-sidecar-d2-step80-kill.md
Related: docs/decisions/2026-08-06-universal-residual-adapter-sidecar-d2d-step80-halflr-decision.md
