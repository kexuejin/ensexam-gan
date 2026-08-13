# Universal Sidecar D3 Baseline-Tail Support Cache Verification

```text
subgoal_terminal = PREREQUISITE_SATISFIED
prerequisite = exact train-manifest baseline-support cache (active bucket)
training_handoff = disabled
d3_candidate_status = not_opened
product_default = artifacts/current-primary
```

## Scope

The active bucket
`cross_domain_residual_headroom_vs_source_solved_pixel_regression` names one
prerequisite for its admissible causal change: an exact train-manifest
baseline-support cache or an equivalent validated online frozen-teacher
signal. A cache for the sidecar train manifest was built on 2026-08-07 but had
no durable record. This subgoal verified it fail-closed and fixes the result;
it ran no training, no candidate inference, and read no evaluation split.

## Cache Identity

```text
cache_dir = artifacts/caches/baseline-tail-universal-sidecar-d3-mixed-scut130-hw5k260-20260807
protocol = train_only_cached_baseline_tail_support
train_file_list = hardcase_lists/mixed_scut130_hw5k260_20260729.txt
train_file_list_sha256 = 0385fb96aa7aee1812b95b90acd4198e2af39e96c895a7cd8cfb2681258470ca
primary_weights_sha256 = e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae
primary_config_sha256 = 8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e
rows_csv_sha256 = 592f6383164af92ec10008881a8b160cee6828132831ac66c4d3316d2742545a
builder_script = scripts/analysis/cache_train_baseline_tail_support.py
thresholds = changed/edit/residual = 12/12/12 px
```

The primary weights and config hashes are identical to the ledger anchors, so
the cache is exactly baseline-supported by the immutable current-primary.

## Verification Results

```text
manifest pages                = 383  (the mixed list is 383 lines; the
                                      scut130+hw5k260 name is nominal)
cache_rows.csv data rows      = 383
pages missing from cache      = 0
rows without residual_safe    = 0
residual_safe files           = 383
outside_safe files            = 383
train_file_list sha256        = match
rows_csv sha256               = match
csv row schema                = file, source/label/baseline_prediction sha256,
                                residual_safe + outside_safe path/sha256,
                                height/width, changed/outside px,
                                safe px and ratios
coverage stats (manifest.json): mean residual_safe_ratio 0.4751,
                                mean outside_safe_ratio 0.9550,
                                min residual_safe_ratio 0.0405,
                                min outside_safe_ratio 0.5285
```

## Prior-Family Caution

The first cached baseline-tail objective on SCUT inner130
(`docs/decisions/2026-07-26-current-primary-trainproxy-inner130-baseline-tail-cache-step160-bs2-lambda05-inner-val15-decision.md`)
improved mean residual but regressed overerase on every inner-val15 page and
was rejected before Dev40. The D3 design must treat overerase stability as a
first-class kill condition, not assume the cached objective is safe as
previously configured.

## Follow-Up Boundary

This record satisfies the cache prerequisite only. It does not open the D3
candidate: a bounded D3 plan must first fix the one-change hypothesis, the
frozen training surface (architecture, data, seed, step budget, optimizer,
learning-rate schedule, matched-copy inference), structure/gradient checks,
and the SCUT inner-val15 zero-page-regression kill gate before any training
step runs. SCUT115, holdout40, and reserved blind remain disabled. No
threshold rescue, routing, domain labels, base unfreezing, or
current-primary mutation is authorized.
