# Universal Residual Adapter Sidecar R1 Root Cause

```text
r1_terminal = PASS
r1_result = ROOT_CAUSE_IDENTIFIED
initial_u4c_kill_status = superseded_by_protocol_mismatch
matched_copy_scut15 = equivalent_pass
actual_root_cause = zero_init_gradient_dead_zone
successor_handoff = D1_GRADIENT_ALIVE_ZERO_EQUIVALENCE_DESIGN
product_default = artifacts/current-primary
fresh_blind = disabled
promotion = disabled
```

## Scope

R1 re-examined the U4C source-guard failure before deciding whether to continue
or kill the universal sidecar direction. The review stayed inside the universal
capability boundary:

- single external `clean(image)`-style inference surface
- no domain label, caller hint, source selector, or hard route
- internal continuous soft residual adapter mixing only
- `artifacts/current-primary` remains the product default
- fresh blind and promotion remain separate future goals

## Evidence

### Initial U4C comparison was protocol-mismatched

The initial U4C candidate inference used the default outside-mask copy behavior,
while the baseline CSV was generated with the matched current-primary copy
protocol:

```text
copy_input_outside_mask = mb
copy_mask_threshold_auto = mb_cov8_step
```

That mismatch produced apparent source-guard movement on 15 SCUT inner-val pages:

| Metric | Baseline Mean | Candidate Mean | Delta Mean | Delta P95 | Delta Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual_ratio | 0.176948604902 | 0.176724309843 | -0.000224295059 | +0.000139394888 | +0.000242385069 |
| overerase_ratio | 0.002324671360 | 0.002378996104 | +0.000054324743 | +0.000180469425 | +0.000280250542 |

Initial output:

```text
outputs/universal_sidecar_u4c_inner_val15_step20_20260806/post_freeze_metrics.csv
```

### Matched-copy replay is exactly equivalent

Candidate inference was replayed with the same copy protocol as the current-primary
baseline:

```text
outputs/universal_sidecar_u4c_inner_val15_step20_matched_copy_20260806/frozen_predictions/metrics.csv
outputs/universal_sidecar_u4c_inner_val15_step20_matched_copy_20260806/post_freeze_metrics.csv
```

Matched-copy replay produced no per-page deltas:

| Metric | Baseline Mean | Candidate Mean | Delta Mean | Delta P95 | Delta Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual_ratio | 0.176948604902 | 0.176948604902 | 0.000000000000 | 0.000000000000 | 0.000000000000 |
| overerase_ratio | 0.002324671360 | 0.002324671360 | 0.000000000000 | 0.000000000000 | 0.000000000000 |

```text
matched_copy_rows = 15
matched_copy_nonzero_delta_files = 0
source_guard_failures = []
```

### The sidecar did not learn

The U4B step20 checkpoint still had zero residual-producing parameters:

```text
artifacts/trials/universal-sidecar-u4a-mixed-scut130-hw5k260-step20-20260806/ensexam/20260806_100131/epoch_1.pth
```

Checkpoint audit:

```text
sidecar_key_count = 17
universal_residual_adapter_sidecar.adapters.0.2.bias norm=0.0 maxabs=0.0
universal_residual_adapter_sidecar.adapters.0.2.weight norm=0.0 maxabs=0.0
universal_residual_adapter_sidecar.adapters.1.2.bias norm=0.0 maxabs=0.0
universal_residual_adapter_sidecar.adapters.1.2.weight norm=0.0 maxabs=0.0
universal_residual_adapter_sidecar.adapters.2.2.bias norm=0.0 maxabs=0.0
universal_residual_adapter_sidecar.adapters.2.2.weight norm=0.0 maxabs=0.0
universal_residual_adapter_sidecar.global_residual_scale norm=0.0 maxabs=0.0
```

The gate final layer had nonzero initialized parameters, but the residual-producing
adapter final projections and global scale were still zero. The model therefore
remained equivalent to current-primary under the matched inference protocol.

## Root Cause

The sidecar was initialized with both:

- `global_residual_scale = 0`
- zero final projections for every residual adapter

The forward path then short-circuited exactly-zero residuals back to
`baseline_output`. Under this parameterization, the first training step had no
useful gradient path into the residual-producing final projections. The issue is
therefore a gradient-dead initialization, not a true source-guard regression.

## Decision

Continue the direction only through a D1 successor that preserves exact
zero-output equivalence while making the first training step gradient-alive.

Reject these as default next actions:

- threshold-only rescue
- copy-mask protocol tuning as a quality fix
- domain labels, caller hints, source/path selectors, or hard routing
- base generator unfreeze
- fresh blind or promotion evaluation before a gradient-alive successor passes
  bounded source guards

## Successor Handoff

D1 must:

- keep the same public generator/cleaning interface
- keep current-primary default unchanged
- preserve eval-time zero-output equivalence at initialization
- ensure synthetic loss backprop produces nonzero gradients on residual final
  projections
- fail closed on sidecar training configs that touch base generator parameters
