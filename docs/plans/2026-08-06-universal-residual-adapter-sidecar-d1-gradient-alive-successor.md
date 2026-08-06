# Universal Residual Adapter Sidecar D1 Gradient-Alive Successor

**status = successor_candidate_selected**
**r1_terminal = PASS**
**r1_result = ROOT_CAUSE_IDENTIFIED**
**initial_u4c_kill_status = superseded_by_protocol_mismatch**
**matched_copy_scut15 = equivalent_pass**
**actual_root_cause = zero_init_gradient_dead_zone**
**d1_terminal = PASS**
**d1_result = D1_GRADIENT_ALIVE_ZERO_EQUIVALENCE_DESIGN**
**h0_terminal = PASS**
**fresh_blind = disabled**
**promotion = disabled**

## Objective

Fix the gradient dead zone in `UniversalResidualAdapterSidecar` while preserving exact output equivalence at initialization.  
Keep `single clean(image)` external interface, no domain label/caller hint/hard routing, `current-primary` default unchanged.

## Root Cause (from R1)

- `global_residual_scale = nn.Parameter(torch.zeros(()))`
- Final projections initialized to zero: `nn.init.zeros_(...)`
- Training forward returned `baseline_output` directly when the residual was exactly zero.
- Together, the zero scale and zero-residual shortcut created a gradient dead zone:
  no learning signal reached the sidecar final projections.

## Design Decisions

**Option A (recommended)** - Minimal change, preserves exact equivalence until parameters move.

- Initialize `global_residual_scale` to a small positive fixed value: `torch.tensor(1e-3)`
- Keep final projections zero-init (to preserve exact equivalence at init).
- Keep the training path differentiable even when the residual tensor is exactly zero.
- This gives immediate nonzero gradient into the final projection parameters. The
  global scale may receive zero gradient on the very first step while the residual
  itself is still exactly zero; that is acceptable because final projections can move.

**Rejected Options**
- Option B (remove global scale): Would require changing the entire residual path; risk of sidecar behavior drift.
- Option C (constant positive cap): More complex, unnecessary.

## Implementation Changes

### 1. `networks/generator.py`

```diff
- self.global_residual_scale = nn.Parameter(torch.zeros(()))
+ self.global_residual_scale = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))

- candidate = baseline_output if residual_is_zero else torch.clamp(...)
+ candidate = torch.clamp(baseline_output + scaled_residual, -1.0, 1.0)  # training
```

### 2. Add gradient liveness test

Add a synthetic gradient liveness regression to
`tests/test_universal_residual_adapter_sidecar.py`.

```python
import torch
from networks.generator import UniversalResidualAdapterSidecar

class TestGradientLiveness:
    def test_sidecar_gradient_flow(self):
        sidecar = UniversalResidualAdapterSidecar(feature_channels=16).train()
        feature = torch.randn(2, 16, 8, 8)
        baseline = torch.zeros(2, 3, 8, 8)
        target = torch.full_like(baseline, 0.25)

        candidate, telemetry = sidecar(feature, baseline)
        assert torch.equal(candidate.detach(), baseline)

        loss = torch.nn.functional.mse_loss(candidate, target)
        loss.backward()

        final_bias_grads = [adapter[-1].bias.grad.abs().sum() for adapter in sidecar.adapters]
        assert all(float(grad_sum) > 0.0 for grad_sum in final_bias_grads)
```

### 3. Bounded V1 smoke

Run step20 training with new checkpoint and verify:

- Synthetic gradient liveness passes
- Pre-training zero-init matched-copy SCUT15 residual/overerase deltas remain 0.0
- After bounded training, matched-copy SCUT15 source guards do not regress

## Verification Protocol

- Run synthetic gradient liveness test (must pass).
- Run pre-training zero-init matched-copy SCUT15 equivalence check when producing
  a new initialized checkpoint (residual delta = 0.0, overerase delta = 0.0).
- Run post-training matched-copy SCUT15 guard evaluation for bounded V1 candidates.
- Commit only if all gates pass.

**Next step:** Implement the changes in the clean worktree, run tests, and commit as next clean replay.
