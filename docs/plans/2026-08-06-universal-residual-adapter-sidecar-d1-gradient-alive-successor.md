# Universal Residual Adapter Sidecar D1 Gradient-Alive Successor

**status = successor_candidate_selected**  
**r1_terminal = ROOT_CAUSE_IDENTIFIED**  
**initial_u4c_kill_status = superseded_by_protocol_mismatch**  
**matched_copy_scut15 = equivalent_pass**  
**actual_root_cause = zero_init_gradient_dead_zone**  
**d1_terminal = D1_GRADIENT_ALIVE_ZERO_EQUIVALENCE_DESIGN**  
**h0_clean_replay = complete**  
**fresh_blind = disabled**  
**promotion = disabled**

## Objective

Fix the gradient dead zone in `UniversalResidualAdapterSidecar` while preserving exact output equivalence at initialization.  
Keep `single clean(image)` external interface, no domain label/caller hint/hard routing, `current-primary` default unchanged.

## Root Cause (from R1)

- `global_residual_scale = nn.Parameter(torch.zeros(()))`
- Final projections initialized to zero: `nn.init.zeros_(...)`
- This creates a gradient dead zone: no learning signal reaches the sidecar parameters.

## Design Decisions

**Option A (recommended)** - Minimal change, preserves exact equivalence until parameters move.

- Initialize `global_residual_scale` to a small positive fixed value: `torch.tensor(1e-3)`
- Keep final projections zero-init (to preserve exact equivalence at init).
- This gives immediate nonzero gradient into final projections and global scale.

**Rejected Options**
- Option B (remove global scale): Would require changing the entire residual path; risk of sidecar behavior drift.
- Option C (constant positive cap): More complex, unnecessary.

## Implementation Changes

### 1. `networks/generator.py`

```diff
- self.global_residual_scale = nn.Parameter(torch.zeros(()))
+ self.global_residual_scale = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))
```

### 2. Add gradient liveness test

Create `tests/test_universal_residual_adapter_gradient_liveness.py`

```python
import torch
import pytest
from networks.generator import Generator

class TestGradientLiveness:
    def test_sidecar_gradient_flow(self):
        """Verify gradients flow to final projections and global scale when sidecar is enabled."""
        generator = Generator({'universal_residual_adapter_sidecar': {'enabled': True}})
        
        # Baseline frozen, sidecar enabled
        baseline = torch.zeros((1, 3, 64, 64))
        recon = torch.zeros((1, 3, 64, 64))
        
        # Forward
        candidate, telemetry = generator(recon, baseline)
        
        # Backward
        loss = candidate.sum()
        loss.backward()
        
        # Assert gradients nonzero on sidecar parameters
        grad_scale = generator.universal_residual_adapter_sidecar.global_residual_scale.grad
        assert grad_scale is not None
        assert grad_scale.abs().sum() > 0
        
        # Check one final projection
        proj_weight = generator.universal_residual_adapter_sidecar.adapters[0][-1].weight.grad
        assert proj_weight is not None
        assert proj_weight.abs().sum() > 0
```

### 3. Bounded V1 smoke

Run step20 training with new checkpoint and verify:

- Synthetic gradient liveness passes
- Matched-copy SCUT15 residual/overerase deltas = 0.0
- No source guard regression

## Verification Protocol

- Run synthetic gradient liveness test (must pass).
- Run matched-copy SCUT15 evaluation (residual delta = 0.0, overerase delta = 0.0).
- Commit only if all gates pass.

**Next step:** Implement the changes in the clean worktree, run tests, and commit as next clean replay.
