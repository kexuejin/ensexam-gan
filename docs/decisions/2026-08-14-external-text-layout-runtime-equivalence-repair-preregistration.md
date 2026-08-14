# External Text Layout Runtime Equivalence Repair Preregistration

## Decision

`PREREQUISITE_NEEDED`. Preregister and implement one repository-local,
in-memory repair for the frozen CPU Transformers detector runtime. The repair
removes only the first construction of `upsampled` in
`PPOCRV6MediumDetNeck.forward`. That list is overwritten before any read by an
equivalent list comprehension, so retaining both constructions creates three
additional high-resolution interpolation outputs without changing the returned
tensor.

This is implementation repair, not detector tuning. The frozen model, weights,
dtype, CPU/Transformers engine, preprocessing geometry, thresholds, scale
factors, reverse output order, and `torch.cat(..., dim=1)` remain unchanged.
The original external-layout preregistration and its SHA-256 remain unchanged.

## Frozen Runtime Binding

Apply the repair only to Transformers `5.12.1` and only when the installed
model source has SHA-256:

```text
4bb27b16b04056ee00779391a4943efa5b5c2745e9431e4e9aa652423b271210
```

Before importing PaddleOCR, validate that the source still contains exactly
the registered two-assignment AST: the first empty-list/loop construction at
line `306`, followed by the replacement list comprehension at line `314` and
the unchanged reverse-order concatenation. Import the hashed module only after
these static checks pass. Replace only the class's `forward` function in
memory; never modify global `site-packages`.

Fail closed on version, source hash, AST, loaded-module path, original-forward
identity, or repair-marker drift. A second identical application must be a
no-op and return `already_applied`.

## Static Verification

Fake-feature execution establishes the exact intended control-flow change:

```text
original and repaired returned value            identical
original target-region interpolation scales     2, 4, 8, 2, 4, 8
repaired target-region interpolation scales     2, 4, 8
scale-1 feature identity                         preserved
reverse output ordering / cat dimension         preserved / 1
version, source-hash, and AST drift              rejected before module import
repair repeated in one process                   already_applied
PaddleOCR TextDetection construction             after repair only
```

The test uses temporary source text, fake tensors, fake interpolation, and a
fake module. It does not import or execute the real detector, read any target or
label, create a formal layout output, or access a quality surface.

## Next Boundary

Do not run OCR, MPS, primary inference, second-stage inference, or training
while swap exceeds `512 MiB`. After the host passes that unchanged launch gate,
run exactly one nonformal target-free `hw5k_1011.jpg` safety probe under the
existing process isolation and health monitor. Success requires system free
memory at least `35%`, process-tree RSS at most `10 GiB`, swap at most
`512 MiB`, successful page completion, and zero residual processes.

That probe remains runtime-safety evidence only. Formal 275-page
materialization and the train-only diagnostic remain disabled until the safe
probe passes and the missing frozen prediction caches are reconstructed with
their historical content hashes. Product default, promotion, and reserved
blind remain unchanged.

## Registered Surface

```text
contract:
  docs/external-text-layout-runtime-equivalence-repair-v1.json
repair:
  scripts/analysis/external_text_layout_transformers_runtime_repair.py
integration:
  scripts/analysis/materialize_external_text_layout_support_train_only.py
test:
  tests/test_external_text_layout_runtime_equivalence_repair.py
```

Intent: Remove a provably overwritten high-resolution activation construction before another bounded detector safety probe.
Constraint: The prior detector initialization crossed the fixed memory floor, and swap remains above the fixed launch gate.
Rejected: Edit the generated Transformers file in global site-packages | creates an untracked machine-local fork and weakens provenance.
Rejected: Change model, dtype, engine, geometry, thresholds, scale factors, output ordering, or concatenation | changes the frozen producer instead of repairing equivalent control flow.
Rejected: Retry the detector before static verification or while swap exceeds 512 MiB | violates the existing runtime-safety boundary.
Confidence: high for control-flow equivalence; medium for runtime memory improvement; low for full root-cause attribution.
Scope-risk: narrow
Reversibility: clean
Directive: Keep the source/version/AST guards fail closed and do not generalize this into an unbound Transformers monkeypatch.
Tested: Fake-feature original-versus-repaired execution, exact interpolation call counts, scale-one identity, reverse concatenation, idempotence, source/version/AST drift rejection, and detector construction order without real model import or execution.
Not-tested: Real detector import, one-page detector completion, measured RSS reduction, formal materialization, cache reconstruction, train-only audit, training, candidate inference, quality gates, or promotion.
Related: docs/decisions/2026-08-13-external-text-layout-runtime-safety-prerequisite.md
