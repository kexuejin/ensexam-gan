# Sign-Separated Residual Repair Synthetic Prerequisite PASS

## Decision

`PASS`. The preregistered identity/brighten/darken representation satisfies its
deterministic CPU synthetic prerequisite. This authorizes only a separate
metadata-only split/data-role preflight. It does not authorize opening image or
target pixels, constructing target-derived patches, enabling the training CLI,
training, checkpoint generation, page inference, visual review, or any quality
gate.

## Compatibility And Identity

The audit proves:

~~~text
legacy erasemap parameters = 384612
legacy erasemap state tensors = 32
legacy erasemap checkpoint roundtrip = exact

legacy residual_delta parameters = 384612
legacy residual_delta state tensors = 32
legacy residual_delta checkpoint roundtrip = exact

sign-separated parameters = 389253
sign-separated state tensors = 36
sign-separated forward parameters = x only
training CLI enabled = false
global residual scale present = false
zero magnitude projections = exact
zero-init candidate = bit-exact identity
~~~

The existing `infer_full_page` public parameters are unchanged. Nonzero new
model state also survives checkpoint save/load with bit-exact output; the
round-trip fixture has maximum delta `0.013034403324127197`, so the check cannot
pass merely because both sides default to identity.

## Direction And Gradient Evidence

Forced routes satisfy the registered shared `0.08` bound:

| Route | Delta min | Delta max | Opposing pixels |
| --- | ---: | ---: | ---: |
| identity | 0.0 | 0.0 | 0 |
| brighten | 0.07999998331069946 | 0.07999998331069946 | 0 |
| darken | -0.07999998331069946 | -0.07999998331069946 | 0 |

The forced route probabilities are exactly one-hot in the selected synthetic
case. At zero initialization, branch-isolated supervision yields:

| Target | Matching magnitude gradient | Opposite magnitude gradient | Route gradient |
| --- | ---: | ---: | ---: |
| brighten | 0.07999999821186066 | 0.0 | 1.3333332538604736 |
| darken | 0.07999999821186066 | 0.0 | 1.3333332538604736 |

After two SGD steps, brighten output is strictly nonnegative and darken output
is strictly nonpositive, with `0` opposing pixels. The maximum synthetic
movement remains below the registered bound. A two-step identity target keeps
both magnitude tensors exactly zero and returns the input bit-for-bit.

## Evidence

~~~text
scripts/infer/patch_cleanup_erasemap.py
sha256 = dfa4b9946e6645fbc19f2267fb55cc26a61b5c8391c604eee691c7f229440727

scripts/analysis/audit_sign_separated_residual_repair.py
sha256 = 41e3d4544c590d657a440164014ab731bc1c259064cb064981ba7e4caf2d88eb

tests/test_sign_separated_residual_repair.py
sha256 = 6ce1e3c43c95a019a7d96f47e8ceb5449634ca65cfb9567397f59beb67a10e4b

outputs/sign-separated-residual-repair-synthetic-preflight-20260809/audit-final.json
sha256 = b448a17381a949c621e00c5fd9eeedf3404b4f7de26356043e3a80a56ebfebe5
~~~

Focused verification passed with `9` tests and `11` subtests. Final full
repository verification passed with `102` tests and `55` subtests.

## Next Boundary

The next admissible action is a metadata-only preflight that selects mutually
exclusive page roles and freezes artifact/config identities. It may read
manifests, path names, CSV metadata, and hashes. It must not decode an image,
mask, label, or target.

That preflight must prove:

- the current-primary and current-second-stage baseline artifacts are present
  and hashed without changing either default;
- train, inner-val15, development, SCUT115, holdout40, and reserved-blind page
  identities are mutually exclusive;
- no target-derived patch manifest exists before roles are frozen;
- any future training source is development-only and excludes every gate page;
- the future training CLI/config change is uniquely preregistered and absent;
- output directories for data materialization and training are absent;
- inner-val15 remains first with minimum residual gain `0.0005`, measurable
  movement, and zero aggregate/page-level residual or overerase regression.

Only a passing metadata preflight may authorize implementation of the
train-only branch losses and construction of target-derived train patches. A
separate real training preflight remains required after that implementation.

Intent: Admit metadata-only split design after synthetic evidence proves the sign-separated representation without opening a hidden training path.
Constraint: No image, mask, label, or target pixels were read by the synthetic prerequisite.
Constraint: Training CLI support remains intentionally absent.
Rejected: Treat synthetic movement as quality evidence | synthetic signs and gradients do not establish residual or overerase generalization.
Confidence: high for the synthetic contract; medium for future product lift
Scope-risk: moderate
Reversibility: clean
Directive: Freeze mutually exclusive metadata roles next; do not enable training or derive patches until that preflight passes.
Tested: Deterministic CPU identity, forced routes, branch gradients, two-step direction, bound, legacy/new checkpoint round-trips, signatures, and training-CLI closure.
Not-tested: Data roles, real images or targets, train-only loss integration, training, inner-val15, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-09-sign-separated-residual-repair-preregistration.md
Related: docs/current-primary-quality-loop-ledger.json
