# Monotonic Residual Erase Synthetic Prerequisite PASS

## Decision

`PASS`. The monotonic residual-erase representation satisfies its deterministic
CPU contract without accessing real data or enabling training. This authorizes
only the next metadata/data-role preflight.

It does not establish model quality and does not authorize image, mask, label,
or target decode; patch construction; training; checkpoint generation;
prediction; inner-val15; development gates; SCUT115; holdout40; visual review;
reserved blind; parameter sweeps; or promotion. `artifacts/current-primary`
and the current second-stage baseline remain unchanged.

## Proven Representation

~~~text
model type:                 monotonic_residual_erase
output capability:          preserve or brighten only
residual bound:             0.08
forward parameters:         x
route/dark branch:          absent
global scale:               absent
parameter count:            384,578
state tensors:              32
exact identity init:        true
training CLI enabled:       false
real data accessed:         false
quality gate started:       false
~~~

The model retains the cleanup tuple interface: candidate, edit alpha, and clean
candidate. Its edit-support and brighten-magnitude final projections are zero,
so both candidate forms are bit-exact copies of the input at initialization.

## Gradient And Movement Evidence

~~~text
brighten target +0.05:
  edit pixels:              64
  support bias gradient:    0.5
  magnitude bias gradient:  0.0799999982
  two-step delta max:       +0.0006752610
  negative pixels:          0

target-darker -0.05:
  preserve pixels:          64
  support bias gradient:    0.5
  magnitude bias gradient:  0
  two-step result:          exact no-op

identity target:
  preserve pixels:          64
  support bias gradient:    0.5
  magnitude bias gradient:  0
  two-step result:          exact no-op

forced-output delta range:
  +0.0799999833 to +0.0800000131
~~~

The tiny floating-point excess above `0.08` is `1.31e-8`, within the registered
`1e-7` numerical tolerance. No forced or optimized output contains a negative
pixel.

## Compatibility Evidence

Existing model construction and checkpoint loading remain exact:

~~~text
erasemap:                       384,612 params / 32 tensors / round-trip exact
residual_delta:                 384,612 params / 32 tensors / round-trip exact
sign_separated_residual_delta:  389,253 params / 36 tensors / round-trip exact
monotonic serialization:       exact, nonzero delta=0.0143240094
historical trainer sha256:      ce45f17c7d377aa665c9583215baead7ca555858cfe291ac089072ca8e51dc16
~~~

The historical trainer is unchanged and does not accept
`monotonic_residual_erase`. The synthetic model lives in a dedicated module so
the hash-frozen shared cleanup factory and all historical model records remain
unchanged. No training or real-data path was added.

## Next Boundary

The only admissible next action is a metadata/data-role preflight that:

- reuses the frozen train275, inner-val15, development, SCUT115, holdout40,
  and unavailable reserved-blind roles;
- proves pairwise role isolation without pixel decode;
- freezes current-primary and current-second-stage hashes;
- defines target-lighter support from train targets only;
- maps target-darker and identity pixels to preserve negatives;
- keeps all label/target information outside inference;
- leaves training and all quality gates closed.

## Evidence Hashes

~~~text
docs/decisions/2026-08-10-monotonic-residual-erase-preregistration.md
sha256 = 17d0ec3aebb4b2937022bb7b0d67f52a8b69e61f3daf7acd54e2a93edf421544

scripts/infer/monotonic_residual_erase.py
sha256 = 4891a59ae60696f77d255540fb53e36ad4ea49466e07905fe642e3b5ff5d4f0b

scripts/analysis/audit_monotonic_residual_erase_prerequisite.py
sha256 = b5151cbc43dcfa07fc500fa30330eeac9a86f7f90f8905c0edddfe9ae89f2aad

tests/test_monotonic_residual_erase_prerequisite.py
sha256 = 68d32074657bf2b45aa5f075b1f1ba6b6f82718f64dfc8e1e6716d9dd2d448c7

outputs/monotonic-residual-erase-synthetic-prerequisite-20260810/audit.json
sha256 = 3a43b2a180642ebfe3edb1a7ef75f443e128ea760f2d2c17ccc40097beacd601

docs/sign-separated-residual-data-roles.json
sha256 = 1c57978a3231d87970d90fa685ca532aeb3f155009c44a706edcafcb418e810d

scripts/train/train_patch_cleanup_erasemap_probe.py
sha256 = ce45f17c7d377aa665c9583215baead7ca555858cfe291ac089072ca8e51dc16
~~~

Intent: Prove a preserve-or-brighten representation before allowing any new residual-erasure data path.
Constraint: The prerequisite is CPU synthetic-only and cannot authorize training or quality evaluation.
Rejected: Treat positive synthetic movement as quality lift | no real page or target was accessed.
Confidence: high for the synthetic contract, unknown for real quality
Scope-risk: moderate
Reversibility: clean
Directive: Reuse the exact model and audit hashes in the metadata/data-role preflight; do not expose this model in a trainer yet.
Tested: 17 focused tests and 13 subtests; exact identity; nonnegative bounded output; brighten and preserve gradients; two-step cases; serialization; three legacy model round-trips; immutable trainer hash.
Not-tested: Real pixels, data-role derivation, patch construction, training, checkpoints, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-10-monotonic-residual-erase-preregistration.md
