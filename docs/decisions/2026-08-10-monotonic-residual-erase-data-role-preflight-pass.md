# Monotonic Residual Erase Data-Role Preflight PASS

## Decision

`PASS`. The monotonic residual-erase iteration reuses the exact frozen page
roles and baselines without decoding an image, mask, label, or target pixel.
Its supervision semantics are now frozen before a trainer or target-derived
patch index exists.

This result authorizes only a dedicated training/config preflight. It does not
authorize pixel decode, patch construction, training, checkpoint generation,
prediction, inner-val15, development gates, SCUT115, holdout40, visual review,
reserved blind, parameter sweeps, or replacement of
`artifacts/current-primary`.

## Frozen Effective Roles

~~~text
train:                  275 pages (253 HW5K + 22 SCUT)
inner_val15:             15 pages
development_train160:   156 pages
development_next120:    112 pages
SCUT115:                115 pages
holdout40:               40 pages
reserved blind:           0 pages, unavailable and unauthorized
pairwise overlap:          0
~~~

The role plan references the existing hash-frozen role contract rather than
copying manifests or recomputing a new split. Current-primary, the current
second-stage checkpoint, their inference protocols, every manifest, and every
effective identity hash were revalidated.

## Frozen Supervision Semantics

Only the effective train role may expose targets during a later authorized
materialization step:

~~~text
input:
  frozen current-primary plus current-second-stage prediction

edit positive:
  target_luma - input_luma > 2 gray

preserve negative:
  target_luma - input_luma <= 2 gray
  includes target-darker, identity, and submargin target-lighter pixels

model output:
  nonnegative luminance delta bounded by 0.08

inference forbidden:
  target, label, mask, split, domain, route override
~~~

Targets remain forbidden for inner-val15, both development roles, SCUT115,
holdout40, and reserved blind until each role is opened by its quality-gate
position. The inference surface remains target-free at every stage.

## Closed Surfaces

The validator imported no `cv2`, `imageio`, `numpy`, `PIL`, or `torch`. No
training script exposes `monotonic_residual_erase`. The following planned
outputs were absent:

~~~text
hardcase_lists/monotonic-residual-erase-train-patches-v1.csv
artifacts/trials/monotonic-residual-erase-v1
docs/monotonic-residual-erase-training-plan.json
outputs/monotonic-residual-erase-training-preflight-20260810
~~~

Stage progression note: after the training/config preflight records PASS, this
validator remains replayable but permits exactly the dedicated trainer whose
path and SHA-256 are present in that PASS record. It still rejects the trainer
while the prerequisite is pending and rejects every additional or drifted
training CLI. The persisted data-role preflight JSON above remains the
historical pre-trainer result; it is not rewritten by later stages.

## First Quality Gate

Inner-val15 remains first and unchanged:

~~~text
minimum aggregate residual gain:      0.0005
measurable movement:                  required
aggregate residual regression:        prohibited
page residual regression:             prohibited
aggregate overerase regression:       prohibited
page overerase regression:            prohibited
~~~

No later split opens unless this gate eventually passes.

## Next Boundary

The next training/config preflight must freeze exactly one training attempt,
including a dedicated trainer, train-only patch semantics, identity
initialization, device, seed, step count, learning rate, loss terms, output
paths, and application protocol. It must prove these surfaces without decoding
real pixels, generating a patch, starting training, or opening a quality gate.

## Evidence Hashes

~~~text
docs/monotonic-residual-erase-data-roles.json
sha256 = f2555ddec01981e44ad5ce965977ef2c88003bae3ca5966c60437c93f91a110a

scripts/analysis/validate_monotonic_residual_erase_data_roles.py
sha256 = 2d617657c3e2b5094f2d841a0d8d72a5bef8a71560248bf9765fd7943814edc4

tests/test_validate_monotonic_residual_erase_data_roles.py
sha256 = 32336df1ade448a3df5cb55dfd01fb8891fbd17a6134e0d8659fd0b35ede9dfa

outputs/monotonic-residual-erase-data-role-preflight-20260810/preflight.json
sha256 = e0f3962c45faf25dbbf9ba9281d1a629e64a677d4d33e0657ce913678bfdb28b

docs/sign-separated-residual-data-roles.json
sha256 = 1c57978a3231d87970d90fa685ca532aeb3f155009c44a706edcafcb418e810d

scripts/infer/monotonic_residual_erase.py
sha256 = 4891a59ae60696f77d255540fb53e36ad4ea49466e07905fe642e3b5ff5d4f0b

outputs/monotonic-residual-erase-synthetic-prerequisite-20260810/audit.json
sha256 = 3a43b2a180642ebfe3edb1a7ef75f443e128ea760f2d2c17ccc40097beacd601
~~~

Intent: Freeze leakage-safe preserve-or-brighten supervision before any real-data or training path exists.
Constraint: This PASS is metadata-only and authorizes only a training/config preflight.
Rejected: Create new split manifests | the existing effective roles are already mutually exclusive and hash-frozen.
Rejected: Decode targets to inspect class balance now | pixel access is outside this gate's authority.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Reuse this exact role-plan hash and keep targets train-only in the training/config preflight.
Tested: 16 focused tests and 2 subtests; exact role counts and hashes; zero overlap; baseline hashes; reserved-blind closure; supervision semantics; absent outputs; no pixel-decoder imports; closed trainer surface.
Not-tested: Real pixels, patch balance, training configuration, training, checkpoint behavior, application reachability, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-10-monotonic-residual-erase-synthetic-prerequisite-pass.md
