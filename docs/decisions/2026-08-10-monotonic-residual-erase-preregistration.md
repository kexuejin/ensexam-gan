# Monotonic Residual Erase Preregistration

## Decision

`PREREQUISITE_NEEDED`. The next bounded uncertainty is whether residual
handwriting can be separated from printed-text and paper preservation by
removing darkening as an output capability, not by retuning the killed
sign-separated route/magnitude family.

This record authorizes only a deterministic CPU synthetic prerequisite. It
does not authorize real image, mask, label, or target decode; patch building;
training; checkpoint generation; prediction; inner-val15; development gates;
SCUT115; holdout40; visual review; reserved blind; threshold, learning-rate,
step-count, or loss-weight sweeps; or replacement of `artifacts/current-primary`.

## Named Failure Bucket

`residual_brighten_support_competes_with_dark_preservation_correction`

The sign-separated v2 checkpoint passed representation, role, materialization,
application, and training prerequisites, but its only real run routed all
`33,554,432` audited train-patch pixels to `darken`. No pixel selected identity
or brighten, so inner-val15 remained closed.

The 512 frozen train patches were balanced by selected patch label, not by
within-patch directional pixels:

~~~text
selected brighten patches: 256
  mean brighten ratio: 0.650198
  mean darken ratio:   0.236699

selected darken patches: 256
  mean brighten ratio: 0.077286
  mean darken ratio:   0.862852
~~~

This mixed support makes dark-preservation correction compete directly with
the residual-handwriting brighten path. Product labels independently show
that residual removal can create slight wins, while overerase and paper-tone
damage remain explicit loss buckets. The next representation therefore makes
preservation a negative edit decision rather than another output direction.

## Single Causal Change

Add an identity-initialized `monotonic_residual_erase` cleanup model over the
frozen current-primary plus current-second-stage pipeline prediction:

- one edit-support logit head;
- one nonnegative brighten-magnitude head;
- one shared luminance delta bounded by `0.08`;
- target-lighter pixels are edit positives;
- target-darker and identity pixels are preserve negatives;
- no route softmax, darken branch, global scale, RGB free residual, or target
  input at inference;
- unchanged three-value cleanup forward interface.

The model may only preserve or brighten. It cannot learn the all-darken state
that killed v2. This does not claim that monotonic brightening will preserve
printed text or paper tone on real pages; those remain quality-gate questions.

## Frozen Baseline And Roles

~~~text
current-primary checkpoint:
  e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae
current-primary config:
  8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e
current-second-stage checkpoint:
  36dd96a7efb8145a010b37a2e5351b6e1efda8fa329ee33a81e48511a8e400b7
inner-val15 manifest:
  fb25bb2aef2f9285403f908deb3da6d88b07b5d1c2c812965ce9e0636ddc172e
train effective identities:
  275 pages, 253 HW5K + 22 SCUT
  da9b90b9d344711d3dac2b202b7049ed4f921d003c8ee5a6c3531f1a9367e785
inner-val15 effective identities:
  86098752d80185792a1cbc627381139caedc006757c5d5ac6daeedf4905f1816
SCUT115 effective identities:
  500baa2493a4746d2e1cd393a709e8e05820ac6ad28d1a1bc7a175c56243baa5
holdout40 effective identities:
  552f6e3194ad5638ae957bd3d14d0ed8cd9e32380ca9805c9fac17915988d657
reserved blind:
  unavailable
~~~

The existing mutually exclusive roles are reused. No role, baseline, or
matched-copy gate is reopened by the synthetic prerequisite.

## Synthetic Acceptance Contract

The prerequisite is `PASS` only if deterministic CPU evidence proves:

- bit-exact identity initialization for output and clean candidate;
- final support and magnitude projections initialized to zero;
- public `forward` accepts only `x`;
- no dark, route, or global-scale parameter exists;
- forced output is nonnegative and bounded by `0.08`;
- a brighten target produces nonzero support-head and magnitude-head gradients;
- two synthetic brighten updates create positive movement without a negative
  pixel;
- target-darker and identity optimization remain exact no-ops with zero
  magnitude;
- checkpoint serialization is exact;
- erasemap, residual-delta, and sign-separated model construction and loading
  remain unchanged;
- the immutable historical trainer does not expose this model type.

Any missing or non-finite evidence, negative delta, bound violation, dead
brighten gradient, moved preserve case, legacy regression, changed historical
trainer, real-data access, or training CLI enablement is
`PREREQUISITE_NEEDED`.

## Next Boundary

A synthetic `PASS` may authorize only a metadata/data-role preflight. That
preflight must reuse the exact frozen roles and prove that target-lighter
support is derived from train targets only while target-darker and identity
pixels become preserve negatives. Training remains prohibited until a later,
separate training/config preflight passes.

## Alternatives Not Selected

- **Sign-separated rescue:** explicitly prohibited after the all-darken KILL.
- **Another signed loss or scalar sweep:** repeats closed loss/scale families.
- **t4 component selector continuation:** its target-aware oracle ceiling is
  below `0.001` SCUT115 and `0.0005` holdout40 residual gain.
- **Correction-fluid harmonization:** the six-page evidence is too narrow for
  the fixed inner-val15-first path and prior inpainting was visually rejected.
- **Broad primary continuation:** previous broad and patch-only updates damaged
  gate features or held-out residual.

## Registered Evidence

~~~text
docs/decisions/2026-08-10-sign-separated-residual-v2-checkpoint-kill.md
sha256 = 54735b3b1579b923df2c25bef80b43ee4354ed068e6a839c1f5997b8c1041052

outputs/archive/sign-separated-residual-repair-20260810/checkpoint-audit/audit.json
sha256 = 3889612efe6ded767e1ecbb1fb44cad563dfbd6f04b0a76107b098e7060b6b3d

hardcase_lists/archive/sign-separated-residual-repair-20260810-train-patches-v1.csv
sha256 = 62ac367251c0dc27f507f51c71dfc588c6ae3df70fd6c60005b226e2a5aef7d9

docs/product-quality-labels.csv
sha256 = 67d34e5b4a5d81dd846c523002c26e106ec288afc1d0d9e8438a8aa42e24d972

docs/sign-separated-residual-data-roles.json
sha256 = 1c57978a3231d87970d90fa685ca532aeb3f155009c44a706edcafcb418e810d

scripts/train/train_patch_cleanup_erasemap_probe.py
sha256 = ce45f17c7d377aa665c9583215baead7ca555858cfe291ac089072ca8e51dc16
~~~

Intent: Remove dark-preservation correction from the residual-erasure output space so it cannot overwhelm the brighten path again.
Constraint: Only deterministic CPU synthetic evidence is authorized; all real-data, training, evaluation, visual, and promotion surfaces remain closed.
Rejected: Rescue sign-separated v2 | every audited route collapsed to darken and the family is closed.
Rejected: Continue t4 selector tuning | the target-aware oracle ceiling is too low for product-quality lift.
Confidence: medium-high
Scope-risk: moderate
Reversibility: clean
Directive: Treat target-darker pixels as preserve negatives, never as a trainable darken output, unless a new preregistration overturns this evidence.
Tested: Existing checkpoint, patch-index, role, baseline, and product-label evidence reviewed; no implementation or real-data execution authorized by this record.
Not-tested: Synthetic model contract, data-role mapping, training, checkpoint behavior, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, promotion.
