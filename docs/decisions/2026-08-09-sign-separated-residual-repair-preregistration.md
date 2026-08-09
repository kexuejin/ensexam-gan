# Sign-Separated Residual Repair Preregistration

## Status

`PREREQUISITE_NEEDED`. This record selects the first materially different
post-D5 mechanism and authorizes only a CPU synthetic prerequisite. It does not
authorize reading a real training sample, constructing a train manifest,
training, checkpoint generation, page inference, visual review, or any quality
gate. `artifacts/current-primary` remains unchanged and is still the product
default.

## Selected Failure Bucket

~~~text
id = sign-separated-residual-repair
failure_bucket = single_head_residual_brighten_darken_entanglement
stage = optional identity-initialized repair after the frozen product pipeline
first_future_quality_gate = scut_inner_val15
~~~

The named failure is not merely generic overerase. Existing residual-delta
cleanup candidates can produce real residual gains, but the same single signed
delta/alpha surface also produces harmful brightening and unstable
brighten/darken behavior across held-out splits. Loss-only signed-direction and
dark-preservation variants changed this tradeoff without making it stable.

## Evidence And Inference

Direct evidence:

- the residual-delta representation is safer than full clean-image heads, but
  direct promotion remained worse than the current second-stage baseline;
- the original residual-delta family contains page-level residual wins, while
  dark-preservation and signed-delta loss variants still regress SCUT115 or
  holdout40 residual and change useful accepted behavior;
- hand-written and learned component selectors retain high held-out reject
  ratios, so the current feature-only selector surface does not reliably
  separate harmful local edits;
- reviewed product labels name both `residual_handwriting` wins and
  `overerase` / printed-content or paper-texture losses;
- D2-D5 universal sidecars are closed: unrestricted residuals regress, while
  direction restrictions terminate in zero support or zero applied scale.

Inference, with `medium-high` confidence:

The next useful uncertainty is whether making brighten, darken, and identity
explicit competing output states can retain residual removal while learning a
separate restoration path for harmful brightening. This is stronger than
another scalar loss because the output representation and branch supervision
change together; it remains unproven until synthetic gradient isolation and
later leakage-safe data gates pass.

## Exact New Representation

Only one new cleanup model type is admissible:

~~~text
model_type = sign_separated_residual_delta
residual_delta_bound = 0.08
route_classes = identity, brighten, darken
global_residual_scale = absent
~~~

For shared decoder feature `h`, the registered forward path is:

~~~python
route_prob = softmax(route_logits(h), dim=1)

bright_raw = bright_magnitude_head(h)
dark_raw = dark_magnitude_head(h)

bright_magnitude = residual_delta_bound * tanh(
    where(bright_raw >= 0, bright_raw, -bright_raw)
)
dark_magnitude = residual_delta_bound * tanh(
    where(dark_raw >= 0, dark_raw, -dark_raw)
)

signed_delta = (
    route_prob[:, brighten:brighten + 1] * bright_magnitude
    - route_prob[:, darken:darken + 1] * dark_magnitude
)
candidate = clamp(input + signed_delta, 0, 1)
~~~

Both magnitude heads are scalar per pixel and broadcast equally across RGB,
so the synthetic mechanism changes luminance without inventing a color cast.
The route probabilities share a simplex; therefore brighten and darken cannot
independently exceed the registered total bound. There is no learned global
scalar capable of silently disabling all branches.

The final projection of each magnitude head is initialized exactly to zero.
The route projection may initialize to zero logits; zero magnitude still makes
the complete stage an exact identity mapping.

## Registered Synthetic Supervision

The synthetic prerequisite may add only branch-isolated target construction
and losses needed to prove the representation:

~~~python
target_delta = target - input
target_luma_delta = mean_rgb(target_delta)

bright_mask = target_luma_delta > direction_margin
dark_mask = target_luma_delta < -direction_margin
identity_mask = not (bright_mask or dark_mask)

route_target = identity / brighten / darken from those masks
route_loss = cross_entropy(route_logits, route_target)

bright_target = clamp(mean_rgb(target_delta), 0, residual_delta_bound)
dark_target = clamp(-mean_rgb(target_delta), 0, residual_delta_bound)

bright_loss = masked_l1(bright_magnitude, bright_target, bright_mask)
dark_loss = masked_l1(dark_magnitude, dark_target, dark_mask)
identity_loss = masked_l1(abs(signed_delta), 0, identity_mask)
~~~

Branch magnitude losses are masked by target direction. An opposite branch
must not receive a magnitude gradient merely because folded absolute value has
a chosen derivative at zero. Labels and targets are training-only; the model
forward and inference surface accept only the image tensor.

## Fail-Closed Synthetic Prerequisite

No real data or training may be admitted until one deterministic CPU audit
proves all of the following:

- existing `erasemap` and `residual_delta` model construction, checkpoint
  loading, and forward behavior remain unchanged;
- the new model has no global residual-scale parameter;
- final magnitude projections are exactly zero and the complete stage is
  bit-exact identity at initialization;
- `forward` exposes no target, label, split, domain, route override, or caller
  argument;
- forced brighten routing yields only nonnegative RGB deltas;
- forced darken routing yields only nonpositive RGB deltas;
- identity routing is a no-op;
- all forced outputs remain within `0.08` of the input per channel;
- zero-branch route and matching magnitude gradients are nonzero for synthetic
  brighten and darken targets;
- the opposite magnitude branch receives exactly zero supervised gradient for
  each directional target;
- a two-step synthetic update creates nonzero correctly signed output for
  brighten and darken cases without an opposing pixel;
- identity-target optimization keeps both magnitudes at zero;
- state serialization and reload preserve exact output;
- no public inference interface regression occurs.

The audit must emit machine-readable per-case gradients, route probabilities,
signed-delta extrema, bound checks, state-shape counts, and public signatures.
Any missing field, non-finite value, opposing output, dead matching branch,
live opposite branch, or non-identity initialization is
`PREREQUISITE_NEEDED`.

## Future Data Boundary

Synthetic PASS will not authorize training directly. It may authorize only a
separate data/config preflight that must:

- freeze current-primary and the current second-stage baseline hashes;
- assign mutually exclusive train, inner-val15, development, SCUT115,
  holdout40, and reserved-blind roles before target-derived patches are built;
- prove zero page overlap across those roles;
- derive brighten/darken/identity supervision from train targets only;
- keep targets and labels inaccessible to inference and selectors;
- define one bounded attempt and an absent output directory;
- retain inner-val15 as the first quality gate with minimum residual gain
  `0.0005`, no aggregate or page-level residual regression, no aggregate or
  page-level overerase regression, and measurable movement.

No later split opens unless that first gate passes. Paper-tone and printed-text
review is required only after local metrics identify changed pages and before
promotion; visual review cannot rescue a failed metric gate.

## Alternatives Not Selected

- **D5 scale floor / more steps / learning-rate change:** exact D5 is closed and
  scalar coercion would repeat a measured zero-output family.
- **Another signed-delta or dark-preservation weight:** those loss-only variants
  already failed to stabilize held-out residual behavior in the single-head
  representation.
- **Hand-tuned or lightweight component selector:** current component features
  retain high held-out reject ratios and need new reviewed labels or context.
- **Correction-fluid paper-tone harmonization:** the available six-page review
  bucket is too narrow for the required inner-val15-first quantitative path,
  and prior inpainting was visually rejected.
- **Another primary/full-generator fine-tune:** broad and patch-only variants
  repeatedly damaged gate features or overerase and are not the lightest
  materially new mechanism.

## Registered Evidence

~~~text
docs/rejected-directions.md
sha256 = 0a3a098355a65a3a3ae210d2e221d30fea1b0894b395b8e97a016e0d96f3bff9

docs/product-quality-labels.csv
sha256 = 67d34e5b4a5d81dd846c523002c26e106ec288afc1d0d9e8438a8aa42e24d972

scripts/infer/patch_cleanup_erasemap.py
sha256 = af06c5f8829ac8d29febc6ac60819fed4142bcb02cfe2d26350883a7a2c26111

scripts/train/train_patch_cleanup_erasemap_probe.py
sha256 = ce45f17c7d377aa665c9583215baead7ca555858cfe291ac089072ca8e51dc16

docs/decisions/2026-08-09-universal-sidecar-d5-folded-direction-inner-val15-kill.md
sha256 = d90127ff58bf88e4606d741fb8415de8195b894ddc68e4a68605f45c2bc3cff8

docs/model-registry.md
sha256 = 3521acbb247e32fcb5961c867b9e42234258982a1c01a14961e48d4f21a981f5
~~~

Intent: Replace the exhausted universal-sidecar and single-head scalar-loss loops with one sign-separated, identity-initialized residual representation gated behind synthetic branch-isolation proof.
Constraint: This record authorizes CPU synthetic model/tests/audit only; no real sample, training, prediction, or quality gate may be opened.
Constraint: Existing cleanup model types and current-primary/current-second-stage artifacts remain unchanged.
Rejected: Global scale or scalar-loss rescue | those families have direct held-out or zero-output failure evidence.
Rejected: Selector-only continuation | existing page/component selector evidence is too narrow or unsafe on held-out pages.
Confidence: medium-high
Scope-risk: moderate
Reversibility: clean
Directive: Implement and execute only the fail-closed synthetic prerequisite next; require a separate split/data preflight before any training.
Tested: Preregistration evidence and repository search only.
Not-tested: New model implementation, synthetic gradients, data-role preflight, training, inner-val15, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-09-universal-sidecar-d5-folded-direction-inner-val15-kill.md
Related: docs/current-primary-quality-loop-ledger.json
