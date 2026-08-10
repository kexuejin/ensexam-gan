# Sign-Separated Residual Training Preflight PASS

## Decision

`PASS`. The sign-separated residual repair iteration now has one runnable,
bounded training configuration and a dedicated trainer. The preflight checked
metadata, hashes, train-role membership, CLI closure, MPS availability, and
synthetic gradients only. It did not decode a real image or target, construct
a target-derived patch, generate a prediction, train, create a checkpoint,
open a quality gate, or change `artifacts/current-primary`.

The PASS authorizes only materializing the frozen train275 sample manifest,
current-primary predictions, frozen current-second-stage predictions, and the
registered target-difference patch index. Actual training remains prohibited
until `sign_separated_residual_train_materialization_audit` proves that those
artifacts contain exactly the effective train275 role and match the registered
pipeline and content hashes.

## Frozen Attempt

The only admitted attempt remains:

~~~text
model                       sign_separated_residual_delta
residual bound              0.08
steps                       80
device                      mps
batch size                  1
learning rate               0.00002
seed                        42
direction margin            2 px
route loss weight           1.0
bright magnitude weight     1.0
dark magnitude weight       1.0
identity delta weight       1.0
validation during training  disabled
patch tile / overlap        256 / 96
patch support floor         0.001
patch selection             top 256 brighten + top 256 darken
~~~

The training input is the frozen current-primary plus current-second-stage
pipeline output over the effective train role only. The dedicated trainer has
no model-selection option, explicit-mask mode, initialization checkpoint, or
training-time validation path. Its checkpoint metadata identifies the model
type and target-difference source for inference compatibility.

## Machine-Checked Evidence

The preflight returned:

~~~text
terminal = PASS
runnable = true
effective train pages = 275
train domains = 253 HW5K + 22 SCUT
MPS available = true
exact identity initialization = true
matching magnitude gradient = 0.0799999982
opposite magnitude gradient = 0.0
route gradient = 1.3333332539
real image decode = false
target decode = false
target patch materialized = false
training started = false
prediction artifacts generated = false
promotion enabled = false
reserved blind = unavailable
~~~

Focused verification passed with `32` tests and `13` subtests, including the
historical synthetic audit and data-role preflight. The immutable historical
trainer remains at SHA-256
`ce45f17c7d377aa665c9583215baead7ca555858cfe291ac089072ca8e51dc16`.

Evidence hashes:

~~~text
docs/sign-separated-residual-training-plan.json
sha256 = b298f0faf2de2c6a9e0f7c2c9a44dd64a7457962d1fdd42226b24e99a8ad470b

scripts/train/train_sign_separated_residual_probe.py
sha256 = b2e018f46d797335688858455cc9807757afad2756f3acf8eb48205ee5d2e829

scripts/analysis/build_sign_separated_residual_patch_index.py
sha256 = 36aa768fd5df5ea6573416ef79685be8339d9ebcb2831e5635c5a40c9c2f3056

scripts/analysis/validate_sign_separated_training_preflight.py
sha256 = 82862ffff19e24a12261ce02d5ab759111533191e819be532346ba60012ace7a

tests/test_sign_separated_residual_training.py
sha256 = 31bba18d2529d7bfea30f04899c0a701ef92e8d5b9eeb4702e622e167f9e2c18

tests/test_validate_sign_separated_training_preflight.py
sha256 = 1056097f04bb14bffb52d0c546e5295cb784b4a93be0dd6463b1593a558c7e5d

outputs/sign-separated-residual-repair-training-preflight-20260810/preflight.json
sha256 = 80511bc9d117fb61a4276dee05fa10a04c40b471d64dc494e09a3d540238246b
~~~

## Next Boundary

Materialization may read train-role sources and labels and generate only the
registered manifest, primary predictions, frozen second-stage predictions,
patch index, and patch summary. The audit must fail closed on missing, extra,
duplicate, or non-train pages; wrong prediction counts; pipeline checkpoint or
script drift; malformed or out-of-bounds patches; missing brighten/darken
support; or content-hash disagreement.

Inner-val15, development gates, SCUT115, holdout40, visual review,
reserved-blind access, threshold or parameter sweeps, promotion, and
`artifacts/current-primary` replacement remain closed.

Intent: Admit one reproducible train275 materialization path without admitting training prematurely.
Constraint: Reserved blind data is unavailable, and current-primary/current-second-stage remain immutable defaults.
Rejected: Enable the modified historical trainer | it would invalidate prior fail-closed evidence and mix legacy and new causal surfaces.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Do not invoke the dedicated trainer until the named materialization audit records PASS for the exact train275 manifest, frozen predictions, and patch index.
Tested: Thirty-two focused tests, thirteen subtests, one metadata/hash/synthetic training preflight, historical trainer SHA-256, absent planned outputs, and MPS availability.
Not-tested: Real prediction materialization, target-derived patch contents, training, checkpoint movement, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-09-sign-separated-residual-repair-data-role-preflight-pass.md
Related: docs/current-primary-quality-loop-ledger.json
