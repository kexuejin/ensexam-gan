# Sign-Separated Residual Repair Data-Role Preflight PASS

## Decision

`PASS`. The sign-separated residual repair iteration has a frozen,
metadata-only data-role plan. The preflight reads manifest rows, page
identities, and registered artifact hashes; it does not decode images, masks,
labels, or targets. It does not construct target-derived patches, enable the
training CLI, train, generate checkpoints or predictions, run visual review,
open a quality gate, or change `artifacts/current-primary`.

The active iteration advances from `PREREQUISITE_NEEDED` to `PENDING` only for
the next named prerequisite: `sign_separated_residual_training_preflight`.
That preflight must pass before any real-data decode or training is admitted.

## Frozen Roles

The effective page-level roles are pairwise disjoint (`overlap_count = 0`):

| Role | Effective pages | Notes |
| --- | ---: | --- |
| `train` | 275 | `253` HW5K + `22` SCUT after all gate exclusions |
| `inner_val15` | 15 | first quality gate |
| `development_train160` | 156 | source manifest minus `inner_val15` |
| `development_next120` | 112 | source manifest minus `inner_val15` |
| `scut115` | 115 | later gate, still closed |
| `holdout40` | 40 | later gate, still closed |
| `reserved_blind` | 0 | unavailable and unauthorized |

The role plan freezes these effective identity hashes:

~~~text
inner_val15              86098752d80185792a1cbc627381139caedc006757c5d5ac6daeedf4905f1816
development_train160     ec28d39f691d1229c9609232189a73f005108f2b7e514e8bf77f1b66205f85df
development_next120      da8f8d5caa31ab73713f8cd158f88659110a3d1c16f4abdb74d2e4691dba2864
scut115                  500baa2493a4746d2e1cd393a709e8e05820ac6ad28d1a1bc7a175c56243baa5
holdout40               552f6e3194ad5638ae957bd3d14d0ed8cd9e32380ca9805c9fac17915988d657
train                   da9b90b9d344711d3dac2b202b7049ed4f921d003c8ee5a6c3531f1a9367e785
reserved_blind          unavailable
~~~

The source manifest hashes are also frozen by the role plan:

~~~text
inner_val15              fb25bb2aef2f9285403f908deb3da6d88b07b5d1c2c812965ce9e0636ddc172e
development_train160     7d3ad05218462edf31c31f7ebbbf0e0a4bb495d58b5fe51712049f91830abe19
development_next120      7410febb1a085c3e5ecc90450eac150fc583eceea37648a077256f25eda8fe58
scut115                  68e0ce32be4520f04d939506f3ed53a15b7b73cd4feaf74ec793f79fd8928e9b
holdout40               f76e1d4326fe40aac9cdcaf005c9ccd1c79f8622f81c12b3073bc1ef840f475d
train                   0385fb96aa7aee1812b95b90acd4198e2af39e96c895a7cd8cfb2681258470ca
~~~

## Baseline And Gate

The product default remains `artifacts/current-primary`. The preflight
revalidated the registered baseline identities:

~~~text
current-primary config       8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e
current-primary checkpoint   e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae
second-stage checkpoint      36dd96a7efb8145a010b37a2e5351b6e1efda8fa329ee33a81e48511a8e400b7
second-stage inference       3b066db93a7f16896e3c2b432098d08d82ae6020c7e9defa87f6cc46e9d9560d
~~~

`inner_val15` remains the first quality gate. A candidate must show residual
gain of at least `0.0005`, measurable movement, no aggregate residual or
overerase regression, and no page-level residual or overerase regression.
SCUT115, holdout40, reserved blind, threshold rescue, parameter sweeps, and
promotion remain closed until the earlier gates pass.

## Machine-Checked Evidence

The metadata-only validator returned:

~~~text
terminal = PASS
metadata_only = true
pixel_decoder_imports = []
training_cli_enabled = false
overlap_count = 0
reserved_blind_state = unavailable
reserved_blind_authorized = false
planned outputs = absent
~~~

Evidence hashes:

~~~text
docs/sign-separated-residual-data-roles.json
sha256 = 1c57978a3231d87970d90fa685ca532aeb3f155009c44a706edcafcb418e810d

scripts/analysis/validate_sign_separated_data_roles.py
sha256 = 23b0fd092bb824cf9143fadaf4ac2265117f5c9ca14c2b2b712bd30f0f025c2f

tests/test_validate_sign_separated_data_roles.py
sha256 = cb47027b4d2119bb0d589edccc205553c989345aec0106cbd5e1160fd58e3ae8

outputs/sign-separated-residual-repair-data-role-preflight-20260809/preflight.json
sha256 = 9bb38b018543db08f4e047bde08d2f0b2270bba1c2597e755917934f17539f0e
~~~

Focused verification passed with `9` tests. The validator is standard-library
only and uses path and text metadata; its AST import audit found no forbidden
pixel-decoder modules.

## Next Boundary

The only newly admissible action is to implement and execute
`sign_separated_residual_training_preflight`. It must verify one uniquely
registered configuration, frozen baseline initialization, the train-only
effective role, exact device and step bounds, closed evaluation/promotion
surfaces, and absent output destinations. Until that PASS, keep image/target
decoding, target-derived patch construction, training, inference, and all
quality gates prohibited.

Intent: Freeze leakage-safe page roles before admitting a real-data training path.
Constraint: This is metadata-only evidence; reserved blind data is unavailable and unauthorized.
Rejected: Treat role planning as a quality result | role isolation proves provenance, not model lift.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Do not open training or inner-val15 from this PASS; run the named training preflight first.
Tested: Nine focused tests and one metadata-only preflight; role counts, identity hashes, baseline hashes, zero overlap, forbidden imports, training CLI closure, reserved-blind state, and absent outputs.
Not-tested: Real training, target-derived patches, checkpoint movement, predictions, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-09-sign-separated-residual-repair-preregistration.md
Related: docs/current-primary-quality-loop-ledger.json
