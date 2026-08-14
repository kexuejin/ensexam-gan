# Sign-Separated Residual Train275 Materialization PASS

## Decision

`PASS`. The exact effective train275 role has been materialized through the
frozen current-primary and current-second-stage pipeline. The registered
target-difference builder produced a deterministic, direction-balanced patch
index, and an independent audit rebuilt all candidates from the frozen
predictions and train-only targets with exact agreement.

No sign-separated training, checkpoint generation, inner-val15 evaluation,
later quality gate, visual review, reserved-blind access, promotion, or
`artifacts/current-primary` replacement occurred.

## Materialized Evidence

~~~text
effective train pages         275
domains                       253 HW5K + 22 SCUT
primary predictions           275
frozen pipeline predictions   275
target-difference candidates  53,978
selected patches              512
selected brighten             256
selected darken               256
selected pages                42
~~~

The audit proved exact filename sets, row counts, source/prediction hashes,
frozen primary parameters and adaptive mask policy, frozen second-stage
command parameters, train-role membership, patch coordinates, per-patch
direction metrics, deterministic ranking, summary hashes, and absent training
and quality-gate outputs.

Content identities:

~~~text
train filename SHA-256
e9ac4d6f700f41ef3a9b7c3f04ce0593f593324a881a0f9fc387901a497f9039

sample manifest SHA-256
ba31900496161322f839f366fa40765d71182d99a59ddad2537786310aae432f

primary prediction content SHA-256
6400c9413af963e3de280e348bd635cd962e5387c2e975e930036d320214274a

frozen pipeline prediction content SHA-256
2ffa40fc0c9b2a7e721d560f6f12edfe2ccdc1c1988582fa7a8104665cdc088a

train target content SHA-256
dfd459f552bd0828221c90258f33f4eacc54220494c7e02b21a179894853e99e

patch index SHA-256
62ac367251c0dc27f507f51c71dfc588c6ae3df70fd6c60005b226e2a5aef7d9
~~~

## Candidate Application Prerequisite

The materialization audit also makes the candidate application boundary
concrete. The registered model bound is `0.08`, so its maximum per-channel
change is approximately `20.4` gray levels. Reusing the current second-stage
application threshold `second_delta_threshold=32` would therefore reject
every possible candidate edit by construction.

Training is not authorized directly from this PASS. First run
`sign_separated_residual_candidate_application_preflight` and freeze a
candidate-specific third-stage application protocol. The protocol must use the
current frozen pipeline output as model input, select a data-independent
minimum meaningful delta no larger than the model bound, preserve exact
identity initialization as a no-op, prove synthetic correctly signed movement,
and keep all quality gates closed. This is protocol compatibility for the new
representation, not a post-result threshold sweep.

## Evidence Hashes

~~~text
scripts/analysis/materialize_sign_separated_train_inputs.py
sha256 = cbc3d107f9410f83d1698b17cd56c3b6121b3b698cdff7f66914ff960bcf19ec

scripts/analysis/audit_sign_separated_train_materialization.py
sha256 = d57b351f7ea1d14e420e208ce77d6647d551d377f88e5e32e3cc07b1d36c352c

tests/test_sign_separated_train_materialization.py
sha256 = 76e997a827816e1875883429897f5a21e5de2279ebbf86a2503f6aa0a4eeb343

outputs/sign-separated-residual-repair-train275-materialization-audit-20260810/audit.json
sha256 = 571fe09e0d17c80c3b4e4d726562b00a63ace63723b37c9102f021aea7d24427

hardcase_lists/sign-separated-residual-repair-train275-v1.txt
sha256 = ba31900496161322f839f366fa40765d71182d99a59ddad2537786310aae432f

hardcase_lists/sign-separated-residual-repair-train-patches-v1.csv
sha256 = 62ac367251c0dc27f507f51c71dfc588c6ae3df70fd6c60005b226e2a5aef7d9

outputs/sign-separated-residual-repair-train-patches-v1/summary.json
sha256 = b45e58f6ef4acf71cb0f1d78f291a33e1078ba27e8fe7995020bfdd66c9bcef6

outputs/sign-separated-residual-repair-train275-primary-v1/metrics.csv
sha256 = efd58814583089e888482a7e1604efc1d19ee5f514085cbef0e0c6cabf479846

outputs/sign-separated-residual-repair-train275-frozen-pipeline-v1/metrics.csv
sha256 = b800fdf385075bac46cc50db08a726dc2b9a6201b11a1229a164738b595a708d
~~~

Intent: Prove exact train275 pipeline and patch materialization before allowing any optimization.
Constraint: The registered 0.08 output bound is incompatible with the legacy 32-gray candidate delta gate.
Rejected: Train first and discover a guaranteed gate no-op later | the incompatibility is analytically provable before optimization.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Freeze and pass the candidate application preflight before invoking the sign-separated trainer; do not tune the application threshold after seeing gate results.
Tested: Thirty-eight focused tests, thirteen subtests, 275 primary predictions, 275 frozen pipeline predictions, independent reconstruction of 53,978 candidates, exact 512-patch comparison, content hashes, and absent training/gate outputs.
Not-tested: Candidate application implementation, real training, checkpoint movement, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-10-sign-separated-residual-training-preflight-pass.md
Related: docs/current-primary-quality-loop-ledger.json
