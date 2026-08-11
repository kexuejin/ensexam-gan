# Monotonic Residual Erase Candidate Application Preflight PASS

## Decision

`PASS`. The monotonic branch now has one target-free third-stage application
protocol whose meaningful-delta gate is reachable under the registered model
bound. The protocol consumes only the frozen current-primary plus current
second-stage prediction, requires edit probability at least `0.5`, requires a
mean brighten delta of at least `12` gray, and rejects a candidate if any color
channel is darker than the frozen baseline.

The original `2e-5` learning rate is not application-compatible: after the
registered 80 synthetic steps it produces only `0.023543` gray maximum
movement, which remains an integer-image no-op. The v2 plan therefore
supersedes only the learning rate and output path, registering `1e-4` for the
same model, data, 80 steps, seed, losses, and materialized patch index. The
registered case reaches `20.399681` gray without exceeding the `20.4` bound.

No real image or target was decoded, no training or checkpoint was started,
and no quality gate, visual review, reserved-blind access, promotion, or
default artifact replacement occurred.

## Post-KILL Replay

After the exact v2 run was audited and recorded as KILL, this preflight remains
replayable by accepting only the hash-registered v2 training output and
checkpoint audit from that KILL record. Candidate inference and all quality
outputs remain absent and closed. This continuity rule records stage
progression; it does not reopen the killed family.

## Frozen V2 Contract

~~~text
model type                    monotonic_residual_erase
baseline input                frozen primary + frozen second stage
training patches              exact audited train275 top-256 index
learning rate / steps         1e-4 / 80
residual bound                0.08 (20.4 gray)
edit probability threshold    0.5
minimum brighten delta        12 gray
direction guard               no candidate channel may darken baseline
training output               artifacts/trials/monotonic-residual-erase-v2
first quality role            inner_val15
~~~

## Synthetic Evidence

~~~text
legacy 2e-5 max / mean delta  0.023543 / 0.022658 gray
v2 1e-4 max / mean delta      20.399681 / 19.787786 gray
v2 edit probability mean      0.979236
negative-delta pixels         0
identity target output        exact no-op
target-darker output          exact no-op
reachable +13-gray candidate  applied
darker candidate              rejected
~~~

## Next Boundary

Run the exact v2 trainer once on the already audited train275 inputs. The
resulting checkpoint must pass a separate checkpoint audit before candidate
inference or inner-val15 begins. Do not tune the learning rate, step count,
application thresholds, patch index, or loss weights after seeing results.

## Evidence Hashes

~~~text
docs/monotonic-residual-erase-candidate-plan-v2.json
sha256 = a5089e8a2a0877ef9e34966f7868d82efeeb24d5057d0e5d594c8eda0b1dfc56

scripts/infer/run_monotonic_residual_erase_candidate.py
sha256 = 87ff65ae14bbb3eb821434f56381d5f2a84b77eece8d1237e7be12159775b636

scripts/analysis/validate_monotonic_residual_erase_candidate_application_preflight.py
sha256 = ffc753cfd2ca20335158197d776eebdc1afa8909dfeb72c6cd0df2b31f66fb39

tests/test_monotonic_residual_erase_candidate_application_preflight.py
sha256 = 87406a9b610133cc385467814afeb8e0b64502148afc7644db34ae4604950ed6

outputs/monotonic-residual-erase-candidate-application-preflight-20260810/preflight.json
sha256 = 7e2a519210244544c94d5753980e9945690a46b964d6a7801eebc337168e27e8
~~~

Intent: Freeze a reachable target-free monotonic application path before optimization.
Constraint: The model can brighten by at most 20.4 gray, and v1 training moves less than one integer gray level in the deterministic reachability case.
Rejected: Reuse the 32-gray legacy gate | it is analytically impossible under the 0.08 model bound.
Rejected: Keep the v1 learning rate and lower the gate below one gray | the resulting integer-image candidate remains a no-op.
Rejected: Use 5e-4 or 1e-3 | 1e-4 is the lowest bounded probe that already reaches the registered meaningful gate.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Run the exact v2 training configuration once, then audit the checkpoint before opening inner-val15.
Tested: Deterministic 80-step legacy and v2 reachability, exact identity and target-darker preservation, output bound, target-free CLI, probability/delta gating, darker-candidate rejection, MPS availability, and absent training/gate outputs.
Not-tested: Real training, checkpoint behavior, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-10-monotonic-residual-erase-train275-materialization-pass.md
Related: docs/current-primary-quality-loop-ledger.json
