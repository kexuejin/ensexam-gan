# Universal Sidecar D3 Baseline-Tail Preflight Pass

## Result

`PASS` for bounded training readiness. The first authorized downstream action
is the D3 80-step training run, followed by leakage-safe SCUT inner-val15 only.
This preflight is not evidence of quality lift or promotion eligibility.

## Evidence

~~~text
config_sha256 = a0916179bb3cf3e8923e17d1887d412de2a1a05109c457729b25adb315f6d66f
validator_sha256 = 3a2301dc96a5e693bd8ca70635a659a9893e19da7c5ef25714675ba635adf6e3
preflight_summary = outputs/universal-sidecar-d3-preflight-20260808/preflight.json
preflight_summary_sha256 = 4ef5f5eae355bba418babd6db285778aeafb1658b972e660f33d40f434f4f471
terminal = PASS
first_gate = scut_inner_val15
cache_manifest_sha256 = 92c78488cbc59e5b380fa0496f395dcfd69624b8aff58186e1559bcc66bfa21b
cache_rows_sha256 = 592f6383164af92ec10008881a8b160cee6828132831ac66c4d3316d2742545a
cache_sample_count = 383
inner_val15_name_overlap = 0
~~~

The validator checked the D3/D2 normalized semantic diff, all preregistered
loss values, runtime support, cache manifest/rows hashes, every source/label/
residual-safe/outside-safe row hash, cache row order, mask completeness,
current-primary config/weight hashes, sidecar-only trainable scope, frozen
BatchNorm, disabled validation/final-test modes, disabled later gates, and an
empty D3 output directory.

## Boundary

The training run must use only the registered D3 config. No change to step
budget, learning rate, scheduler, threshold, selector, product default, or
evaluation split is authorized. A non-passing inner-val15 result terminates D3
as `KILL`; a passing result still does not authorize SCUT115, holdout40, or
reserved blind without the development gate decision.

Intent: Admit the single preregistered D3 causal change only after its training support and gate isolation are reproducibly verified.
Constraint: Cache payload remains local and all later evaluation gates remain disabled until inner-val15 passes.
Rejected: Treat a cache manifest hash alone as sufficient | every declared source, label, and support mask hash was rechecked.
Confidence: high
Scope-risk: narrow
Reversibility: clean
Directive: Run exactly the registered step80 command next; do not parameter-sweep after this PASS.
Tested: `py_compile`; 25 focused tests/subtests; real 383-row cache preflight PASS.
Not-tested: D3 optimization trajectory, candidate prediction hashes, inner-val15 metrics, visual review, promotion splits, reserved blind.
Related: docs/decisions/2026-08-08-universal-sidecar-d3-baseline-tail-preregistration.md
Related: docs/current-primary-quality-loop-ledger.json
