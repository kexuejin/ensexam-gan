# External Text Layout Baseline-Relative One-Page Probe Pass

## Decision

`PASS`. The single preregistered target-free `hw5k_1011.jpg` CPU detector
attempt completed under the v2 baseline-relative safety contract. The result
path now exists with `attempt_count=1`; this exact probe must never be repeated
or overwritten.

The locked `60s` launch window sampled health `61` times with `85%` minimum
free memory, `71,073,792` bytes peak parent process-tree RSS, and zero swap
growth from an absolute launch baseline of `1,108,994,949` bytes. The detector
page then completed with:

```text
minimum free memory             63%
peak detector process-tree RSS  7,429,668,864 bytes (6.92 GiB)
peak swap growth                0 bytes
post-run free memory            73%
residual model processes        0
temporary outputs retained      false
```

All frozen safety limits passed: launch free memory remained at least `70%`,
runtime free memory remained at least `45%`, process-tree RSS remained below
`8 GiB`, and runtime swap growth remained below `512 MiB`. The spawned process
group exited normally, temporary page data was removed, and a post-run process
check found no conflicting model process.

## Authority

This PASS proves runtime safety for the exact tiled one-page detector path. It
does not create formal layout evidence and did not access targets, labels,
recognition text, or routing metadata. It did not use MPS and did not run cache
reconstruction, formal materialization, training, quality evaluation, visual
review, candidate inference, promotion, or reserved-blind verification.

PASS authorizes only the next preregistered step: adapt the already integrated
cache-stage process monitor to measure swap growth from a locked absolute
baseline. Cache execution remains closed until that separate implementation
passes both registered Python test surfaces and a static readiness check.

Intent: Convert the unconsumed runtime prerequisite into one-shot evidence without weakening host protections.
Constraint: The exact result path is terminal and cannot be retried, replaced, or interpreted as quality evidence.
Rejected: Run cache reconstruction immediately after probe PASS | its current validator and monitor still use the superseded absolute swap/result schema.
Rejected: Treat low swap growth as detector quality evidence | the probe did not inspect targets, labels, or formal metrics.
Confidence: high for the recorded runtime safety result; none implied for quality lift.
Scope-risk: narrow
Reversibility: irreversible for the consumed one-shot attempt; clean for subsequent cache integration.
Directive: Never overwrite or repeat this probe. Bind downstream cache authority to the exact result hash and v2 contract schema.
Tested: One real target-free CPU page, locked readiness window, continuous resource monitor, page record validation, process cleanup, result persistence, and residual-process check.
Not-tested: Cache reconstruction, formal layout materialization, train275 diagnostic, quality gates, visual review, candidate inference, promotion, or reserved-blind access.
Related: docs/external-text-layout-tiled-9x9-one-page-safety-probe-v2.json
Related: docs/external-text-layout-baseline-relative-swap-gate-integration-verification-20260814.json

Result: outputs/external-text-layout-runtime-safety-probe-tiled-9x9-20260814/result.json
Result-SHA256: 1909d66f29d18ca5805fb29b8b89ac054e240bc9231b2a7fa96121466e0ad550
