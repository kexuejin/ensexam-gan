# External Text Layout Recovered Materializer Formal Memory Launcher V5 Preregistration

## Decision

`PREREQUISITE_NEEDED`. V4 resumed correctly, skipped the retained first page,
and completed pages 2 through 8. Page 9 was terminated when free memory reached
`39%`, below v4's retained one-page probe floor of `45%`. After termination the
host recovered to `69%` free memory, swap growth remained exactly zero, no
model process remained, and all eight page/record pairs are hash-bound.

As with the prior RSS correction, `45%` is a probe-specific strengthened value.
The unchanged shared formal materializer freezes a `35%` runtime free-memory
floor. V5 may restore only that formal runtime value while retaining `70%`
launch readiness, `10 GiB` detector RSS, `0.25s` monitoring, `512 MiB` swap
growth, CPU one-page isolation, Simulator checks, timeout, and process-group
termination. Any sample below `35%` remains terminal.

Only launcher/test implementation and synthetic verification are authorized.
Resume from `hw5k_1214.jpg` remains closed until v5 integration PASS is pushed;
the eight completed pages must not rerun.

Intent: Use the frozen formal runtime memory floor while preserving stronger launch readiness and hard growth limits.
Constraint: A real page reached 39% free with zero swap growth and clean recovery, while the shared formal floor is 35%.
Rejected: Lower below 35% or remove the floor | no formal contract authorizes weaker protection.
Rejected: Retry page 9 under 45% | the exact run already proved that gate blocks progress without indicating swap pressure.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Resume from page 9 only after integration PASS; preserve all eight completed records and every other v4 safety limit.
Tested: Eight exact completed pages, runtime free-memory rejection, zero swap growth, post-stop host recovery, no model process, and retained content hashes.
Not-tested: V5 implementation, page-nine completion, remaining pages, final publication, diagnostic, or quality gates.
Related: docs/external-text-layout-recovered-materializer-formal-memory-launch-v5.json
Related: docs/external-text-layout-recovered-materializer-formal-rss-launch-v4-integration-verification-20260815.json
