# External Text Layout Host Readiness Recovery

## Decision

`PREREQUISITE_NEEDED`. Safe in-session cleanup did not bring the host below
the unchanged `512 MiB` swap launch gate. The single tiled detector probe was
not started or consumed, and no cache, formal materialization, quality, or
promotion output was created.

The host had no Booted iOS Simulator, conflicting model process, or surviving
OMX team state. User-owned CoreSimulator services were terminated, leaving
only the small system-owned on-demand disk image service. No user application,
Nexus service, browser, development server, or current Codex process was
terminated. A non-interactive system cache purge was unavailable, and no
privilege bypass was attempted.

Five 30-second samples remained at `1,370.62 MiB` swap after cleanup. A final
reading later fell to `1,362.62 MiB`, still more than twice the frozen maximum,
with `87%` free memory. The largest visible unrelated service, Nexus, held
about `77 MiB` of swapped pages; stopping unrelated applications could not
reliably close the remaining gap and would introduce avoidable user-facing
side effects.

## Next Action

Use a fresh clean host restart as the next recovery boundary. Before opening
unrelated applications, recheck the same swap, free-memory, Simulator, model
process, probe-result, and cache-absence gates. A clean reading may authorize
the already preregistered single probe, but must not launch it automatically.
Do not weaken the absolute swap gate based on the current high free-memory
percentage.

Intent: Prevent repeated low-value process cleanup from consuming or weakening the one-shot detector safety path.
Constraint: The tiled probe is single-attempt and remains closed above 512 MiB swap.
Rejected: Treat zero Booted Simulators as sufficient recovery | CoreSimulator cleanup did not materially close the swap gap.
Rejected: Stop Nexus, browsers, or the current Codex session | their attributable swapped pages are insufficient and termination would disrupt unrelated work.
Rejected: Replace the absolute swap gate with free-memory-only readiness | would invalidate the frozen safety contract after prior host instability.
Rejected: Run an unattended watcher that launches after the gate passes | could consume the unique attempt without active verification of all launch conditions.
Confidence: high for the observed gate failure and absence of model execution; medium that a fresh clean restart will remain below the gate long enough for the probe.
Scope-risk: narrow
Reversibility: clean
Directive: Recheck readiness immediately after a clean restart; do not repeat Simulator or OMX cleanup unless new evidence shows those resources returned.
Tested: Git/branch state, boot time, memory pressure, exact swap usage, top-process RSS, bounded per-process vmmap summaries, OMX team absence, Simulator inventory, conflicting model process absence, five post-cleanup swap samples, and final probe/cache path absence.
Not-tested: Fresh clean restart readiness, detector execution, MPS execution, tiled probe, historical cache reconstruction, quality evaluation, visual review, promotion, or reserved-blind access.
Related: docs/external-text-layout-host-readiness-recovery-20260814.json
Related: docs/external-text-layout-tiled-9x9-one-page-safety-probe-v1.json
Related: docs/external-text-layout-cache-reconstruction-runtime-monitor-v1.json
