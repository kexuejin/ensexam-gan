# Zero-Loss 112 Review-First Design

## Goal

Turn `zero_reject_veto_112_zero_target_loss` into a review-complete, auditable
release candidate before attempting to expand its selected-page coverage.

## Decision

Keep the current fixed selector and all inference thresholds unchanged. Its
112 selected pages already have zero target losses, metric losses, and local
rejects, but 58 selected pages remain target-borderline. Resolve those pages
through bounded, durable review shards instead of treating them as automatic
wins or widening the selector.

Review priority is:

1. `auto_win_candidate` target-borderline pages.
2. Low- and moderate-absolute-risk `ratio_noise_review` pages.
3. Remaining selected target-borderline pages.
4. Post-125 `promote_candidate` rows, which are the only source of later
   coverage expansion.

Each review turn is one persisted shard: at most four downscaled page previews
or at most twenty fixed-size crops. Labels must use the established vocabulary
and include a per-sample reason, local metric summary, and evidence paths.

## Promotion Rules

- A blank, ambiguous, or `needs_targeted_review` label promotes no page.
- A reviewed selected borderline page may remain in the 112-page candidate only
  when it has an explicit acceptance label; rejected pages are removed through
  the existing target-quality overlay path.
- A new page may be added only from a post-125 `promote_candidate` row with an
  explicit positive review label and after materialization verifies no target
  loss, metric loss, local reject, source mismatch, or missing output.
- No model retraining, threshold search, or replacement of current artifacts is
  permitted in this phase.

## Completion Gate

The phase completes only when the quality status reports zero target losses,
metric losses, local rejects, missing-quality rows, unresolved selected
target-borderline pages, and blank required post-125 decisions. Materialized
changed pages must also pass the documented local page/crop review and
readiness-smoke comparison.

