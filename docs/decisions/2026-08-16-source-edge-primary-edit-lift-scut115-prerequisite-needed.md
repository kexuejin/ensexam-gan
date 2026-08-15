# Source-Edge Primary-Edit Lift SCUT115 Prerequisite Needed

## Decision

`PREREQUISITE_NEEDED`. The fixed `source_edge_primary_edit_lift_v1` candidate
is authorized for SCUT115 after passing Dev40 and holdout40, but the local
current-primary SCUT115 baseline prediction PNG surface is absent.

Do not fabricate SCUT115 candidate PNGs from metrics CSVs, stale selector
outputs, or review-pack composites. The deterministic source-edge candidate
requires the actual baseline prediction image for each page.

## Evidence

The three local SCUT115 current-primary baseline directories each retain
`post_freeze_metrics.csv` but contain zero baseline prediction PNGs under
`frozen_predictions/pred`:

| Directory | PNG Count | Metrics CSV |
| --- | ---: | --- |
| `outputs/scut115_current_primary_baseline_20260726_primaryonly` | 0 | present |
| `outputs/scut115_current_primary_baseline_20260726_primaryonly_goal_185024` | 0 | present |
| `outputs/scut115_current_primary_baseline_20260726_goal_primaryonly` | 0 | present |

Repository/local search also found no `pred/13.png` or
`frozen_predictions/pred/13.png` candidate surface that could serve as the
first SCUT115 baseline prediction for the 115-page manifest.

## Required Recovery

Restore or rebuild a complete current-primary SCUT115 baseline prediction
surface for the existing 115-page sample list:

- Source list: `docs/scut-test115-relative.txt`
- Current-primary config: `artifacts/current-primary/config.yaml`
- Current-primary weights: `artifacts/current-primary/micro_region_probe_step0001.pth`
- Required prediction count: 115 PNG files
- Required split boundary: SCUT115 labels may be read only after baseline and
  candidate predictions exist for scoring

After recovery, run the same fixed source-edge candidate exactly once on
SCUT115. Do not change the source-edge threshold, primary-edit floor, alpha,
lift cap, component bounds, median kernel, or page set.

## Boundary

Holdout40 has passed. SCUT115 has not run. Reserved blind, visual review,
promotion, and `artifacts/current-primary` replacement remain closed.

Intent: Prevent false SCUT115 completion while preserving the authorized fixed candidate.
Constraint: Candidate generation requires baseline prediction PNGs; metrics CSVs cannot reconstruct image pixels.
Rejected: Generate SCUT115 candidate PNGs from metrics, stale selector outputs, or review packs | that would fabricate the required baseline-dependent image surface.
Rejected: Advance to reserved blind or promotion before SCUT115 | the held-out gate order is not complete.
Confidence: high
Scope-risk: narrow
Directive: Recover or rebuild the SCUT115 current-primary baseline prediction surface, then run exactly one fixed source-edge SCUT115 gate with unchanged parameters.
Tested: Local filesystem check over the three known SCUT115 current-primary baseline directories.
Not-tested: SCUT115 candidate gate, visual review, reserved blind, promotion.
Related: docs/decisions/2026-08-16-source-edge-primary-edit-lift-holdout40-pass.md
