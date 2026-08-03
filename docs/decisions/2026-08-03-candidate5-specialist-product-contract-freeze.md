# Candidate 5 Specialist Product Contract Freeze

## Decision

M2 terminal: `CONTRACT_PASS`.

Freeze the caller-known Candidate 5 claim, artifact identities, wrong-route risk,
fresh-set custody requirements, dual-inference ordering, and six-metric
promotion-readiness gate in
`docs/plans/2026-08-03-candidate5-specialist-product-contract.md`.

This is a contract decision, not a product promotion. Candidate 5 remains
`research_only/gate_qualified_nonpromotion`; current-primary remains the default;
the data lane remains `external_data_prerequisite`; automatic routing remains
unauthorized.

## Why The Contract Passes

- The explicit caller-domain harness already fails closed, preserves manifest
  order, serializes the two MPS branches, pins artifacts, and verifies output
  hashes.
- Candidate 5 passed the existing preregistered HW5K Gate A by `22.281%` mean
  residual improvement with mean overerase improvement.
- A new six-metric preflight confirms that mean, linear-p95, and max residual and
  overerase all move in the correct direction across 232/232 HW5K development
  pages. The result is `accept`, SHA-256
  `708db2f562087357de86261566bfff855d7836b095d319edf00eab37e2d661e8`.
- Wrong-route risk is already measurable and is handled by the immutable
  `default|unknown -> current-primary` mapping, not by weakening source gates.

## Remaining Prerequisites

1. Implement a paired-checkpoint blind preparation/completion protocol that
   proves both source-only prediction sets were frozen before either label score.
2. Add the exact `>=20%` mean-residual comparison to a reusable fixed gate while
   preserving the other five non-regression checks.
3. Admit and register a genuinely fresh paired source with at least 200 pages.
4. Execute the contract once, then either advance to system validation or close
   Candidate 5 without rescue.

Only items 1-2 are locally executable while external data is unavailable. They
form the next bounded implementation milestone. Item 3 remains parked.

## Verification

- No training or inference was run.
- Existing frozen HW5K development CSVs were compared with the repository gate
  implementation in `--six-metric-only` mode.
- All six checks passed; command exit code was zero.
- No threshold, checkpoint, manifest, or current-primary artifact changed.
