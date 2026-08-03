# Explicit-Domain Dual-Checkpoint Research Harness Design

## Status

Approved on 2026-08-03 for implementation as a research-only harness.

This design closes the current Candidate 5 universal-checkpoint training line.
It does not promote Candidate 5, change `artifacts/current-primary`, authorize a
new specialist training run, or introduce automatic domain detection.

## Context

Candidate 5 materially improves HW5K Gate A but systematically regresses SCUT
Gate B. It is therefore useful as a provisional HW5K research specialist but
is not a universal replacement. The lower-risk next step is to let an external
caller provide the domain explicitly and dispatch each page to one of two
frozen checkpoints:

- `default` and `unknown` use `artifacts/current-primary`;
- `hw5k` uses the fixed Candidate 5 checkpoint as a provisional research
  specialist.

This separates checkpoint selection from image-content inference. It avoids the
unvalidated source-feature router proposed earlier, including label-dependent
statistics that are unavailable at inference time.

## Goals

1. Provide one auditable primary-only inference entry point for an explicit
   caller-provided domain manifest.
2. Preserve current-primary as the safe default for `default` and `unknown`.
3. Require explicit acknowledgement before the rejected-as-universal Candidate
   5 checkpoint can be used for `hw5k` research.
4. Run the two checkpoint branches serially and merge their predictions without
   changing prediction bytes.
5. Persist enough provenance to reproduce every dispatch and prediction.
6. Fail closed on malformed manifests, ambiguous output names, artifact drift,
   partial branch failure, or output collision.

## Non-Goals

- No automatic router, feature extraction, threshold fitting, or domain
  classifier.
- No full-data HW5K specialist training or any other training.
- No second-stage cleanup orchestration in the first version.
- No product promotion or unknown-paper generalization claim.
- No mutation of `artifacts/current-primary` or checkpoint payloads.
- No fallback from an invalid or missing domain to a guessed branch.

## Considered Approaches

### 1. Audited subprocess orchestrator — selected

A new Python entry point validates the explicit manifest, materializes one
source-only sample list per non-empty branch, invokes the existing frozen
primary inference script sequentially, verifies branch outputs, and copies
predictions into a unified directory while checking SHA-256 equality.

This reuses the established label-free inference boundary and keeps changes
outside the model and core page inference implementation.

### 2. In-process dual-model inference — rejected for the first version

Refactor common inference functions and load/switch models inside one process.
This may eventually reduce startup cost, but it expands the regression surface
and makes the first research harness harder to audit.

### 3. Caller-managed shell commands — rejected

The caller could split pages and invoke the existing script twice. This avoids
new code but provides no canonical manifest validation, failure aggregation,
merged provenance, or prediction-byte verification.

## Input Contract

The harness accepts a UTF-8 CSV with exactly these required columns:

```text
image_path,domain
```

Optional comments and implicit routing are not supported. Every non-empty row
must satisfy all rules before inference begins:

- `image_path` resolves to a source image and must not resolve through a path
  component named `label`, `labels`, `target`, `targets`, or `all_labels`;
- resolved image paths are unique;
- prediction filenames derived from image stems are globally unique;
- `domain` is exactly one of `default`, `unknown`, or `hw5k`;
- missing or unknown fields are errors rather than fallback signals.

The mapping is immutable:

| Caller domain | Branch | Checkpoint status |
| --- | --- | --- |
| `default` | current-primary | product default |
| `unknown` | current-primary | safe default |
| `hw5k` | Candidate 5 | provisional research specialist |

The CLI must require `--ack-research-specialist` whenever any `hw5k` row is
present. An acknowledgement is not promotion evidence; it only prevents the
research checkpoint from being selected accidentally.

## Artifact Contract

The first implementation freezes these registered artifacts:

```text
default config:
  artifacts/current-primary/config.yaml
default checkpoint:
  artifacts/current-primary/micro_region_probe_step0001.pth
default checkpoint SHA-256:
  e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae

hw5k config:
  artifacts/trials/hw5k-mixed-scut130-hw5k130-50pct-guard-jointtail-lite-step6400-respress-bs4-20260730/ensexam/20260801_183409/config.yaml
hw5k checkpoint:
  artifacts/trials/hw5k-mixed-scut130-hw5k130-50pct-guard-jointtail-lite-step6400-respress-bs4-20260730/ensexam/20260801_183409/epoch_1.pth
hw5k checkpoint SHA-256:
  8da25117dd883f95059b6d7067e3dc3580da11339de365ef904f711db4a1f490
```

The harness accepts explicit config/checkpoint paths for testability, but it
must record their resolved paths and SHA-256 values. The research-specialist
acknowledgement remains mandatory for every non-default specialist path.

## Execution And Data Flow

1. Parse and validate the entire caller manifest before creating inference
   outputs.
2. Resolve and hash both config/checkpoint pairs.
3. Acquire one non-blocking repository-local serial inference lock.
4. Create the requested output directory and write a `running` manifest that
   contains the frozen input and artifact identities.
5. Materialize source-only branch sample lists in original manifest order.
6. Invoke `scripts/infer/run_primary_full_page.py` for each non-empty branch,
   sequentially, with the same frozen inference arguments:
   - page overlap `32`;
   - batch size `8`;
   - copy mask `mb`;
   - automatic threshold `mb_cov8_step`;
   - fallback threshold `70`;
   - dilation `0`;
   - `--skip-label-metrics`.
7. Validate that every expected branch prediction exists exactly once and that
   branch metrics report the expected config/checkpoint SHA.
8. Copy each prediction into a unified `pred/` directory and prove the copied
   file SHA equals the branch prediction SHA.
9. Write `route_decisions.csv` in caller-manifest order and atomically replace
   the run manifest with `status=complete`.
10. Release the serial lock.

The harness never reads labels and never computes target-aware metrics.
Post-freeze scoring remains a separate workflow.

## Output Contract

The output directory contains:

```text
run_manifest.json
route_decisions.csv
pred/
branches/default/       # present only when non-empty
branches/hw5k/          # present only when non-empty
```

Each route record contains at least:

```text
row_index
image_path
image_sha256
caller_domain
selected_branch
primary_config_path
primary_config_sha256
primary_weights_path
primary_weights_sha256
branch_prediction_path
merged_prediction_path
prediction_sha256
```

`run_manifest.json` records the protocol version, status, command, input
manifest path/SHA, row/branch counts, fixed inference parameters, artifact
identities, branch commands, lock evidence, output paths, and research-only
promotion state.

## Failure Semantics

- Validation fails before inference for malformed rows, forbidden paths,
  duplicates, output collisions, missing artifacts, or missing acknowledgement.
- Lock contention fails immediately; it does not wait or launch a second MPS
  inference process.
- A branch subprocess failure, missing prediction, extra prediction, metrics
  mismatch, or SHA mismatch makes the whole run fail.
- Failed runs keep an auditable `status=failed` manifest and must never be
  interpreted as a complete routed prediction set.
- There is no branch fallback after inference begins. In particular, an `hw5k`
  failure does not silently reroute pages to current-primary.

## Promotion And Stop Contract

Candidate 5 remains `research_only/gate_qualified_nonpromotion`. This harness
does not turn a caller domain label into evidence of product safety.

A future explicit HW5K-domain product checkpoint requires a separate,
pre-registered domain-development gate, caller-provided domain contract,
source-domain risk report, contamination audit, and a fresh unseen HW5K-domain
blind set. Automatic routing requires an additional independent routing set,
unknown-domain policy, false-route cost threshold, and rejection behavior.

This implementation stops after the research harness and bounded verification.
It must not create a specialist-training configuration or automatic router as a
follow-on convenience.

## Testing And Verification

Unit and integration tests must cover:

- accepted domain mapping and original row ordering;
- missing/invalid domains and unknown CSV columns;
- forbidden label/target paths, including resolved symlinks;
- duplicate source paths and colliding prediction filenames;
- mandatory research-specialist acknowledgement;
- missing/stale artifact identity;
- empty default or specialist partitions;
- deterministic branch commands and strictly sequential subprocess execution;
- complete prediction-set validation;
- branch metrics artifact-SHA mismatch;
- copy/merge SHA equivalence;
- subprocess failure propagation and failed-run manifest state.

After automated tests pass, run one bounded serial MPS smoke with one
current-primary page and one explicit HW5K page. The smoke proves dispatch,
checkpoint loading, label-free inference, merge integrity, and complete audit
artifacts. It is not a quality gate or promotion result.
