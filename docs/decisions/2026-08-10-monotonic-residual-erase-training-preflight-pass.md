# Monotonic Residual Erase Training Preflight PASS

## Decision

`PASS`. The monotonic residual-erase iteration now has one dedicated model,
train-only supervision contract, patch-selection rule, and bounded training
schedule. The preflight checked metadata, content hashes, effective train-role
membership, CLI closure, MPS availability, class-balanced synthetic gradients,
and absent output paths. It did not decode a real image or target, materialize a
target-derived patch, generate a prediction, train, create a checkpoint, open a
quality gate, or change either current default artifact.

This PASS authorizes only materializing the exact train275 manifest, frozen
current-primary predictions, frozen current-second-stage predictions, and the
registered top-target-lighter patch index. Training remains prohibited until a
separate materialization audit proves page membership, pipeline hashes, content
hashes, patch bounds, target-lighter support, and preserve-negative coverage.

## Frozen Attempt

~~~text
model                         monotonic_residual_erase
allowed output               nonnegative luminance delta only
residual bound                0.08
steps                         80
device                        mps
batch size                    1
learning rate                 0.00002
seed                          42
luminance margin              2 gray levels
support positive weight       1.0
support preserve weight       1.0
bright magnitude weight       1.0
preserve delta weight         1.0
support class balance         separate per-sample positive/preserve means
validation during training    disabled
patch tile / overlap          256 / 96
patch positive support floor  0.001
patch selection               top 256 target-lighter support only
~~~

The separate support reductions are causal, not a tuning option. A global
pixel mean would let the much larger preserve region drown out sparse residual
positives and recreate a no-op collapse. Target-darker, identity, and submargin
target-lighter pixels supervise preserve behavior; no darken route or signed
output exists.

## Machine-Checked Evidence

~~~text
terminal = PASS
runnable = true
metadata_only = true
effective train pages = 275
train domains = 253 HW5K + 22 SCUT
MPS available = true
parameter / state tensor count = 384578 / 32
exact identity initialization = true
target-lighter support gradient = 0.5
target-lighter magnitude gradient = 0.0799999982
identity support / magnitude gradient = 0.5 / 0.0
target-darker support / magnitude gradient = 0.5 / 0.0
one-step target-lighter delta max = 0.0003297925
negative one-step delta pixels = 0
real image decode = false
target decode = false
target patch materialized = false
training started = false
checkpoint generated = false
prediction artifacts generated = false
promotion enabled = false
reserved blind = unavailable
~~~

Evidence hashes:

~~~text
docs/monotonic-residual-erase-training-plan.json
sha256 = 904f25be15bf65f6dcd847e84e71d505699c987f83f825cfc57959537b521457

scripts/train/train_monotonic_residual_erase.py
sha256 = ebad41cb0b6518e494bb0dae18b7305dff1b1420a65e630046da597417261197

scripts/analysis/build_monotonic_residual_erase_patch_index.py
sha256 = 2e8179cafb459a647aa27189d9d3cc73caf0fe3c9cf53ba8291319b7d7c925b8

scripts/analysis/validate_monotonic_residual_erase_training_preflight.py
sha256 = d859b478b0ce717aaa2161a9b4b84a2a4c2085ac3013bdf51470e0c5003458eb

tests/test_monotonic_residual_erase_training.py
sha256 = e0bc7ca8e05298117ba16fe9cf32ce09e4950bb9d514de56c96e1203ddf05a0b

tests/test_validate_monotonic_residual_erase_training_preflight.py
sha256 = 36f03a0a71bc7d8a2c606230631a1d2b8a5cccd4bd1379b6ff50a15cc45a8bde

outputs/monotonic-residual-erase-training-preflight-20260810/preflight.json
sha256 = eb34260fd05e3f56fac5a9d2ddbddb146d049692ce2b5d8f87649995a4fe12f5
~~~

## Next Boundary

Materialization may read only train-role sources and labels and may create only
the registered manifest, current-primary predictions, frozen second-stage
predictions, patch index, and patch summary. The audit must fail closed on
missing, extra, duplicate, or non-train pages; wrong prediction counts;
pipeline checkpoint or script drift; malformed or out-of-bounds patches;
missing target-lighter support; insufficient preserve-negative coverage; or
content-hash disagreement.

Training, checkpoint creation, inner-val15, development gates, SCUT115,
holdout40, visual review, reserved-blind access, threshold or parameter sweeps,
promotion, and default artifact replacement remain closed.

Stage progression note: after the materialization audit records PASS, this
preflight remains replayable and permits only the exact manifest, frozen
prediction links, patch index, and summary whose hashes are present in that
PASS record. It continues to require the training output and first quality
gate output to be absent.

Intent: Admit one class-balanced train275 materialization path without admitting training prematurely.
Constraint: Reserved blind data is unavailable, and current-primary/current-second-stage remain immutable defaults.
Rejected: Global pixel-mean support BCE | sparse target-lighter residuals would be dominated by preserve negatives and could collapse to no-op.
Rejected: Direction-balanced brighten/darken patches | the candidate has no darken output and target-darker pixels are preserve negatives.
Confidence: high
Scope-risk: moderate
Reversibility: clean
Directive: Do not invoke the trainer until the exact train275 materialization audit records PASS.
Tested: Metadata/hash preflight, synthetic class-balance and gradient contracts, MPS availability, train-role file existence, and absent outputs.
Not-tested: Real prediction materialization, target-derived patch contents, training, checkpoints, inner-val15, development gates, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-10-monotonic-residual-erase-data-role-preflight-pass.md
Related: docs/current-primary-quality-loop-ledger.json
