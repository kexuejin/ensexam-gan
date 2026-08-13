# External Text Layout Support Preregistration

## Decision

`PREREQUISITE_NEEDED`. The next bounded uncertainty is whether a frozen external
text detector contributes target-aligned support beyond final second-stage RGB.
The source is the locally cached official `PP-OCRv6_medium_det` safetensors
release, not another EnsExam-GAN checkpoint or another encoding of a closed
pipeline surface.

This source is materially different because it was trained for text-region
localization across printed and handwritten document text. Its fixed polygon
occupancy and confidence can expose layout evidence that cleaned RGB does not
explicitly identify. Only target-free train275 detector materialization and one
fixed train-label separability diagnostic are authorized. No recognition,
training, checkpoint, candidate inference, quality split, visual review,
reserved-blind access, promotion, or current-primary replacement is authorized.

## Frozen Producer

Use the official Apache-2.0 PaddleOCR release already present at:

```text
/Users/kexuejin/.paddlex/official_models/PP-OCRv6_medium_det_safetensors
```

The frozen model hashes are:

```text
model.safetensors       bd393266c02e1a680b1b34c301d5d0d81e6290440b7f8ab0f5d5032276b17eb1
config.json             3ea7e760f64255c152ba139e2f5f798303f7189b87c923cf498e2b2e830cadd3
inference.yml           7298d5ead546584af2504d03355f881ac7a7bc0eb1e282d3e159277c1d0af871
preprocessor_config.json 74421569a28a78d417db320f9bc039ce0997a1defe2b35f259dfc74299c9f1ed
README.md               cdba9f31531fe1a4ff1323a21d7cf671a7a042b23a788ffeb081d1845acccdb6
```

Freeze Python `3.13.1`, PaddleOCR `3.7.0`, PaddleX `3.7.2`, Paddle `3.0.0`,
Torch `2.12.0`, Transformers `5.12.1`, NumPy `2.3.5`, and OpenCV `4.13.0`.
Use the Transformers engine on CPU with batch size `1`, `limit_side_len=736`,
`limit_type=min`, `max_side_limit=4000`, `thresh=0.2`, `box_thresh=0.45`, and
`unclip_ratio=1.4`. These values come from the frozen release files and are not
a search space.

## Frozen Materialization

Process the exact 275 raw input pages in
`hardcase_lists/monotonic-residual-erase-train275-v1.txt` order. The
materializer must validate the manifest, frozen roles, source content identity,
external model files, runtime versions, and product authority before decoding
the first page. It must never resolve or open a label, target, quality split,
route, domain field, or caller metadata.

Persist only detection quadrilaterals, detection scores, and two aligned grids:

1. `text_occupancy`: `uint8` zero or one, rasterized with
   `cv2.fillPoly`, `LINE_8`, and shift zero;
2. `text_confidence`: `float32` pixelwise maximum detection score.

Clip integer points to page bounds. Sort detections deterministically by
minimum y, minimum x, maximum y, maximum x, flattened coordinates, then score
before rasterization and persistence. OCR recognition and recognized text are
prohibited. Polygon coordinates are persisted for reproducibility but are not
diagnostic features.

## Frozen Diagnostic

After target-free materialization passes, train-role targets may define positive
pixels exactly as `target_luma - second_stage_luma > 2` gray; all other pixels
are preserve. The full representation is exactly second-stage RGB divided by
255, occupancy in `{0,1}`, and confidence in `[0,1]`. The ablation uses the
same pixels, folds, normalization, ridge implementation, and labels with only
the two external-layout channels removed.

Reuse the established five basename-hash page folds and deterministic
SplitMix64 sampling with at most 1024 pixels per class per page. Fit float64
closed-form ridge with `lambda=1.0`, fitting-fold-only standardization, and an
unpenalized intercept. No detector, feature, fold, sample, threshold, or probe
parameter may be learned from the result.

`PASS` requires all conditions:

- exactly 275 pages and five nonempty held-out folds;
- mean held-out fold AUC at least `0.65`;
- every held-out fold AUC at least `0.55`;
- macro median per-page AUC at least `0.60`;
- mean AUC at least `0.03` above second-stage RGB;
- positive mean score above preserve in at least four of five folds.

Any metric failure is `KILL`. Provenance or exact implementation drift is
`PREREQUISITE_NEEDED`. `PASS` authorizes only a separately preregistered,
leakage-aware text-layout-conditioned data/training/application preflight.

## Leakage Boundary

The external detector was not trained or tuned in this repository and receives
no train275 targets. Its published training-corpus overlap with SCUT or HW5K is
not established, however. Therefore this diagnostic can screen incremental
train-role support only. Even a `PASS` cannot establish product generalization,
SCUT/HW5K safety, or promotion eligibility.

## Terminal Successors

- `PASS`: freeze a separate leakage-aware data/training/application preflight.
- `KILL`: close external text layout without threshold, geometry, transform,
  detector, runtime, probe, or training rescue.
- `PREREQUISITE_NEEDED`: repair registered provenance or implementation only.

## Registered Surface

```text
plan:
  docs/external-text-layout-support-prerequisite-v1.json
future materializer:
  scripts/analysis/materialize_external_text_layout_support_train_only.py
future audit:
  scripts/analysis/audit_external_text_layout_support.py
future test:
  tests/test_external_text_layout_support_prerequisite.py
future materialization:
  outputs/external-text-layout-support-materialization-20260813/
future audit:
  outputs/external-text-layout-support-prerequisite-20260813/audit.json
```

Intent: Test an external text-layout producer after every registered pipeline-derived support source failed its incremental-margin gate.
Constraint: The upstream training corpus cannot be proven disjoint from SCUT or HW5K, so this is train-role support evidence only and cannot establish generalization.
Rejected: OCR recognition or recognized text | introduces content features and a larger uncontrolled producer surface.
Rejected: Tune detector thresholds, geometry features, transforms, runtime, folds, sampling, lambda, or probe class | post-registration feature rescue.
Rejected: Treat detector polygons as target masks | the detector localizes text generally and is not residual-handwriting supervision.
Confidence: medium
Scope-risk: moderate
Reversibility: clean
Directive: Keep detector inference and the five-channel audit exact; a failed margin closes this family, while a pass opens only a separate leakage-aware preflight.
Tested: Official model provenance, license, file hashes, installed runtime versions, and stable detector result fields `dt_polys` and `dt_scores` were verified before registration.
Not-tested: Train275 materialization, support separability, training, candidate inference, inner-val15, development, SCUT115, holdout40, visual review, reserved blind, or promotion.
Related: docs/decisions/2026-08-13-independent-hw5k-expert-disagreement-support-diagnostic-kill.md
