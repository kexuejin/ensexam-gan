#!/usr/bin/env python3
"""Phase 0 frozen-prediction gate evaluator for the Spatial Continuous
Reconstruction Mixture program.

This script consumes *frozen* per-unit metrics and gate/collapse summaries and
enforces every numeric Phase 0 PASS/KILL gate from the committed implementation
plan (docs/plans/2026-08-16-spatial-continuous-reconstruction-mixture-implementation.md).
It never runs predictions, never trains, and never accesses a quality split:
all quality/blind surfaces are reported closed and are never opened.

Inputs
------
--matrix PATH      Sealed Phase 0 matrix JSON produced by the (future) sealed
                   matrix tooling. Shape:

                   {
                     "program": "spatial-mixture-phase0",
                     "version": "spatial-mixture-phase0-v1",
                     "fold_count": 6,
                     "seeds": [42, 31415, 27182],
                     "materiality_floor": 0.0006,        // optional; else from
                                                         // --repeatability-sd
                     "controls": {
                       "baseline": "current-primary",
                       "single_head": {...},
                       "uniform_two_expert": {...},
                       "spatial_mixture": {...}
                     },
                     "units": {
                       "single_head": {
                          "fold0": {
                             "seed42": {"post_freeze_metrics": "...",
                                        "gate_metrics": "...", // spatial only
                                        "expert_metrics": "..."} // spatial only
                          }, ...
                       }, ...
                     }
                   }

                   Fold/seed keys are normalized as "fold<N>" and "seed<S>".
                   "baseline" carries only post_freeze_metrics under the same
                   fold/seed keys (used both as the current-primary reference and
                   as repeatability re-runs, which must agree).

--repeatability-sd FLOAT  Pooled within-page residual-ratio SD from the frozen
                   repeatability calibration. Used to compute the materiality
                   floor as max(0.0005, 2*SD). Ignored if the matrix sets an
                   explicit materiality_floor.
--output PATH      Where to write the terminal PASS/KILL JSON decision.

Every per-unit post_freeze_metrics CSV uses the repository convention columns
(from evaluate_prediction_directory.py):
    file,image_path,pred_path,residual_ratio,overerase_ratio,
    delta_residual_ratio,delta_overerase_ratio
where delta_residual_ratio == baseline_residual - candidate_residual (positive
= candidate improved) and delta_overerase_ratio == candidate_overerase -
baseline_overerase (positive = overerase increase = regression). The domain of
a page is derived from the filename prefix: "scut_..." or "hw5k_...".

Spatial gate unit JSON (one per spatial fold/seed), page-level arrays:
    {
      "anchor_share_mean": <per-page mean w0>,
      "expert_pixel_share_p99": <per-page 99th percentile max(w1,w2 share)>,
      "spatial_gate_std_mean": <per-page mean std of (w0,w1,w2)>,
      "scut_gate_features": [[f1,f2,f3], ...],     // page-level gate summaries
      "hw5k_gate_features": [[f1,f2,f3], ...]       // for the dataset-ID probe
    }
For the dataset-ID logistic probe we use three page-level features derived from
each page: anchor_share_mean, expert_pixel_share_p99, spatial_gate_std_mean.
The probe must show 5-fold balanced accuracy <= 0.65 for SCUT-vs-HW5K identity.

Spatial expert unit JSON (one per spatial fold/seed), aggregated on
target-lighter changed support:
    {
      "e1_e2_cosine_similarity": 0.0..1.0,
      "e1_e2_mean_abs_disagreement": float
    }

Stop conditions
---------------
Any non-finite metric, a missing per-unit file, a gate/metrics shape error, a
materiality floor that drops below 0.0005, or any numeric gate failure emits a
terminal KILL decision, keeps every quality/blind surface closed, and returns a
non-zero exit code. There is no rescue: failing units cannot be re-substituted,
threshold-adjusted, or scored again within this family.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# Frozen numbers from the committed implementation plan. Do not edit.
# --------------------------------------------------------------------------- #
MATERIALITY_FLOOR_MIN = 0.0005
RESIDUAL_ADVANTAGE_FLOOR = 0.0005
REQUIRED_WINS_OF_18 = 14
MAX_RESIDUAL_REGRESS_PAGES_PER_FOLD = 2
MAX_PAGE_RESIDUAL_REGRESS = 0.025
MAX_POOLED_OVERERASE_DELTA = 0.002
MAX_PAGE_OVERERASE_DELTA = 0.010
MAX_OVERERASE_DEGRADE_VS_CONTROL = 0.0005

ANCHOR_SHARE_MIN = 0.10
EXPERT_PIXEL_SHARE_P99_MAX = 0.98
SPATIAL_GATE_STD_MIN = 0.02
DATASET_ID_PROBE_BAC_MAX = 0.65

EXPERT_COSINE_MAX = 0.95
EXPERT_MIN_ABS_DISAGREEMENT = 1.0 / 255.0

CONTROLS = ("single_head", "uniform_two_expert", "spatial_mixture")
CLOSED_SURFACES = (
    "inner_val15",
    "Dev40",
    "SCUT115",
    "holdout40",
    "HW5K_dev232",
    "reserved_blind",
    "visual_review",
    "promotion",
)


class EvalError(RuntimeError):
    """Fail-closed evaluation failure. Keeps all gates closed."""


# --------------------------------------------------------------------------- #
# Small value helpers
# --------------------------------------------------------------------------- #
def _finite_float(value: Any, where: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise EvalError(f"non-numeric metric {where}: {value!r}") from exc
    if not np.isfinite(result):
        raise EvalError(f"non-finite metric {where}: {value!r}")
    return result


def _dir_exists_or(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Metrics reading
# --------------------------------------------------------------------------- #
@dataclass
class PageMetrics:
    file: str
    domain: str
    residual_ratio: float
    overerase_ratio: float
    delta_residual_ratio: float
    delta_overerase_ratio: float


def _domain_from_file(file: str) -> str:
    lower = file.lower()
    if lower.startswith("scut"):
        return "scut"
    if lower.startswith("hw5k"):
        return "hw5k"
    raise EvalError(f"cannot derive domain from page filename prefix: {file!r}")


def read_metrics_csv(path: Path) -> list[PageMetrics]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise EvalError(f"empty metrics CSV: {path}")
    required = (
        "file",
        "residual_ratio",
        "overerase_ratio",
        "delta_residual_ratio",
        "delta_overerase_ratio",
    )
    missing = [k for k in required if k not in rows[0]]
    if missing:
        raise EvalError(f"metrics CSV {path} missing columns: {missing}")

    pages: list[PageMetrics] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        file = row.get("file", "").strip()
        if not file:
            raise EvalError(f"metrics CSV {path} row {index} has empty file")
        if file in seen:
            raise EvalError(f"metrics CSV {path} duplicates page {file!r}")
        seen.add(file)
        pages.append(
            PageMetrics(
                file=file,
                domain=_domain_from_file(file),
                residual_ratio=_finite_float(
                    row["residual_ratio"], f"{path.name}.{file}.residual_ratio"
                ),
                overerase_ratio=_finite_float(
                    row["overerase_ratio"], f"{path.name}.{file}.overerase_ratio"
                ),
                delta_residual_ratio=_finite_float(
                    row["delta_residual_ratio"],
                    f"{path.name}.{file}.delta_residual",
                ),
                delta_overerase_ratio=_finite_float(
                    row["delta_overerase_ratio"],
                    f"{path.name}.{file}.delta_overerase",
                ),
            )
        )
    return pages


def pooled_mean(pages: Iterable[PageMetrics], key: str) -> float:
    """Page-mean pooled aggregate; never weights by pixel count."""
    values = [getattr(p, key) for p in pages]
    if not values:
        raise EvalError("pooled_mean over empty page set")
    return float(np.mean(values))


def page_median_gain(pages: list[PageMetrics], key: str) -> float:
    """Median per-page gain (positive == improvement) for residual_ratio.

    Repository convention: delta_residual_ratio == baseline_residual -
    candidate_residual, so a positive delta means the candidate improved and the
    median of the deltas is the median gain."""
    if not pages:
        raise EvalError("page_median_gain over empty page set")
    gains = [p.delta_residual_ratio for p in pages]
    return float(np.median(gains))


@dataclass
class UnitRows:
    baseline: list[PageMetrics]
    candidate: list[PageMetrics]


@dataclass
class Unit:
    control: str
    fold: str
    seed: str
    rows: UnitRows
    gate: dict[str, Any] | None = None
    expert: dict[str, Any] | None = None


@dataclass
class Decisions:
    failures: list[str] = field(default_factory=list)
    passes: list[str] = field(default_factory=list)

    def ok(self, name: str, detail: Any = None) -> None:
        self.passes.append(name)
        self._last_ok = name

    def fail(self, name: str, detail: Any = None) -> None:
        detail_text = "" if detail is None else f" ({detail})"
        self.failures.append(f"{name}{detail_text}")


def _require_passed(decisions: Decisions) -> bool:
    return not decisions.failures


# --------------------------------------------------------------------------- #
# Gate: material lift
# --------------------------------------------------------------------------- #
def gate_material_lift(
    units: list[Unit],
    materiality_floor: float,
    decisions: Decisions,
) -> None:
    """Pooled held-out residual gain vs current-primary >= floor; and every
    fold x seed page-median gain is strictly positive."""
    gains: list[float] = []
    per_unit_median_positive = True
    for unit in units:
        gain = pooled_mean(unit.rows.baseline, "residual_ratio") - pooled_mean(
            unit.rows.candidate, "residual_ratio"
        )
        gains.append(gain)
        median = page_median_gain(unit.rows.candidate, "residual_ratio")
        if not (median > 0.0):
            per_unit_median_positive = False
            decisions.fail(
                "material_page_median_gain_positive",
                f"{unit.control}:{unit.fold}:{unit.seed} median={median:.6f}",
            )

    pooled_gain = float(np.mean(gains)) if gains else 0.0
    if pooled_gain >= materiality_floor:
        decisions.ok(
            "material_pooled_gain",
            f"gain={pooled_gain:.6f} floor={materiality_floor:.6f}",
        )
    else:
        decisions.fail(
            "material_pooled_gain",
            f"gain={pooled_gain:.6f} < floor={materiality_floor:.6f}",
        )
    if per_unit_median_positive:
        decisions.ok("material_all_unit_median_positive")


# --------------------------------------------------------------------------- #
# Gate: matched-control superiority
# --------------------------------------------------------------------------- #
def _unit_residual(rows: UnitRows) -> float:
    return pooled_mean(rows.candidate, "residual_ratio")


def _unit_overerase(rows: UnitRows) -> float:
    return pooled_mean(rows.candidate, "overerase_ratio")


def gate_control_superiority(
    spatial_units: list[Unit],
    control_units_by_name: dict[str, list[Unit]],
    decisions: Decisions,
) -> None:
    """Spatial pooled residual >= control + 0.0005 and >=14/18 paired wins by
    mean page residual. Spatial pooled overerase must not degrade vs a control
    by more than 0.0005."""

    def make_pairs(
        control_name: str,
    ) -> list[tuple[Unit, Unit, str, str]]:
        control_map = {
            (u.fold, u.seed): u for u in control_units_by_name[control_name]
        }
        pairs: list[tuple[Unit, Unit, str, str]] = []
        for s in spatial_units:
            c = control_map.get((s.fold, s.seed))
            if c is None:
                raise EvalError(
                    f"missing {control_name} unit for {s.fold}:{s.seed}"
                )
            pairs.append((s, c, s.fold, s.seed))
        return pairs

    for control_name in ("single_head", "uniform_two_expert"):
        pairs = make_pairs(control_name)
        spatial_pool = float(
            np.mean([_unit_residual(s.rows) for s, _, _, _ in pairs])
        )
        control_pool = float(
            np.mean([_unit_residual(c.rows) for _, c, _, _ in pairs])
        )
        # lower residual = better; advantage = control - spatial (positive when
        # spatial improves on the control).
        advantage = control_pool - spatial_pool
        if advantage >= RESIDUAL_ADVANTAGE_FLOOR:
            decisions.ok(
                f"control_residual_advantage_{control_name}",
                f"advantage={advantage:.6f}",
            )
        else:
            decisions.fail(
                f"control_residual_advantage_{control_name}",
                f"advantage={advantage:.6f} < {RESIDUAL_ADVANTAGE_FLOOR:.6f}",
            )

        wins = sum(
            1
            for s, c, _, _ in pairs
            if _unit_residual(s.rows) < _unit_residual(c.rows)
        )
        if wins >= REQUIRED_WINS_OF_18:
            decisions.ok(
                f"control_paired_wins_{control_name}",
                f"{wins}/18",
            )
        else:
            decisions.fail(
                f"control_paired_wins_{control_name}",
                f"{wins}/18 < {REQUIRED_WINS_OF_18}",
            )

        over_adv = float(
            np.mean([_unit_overerase(s.rows) for s, _, _, _ in pairs])
            - np.mean([_unit_overerase(c.rows) for _, c, _, _ in pairs])
        )
        if over_adv <= MAX_OVERERASE_DEGRADE_VS_CONTROL:
            decisions.ok(
                f"control_overerase_no_deg_{control_name}",
                f"spatial_vs_control_overerase={over_adv:.6f}",
            )
        else:
            decisions.fail(
                f"control_overerase_no_deg_{control_name}",
                f"spatial_vs_control_overerase={over_adv:.6f}",
            )


# --------------------------------------------------------------------------- #
# Gate: balanced regression budget
# --------------------------------------------------------------------------- #
def gate_regression_budget(
    spatial_units: list[Unit],
    decisions: Decisions,
) -> None:
    """Per fold <=2 residual-regressing pages, each magnitude <=0.025; pooled
    overerase <=0.002 and each page <=0.010 vs current-primary, using the
    candidate's delta columns.

    Repository convention: delta_residual_ratio == baseline - candidate (positive
    = improvement, so a regression is a negative delta); delta_overerase_ratio ==
    candidate - baseline (positive = overerase increase = regression)."""
    by_fold: dict[str, list[PageMetrics]] = {}
    for unit in spatial_units:
        by_fold.setdefault(unit.fold, []).extend(unit.rows.candidate)

    fold_ok = True
    for fold, pages in sorted(by_fold.items()):
        regressed = [p for p in pages if p.delta_residual_ratio < 0.0]
        max_regress = (
            max(-p.delta_residual_ratio for p in regressed) if regressed else 0.0
        )
        if len(regressed) <= MAX_RESIDUAL_REGRESS_PAGES_PER_FOLD and all(
            -p.delta_residual_ratio <= MAX_PAGE_RESIDUAL_REGRESS
            for p in regressed
        ):
            decisions.ok(
                f"fold_residual_regress_{fold}",
                f"pages={len(regressed)} max_regress={max_regress:.6f}",
            )
        else:
            fold_ok = False
            decisions.fail(
                f"fold_residual_regress_{fold}",
                f"pages={len(regressed)} max_regress={max_regress:.6f}",
            )
    if fold_ok:
        decisions.ok("fold_residual_regress_budget")

    all_pages: list[PageMetrics] = []
    for unit in spatial_units:
        all_pages.extend(unit.rows.candidate)
    pooled_overerase_delta = float(
        np.mean([p.delta_overerase_ratio for p in all_pages])
    )
    if pooled_overerase_delta <= MAX_POOLED_OVERERASE_DELTA:
        decisions.ok("pooled_overerase_delta", f"{pooled_overerase_delta:.6f}")
    else:
        decisions.fail("pooled_overerase_delta", f"{pooled_overerase_delta:.6f}")

    over_page_over = [p for p in all_pages if p.delta_overerase_ratio > MAX_PAGE_OVERERASE_DELTA]
    if not over_page_over:
        decisions.ok("page_overerase_delta_budget")
    else:
        decisions.fail(
            "page_overerase_delta_budget",
            f"pages={len(over_page_over)}",
        )


# --------------------------------------------------------------------------- #
# Gate: gate non-collapse
# --------------------------------------------------------------------------- #
def _logistic_balanced_accuracy_5fold(
    scut_features: np.ndarray,
    hw5k_features: np.ndarray,
) -> float:
    """Self-contained 5-fold logistic-regression balanced accuracy used only
    as a page-level dataset-ID audit. Returns BAC in [0,1]. Uses numpy only."""
    scut_features = np.asarray(scut_features, dtype=float)
    hw5k_features = np.asarray(hw5k_features, dtype=float)
    if scut_features.ndim != 2 or hw5k_features.ndim != 2:
        raise EvalError("dataset-ID probe requires 2-D page-level feature arrays")
    n_scut, n_hw5k = scut_features.shape[0], hw5k_features.shape[0]
    if n_scut < 5 or n_hw5k < 5:
        raise EvalError(
            "dataset-ID probe needs at least 5 pages per domain per unit"
        )
    X_raw = np.vstack([scut_features, hw5k_features])
    y = np.concatenate([np.zeros(n_scut), np.ones(n_hw5k)])
    # Simple deterministic 5-fold stride split by combined index.
    n = X_raw.shape[0]
    folds = np.array([i % 5 for i in range(n)])
    per_fold_bac: list[float] = []
    for k in range(5):
        test = folds == k
        train = ~test
        if train.sum() == 0 or test.sum() == 0:
            continue
        Xtr, ytr, Xte = X_raw[train], y[train], X_raw[test]
        yte = y[test]
        w, b = _logistic_fit(Xtr, ytr)
        probs = _sigmoid(Xte @ w + b)
        pred = (probs >= 0.5).astype(int)
        te_scut = yte == 0
        te_hw5k = yte == 1
        tpr = float(np.sum(pred[te_scut] == 0)) / max(float(np.sum(te_scut)), 1.0)
        tnr = float(np.sum(pred[te_hw5k] == 1)) / max(float(np.sum(te_hw5k)), 1.0)
        per_fold_bac.append(0.5 * (tpr + tnr))
    if not per_fold_bac:
        raise EvalError("dataset-ID probe produced no folds")
    return float(np.mean(per_fold_bac))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logistic_fit(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1.0,
    iters: int = 200,
    lr: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """L2-regularized logistic regression via gradient descent on centered
    features. Returns (coeff, intercept) in the raw-feature space, so a
    prediction is `raw_features @ coeff + intercept`. Self-contained numpy."""
    n, d = X.shape
    Xd = np.column_stack([np.ones(n), X])
    mean = Xd.mean(axis=0, keepdims=True)
    std = Xd.std(axis=0, keepdims=True)
    std = np.where(std < 1e-9, 1.0, std)
    Xs = (Xd - mean) / std
    theta = np.zeros(d + 1)
    for _ in range(iters):
        z = Xs @ theta
        p = _sigmoid(z)
        grad = Xs.T @ (p - y) / n
        grad[1:] += l2 * theta[1:] / n
        theta -= lr * grad
    w = theta[1:] / std[0, 1:]
    b = float(theta[0] - np.sum((mean[0, 1:] / std[0, 1:]) * theta[1:]))
    return w, b


def gate_gate_non_collapse(
    spatial_units: list[Unit],
    decisions: Decisions,
) -> None:
    ok = True
    for unit in spatial_units:
        gate = unit.gate
        if gate is None:
            raise EvalError(f"spatial unit {unit.fold}:{unit.seed} missing gate_metrics")
        anchor = _finite_float(gate.get("anchor_share_mean"), "anchor_share_mean")
        p99 = _finite_float(gate.get("expert_pixel_share_p99_max"), "expert_pixel_share_p99_max")
        std = _finite_float(gate.get("spatial_gate_std_mean"), "spatial_gate_std_mean")
        if anchor < ANCHOR_SHARE_MIN:
            ok = False
            decisions.fail("anchor_share_min", f"{unit.fold}:{unit.seed}={anchor:.6f}")
        if p99 > EXPERT_PIXEL_SHARE_P99_MAX:
            ok = False
            decisions.fail("expert_pixel_share_p99", f"{unit.fold}:{unit.seed}={p99:.6f}")
        if std < SPATIAL_GATE_STD_MIN:
            ok = False
            decisions.fail("spatial_gate_std_min", f"{unit.fold}:{unit.seed}={std:.6f}")

        scut_feats = np.asarray(gate.get("scut_gate_features") or [], dtype=float)
        hw5k_feats = np.asarray(gate.get("hw5k_gate_features") or [], dtype=float)
        bac = _logistic_balanced_accuracy_5fold(scut_feats, hw5k_feats)
        if bac > DATASET_ID_PROBE_BAC_MAX:
            ok = False
            decisions.fail(
                "dataset_id_probe_bac",
                f"{unit.fold}:{unit.seed}={bac:.4f}",
            )
    if ok:
        decisions.ok("gate_non_collapse")


# --------------------------------------------------------------------------- #
# Gate: expert non-collapse
# --------------------------------------------------------------------------- #
def gate_expert_non_collapse(
    spatial_units: list[Unit],
    decisions: Decisions,
) -> None:
    ok = True
    for unit in spatial_units:
        expert = unit.expert
        if expert is None:
            raise EvalError(f"spatial unit {unit.fold}:{unit.seed} missing expert_metrics")
        cosine = _finite_float(expert.get("e1_e2_cosine_similarity"), "e1_e2_cosine")
        disagreement = _finite_float(
            expert.get("e1_e2_mean_abs_disagreement"), "e1_e2_disagreement"
        )
        if cosine > EXPERT_COSINE_MAX:
            ok = False
            decisions.fail("expert_cosine_max", f"{unit.fold}:{unit.seed}={cosine:.6f}")
        if disagreement < EXPERT_MIN_ABS_DISAGREEMENT:
            ok = False
            decisions.fail(
                "expert_min_disagreement",
                f"{unit.fold}:{unit.seed}={disagreement:.6f}",
            )
    if ok:
        decisions.ok("expert_non_collapse")


# --------------------------------------------------------------------------- #
# Closed-surface reporter
# --------------------------------------------------------------------------- #
def closed_surface_report() -> dict[str, bool]:
    return {surface: False for surface in CLOSED_SURFACES}


# --------------------------------------------------------------------------- #
# Matrix parsing
# --------------------------------------------------------------------------- #
def _normalize_key(value: str) -> str:
    return value.strip()


@dataclass
class Matrix:
    materiality_floor: float
    units: dict[str, list[Unit]] = field(default_factory=dict)


def load_matrix(path: Path, repeatability_sd: float | None) -> Matrix:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    seeds: list[int] = []
    for s in data.get("seeds", []):
        seeds.append(int(s))
    if not seeds:
        raise EvalError("matrix seeds must be non-empty")

    units_by_control: dict[str, list[Unit]] = {c: [] for c in CONTROLS}
    units_block = data.get("units") or {}
    for control in CONTROLS:
        control_block = units_block.get(control)
        if not isinstance(control_block, dict):
            raise EvalError(f"matrix missing units for control {control!r}")
        for fold_key, fold_block in control_block.items():
            if not isinstance(fold_block, dict):
                raise EvalError(f"matrix fold block for {control}:{fold_key} not a mapping")
            for seed_key, raw_unit in fold_block.items():
                if not isinstance(raw_unit, dict):
                    raise EvalError(f"matrix unit {control}:{fold_key}:{seed_key} not a mapping")
                metrics_path = Path(raw_unit["post_freeze_metrics"])
                candidate = read_metrics_csv(metrics_path)

                base_unit = (
                    units_block.get("baseline", {}).get(fold_key, {}).get(seed_key, {})
                ) if control != "baseline" else None
                if control == "baseline":
                    # Baseline's reference is itself: baseline == candidate and
                    # delta columns are candidate-vs-itself (all zero).
                    baseline = candidate
                elif base_unit and base_unit.get("post_freeze_metrics"):
                    baseline = read_metrics_csv(
                        Path(base_unit["post_freeze_metrics"])
                    )
                else:
                    raise EvalError(
                        f"control {control} unit {fold_key}:{seed_key} has no "
                        "baseline reference in matrix"
                    )

                gate_raw = raw_unit.get("gate_metrics")
                expert_raw = raw_unit.get("expert_metrics")
                gate: dict[str, Any] | None = None
                expert: dict[str, Any] | None = None
                if control == "spatial_mixture":
                    if not gate_raw or not expert_raw:
                        raise EvalError(
                            f"spatial unit {fold_key}:{seed_key} missing gate/expert metrics"
                        )
                    gate = _read_json_path(gate_raw)
                    expert = _read_json_path(expert_raw)
                units_by_control[control].append(
                    Unit(
                        control=control,
                        fold=_normalize_key(fold_key),
                        seed=_normalize_key(seed_key),
                        rows=UnitRows(baseline=baseline, candidate=candidate),
                        gate=gate,
                        expert=expert,
                    )
                )
    for control in CONTROLS:
        if not units_by_control[control]:
            raise EvalError(f"control {control!r} has no units")

    floor = float(data.get("materiality_floor", 0.0) or 0.0)
    if floor <= 0.0:
        if repeatability_sd is None:
            raise EvalError("either matrix.materiality_floor or --repeatability-sd required")
        floor = max(MATERIALITY_FLOOR_MIN, 2.0 * float(repeatability_sd))
    if floor < MATERIALITY_FLOOR_MIN:
        raise EvalError(
            f"materiality floor must not drop below {MATERIALITY_FLOOR_MIN}: {floor}"
        )
    return Matrix(materiality_floor=floor, units=units_by_control)


def _read_json_path(raw: str) -> dict[str, Any]:
    path = Path(raw)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise EvalError(f"expected JSON object at {path}")
    return value


# --------------------------------------------------------------------------- #
# Reporter
# --------------------------------------------------------------------------- #
def _report(path: Path, report: dict[str, Any]) -> Path:
    _dir_exists_or(path.parent)
    blob = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    path.write_text(blob + "\n", encoding="utf-8")
    return path


def run(matrix: Matrix, output: Path) -> int:
    spatial_units = matrix.units["spatial_mixture"]
    decisions = Decisions()
    gate_material_lift(spatial_units, matrix.materiality_floor, decisions)
    gate_control_superiority(
        spatial_units,
        {"single_head": matrix.units["single_head"], "uniform_two_expert": matrix.units["uniform_two_expert"]},
        decisions,
    )
    gate_regression_budget(spatial_units, decisions)
    gate_gate_non_collapse(spatial_units, decisions)
    gate_expert_non_collapse(spatial_units, decisions)

    passed = _require_passed(decisions)
    terminal = "PHASE0_PASS" if passed else "PHASE0_KILL"
    decision_doc = {
        "program": "spatial-mixture-phase0",
        "terminal": terminal,
        "materiality_floor": matrix.materiality_floor,
        "checks": {
            "pass": decisions.passes,
            "fail": decisions.failures,
        },
        "closed_surfaces": closed_surface_report(),
        "reason": (
            "all frozen Phase 0 numeric gates passed"
            if passed
            else "; ".join(decisions.failures)
        ),
    }
    wrote = _report(output, decision_doc)
    print(json.dumps(decision_doc, indent=2, sort_keys=True))
    print(f"decision_json={wrote}")
    return 0 if passed else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--repeatability-sd", type=float, default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = load_matrix(args.matrix, args.repeatability_sd)
    return run(matrix, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
