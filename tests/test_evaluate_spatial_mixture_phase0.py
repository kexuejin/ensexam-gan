"""Phase 0 frozen-prediction gate evaluator tests.

Exercises the Phase 0 PASS/KILL gate logic in
scripts/eval/evaluate_spatial_mixture_phase0.py using synthetic CSV/JSON
fixtures. This test never runs real predictions, never trains, and never
touches a quality split.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/eval/evaluate_spatial_mixture_phase0.py"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_spatial_mixture_phase0", SCRIPT
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
# The script defines `@dataclass` classes; dataclass processing requires the
# module to be registered in sys.modules during class body execution.
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

SECONDS = ("seed42", "seed31415", "seed27182")


def _page_name(control: str, fold: int, seed: int, idx: int, domain: str) -> str:
    # deterministic, domain-prefixed filenames
    return f"{domain}_{control}_{fold}_{seed}_{idx}.jpg"


def _write_metrics(
    root: Path,
    control: str,
    fold: int,
    seed: int,
    n_scut: int,
    n_hw5k: int,
    residual: float,
    overerase: float,
    delta_residual: float,
    delta_overerase: float,
) -> Path:
    path = root / "metrics" / control / f"fold{fold}" / f"seed{seed}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "residual_ratio",
        "overerase_ratio",
        "delta_residual_ratio",
        "delta_overerase_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        idx = 0
        for _ in range(n_scut):
            writer.writerow(
                {
                    "file": _page_name(control, fold, seed, idx, "scut"),
                    "residual_ratio": f"{residual:.8f}",
                    "overerase_ratio": f"{overerase:.8f}",
                    "delta_residual_ratio": f"{delta_residual:.8f}",
                    "delta_overerase_ratio": f"{delta_overerase:.8f}",
                }
            )
            idx += 1
        for _ in range(n_hw5k):
            writer.writerow(
                {
                    "file": _page_name(control, fold, seed, idx, "hw5k"),
                    "residual_ratio": f"{residual:.8f}",
                    "overerase_ratio": f"{overerase:.8f}",
                    "delta_residual_ratio": f"{delta_residual:.8f}",
                    "delta_overerase_ratio": f"{delta_overerase:.8f}",
                }
            )
            idx += 1
    return path


def _write_gate(control: str, path: Path, anchor: float, p99: float, std: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # page-level gate features (one row per page); make scut/hw5k identical so
    # the dataset-ID probe has no signal (BAC ~0.5) unless a test overrides
    scut_rows = [[anchor, p99, std] for _ in range(10)]
    hw5k_rows = [[anchor, p99, std] for _ in range(10)]
    payload = {
        "anchor_share_mean": anchor,
        "expert_pixel_share_p99_max": p99,
        "spatial_gate_std_mean": std,
        "scut_gate_features": scut_rows,
        "hw5k_gate_features": hw5k_rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_expert(
    path: Path, cosine: float, disagreement: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "e1_e2_cosine_similarity": cosine,
        "e1_e2_mean_abs_disagreement": disagreement,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_fixture(
    root: Path,
    *,
    spatial_residual: float = 0.1400,
    spatial_overerase: float = 0.0025,
    spatial_delta_resid: float = 0.0100,
    spatial_delta_over: float = 0.0,
    control_residual: float = 0.1440,
    single_delta_resid: float = 0.0060,
    uniform_delta_resid: float = 0.0055,
    anchor: float = 0.20,
    p99: float = 0.60,
    std: float = 0.05,
    cosine: float = 0.80,
    disagreement: float = 0.010,
    baseline_residual: float = 0.1500,
    n_scut: int = 10,
    n_hw5k: int = 10,
) -> dict:
    """Build a full 6-fold x 3-seed matrix fixture. Returns a dict with the
    matrix JSON structure and paths. Learned-control residual defaults make the
    spatial candidate beat both controls on every paired unit."""
    matrix_dir = root / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    controls = {
        "single_head": single_delta_resid,
        "uniform_two_expert": uniform_delta_resid,
        "spatial_mixture": spatial_delta_resid,
    }
    resid_over = {
        "single_head": control_residual,
        "uniform_two_expert": control_residual,
        "spatial_mixture": spatial_residual,
    }
    over_map = {
        "single_head": spatial_overerase,
        "uniform_two_expert": spatial_overerase,
        "spatial_mixture": spatial_overerase,
    }

    units: dict = {
        "baseline": {},
        "single_head": {},
        "uniform_two_expert": {},
        "spatial_mixture": {},
    }
    for fold in range(6):
        for seed_key in SECONDS:
            base_path = _write_metrics(
                matrix_dir,
                "baseline",
                fold,
                seed_key,
                n_scut,
                n_hw5k,
                baseline_residual,
                0.0025,
                0.0,
                0.0,
            )
            units["baseline"].setdefault(f"fold{fold}", {})[seed_key] = {
                "post_freeze_metrics": str(base_path)
            }
            for control in ("single_head", "uniform_two_expert", "spatial_mixture"):
                candidate_path = _write_metrics(
                    matrix_dir,
                    control,
                    fold,
                    seed_key,
                    n_scut,
                    n_hw5k,
                    resid_over[control],
                    over_map[control],
                    controls[control],
                    spatial_delta_over,
                )
                entry: dict = {"post_freeze_metrics": str(candidate_path)}
                if control == "spatial_mixture":
                    gate_path = matrix_dir / "gate" / f"{control}" / f"fold{fold}" / f"{seed_key}.json"
                    expert_path = matrix_dir / "expert" / f"{control}" / f"fold{fold}" / f"{seed_key}.json"
                    _write_gate(control, gate_path, anchor, p99, std)
                    _write_expert(expert_path, cosine, disagreement)
                    entry["gate_metrics"] = str(gate_path)
                    entry["expert_metrics"] = str(expert_path)
                units[control].setdefault(f"fold{fold}", {})[seed_key] = entry

    matrix = {
        "program": "spatial-mixture-phase0",
        "version": "spatial-mixture-phase0-v1",
        "fold_count": 6,
        "seeds": [42, 31415, 27182],
        "materiality_floor": 0.0006,
        "controls": {
            "baseline": "current-primary",
            "single_head": "equal-reconstruction",
            "uniform_two_expert": "uniform-mixture",
            "spatial_mixture": "spatial-mixture",
        },
        "units": units,
    }
    matrix_path = matrix_dir / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "matrix": matrix_path,
        "output": root / "out",
        "matrix_dir": matrix_dir,
        "matrix_payload": matrix,
    }


class SpatialMixturePhase0EvaluatorTest(unittest.TestCase):
    def _evaluate(self, fixture: dict, **kwargs) -> int:
        return MODULE.main(
            [
                "--matrix",
                str(fixture["matrix"]),
                "--output",
                str(fixture["output"]),
            ]
        )

    def _read_decision(self, fixture: dict) -> dict:
        with fixture["output"].open(encoding="utf-8") as handle:
            return json.load(handle)

    def _fail_checks(self, fixture: dict) -> list:
        return self._read_decision(fixture)["checks"]["fail"]

    def test_clean_pass_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = build_fixture(Path(tmp))
            code = self._evaluate(fixture)
            self.assertEqual(code, 0)
            decision = self._read_decision(fixture)
            self.assertEqual(decision["terminal"], "PHASE0_PASS")
            self.assertEqual(decision["checks"]["fail"], [])
            # every quality/blind surface stays closed
            for key, val in decision["closed_surfaces"].items():
                self.assertFalse(val, f"{key} unexpectedly open")

    def test_paired_advantage_failure_kills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # spatial residual == control residual -> no pooled advantage
            fixture = build_fixture(
                Path(tmp), spatial_residual=0.1440, control_residual=0.1440
            )
            code = self._evaluate(fixture)
            self.assertEqual(code, 1)
            self.assertTrue(
                any(
                    "control_residual_advantage_single_head" in f
                    for f in self._fail_checks(fixture)
                )
            )

    def test_paired_wins_failure_kills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # spatial residual higher than control on every page -> 0/18 wins
            fixture = build_fixture(
                Path(tmp), spatial_residual=0.1500, control_residual=0.1440
            )
            code = self._evaluate(fixture)
            self.assertEqual(code, 1)
            self.assertTrue(
                any("control_paired_wins_" in f for f in self._fail_checks(fixture))
            )

    def test_regression_budget_failure_kills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Force 3 residual-regressing pages across all fixtures by using
            # negative delta_residual on every page (candidate worse on residual)
            # while still pooling to a gain via the residual columns is not
            # possible here; instead bake regression directly: set the spatial
            # per-page delta small but rely on a fold that has >2 regressors.
            # To keep it deterministic, craft spatial residual worse than baseline
            # so deltas are negative everywhere -> every page regresses.
            fixture = build_fixture(
                Path(tmp),
                spatial_residual=0.1550,
                baseline_residual=0.1500,
                spatial_delta_resid=-0.0050,
            )
            code = self._evaluate(fixture)
            self.assertEqual(code, 1)
            self.assertTrue(
                any(
                    "fold_residual_regress" in f
                    for f in self._fail_checks(fixture)
                )
            )

    def test_overerase_page_budget_failure_kills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = build_fixture(
                Path(tmp), spatial_delta_over=0.020
            )
            code = self._evaluate(fixture)
            self.assertEqual(code, 1)
            self.assertTrue(
                any(
                    "page_overerase_delta_budget" in f
                    for f in self._fail_checks(fixture)
                )
            )

    def test_gate_collapse_failure_kills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = build_fixture(Path(tmp), anchor=0.05)  # below 0.10 floor
            code = self._evaluate(fixture)
            self.assertEqual(code, 1)
            self.assertTrue(
                any("anchor_share_min" in f for f in self._fail_checks(fixture))
            )

    def test_expert_collapse_failure_kills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = build_fixture(Path(tmp), cosine=0.99)  # above 0.95
            code = self._evaluate(fixture)
            self.assertEqual(code, 1)
            self.assertTrue(
                any("expert_cosine_max" in f for f in self._fail_checks(fixture))
            )

    def test_misaligned_min_disagreement_failure_kills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = build_fixture(Path(tmp), disagreement=0.0001)  # below 1/255
            code = self._evaluate(fixture)
            self.assertEqual(code, 1)
            self.assertTrue(
                any(
                    "expert_min_disagreement" in f
                    for f in self._fail_checks(fixture)
                )
            )

    def test_material_floor_failure_kills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # pooled gain below floor
            fixture = build_fixture(
                Path(tmp),
                spatial_residual=0.1495,  # gain 0.0005 < ... keep below floor
            )
            code = self._evaluate(fixture)
            self.assertEqual(code, 1)
            self.assertTrue(
                any("material_pooled_gain" in f for f in self._fail_checks(fixture))
            )

    def test_kill_still_reports_closed_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = build_fixture(Path(tmp), anchor=0.05)  # guaranteed KILL
            self._evaluate(fixture)
            decision = self._read_decision(fixture)
            self.assertEqual(decision["terminal"], "PHASE0_KILL")
            for key, val in decision["closed_surfaces"].items():
                self.assertFalse(val, f"{key} unexpectedly open on KILL")

    def test_non_finite_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = build_fixture(Path(tmp))
            # introduce a NaN residual in the spatial candidate CSV for one unit
            spatial_csv = next(
                p for p in (Path(tmp) / "matrix" / "metrics" / "spatial_mixture").glob("**/*.csv")
            )
            with spatial_csv.open() as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            rows[0]["residual_ratio"] = "nan"
            with spatial_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(MODULE.EvalError):
                self._evaluate(fixture)

    def test_missing_spatial_gate_metrics_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = build_fixture(Path(tmp))
            matrix = fixture["matrix_payload"]
            matrix["units"]["spatial_mixture"]["fold0"]["seed42"].pop("gate_metrics")
            matrix_path = fixture["matrix"]
            matrix_path.write_text(json.dumps(matrix, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(MODULE.EvalError):
                self._evaluate(fixture)

    def test_dataset_id_probe_bac_reporting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = build_fixture(Path(tmp), anchor=0.30)
            code = self._evaluate(fixture)
            self.assertEqual(code, 0)

    def test_page_median_gain_positive_on_improvement(self) -> None:
        # delta_residual_ratio positive == improvement; median gain must be > 0
        pages = [
            MODULE.PageMetrics(
                file="scut_a.jpg",
                domain="scut",
                residual_ratio=0.1,
                overerase_ratio=0.0,
                delta_residual_ratio=0.02,
                delta_overerase_ratio=0.0,
            )
        ]
        self.assertGreater(MODULE.page_median_gain(pages, "residual_ratio"), 0.0)


if __name__ == "__main__":
    unittest.main()
