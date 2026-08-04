import csv
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analysis" / "gate_dev_candidate_metrics.py"


class GateDevCandidateMetricsTest(unittest.TestCase):
    def write_csv(self, directory: Path, name: str, rows: list[dict[str, object]]) -> Path:
        path = directory / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "file",
                    "image_path",
                    "label_path",
                    "pred_path",
                    "residual_ratio",
                    "overerase_ratio",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def row(
        self,
        directory: Path,
        file_name: str,
        residual: float,
        overerase: float,
        identity_name: str | None = None,
    ) -> dict[str, object]:
        pred = directory / f"{file_name}.png"
        identity = identity_name or file_name
        image = directory / f"{identity}.image.jpg"
        label = directory / f"{identity}.label.jpg"
        pred.write_bytes(b"prediction")
        image.write_bytes(f"source:{identity}".encode("utf-8"))
        label.write_bytes(f"label:{identity}".encode("utf-8"))
        return {
            "file": file_name,
            "image_path": str(image),
            "label_path": str(label),
            "pred_path": str(pred),
            "residual_ratio": residual,
            "overerase_ratio": overerase,
        }

    def run_gate(
        self,
        baseline: Path,
        candidate: Path,
        output: Path,
        *extra_args: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--baseline-csv",
                str(baseline),
                "--candidate-csv",
                str(candidate),
                "--output-json",
                str(output),
                *extra_args,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_accepts_whole_dev_candidate_without_any_regression(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline = self.write_csv(directory, "baseline.csv", [
                self.row(directory, "a", 0.4, 0.003),
                self.row(directory, "b", 0.2, 0.001),
            ])
            candidate = self.write_csv(directory, "candidate.csv", [
                self.row(directory, "a-candidate", 0.3, 0.003, identity_name="a"),
                self.row(directory, "b-candidate", 0.1, 0.001, identity_name="b"),
            ])
            # Keep paired page identity while using distinct prediction files.
            with candidate.open(encoding="utf-8") as handle:
                candidate_rows = list(csv.DictReader(handle))
            candidate_rows[0]["file"] = "a"
            candidate_rows[1]["file"] = "b"
            self.write_csv(directory, "candidate.csv", candidate_rows)
            output = directory / "gate.json"

            result = self.run_gate(baseline, candidate, output)

            self.assertEqual(result.returncode, 0, result.stderr)
            result_json = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result_json["decision"], "accept")
            self.assertEqual(result_json["p95_convention"], "linear")
            self.assertEqual(
                result_json["p95_method"],
                "numpy.quantile(q=0.95, method='linear')",
            )

    def test_uses_frozen_15_page_second_worst_p95_rule(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline = self.write_csv(directory, "baseline.csv", [
                self.row(directory, f"page-{index}", (index + 1) / 100, (index + 1) / 1000)
                for index in range(15)
            ])
            candidate = self.write_csv(directory, "candidate.csv", [
                self.row(directory, f"page-{index}", index / 100, index / 1000)
                for index in range(15)
            ])
            output = directory / "gate.json"

            result = self.run_gate(
                baseline,
                candidate,
                output,
                "--p95-convention",
                "frozen-15page-lower-index",
                "--six-metric-only",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            result_json = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result_json["p95_convention"], "frozen-15page-lower-index")
            self.assertEqual(
                result_json["p95_method"],
                "sorted ascending index 13 (second-worst of 15)",
            )
            self.assertAlmostEqual(result_json["baseline_summary"]["p95_residual_ratio"], 0.14)
            self.assertAlmostEqual(result_json["candidate_summary"]["p95_residual_ratio"], 0.13)
            self.assertEqual(result_json["gate_scope"], "six_metrics_only")

    def test_six_metric_only_does_not_promote_page_deltas_to_extra_gates(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline = self.write_csv(directory, "baseline.csv", [
                self.row(directory, "a", 0.40, 0.001),
                self.row(directory, "b", 0.20, 0.010),
            ])
            candidate = self.write_csv(directory, "candidate.csv", [
                self.row(directory, "a", 0.30, 0.002),
                self.row(directory, "b", 0.10, 0.009),
            ])
            output = directory / "gate.json"

            result = self.run_gate(baseline, candidate, output, "--six-metric-only")

            self.assertEqual(result.returncode, 0, result.stderr)
            result_json = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result_json["decision"], "accept")
            self.assertEqual(len(result_json["checks"]), 6)

    def test_accepts_exact_minimum_mean_residual_improvement_and_persists_bound(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline = self.write_csv(directory, "baseline.csv", [
                self.row(directory, "a", 0.5, 0.001),
                self.row(directory, "b", 0.3, 0.002),
            ])
            candidate = self.write_csv(directory, "candidate.csv", [
                self.row(directory, "a", 0.32, 0.001),
                self.row(directory, "b", 0.32, 0.002),
            ])
            output = directory / "gate.json"

            result = self.run_gate(
                baseline,
                candidate,
                output,
                "--six-metric-only",
                "--min-mean-residual-improvement-percent",
                "20",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            result_json = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result_json["decision"], "accept")
            self.assertEqual(result_json["gate_scope"], "six_metrics_only")
            self.assertEqual(result_json["min_mean_residual_improvement_pct"], 20.0)
            self.assertAlmostEqual(result_json["mean_residual_bound"], 0.32)
            self.assertTrue(
                next(
                    check
                    for check in result_json["checks"]
                    if check["name"] == "mean_residual_improves"
                )["passed"]
            )

    def test_rejects_just_below_minimum_mean_residual_improvement(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline = self.write_csv(directory, "baseline.csv", [
                self.row(directory, "a", 0.5, 0.001),
                self.row(directory, "b", 0.3, 0.002),
            ])
            candidate = self.write_csv(directory, "candidate.csv", [
                self.row(directory, "a", 0.3201, 0.001),
                self.row(directory, "b", 0.3201, 0.002),
            ])
            output = directory / "gate.json"

            result = self.run_gate(
                baseline,
                candidate,
                output,
                "--six-metric-only",
                "--min-mean-residual-improvement-pct",
                "20",
            )

            self.assertEqual(result.returncode, 2, result.stderr)
            result_json = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result_json["decision"], "reject")
            self.assertFalse(
                next(
                    check
                    for check in result_json["checks"]
                    if check["name"] == "mean_residual_improves"
                )["passed"]
            )

    def test_rejects_invalid_minimum_mean_residual_improvement_values(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline = self.write_csv(directory, "baseline.csv", [self.row(directory, "a", 0.5, 0.001)])
            candidate = self.write_csv(directory, "candidate.csv", [self.row(directory, "a", 0.4, 0.001)])

            for value in ("-1", "100", "nan", "inf"):
                with self.subTest(value=value):
                    result = self.run_gate(
                        baseline,
                        candidate,
                        directory / f"gate-{value}.json",
                        "--six-metric-only",
                        "--min-mean-residual-improvement-pct",
                        value,
                    )

                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("finite, non-negative, and less than 100", result.stderr)

    def test_rejects_nonzero_minimum_without_six_metric_only(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline = self.write_csv(directory, "baseline.csv", [self.row(directory, "a", 0.5, 0.001)])
            candidate = self.write_csv(directory, "candidate.csv", [self.row(directory, "a", 0.4, 0.001)])
            output = directory / "gate.json"

            result = self.run_gate(
                baseline,
                candidate,
                output,
                "--min-mean-residual-improvement-pct",
                "20",
            )

            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("requires --six-metric-only", result.stderr)
            self.assertFalse(output.exists())

    def test_omitted_minimum_preserves_backward_compatible_six_metric_gate(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline = self.write_csv(directory, "baseline.csv", [
                self.row(directory, "a", 0.5, 0.001),
                self.row(directory, "b", 0.3, 0.002),
            ])
            candidate = self.write_csv(directory, "candidate.csv", [
                self.row(directory, "a", 0.39, 0.001),
                self.row(directory, "b", 0.39, 0.002),
            ])
            output = directory / "gate.json"

            result = self.run_gate(baseline, candidate, output, "--six-metric-only")

            self.assertEqual(result.returncode, 0, result.stderr)
            result_json = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result_json["decision"], "accept")
            self.assertEqual(result_json["min_mean_residual_improvement_pct"], 0.0)
            self.assertAlmostEqual(result_json["mean_residual_bound"], 0.4)

    def test_rejects_frozen_15_page_p95_rule_for_other_sample_counts(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline = self.write_csv(directory, "baseline.csv", [self.row(directory, "a", 0.2, 0.001)])
            candidate = self.write_csv(directory, "candidate.csv", [self.row(directory, "a", 0.1, 0.001)])

            result = self.run_gate(
                baseline,
                candidate,
                directory / "gate.json",
                "--p95-convention",
                "frozen-15page-lower-index",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires exactly 15 metric rows", result.stderr)

    def test_persists_rejected_quality_gate(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline = self.write_csv(directory, "baseline.csv", [self.row(directory, "a", 0.2, 0.001)])
            candidate = self.write_csv(directory, "candidate.csv", [self.row(directory, "a", 0.3, 0.001)])
            output = directory / "gate.json"

            result = self.run_gate(baseline, candidate, output)

            self.assertEqual(result.returncode, 2)
            result_json = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result_json["decision"], "reject")
            self.assertFalse(next(check for check in result_json["checks"] if check["name"] == "mean_residual_improves")["passed"])

    def test_rejects_mismatched_or_duplicate_csv_identity(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline = self.write_csv(directory, "baseline.csv", [self.row(directory, "a", 0.2, 0.001)])
            candidate = self.write_csv(directory, "candidate.csv", [
                self.row(directory, "a", 0.1, 0.001),
                self.row(directory, "a", 0.1, 0.001),
            ])

            result = self.run_gate(baseline, candidate, directory / "gate.json")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate", result.stderr)

    def test_rejects_equal_filename_with_different_source_or_label_identity(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            baseline = self.write_csv(directory, "baseline.csv", [self.row(directory, "a", 0.2, 0.001)])
            candidate = self.write_csv(
                directory,
                "candidate.csv",
                [self.row(directory, "a", 0.1, 0.001, identity_name="different-a")],
            )

            result = self.run_gate(baseline, candidate, directory / "gate.json")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source/label identity differs", result.stderr)


if __name__ == "__main__":
    unittest.main()
