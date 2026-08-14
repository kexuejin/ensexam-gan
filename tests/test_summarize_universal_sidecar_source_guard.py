import csv
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analysis" / "summarize_universal_sidecar_source_guard.py"


class SummarizeUniversalSidecarSourceGuardTest(unittest.TestCase):
    def write_metrics(
        self,
        directory: Path,
        rows: list[dict[str, object]],
    ) -> Path:
        path = directory / "post_freeze_metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "file",
                    "baseline_residual_ratio",
                    "baseline_overerase_ratio",
                    "residual_ratio",
                    "overerase_ratio",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def write_samples(self, directory: Path, files: list[str]) -> Path:
        path = directory / "samples.txt"
        path.write_text("\n".join(files) + "\n", encoding="utf-8")
        return path

    def run_summary(
        self,
        metrics: Path,
        expected_samples: Path,
        output_json: Path,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--post-freeze-metrics",
                str(metrics),
                "--expected-samples-file",
                str(expected_samples),
                "--output-json",
                str(output_json),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_exact_no_delta_passes_source_guard(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            metrics = self.write_metrics(
                directory,
                [
                    {
                        "file": "100.jpg",
                        "baseline_residual_ratio": 0.2,
                        "baseline_overerase_ratio": 0.001,
                        "residual_ratio": 0.2,
                        "overerase_ratio": 0.001,
                    },
                    {
                        "file": "101.jpg",
                        "baseline_residual_ratio": 0.4,
                        "baseline_overerase_ratio": 0.003,
                        "residual_ratio": 0.4,
                        "overerase_ratio": 0.003,
                    },
                ],
            )
            samples = self.write_samples(directory, ["100.jpg", "101.jpg"])
            output_json = directory / "source_guard_summary.json"

            result = self.run_summary(metrics, samples, output_json)

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["source_guard_status"], "pass")
            self.assertEqual(summary["failures"], [])
            self.assertFalse(summary["measurable_page_delta"])
            self.assertEqual(summary["nonzero_delta_files"], [])
            self.assertEqual(summary["pages"], 2)
            self.assertAlmostEqual(summary["residual_ratio"]["baseline_mean"], 0.3)
            self.assertAlmostEqual(summary["residual_ratio"]["candidate_mean"], 0.3)
            self.assertEqual(summary["residual_ratio"]["delta_max"], 0.0)
            self.assertEqual(summary["overerase_ratio"]["delta_max"], 0.0)
            self.assertEqual(summary["provenance"]["post_freeze_metrics"], str(metrics))
            self.assertEqual(
                summary["provenance"]["expected_samples_file"],
                str(samples),
            )
            self.assertEqual(
                len(summary["provenance"]["post_freeze_metrics_sha256"]),
                64,
            )
            self.assertEqual(
                len(summary["provenance"]["expected_samples_sha256"]),
                64,
            )

    def test_positive_residual_delta_kills_source_guard(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            metrics = self.write_metrics(
                directory,
                [
                    {
                        "file": "300.jpg",
                        "baseline_residual_ratio": 0.2,
                        "baseline_overerase_ratio": 0.001,
                        "residual_ratio": 0.2,
                        "overerase_ratio": 0.001,
                    },
                    {
                        "file": "301.jpg",
                        "baseline_residual_ratio": 0.4,
                        "baseline_overerase_ratio": 0.003,
                        "residual_ratio": 0.400008,
                        "overerase_ratio": 0.003,
                    },
                ],
            )
            samples = self.write_samples(directory, ["300.jpg", "301.jpg"])
            output_json = directory / "source_guard_summary.json"

            result = self.run_summary(metrics, samples, output_json)

            self.assertEqual(result.returncode, 1, result.stdout)
            summary = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["source_guard_status"], "fail")
            self.assertEqual(summary["failures"], ["residual_source_guard_regression"])
            self.assertTrue(summary["measurable_page_delta"])
            self.assertEqual(
                summary["nonzero_delta_files"],
                [
                    {
                        "file": "301.jpg",
                        "delta_residual_ratio": 7.99999999995249e-06,
                        "delta_overerase_ratio": 0.0,
                    }
                ],
            )
            self.assertEqual(summary["residual_ratio"]["nonzero_delta_count"], 1)
            self.assertEqual(summary["overerase_ratio"]["nonzero_delta_count"], 0)
            self.assertEqual(
                summary["residual_ratio"]["delta_max"],
                7.99999999995249e-06,
            )

    def test_measurable_non_regressive_delta_passes_source_guard(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            metrics = self.write_metrics(
                directory,
                [
                    {
                        "file": "400.jpg",
                        "baseline_residual_ratio": 0.2,
                        "baseline_overerase_ratio": 0.001,
                        "residual_ratio": 0.19999,
                        "overerase_ratio": 0.001,
                    }
                ],
            )
            samples = self.write_samples(directory, ["400.jpg"])
            output_json = directory / "source_guard_summary.json"

            result = self.run_summary(metrics, samples, output_json)

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["source_guard_status"], "pass")
            self.assertEqual(summary["failures"], [])
            self.assertTrue(summary["measurable_page_delta"])
            self.assertEqual(len(summary["nonzero_delta_files"]), 1)
            self.assertLess(summary["residual_ratio"]["delta_max"], 0.0)

    def test_positive_overerase_delta_kills_source_guard(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            metrics = self.write_metrics(
                directory,
                [
                    {
                        "file": "500.jpg",
                        "baseline_residual_ratio": 0.2,
                        "baseline_overerase_ratio": 0.001,
                        "residual_ratio": 0.2,
                        "overerase_ratio": 0.00101,
                    }
                ],
            )
            samples = self.write_samples(directory, ["500.jpg"])
            output_json = directory / "source_guard_summary.json"

            result = self.run_summary(metrics, samples, output_json)

            self.assertEqual(result.returncode, 1, result.stdout)
            summary = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["source_guard_status"], "fail")
            self.assertEqual(
                summary["failures"],
                ["overerase_source_guard_regression"],
            )
            self.assertGreater(summary["overerase_ratio"]["delta_max"], 0.0)

    def test_truncated_or_duplicate_input_is_rejected(self) -> None:
        with TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            row = {
                "file": "600.jpg",
                "baseline_residual_ratio": 0.2,
                "baseline_overerase_ratio": 0.001,
                "residual_ratio": 0.2,
                "overerase_ratio": 0.001,
            }
            metrics = self.write_metrics(directory, [row])
            output_json = directory / "source_guard_summary.json"

            with self.subTest(case="truncated"):
                samples = self.write_samples(directory, ["600.jpg", "601.jpg"])
                result = self.run_summary(metrics, samples, output_json)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("does not match expected sample manifest", result.stderr)

            with self.subTest(case="duplicate_csv"):
                metrics = self.write_metrics(directory, [row, row])
                samples = self.write_samples(directory, ["600.jpg"])
                result = self.run_summary(metrics, samples, output_json)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("duplicate files", result.stderr)

            with self.subTest(case="duplicate_manifest"):
                metrics = self.write_metrics(directory, [row])
                samples = self.write_samples(directory, ["600.jpg", "600.jpg"])
                result = self.run_summary(metrics, samples, output_json)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("duplicate samples", result.stderr)

    def test_invalid_ratio_is_rejected(self) -> None:
        cases = (
            ("nan", "non-finite residual_ratio"),
            ("inf", "non-finite residual_ratio"),
            ("-inf", "non-finite residual_ratio"),
            ("-0.01", "out-of-range residual_ratio"),
            ("1.01", "out-of-range residual_ratio"),
        )
        for value, expected_error in cases:
            with self.subTest(value=value), TemporaryDirectory() as raw_directory:
                directory = Path(raw_directory)
                metrics = self.write_metrics(
                    directory,
                    [
                        {
                            "file": "700.jpg",
                            "baseline_residual_ratio": 0.2,
                            "baseline_overerase_ratio": 0.001,
                            "residual_ratio": value,
                            "overerase_ratio": 0.001,
                        }
                    ],
                )
                samples = self.write_samples(directory, ["700.jpg"])
                output_json = directory / "source_guard_summary.json"

                result = self.run_summary(metrics, samples, output_json)

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)


if __name__ == "__main__":
    unittest.main()
