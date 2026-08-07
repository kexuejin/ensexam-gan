import csv
import json
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analysis" / "calibrate_prediction_replay_noise.py"


class CalibratePredictionReplayNoiseTest(unittest.TestCase):
    def write_expected(self, directory: Path, files: list[str] | None = None) -> Path:
        path = directory / "expected.txt"
        path.write_text("\n".join(files or ["a.jpg", "b.jpg"]) + "\n", encoding="utf-8")
        return path

    def write_run(
        self,
        directory: Path,
        name: str,
        residual: list[float],
        overerase: list[float] | None = None,
        *,
        files: list[str] | None = None,
        batch_size: str = "8",
        same_prediction_hashes: bool = True,
    ) -> tuple[Path, Path]:
        files = files or ["a.jpg", "b.jpg"]
        overerase = overerase or [0.01, 0.02]
        run_dir = directory / name
        run_dir.mkdir()
        post_path = run_dir / "post_freeze_metrics.csv"
        with post_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["file", "residual_ratio", "overerase_ratio"],
            )
            writer.writeheader()
            for file, residual_value, overerase_value in zip(files, residual, overerase):
                writer.writerow(
                    {
                        "file": file,
                        "residual_ratio": residual_value,
                        "overerase_ratio": overerase_value,
                    }
                )

        inference_path = run_dir / "inference_metrics.csv"
        fieldnames = [
            "file",
            "image_sha256",
            "pred_sha256",
            "primary_config_sha256",
            "primary_weights_sha256",
            "page_overlap",
            "batch_size",
            "copy_input_outside_mask",
            "copy_mask_threshold",
            "copy_mask_threshold_auto",
            "copy_mask_dilate",
        ]
        with inference_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, file in enumerate(files):
                pred_token = str(index) if same_prediction_hashes else f"{name}-{index}"
                writer.writerow(
                    {
                        "file": file,
                        "image_sha256": str(index + 1) * 64,
                        "pred_sha256": (pred_token * 64)[:64],
                        "primary_config_sha256": "c" * 64,
                        "primary_weights_sha256": "d" * 64,
                        "page_overlap": "32",
                        "batch_size": batch_size,
                        "copy_input_outside_mask": "mb",
                        "copy_mask_threshold": str(8 + index),
                        "copy_mask_threshold_auto": "mb_cov8_step",
                        "copy_mask_dilate": "0",
                    }
                )
        return post_path, inference_path

    def command(
        self,
        expected: Path,
        output: Path,
        runs: list[tuple[str, Path, Path]],
    ) -> list[str]:
        command = [
            sys.executable,
            str(SCRIPT),
            "--expected-samples-file",
            str(expected),
            "--output-json",
            str(output),
        ]
        for name, post, inference in runs:
            command.extend(["--run", f"{name}:{post}:{inference}"])
        return command

    def test_exact_deterministic_replays_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            expected = self.write_expected(directory)
            runs = [
                (name, *self.write_run(directory, name, [0.1, 0.2]))
                for name in ("run1", "run2", "run3")
            ]
            output = directory / "summary.json"
            subprocess.run(self.command(expected, output, runs), check=True)
            summary = json.loads(output.read_text(encoding="utf-8"))

            self.assertTrue(summary["passed"])
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["run_count"], 3)
            self.assertEqual(summary["nonzero_metric_files"], [])
            self.assertEqual(summary["prediction_hashes"]["identical_files"], 2)
            residual = summary["metrics"]["residual_ratio"]
            self.assertEqual(residual["aggregate_replay_stddev"], 0.0)
            self.assertEqual(residual["calibrated_minimum_gain"], 0.0005)

    def test_nonzero_noise_calibrates_three_sigma(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            expected = self.write_expected(directory)
            residuals = ([0.10, 0.20], [0.11, 0.20], [0.09, 0.20])
            runs = [
                (
                    f"run{index}",
                    *self.write_run(
                        directory,
                        f"run{index}",
                        values,
                        same_prediction_hashes=False,
                    ),
                )
                for index, values in enumerate(residuals, start=1)
            ]
            output = directory / "summary.json"
            subprocess.run(self.command(expected, output, runs), check=True)
            summary = json.loads(output.read_text(encoding="utf-8"))

            expected_stddev = statistics.stdev([0.15, 0.155, 0.145])
            residual = summary["metrics"]["residual_ratio"]
            self.assertAlmostEqual(residual["aggregate_replay_stddev"], expected_stddev)
            self.assertAlmostEqual(
                residual["calibrated_minimum_gain"], 3.0 * expected_stddev
            )
            self.assertEqual(summary["nonzero_metric_files"], ["a.jpg"])
            self.assertEqual(summary["prediction_hashes"]["different_files"], 2)

    def test_fewer_than_three_runs_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            expected = self.write_expected(directory)
            runs = [
                (name, *self.write_run(directory, name, [0.1, 0.2]))
                for name in ("run1", "run2")
            ]
            result = subprocess.run(
                self.command(expected, directory / "summary.json", runs),
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("at least three", result.stderr)

    def test_manifest_mismatch_and_duplicates_fail(self) -> None:
        for files, message in ((["a.jpg", "c.jpg"], "does not match"), (["a.jpg", "a.jpg"], "duplicate")):
            with self.subTest(files=files), tempfile.TemporaryDirectory() as tmpdir:
                directory = Path(tmpdir)
                expected = self.write_expected(directory)
                runs = [
                    (
                        name,
                        *self.write_run(directory, name, [0.1, 0.2], files=files),
                    )
                    for name in ("run1", "run2", "run3")
                ]
                result = subprocess.run(
                    self.command(expected, directory / "summary.json", runs),
                    text=True,
                    capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_protocol_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            expected = self.write_expected(directory)
            runs = [
                ("run1", *self.write_run(directory, "run1", [0.1, 0.2])),
                ("run2", *self.write_run(directory, "run2", [0.1, 0.2])),
                (
                    "run3",
                    *self.write_run(
                        directory, "run3", [0.1, 0.2], batch_size="4"
                    ),
                ),
            ]
            result = subprocess.run(
                self.command(expected, directory / "summary.json", runs),
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("protocol differs", result.stderr)

    def test_invalid_metric_values_fail(self) -> None:
        for invalid in (float("nan"), 1.1):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as tmpdir:
                directory = Path(tmpdir)
                expected = self.write_expected(directory)
                runs = [
                    (
                        name,
                        *self.write_run(directory, name, [invalid, 0.2]),
                    )
                    for name in ("run1", "run2", "run3")
                ]
                result = subprocess.run(
                    self.command(expected, directory / "summary.json", runs),
                    text=True,
                    capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid residual_ratio", result.stderr)


if __name__ == "__main__":
    unittest.main()
