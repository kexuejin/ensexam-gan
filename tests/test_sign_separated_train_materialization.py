import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.analysis.audit_sign_separated_train_materialization import (
    AuditError,
    compare_selected_rows,
    validate_prediction_set,
)
from scripts.analysis.materialize_sign_separated_train_inputs import (
    LEDGER_PATH,
    STAGES,
    TRAINING_PLAN_PATH,
    atomic_write_manifest,
    expected_manifest_rows,
    rewrite_metrics_paths,
    validate_authority,
)
from scripts.analysis.validate_sign_separated_training_preflight import (
    PreflightError,
)


ROOT = Path(__file__).resolve().parents[1]


class SignSeparatedTrainMaterializationTest(unittest.TestCase):
    def test_killed_family_cannot_reopen_materialization(self) -> None:
        ledger = json.loads((ROOT / LEDGER_PATH).read_text(encoding="utf-8"))
        with self.assertRaisesRegex(PreflightError, "active iteration"):
            validate_authority(ROOT, ledger)
        self.assertEqual(
            STAGES, ("manifest", "primary", "second_stage", "patch_index")
        )
        self.assertNotIn("training", STAGES)

    def test_actual_manifest_rows_are_exact_train275_sources(self) -> None:
        plan = json.loads(
            (ROOT / TRAINING_PLAN_PATH).read_text(encoding="utf-8")
        )
        rows = expected_manifest_rows(ROOT, plan)
        self.assertEqual(len(rows), 275)
        self.assertEqual(len(rows), len(set(rows)))
        self.assertEqual(
            sum("/hw5k_" in row for row in rows),
            253,
        )
        self.assertEqual(sum("/scut_" in row for row in rows), 22)
        self.assertTrue(all("/all_images/" in row for row in rows))
        self.assertTrue(all("label" not in row.lower() for row in rows))

    def test_manifest_write_is_atomic_and_refuses_overwrite(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "manifest.txt"
            atomic_write_manifest(path, ["a.png", "b.png"])
            self.assertEqual(path.read_text(encoding="utf-8"), "a.png\nb.png\n")
            self.assertFalse((path.parent / ".manifest.txt.materializing").exists())
            with self.assertRaises(FileExistsError):
                atomic_write_manifest(path, ["c.png"])

    def test_metrics_path_rewrite_uses_csv_structure(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            temporary = root / ".stage.materializing"
            final = root / "stage"
            final.mkdir()
            metrics = final / "metrics.csv"
            with metrics.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["file", "pred_path"])
                writer.writeheader()
                writer.writerow(
                    {
                        "file": "a.jpg",
                        "pred_path": str(temporary / "pred/a.png"),
                    }
                )
            rewrite_metrics_paths(metrics, temporary, final)
            with metrics.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["pred_path"], str(final / "pred/a.png"))

    def test_prediction_set_rejects_missing_or_extra_pages(self) -> None:
        with TemporaryDirectory() as raw:
            pred_dir = Path(raw)
            (pred_dir / "a.png").write_bytes(b"a")
            result = validate_prediction_set(pred_dir, ["a.png"])
            self.assertEqual(result["count"], 1)
            with self.assertRaisesRegex(AuditError, "prediction set changed"):
                validate_prediction_set(pred_dir, ["a.png", "b.png"])

    def test_patch_row_comparison_fails_closed_on_metric_drift(self) -> None:
        expected = [
            {
                "rank_score": 0.3,
                "selected_for": "brighten",
                "file": "scut_1.jpg",
                "x1": 0,
                "y1": 0,
                "x2": 8,
                "y2": 8,
                "brighten_ratio": 0.1,
                "darken_ratio": 0.0,
                "brighten_mean_delta": 3.0,
                "darken_mean_delta": 0.0,
                "brighten_score": 0.3,
                "darken_score": 0.0,
            }
        ]
        actual = [{key: str(value) for key, value in expected[0].items()}]
        compare_selected_rows(actual, expected)
        actual[0]["brighten_score"] = "0.4"
        with self.assertRaisesRegex(AuditError, "brighten_score changed"):
            compare_selected_rows(actual, expected)


if __name__ == "__main__":
    unittest.main()
