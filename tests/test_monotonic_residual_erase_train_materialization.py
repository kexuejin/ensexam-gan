import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.analysis.audit_monotonic_residual_erase_train_materialization import (
    AuditError,
    compare_selected_rows,
    run_audit,
)
from scripts.analysis.materialize_monotonic_residual_erase_train_inputs import (
    ARCHIVE_PRIMARY_DIR,
    ARCHIVE_SECOND_STAGE_DIR,
    CONTROL_DIR,
    LEDGER_PATH,
    TRAINING_PLAN_PATH,
    assert_manifest,
    read_json,
    repo_path,
    validate_authority,
    validate_reuse_source,
)


ROOT = Path(__file__).resolve().parents[1]


class MonotonicResidualEraseTrainMaterializationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = read_json(ROOT / TRAINING_PLAN_PATH)
        cls.ledger = read_json(ROOT / LEDGER_PATH)

    def test_actual_materialization_passes_independent_audit(self) -> None:
        result = run_audit(repo_root=ROOT)
        self.assertEqual(result["terminal"], "PASS", result)
        self.assertEqual(result["train_count"], 275)
        self.assertEqual(result["primary_predictions"]["count"], 275)
        self.assertEqual(result["second_stage_predictions"]["count"], 275)
        self.assertEqual(result["patch_summary"]["patch_count"], 256)
        self.assertEqual(result["patch_summary"]["candidate_count"], 52645)
        self.assertGreater(result["patch_summary"]["positive_ratio_min"], 0.0)
        self.assertGreater(result["patch_summary"]["preserve_ratio_min"], 0.0)
        self.assertTrue(result["real_train_pixels_audited"])
        self.assertEqual(result["target_decode_roles"], ["train"])
        self.assertFalse(result["training_started"])
        self.assertFalse(result["first_quality_gate_started"])

    def test_reuse_source_matches_frozen_pipeline_pass(self) -> None:
        source = validate_reuse_source(ROOT, self.plan)
        self.assertEqual(
            source["primary_predictions"]["content_sha256"],
            "6400c9413af963e3de280e348bd635cd962e5387c2e975e930036d320214274a",
        )
        self.assertEqual(
            source["second_stage_predictions"]["content_sha256"],
            "2ffa40fc0c9b2a7e721d560f6f12edfe2ccdc1c1988582fa7a8104665cdc088a",
        )

    def test_registered_prediction_paths_are_relative_archive_links(self) -> None:
        outputs = self.plan["planned_outputs_must_be_absent"]
        cases = (
            ("primary_prediction_dir", ROOT / ARCHIVE_PRIMARY_DIR),
            ("training_input_dir", ROOT / ARCHIVE_SECOND_STAGE_DIR),
        )
        for key, expected in cases:
            link = repo_path(ROOT, outputs[key])
            self.assertTrue(link.is_symlink())
            self.assertFalse(Path(os.readlink(link)).is_absolute())
            self.assertEqual(link.resolve(), expected.resolve())

    def test_manifest_is_exact_train275(self) -> None:
        manifest = repo_path(
            ROOT, self.plan["planned_outputs_must_be_absent"]["sample_manifest"]
        )
        rows = assert_manifest(ROOT, self.plan, manifest)
        self.assertEqual(len(rows), 275)
        self.assertEqual(len(rows), len(set(rows)))

    def test_patch_index_contains_only_registered_support_columns(self) -> None:
        patch = repo_path(ROOT, self.plan["patch_builder"]["output_csv"])
        with patch.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 256)
        self.assertTrue(
            all(float(row["edit_positive_ratio"]) >= 0.001 for row in rows)
        )
        self.assertTrue(
            all(float(row["preserve_negative_ratio"]) > 0.0 for row in rows)
        )
        self.assertNotIn("darken_ratio", rows[0])
        self.assertNotIn("selected_for", rows[0])

    def test_patch_row_drift_fails_closed(self) -> None:
        actual = [
            {
                "file": "a.png",
                "x1": "0",
                "y1": "0",
                "x2": "8",
                "y2": "8",
                "edit_positive_score": "1.0",
                "edit_positive_ratio": "0.5",
                "edit_positive_mean_delta": "2.0",
                "preserve_negative_ratio": "0.5",
            }
        ]
        expected = [dict(actual[0], edit_positive_score=2.0)]
        with self.assertRaisesRegex(AuditError, "edit_positive_score changed"):
            compare_selected_rows(actual, expected)

    def test_training_preflight_must_stay_passed(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "ledger.json"
            ledger = json.loads(json.dumps(self.ledger))
            prerequisite = next(
                item
                for item in ledger["active_iteration"]["prerequisites"]
                if item["id"] == "monotonic_residual_erase_training_preflight"
            )
            prerequisite["status"] = "pending"
            path.write_text(json.dumps(ledger), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training preflight"):
                validate_authority(ROOT, read_json(path))

    def test_stage_records_cover_every_atomic_stage(self) -> None:
        records = sorted((ROOT / CONTROL_DIR).glob("*.json"))
        self.assertEqual(
            [path.stem for path in records],
            ["manifest", "patch_index", "primary", "second_stage"],
        )


if __name__ == "__main__":
    unittest.main()
