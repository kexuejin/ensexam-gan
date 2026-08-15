import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.analysis.audit_external_text_layout_conditioned_patch_materialization import (
    AuditError,
    compare_selected_rows,
    normalize_summary_path,
    validate_forbidden_outputs,
    validate_ledger_authority,
)


class ExternalTextLayoutConditionedPatchMaterializationAuditTest(unittest.TestCase):
    def test_compare_selected_rows_detects_float_drift(self) -> None:
        actual = [
            {
                "edit_positive_mean_delta": "30.0",
                "edit_positive_ratio": "1.0",
                "edit_positive_score": "30.0",
                "file": "a.png",
                "preserve_negative_ratio": "0.0",
                "text_confidence_mean": "0.5",
                "text_confidence_occupied_mean": "0.5",
                "text_occupancy_ratio": "1.0",
                "x1": "0",
                "x2": "2",
                "y1": "0",
                "y2": "2",
            }
        ]
        expected = [dict(actual[0], edit_positive_score=30.1)]
        with self.assertRaisesRegex(AuditError, "score"):
            compare_selected_rows(actual, expected)

    def test_normalize_summary_path_accepts_portable_suffix(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            patch_csv = root / "hardcase_lists" / "patches.csv"
            summary = {"patch_index": f"/tmp/other/{patch_csv.relative_to(root)}"}
            normalized = normalize_summary_path(summary, patch_csv)
        self.assertEqual(normalized["patch_index"], str(patch_csv))

    def test_ledger_requires_preflight_and_surface_pass(self) -> None:
        ledger = {
            "active_iteration": {
                "id": "monotonic-residual-erase-support",
                "prerequisites": [
                    {
                        "id": "external_text_layout_support_train_only_diagnostic",
                        "status": "passed",
                    },
                    {
                        "id": "external_text_layout_conditioned_monotonic_preflight",
                        "status": "passed",
                    },
                    {
                        "id": "external_text_layout_conditioned_monotonic_surface_integration",
                        "status": "passed",
                    },
                ],
            }
        }
        authority = validate_ledger_authority(ledger)
        self.assertEqual(authority["patch_materialization_status"], "pending")
        ledger["active_iteration"]["prerequisites"][1]["status"] = "pending"
        with self.assertRaisesRegex(AuditError, "required prerequisite"):
            validate_ledger_authority(ledger)

    def test_forbidden_outputs_reject_training_or_quality_start(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            training = root / "train-out"
            training.mkdir()
            plan = {
                "planned_outputs_must_be_absent": {
                    "checkpoint_audit": "checkpoint",
                    "first_gate_candidate": "candidate",
                    "first_gate_score": "score",
                    "training_output_dir": "train-out",
                }
            }
            with self.assertRaisesRegex(AuditError, "quality output"):
                validate_forbidden_outputs(root, plan)


if __name__ == "__main__":
    unittest.main()
