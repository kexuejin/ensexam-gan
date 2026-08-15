import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analysis" / "report_current_primary_quality_loop_status.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReportCurrentPrimaryQualityLoopStatusTest(unittest.TestCase):
    def artifact(self, root: Path, relative: str, content: str) -> dict[str, str]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"path": relative, "sha256": digest(path)}

    def ledger(self, root: Path) -> Path:
        config = self.artifact(root, "artifacts/current-primary/config.yaml", "config\n")
        checkpoint = self.artifact(root, "artifacts/current-primary/model.pth", "checkpoint\n")
        manifest = self.artifact(root, "hardcase_lists/inner-val15.txt", "a.jpg\n")
        calibration = self.artifact(root, "docs/decisions/calibration.md", "calibration\n")
        d2 = self.artifact(root, "docs/decisions/d2.md", "d2\n")
        d2d = self.artifact(root, "docs/decisions/d2d.md", "d2d\n")
        data = {
            "schema_version": 1,
            "program": {
                "name": "test loop",
                "product_default": "artifacts/current-primary",
                "promotion_state": "disabled",
                "reserved_blind_state": "disabled",
            },
            "baseline": {
                "product_default": "artifacts/current-primary",
                "config": config,
                "checkpoint": checkpoint,
                "inner_val15_manifest": manifest,
                "matched_copy_protocol": {
                    "copy_input_outside_mask": "mb",
                    "copy_mask_threshold_auto": "mb_cov8_step",
                    "copy_mask_dilate": 0,
                    "page_overlap": 32,
                    "batch_size": 8,
                    "change_threshold": 12,
                    "eval_threshold": 12,
                },
            },
            "calibration": {
                "terminal": "PASS",
                "scope": "scut_inner_val15_current_primary_matched_copy",
                "run_count": 3,
                "pages_per_run": 15,
                "prediction_hashes_identical": True,
                "minimum_residual_gain": 0.0005,
                "decision": calibration,
            },
            "records": [
                {
                    "id": "d2",
                    "terminal": "KILL",
                    "outcome": "source_guard_residual_regression",
                    "repeat_policy": "do_not_repeat",
                    "evidence": [d2],
                },
                {
                    "id": "d2d",
                    "terminal": "PASS",
                    "outcome": "safe_no_lift",
                    "repeat_policy": "do_not_repeat",
                    "evidence": [d2d],
                },
            ],
            "active_iteration": {
                "id": "d3",
                "terminal": "PREREQUISITE_NEEDED",
                "failure_bucket": "source_solved_pixel_regression",
                "causal_change": "baseline-tail non-regression constraint",
                "first_gate": "scut_inner_val15",
                "prerequisites": [
                    {
                        "id": "train_only_support",
                        "status": "pending",
                        "detail": "registered train-only support cache",
                    }
                ],
                "next_action": "build and validate support cache",
                "prohibited_before_first_gate": [
                    "scut115",
                    "holdout40",
                    "reserved_blind",
                ],
                "evidence": [d2, d2d],
            },
        }
        path = root / "docs" / "ledger.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def command(self, root: Path, ledger: Path, output: Path) -> list[str]:
        return [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(root),
            "--ledger",
            str(ledger),
            "--output-json",
            str(output),
        ]

    def evidence_audit_command(self, root: Path, ledger: Path, output: Path) -> list[str]:
        return [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(root),
            "--ledger",
            str(ledger),
            "--evidence-audit-json",
            str(output),
            "--evidence-audit-only",
        ]

    def test_valid_ledger_is_active_and_not_promotion_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self.ledger(root)
            output = root / "report.json"
            subprocess.run(self.command(root, ledger, output), check=True)
            report = json.loads(output.read_text(encoding="utf-8"))

            self.assertTrue(report["baseline_verified"])
            self.assertEqual(report["status"], "active_not_promotion_eligible")
            self.assertFalse(report["candidate_admission_ready"])
            self.assertFalse(report["promotion_eligible"])
            self.assertIn("pending prerequisite: train_only_support", report["blockers"])

    def test_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self.ledger(root)
            data = json.loads(ledger.read_text(encoding="utf-8"))
            data["baseline"]["config"]["sha256"] = "0" * 64
            ledger.write_text(json.dumps(data), encoding="utf-8")

            result = subprocess.run(self.command(root, ledger, root / "report.json"), text=True, capture_output=True)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("baseline.config.sha256 mismatch", result.stderr)

    def test_rejected_direction_cannot_be_marked_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self.ledger(root)
            data = json.loads(ledger.read_text(encoding="utf-8"))
            data["records"][0]["repeat_policy"] = "repeat"
            ledger.write_text(json.dumps(data), encoding="utf-8")

            result = subprocess.run(self.command(root, ledger, root / "report.json"), text=True, capture_output=True)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("repeat_policy=do_not_repeat", result.stderr)

    def test_active_iteration_cannot_skip_inner_val15(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self.ledger(root)
            data = json.loads(ledger.read_text(encoding="utf-8"))
            data["active_iteration"]["first_gate"] = "scut115"
            ledger.write_text(json.dumps(data), encoding="utf-8")

            result = subprocess.run(self.command(root, ledger, root / "report.json"), text=True, capture_output=True)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("first_gate must be scut_inner_val15", result.stderr)

    def test_ledger_cannot_escape_repository_with_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self.ledger(root)
            data = json.loads(ledger.read_text(encoding="utf-8"))
            data["baseline"]["config"]["path"] = "../outside.yaml"
            ledger.write_text(json.dumps(data), encoding="utf-8")

            result = subprocess.run(self.command(root, ledger, root / "report.json"), text=True, capture_output=True)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not contain parent traversal", result.stderr)

    def test_evidence_audit_reports_missing_artifacts_without_status_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self.ledger(root)
            (root / "docs" / "decisions" / "d2.md").unlink()
            output = root / "evidence-audit.json"

            result = subprocess.run(
                self.evidence_audit_command(root, ledger, output),
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            audit = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "evidence_incomplete")
            self.assertEqual(audit["missing_count"], 2)
            self.assertEqual(audit["missing_unique_path_count"], 1)
            self.assertEqual(audit["missing_prefix_counts"], {"docs": 2})
            self.assertEqual(audit["missing_unique_paths"], {"docs/decisions/d2.md": 2})
            self.assertEqual(audit["gap_class_counts"], {"missing_tracked_reference": 2})
            self.assertEqual(
                audit["gap_class_unique_path_counts"],
                {"missing_tracked_reference": 1},
            )
            missing_paths = {item["path"] for item in audit["missing"]}
            self.assertEqual(missing_paths, {"docs/decisions/d2.md"})
            self.assertIn("status=evidence_incomplete", result.stdout)

    def test_evidence_audit_reports_hash_mismatch_without_status_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self.ledger(root)
            (root / "docs" / "decisions" / "d2d.md").write_text("changed\n", encoding="utf-8")
            output = root / "evidence-audit.json"

            result = subprocess.run(
                self.evidence_audit_command(root, ledger, output),
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            audit = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "evidence_incomplete")
            self.assertEqual(audit["mismatch_count"], 2)
            self.assertEqual(audit["mismatch_unique_path_count"], 1)
            self.assertEqual(audit["mismatch_prefix_counts"], {"docs": 2})
            self.assertEqual(audit["mismatch_unique_paths"], {"docs/decisions/d2d.md": 2})
            self.assertEqual(audit["gap_class_counts"], {"tracked_evidence_hash_drift": 2})
            self.assertEqual(
                audit["gap_class_unique_path_counts"],
                {"tracked_evidence_hash_drift": 1},
            )
            mismatched_paths = {item["path"] for item in audit["mismatched"]}
            self.assertEqual(mismatched_paths, {"docs/decisions/d2d.md"})


if __name__ == "__main__":
    unittest.main()
