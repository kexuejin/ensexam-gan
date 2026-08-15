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

    def track_decision_doc(self, root: Path, relative: str, text: str) -> None:
        subprocess.run(
            ["git", "init"],
            cwd=root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        subprocess.run(["git", "add", relative], cwd=root, check=True)

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

    def test_all_exhausted_ledger_reports_terminal_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self.ledger(root)
            data = json.loads(ledger.read_text(encoding="utf-8"))
            data["program"]["lifecycle_state"] = "all_exhausted"
            data["active_iteration"]["terminal"] = "KILL"
            data["active_iteration"]["prerequisites"][0]["status"] = "passed"
            data["active_iteration"]["next_action"] = "none; all named buckets are exhausted or blocked"
            data["active_iteration"]["causal_change"] = (
                "broader durable exhaustion of all named failure buckets"
            )
            ledger.write_text(json.dumps(data), encoding="utf-8")
            output = root / "report.json"

            subprocess.run(self.command(root, ledger, output), check=True)

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "all_failure_buckets_exhausted_or_blocked")
            self.assertEqual(report["program"]["lifecycle_state"], "all_exhausted")
            self.assertFalse(report["candidate_admission_ready"])
            self.assertFalse(report["promotion_eligible"])
            self.assertEqual(report["blockers"], [])

    def test_all_exhausted_ledger_rejects_pending_prerequisites(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self.ledger(root)
            data = json.loads(ledger.read_text(encoding="utf-8"))
            data["program"]["lifecycle_state"] = "all_exhausted"
            data["active_iteration"]["terminal"] = "KILL"
            ledger.write_text(json.dumps(data), encoding="utf-8")

            result = subprocess.run(
                self.command(root, ledger, root / "report.json"),
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no pending prerequisites", result.stderr)

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
            self.assertIn("repeat_policy starting with do_not_repeat", result.stderr)

    def test_rejected_direction_can_use_specific_do_not_repeat_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self.ledger(root)
            data = json.loads(ledger.read_text(encoding="utf-8"))
            data["records"][0]["repeat_policy"] = (
                "do_not_repeat_same_runtime_geometry_without_new_preregistration"
            )
            ledger.write_text(json.dumps(data), encoding="utf-8")
            output = root / "report.json"

            subprocess.run(self.command(root, ledger, output), check=True)

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                report["records"][0]["repeat_policy"],
                "do_not_repeat_same_runtime_geometry_without_new_preregistration",
            )

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
            self.assertEqual(
                audit["successor_readiness"]["status"],
                "blocked_by_unresolved_evidence",
            )
            self.assertEqual(
                audit["successor_readiness"]["blocking_gap_class_counts"],
                {"missing_tracked_reference": 2},
            )
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
            self.assertEqual(
                audit["successor_readiness"]["status"],
                "blocked_by_unresolved_evidence",
            )
            self.assertEqual(
                audit["successor_readiness"]["blocking_gap_class_counts"],
                {"tracked_evidence_hash_drift": 2},
            )
            mismatched_paths = {item["path"] for item in audit["mismatched"]}
            self.assertEqual(mismatched_paths, {"docs/decisions/d2d.md"})

    def test_evidence_audit_keeps_undocumented_missing_ignored_output_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self.ledger(root)
            data = json.loads(ledger.read_text(encoding="utf-8"))
            data["records"][0]["evidence"].append(
                {
                    "path": "outputs/transient-run/metrics.json",
                    "sha256": "1" * 64,
                }
            )
            ledger.write_text(json.dumps(data), encoding="utf-8")
            output = root / "evidence-audit.json"

            result = subprocess.run(
                self.evidence_audit_command(root, ledger, output),
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            audit = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(audit["gap_class_counts"], {"missing_ignored_output": 1})
            self.assertEqual(
                audit["successor_readiness"]["status"],
                "blocked_by_unresolved_evidence",
            )
            self.assertEqual(
                audit["successor_readiness"]["blocking_gap_class_counts"],
                {"missing_ignored_output": 1},
            )

    def test_evidence_audit_marks_documented_missing_ignored_output_nonblocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self.ledger(root)
            expected_sha256 = "1" * 64
            evidence_path = "outputs/transient-run/metrics.json"
            self.track_decision_doc(
                root,
                "docs/decisions/transient-run-kill.md",
                "\n".join(
                    [
                        "# Transient Run KILL",
                        "",
                        "`KILL`. The disposable metrics output is documented here.",
                        f"- path: `{evidence_path}`",
                        f"- sha256: `{expected_sha256}`",
                        "",
                    ]
                ),
            )
            data = json.loads(ledger.read_text(encoding="utf-8"))
            data["records"][0]["evidence"].append(
                {"path": evidence_path, "sha256": expected_sha256}
            )
            ledger.write_text(json.dumps(data), encoding="utf-8")
            output = root / "evidence-audit.json"

            result = subprocess.run(
                self.evidence_audit_command(root, ledger, output),
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            audit = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                audit["gap_class_counts"],
                {"documented_missing_ignored_output": 1},
            )
            self.assertEqual(
                audit["successor_readiness"]["status"],
                "not_blocked_by_evidence_audit",
            )
            self.assertEqual(
                audit["successor_readiness"]["nonblocking_gap_class_counts"],
                {"documented_missing_ignored_output": 1},
            )
            [missing] = audit["missing"]
            self.assertTrue(missing["decision_documented"])
            self.assertEqual(
                missing["decision_documents"][0]["path"],
                "docs/decisions/transient-run-kill.md",
            )

    def test_evidence_audit_marks_documented_ignored_hash_drift_nonblocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self.ledger(root)
            evidence_path = "outputs/transient-run/preflight.json"
            expected_sha256 = hashlib.sha256(b"old\n").hexdigest()
            actual = root / evidence_path
            actual.parent.mkdir(parents=True, exist_ok=True)
            actual.write_text("new\n", encoding="utf-8")
            self.track_decision_doc(
                root,
                "docs/decisions/transient-preflight-pass.md",
                "\n".join(
                    [
                        "# Transient Preflight PASS",
                        "",
                        "`PASS`. The disposable preflight output is documented here.",
                        f"- path: `{evidence_path}`",
                        f"- sha256: `{expected_sha256}`",
                        "",
                    ]
                ),
            )
            data = json.loads(ledger.read_text(encoding="utf-8"))
            data["records"][0]["evidence"].append(
                {"path": evidence_path, "sha256": expected_sha256}
            )
            ledger.write_text(json.dumps(data), encoding="utf-8")
            output = root / "evidence-audit.json"

            result = subprocess.run(
                self.evidence_audit_command(root, ledger, output),
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            audit = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                audit["gap_class_counts"],
                {"documented_ignored_evidence_hash_drift": 1},
            )
            self.assertEqual(
                audit["successor_readiness"]["status"],
                "not_blocked_by_evidence_audit",
            )
            self.assertEqual(
                audit["successor_readiness"]["nonblocking_gap_class_counts"],
                {"documented_ignored_evidence_hash_drift": 1},
            )
            [mismatch] = audit["mismatched"]
            self.assertTrue(mismatch["decision_documented"])
            self.assertEqual(mismatch["actual_sha256"], digest(actual))

    def test_full_status_rejects_undocumented_missing_ignored_record_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self.ledger(root)
            data = json.loads(ledger.read_text(encoding="utf-8"))
            data["records"][0]["evidence"].append(
                {
                    "path": "outputs/transient-run/metrics.json",
                    "sha256": "1" * 64,
                }
            )
            ledger.write_text(json.dumps(data), encoding="utf-8")

            result = subprocess.run(
                self.command(root, ledger, root / "report.json"),
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("path is not a file", result.stderr)

    def test_full_status_accepts_documented_missing_ignored_record_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = self.ledger(root)
            expected_sha256 = "1" * 64
            evidence_path = "outputs/transient-run/metrics.json"
            self.track_decision_doc(
                root,
                "docs/decisions/transient-run-kill.md",
                "\n".join(
                    [
                        "# Transient Run KILL",
                        "",
                        "`KILL`. The disposable metrics output is documented here.",
                        f"- path: `{evidence_path}`",
                        f"- sha256: `{expected_sha256}`",
                        "",
                    ]
                ),
            )
            data = json.loads(ledger.read_text(encoding="utf-8"))
            data["records"][0]["evidence"].append(
                {"path": evidence_path, "sha256": expected_sha256}
            )
            ledger.write_text(json.dumps(data), encoding="utf-8")
            output = root / "report.json"

            subprocess.run(self.command(root, ledger, output), check=True)

            report = json.loads(output.read_text(encoding="utf-8"))
            documented = report["records"][0]["evidence"][-1]
            self.assertEqual(
                documented["status"],
                "documented_missing_ignored_evidence",
            )
            self.assertTrue(documented["decision_documented"])
            self.assertEqual(
                documented["decision_documents"][0]["path"],
                "docs/decisions/transient-run-kill.md",
            )

    def test_full_status_accepts_tracked_code_hash_found_in_git_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.DEVNULL)
            tracked_script = self.artifact(root, "scripts/tool.py", "old\n")
            subprocess.run(["git", "add", "scripts/tool.py"], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Quality Loop Test",
                    "-c",
                    "user.email=quality-loop@example.invalid",
                    "commit",
                    "-m",
                    "record old tool",
                ],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            old_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
            ledger = self.ledger(root)
            data = json.loads(ledger.read_text(encoding="utf-8"))
            data["records"][0]["evidence"].append(tracked_script)
            ledger.write_text(json.dumps(data), encoding="utf-8")
            (root / "scripts" / "tool.py").write_text("new\n", encoding="utf-8")
            output = root / "report.json"

            subprocess.run(self.command(root, ledger, output), check=True)

            report = json.loads(output.read_text(encoding="utf-8"))
            documented = report["records"][0]["evidence"][-1]
            self.assertEqual(documented["status"], "tracked_code_historical_drift")
            self.assertTrue(documented["historical_git_match"])
            self.assertEqual(documented["historical_git_commit"], old_commit)

    def test_evidence_audit_marks_tracked_code_hash_found_in_git_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subprocess.run(["git", "init"], cwd=root, check=True, stdout=subprocess.DEVNULL)
            tracked_script = self.artifact(root, "scripts/tool.py", "old\n")
            subprocess.run(["git", "add", "scripts/tool.py"], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Quality Loop Test",
                    "-c",
                    "user.email=quality-loop@example.invalid",
                    "commit",
                    "-m",
                    "record old tool",
                ],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            old_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
            ledger = self.ledger(root)
            data = json.loads(ledger.read_text(encoding="utf-8"))
            data["records"][0]["evidence"].append(tracked_script)
            ledger.write_text(json.dumps(data), encoding="utf-8")
            (root / "scripts" / "tool.py").write_text("new\n", encoding="utf-8")
            output = root / "evidence-audit.json"

            result = subprocess.run(
                self.evidence_audit_command(root, ledger, output),
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            audit = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(audit["gap_class_counts"], {"tracked_code_historical_drift": 1})
            self.assertEqual(
                audit["gap_class_unique_path_counts"],
                {"tracked_code_historical_drift": 1},
            )
            self.assertEqual(
                audit["successor_readiness"]["status"],
                "not_blocked_by_evidence_audit",
            )
            self.assertEqual(
                audit["successor_readiness"]["blocking_gap_class_counts"],
                {},
            )
            self.assertEqual(
                audit["successor_readiness"]["nonblocking_gap_class_counts"],
                {"tracked_code_historical_drift": 1},
            )
            [mismatch] = audit["mismatched"]
            self.assertEqual(mismatch["path"], "scripts/tool.py")
            self.assertTrue(mismatch["historical_git_match"])
            self.assertEqual(mismatch["historical_git_commit"], old_commit)
            self.assertEqual(mismatch["historical_git_short_commit"], old_commit[:7])


if __name__ == "__main__":
    unittest.main()
