import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.analysis.audit_stroke_only_patch_suppression_inputs import (
    DEFAULT_REQUIRED_BUCKET,
    DEFAULT_REQUIRED_CANDIDATE,
    PLAN_PATH,
    SELECTOR_REPLAY_PATH,
    PreflightError,
    assert_exact_plan,
    run_audit,
    select_authorized_rows,
    validate_selector_replay_alignment,
    validate_outputs_absent,
)


ROOT = Path(__file__).resolve().parents[1]


class StrokeOnlyPatchSuppressionInputAuditTest(unittest.TestCase):
    def registered_plan(self) -> dict:
        return json.loads((ROOT / PLAN_PATH).read_text(encoding="utf-8"))

    def test_registered_plan_contract_is_exact(self) -> None:
        plan = self.registered_plan()
        assert_exact_plan(plan)
        plan["authorization"]["scut115"] = True
        with self.assertRaisesRegex(PreflightError, "authorization"):
            assert_exact_plan(plan)

    def test_select_authorized_rows_fails_closed_on_validation_split(self) -> None:
        rows = [
            {
                "split": "scut115",
                "file": "17.jpg",
                "bucket": DEFAULT_REQUIRED_BUCKET,
                "candidate": DEFAULT_REQUIRED_CANDIDATE,
            }
        ]
        with self.assertRaisesRegex(PreflightError, "outside authority"):
            select_authorized_rows(
                rows,
                allowed_splits={"train", "train160"},
                required_bucket=DEFAULT_REQUIRED_BUCKET,
                required_candidate=DEFAULT_REQUIRED_CANDIDATE,
            )

    def test_select_authorized_rows_requires_matching_source_candidate(self) -> None:
        with self.assertRaisesRegex(PreflightError, "no stroke-only source rows"):
            select_authorized_rows(
                [{"split": "train160", "file": "166.jpg", "bucket": "other", "candidate": "other"}],
                allowed_splits={"train", "train160"},
                required_bucket=DEFAULT_REQUIRED_BUCKET,
                required_candidate=DEFAULT_REQUIRED_CANDIDATE,
            )

    def test_registered_input_audit_reports_current_missing_prediction_artifacts(self) -> None:
        result = run_audit(repo_root=ROOT)
        self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED", result)
        self.assertEqual(result["selected_split_counts"], {"train160": 3})
        self.assertFalse(result["candidate_inference_started"])
        self.assertFalse(result["target_decode"])
        self.assertEqual(
            result["selector_replay_csv"],
            str(SELECTOR_REPLAY_PATH),
        )
        self.assertEqual(len(result["selector_replay_alignment"]), 3)
        missing_paths = {item["path"] for item in result["missing_required_paths"]}
        present_paths = {item["path"] for item in result["present_required_paths"]}
        baseline_path = "outputs/scut_train160_nonholdout_second_stage_baseline_20260706/pred/166.png"
        self.assertIn(baseline_path, missing_paths | present_paths)
        self.assertIn(
            (
                "outputs/eval_scut_train160_nonholdout_exact129_outside_edit_lam16_interval_relaxed_gate_"
                "20260706/candidate/166.png"
            ),
            missing_paths,
        )

    def test_selector_replay_mismatch_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            selector_replay = root / "selector.csv"
            with selector_replay.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "split",
                        "file",
                        "image_path",
                        "baseline_pred_path",
                        "candidate_pred_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "split": "train160",
                        "file": "166.jpg",
                        "image_path": "source.png",
                        "baseline_pred_path": "baseline.png",
                        "candidate_pred_path": "candidate.png",
                    }
                )
            with self.assertRaisesRegex(PreflightError, "does not match"):
                validate_selector_replay_alignment(
                    root,
                    [
                        {
                            "split": "train160",
                            "file": "166.jpg",
                            "source_input": "source.png",
                            "baseline_pred": "baseline.png",
                            "candidate_pred": "wrong.png",
                        }
                    ],
                    selector_replay_path=selector_replay,
                )

    def test_existing_planned_output_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            existing = root / "preflight-output"
            existing.mkdir()
            plan = {
                "planned_outputs_must_be_absent": {
                    "preflight_output": "preflight-output"
                }
            }
            with self.assertRaisesRegex(PreflightError, "must be absent"):
                validate_outputs_absent(root, plan)

    def test_input_audit_passes_when_train_only_paths_exist(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            plan = self.registered_plan()
            plan["evidence"]["implementation"]["path"] = "impl.py"
            plan["evidence"]["implementation"]["sha256"] = "0" * 64
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")

            ledger = {
                "active_iteration": {
                    "id": "monotonic-residual-erase-support",
                    "terminal": "PREREQUISITE_NEEDED",
                    "prerequisites": [
                        {
                            "id": "stroke_only_patch_suppression_preflight",
                            "status": "pending",
                        }
                    ],
                },
                "records": [
                    {
                        "id": "stroke-only-patch-suppression-preregistration",
                        "terminal": "PREREQUISITE_NEEDED",
                        "outcome": (
                            "non_layout_source_dark_stroke_only_patch_suppression_"
                            "frozen_pending_train_only_preflight"
                        ),
                        "evidence": [],
                    }
                ],
            }
            ledger_path = root / "ledger.json"
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

            paths = {}
            for field in ("source_input", "baseline_pred", "candidate_pred", "target"):
                path = root / f"{field}.png"
                path.write_bytes(b"not decoded by audit")
                paths[field] = path.name
            review_csv = root / "review.csv"
            with review_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "split",
                        "file",
                        "bucket",
                        "candidate",
                        "source_input",
                        "baseline_pred",
                        "candidate_pred",
                        "target",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "split": "train160",
                        "file": "166.jpg",
                        "bucket": DEFAULT_REQUIRED_BUCKET,
                        "candidate": DEFAULT_REQUIRED_CANDIDATE,
                        **paths,
                    }
                )

            selector_replay = root / SELECTOR_REPLAY_PATH
            selector_replay.parent.mkdir(parents=True)
            with selector_replay.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "split",
                        "file",
                        "image_path",
                        "baseline_pred_path",
                        "candidate_pred_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "split": "train160",
                        "file": "166.jpg",
                        "image_path": paths["source_input"],
                        "baseline_pred_path": paths["baseline_pred"],
                        "candidate_pred_path": paths["candidate_pred"],
                    }
                )

            result = run_audit(
                repo_root=root,
                plan_path=plan_path,
                ledger_path=ledger_path,
                review_csv=review_csv,
            )
            self.assertEqual(result["terminal"], "PASS", result)
            self.assertTrue(result["runnable"])
            self.assertEqual(result["missing_required_path_count"], 0)


if __name__ == "__main__":
    unittest.main()
