import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from scripts.analysis.audit_source_output_support_separation import (
    ABLATION_CHANNELS,
    CHANNELS,
    AuditError,
    build_page,
    evaluate_fold,
    validate_authority,
    validate_plan,
)
from scripts.analysis.audit_dual_input_support_separation import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def write_rgb(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(values, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write test image: {path}")


class SourceOutputSupportPrerequisiteTest(unittest.TestCase):
    def registered(self) -> tuple[dict, dict]:
        plan = json.loads(
            (ROOT / "docs/source-output-support-prerequisite-v1.json").read_text(
                encoding="utf-8"
            )
        )
        ledger = json.loads(
            (ROOT / "docs/current-primary-quality-loop-ledger.json").read_text(
                encoding="utf-8"
            )
        )
        return plan, ledger

    def test_completed_diagnostic_closes_repeat_execution(self) -> None:
        plan, ledger = self.registered()
        validate_plan(plan)
        with self.assertRaisesRegex(AuditError, "diagnostic is not pending"):
            validate_authority(ledger)
        self.assertEqual(plan["representation"]["channels"], list(CHANNELS))
        self.assertEqual(
            plan["diagnostic"]["ablation_features"], list(ABLATION_CHANNELS)
        )
        self.assertFalse(plan["authorization"]["model_training"])
        self.assertFalse(plan["authorization"]["candidate_inference"])
        self.assertEqual(ledger["program"]["product_default"], "artifacts/current-primary")

    def test_page_features_are_only_raw_source_and_second_stage_rgb(self) -> None:
        source = np.asarray(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )
        second = np.asarray(
            [
                [[20, 30, 40], [50, 60, 70]],
                [[80, 90, 100], [110, 120, 130]],
            ],
            dtype=np.uint8,
        )
        target = second.copy()
        target[0, 0] += 10
        target[1, 0] += 10
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_path = root / "source.png"
            label_dir = root / "labels"
            second_dir = root / "second"
            write_rgb(source_path, source)
            write_rgb(label_dir / "page.png", target)
            write_rgb(second_dir / "page.png", second)
            page = build_page(
                file_name="page.png",
                source_path=source_path,
                label_dir=label_dir,
                second_stage_dir=second_dir,
                primary_row={"image_sha256": sha256_file(source_path)},
                second_stage_row={
                    "base_edit_threshold": "12",
                    "second_delta_threshold": "32",
                    "dark_threshold": "0",
                },
                margin_gray=2.0,
                sample_cap=1,
            )
        self.assertEqual(page["features"].shape, (2, 6))
        positive_source_rows = {tuple(source.reshape(-1, 3)[index]) for index in (0, 2)}
        preserve_source_rows = {tuple(source.reshape(-1, 3)[index]) for index in (1, 3)}
        selected_source = np.rint(page["features"][:, :3] * 255.0).astype(np.uint8)
        self.assertIn(tuple(selected_source[0]), positive_source_rows)
        self.assertIn(tuple(selected_source[1]), preserve_source_rows)
        np.testing.assert_array_equal(page["ablation_features"], page["features"][:, 3:])

    def test_registered_fold_evaluation_covers_full_and_ablation(self) -> None:
        pages = []
        labels = np.asarray([-1, -1, 1, 1], dtype=np.int8)
        source_signal = np.asarray([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
        for fold in range(5):
            for page_index in range(2):
                features = np.column_stack(
                    [
                        source_signal,
                        source_signal * 0.5,
                        source_signal * 0.25,
                        np.zeros((4, 3), dtype=np.float32),
                    ]
                )
                pages.append(
                    {
                        "file": f"fold-{fold}-page-{page_index}.png",
                        "fold": fold,
                        "features": features,
                        "ablation_features": features[:, 3:],
                        "labels": labels,
                        "samples_per_class": 2,
                    }
                )
        result = evaluate_fold(pages, 0, 1.0)
        self.assertEqual(result["test_page_count"], 2)
        self.assertGreater(result["full_auc"], result["ablation_auc"])

    def test_plan_and_authority_drift_fail_closed(self) -> None:
        plan, ledger = self.registered()
        plan["diagnostic"]["lambda"] = 0.1
        with self.assertRaisesRegex(AuditError, "diagnostic field changed"):
            validate_plan(plan)
        for item in ledger["active_iteration"]["prerequisites"]:
            if item["id"] == "source_output_support_train_only_diagnostic":
                item["status"] = "passed"
        with self.assertRaisesRegex(AuditError, "diagnostic is not pending"):
            validate_authority(ledger)


if __name__ == "__main__":
    unittest.main()
