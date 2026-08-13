import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from scripts.analysis.audit_dual_input_support_separation import AuditError
from scripts.analysis.audit_independent_hw5k_expert_disagreement_support import (
    ABLATION_CHANNELS,
    CHANNELS,
    build_page,
    evaluate_fold,
    validate_authority as validate_audit_authority,
    validate_plan as validate_audit_plan,
)
from scripts.analysis.materialize_independent_hw5k_expert_outputs_train_only import (
    MaterializationError,
    atomic_write_json,
    derive_eligible_sources,
    materialize,
    primary_command,
    validate_authority as validate_materialization_authority,
    validate_plan as validate_materialization_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = Path(
    "docs/independent-hw5k-expert-disagreement-support-prerequisite-v1.json"
)


def write_rgb(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(values, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write test image: {path}")


class IndependentHw5kExpertSupportPrerequisiteTest(unittest.TestCase):
    def registered(self) -> tuple[dict, dict]:
        plan = json.loads((ROOT / PLAN_PATH).read_text(encoding="utf-8"))
        ledger = json.loads(
            (ROOT / "docs/current-primary-quality-loop-ledger.json").read_text(
                encoding="utf-8"
            )
        )
        return plan, ledger

    def test_registered_plan_keeps_every_product_surface_closed(self) -> None:
        plan, ledger = self.registered()
        validate_materialization_plan(plan)
        validate_materialization_authority(ledger)
        validate_audit_plan(plan)
        validate_audit_authority(ledger)
        self.assertEqual(plan["representation"]["channels"], list(CHANNELS))
        self.assertEqual(
            plan["diagnostic"]["ablation_features"], list(ABLATION_CHANNELS)
        )
        self.assertFalse(plan["authorization"]["model_training"])
        self.assertFalse(plan["authorization"]["candidate_inference"])
        self.assertFalse(plan["authorization"]["scut115"])
        self.assertFalse(plan["authorization"]["holdout40"])
        self.assertEqual(ledger["program"]["product_default"], "artifacts/current-primary")

    def test_registered_population_reproduces_exact_unseen_hw5k_pages(self) -> None:
        plan, _ledger = self.registered()
        manifest = ROOT / plan["data"]["manifest"]["path"]
        exclusion = Path(plan["data"]["exclusion_manifest"]["external_path"])
        sources = derive_eligible_sources(ROOT, plan, manifest, exclusion)
        names = [name for name, _source, _row in sources]
        self.assertEqual(len(names), 123)
        self.assertTrue(all(name.startswith("hw5k_") for name in names))
        self.assertEqual(
            [sum(int.from_bytes(__import__("hashlib").sha256(name.encode()).digest(), "big") % 5 == fold for name in names) for fold in range(5)],
            [28, 22, 20, 24, 29],
        )

    def test_primary_command_is_exact_and_contains_no_routing_surface(self) -> None:
        plan, _ledger = self.registered()
        command = primary_command(
            repo_root=ROOT,
            plan=plan,
            inference_source=ROOT / "scripts/infer/run_primary_full_page.py",
            samples_file=ROOT / "eligible.txt",
            output_dir=ROOT / "outputs/example",
            config=ROOT / "artifacts/current-primary/config.yaml",
            checkpoint=ROOT / "artifacts/current-primary/micro_region_probe_step0001.pth",
        )
        self.assertIn("--skip-label-metrics", command)
        self.assertEqual(command[command.index("--page-overlap") + 1], "32")
        self.assertEqual(command[command.index("--batch-size") + 1], "8")
        self.assertEqual(command[command.index("--copy-input-outside-mask") + 1], "mb")
        self.assertEqual(
            command[command.index("--copy-mask-threshold-auto") + 1],
            "mb_cov8_step",
        )
        self.assertFalse(
            set(command)
            & {
                "--caller",
                "--domain",
                "--route",
                "--routing",
                "--split",
                "--target",
            }
        )

    def test_plan_and_authority_drift_fail_closed(self) -> None:
        plan, ledger = self.registered()
        plan["diagnostic"]["lambda"] = 0.1
        with self.assertRaisesRegex(AuditError, "diagnostic field changed"):
            validate_audit_plan(plan)
        for item in ledger["active_iteration"]["prerequisites"]:
            if item["id"] == "independent_hw5k_expert_support_train_only_diagnostic":
                item["status"] = "passed"
        with self.assertRaisesRegex(
            MaterializationError, "diagnostic is not pending"
        ):
            validate_materialization_authority(ledger)

    def test_manifest_write_is_atomic_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "manifest.json"
            atomic_write_json(path, {"terminal": "PASS"})
            self.assertEqual(json.loads(path.read_text()), {"terminal": "PASS"})
            self.assertFalse(path.with_name(".manifest.json.materializing").exists())
            with self.assertRaisesRegex(MaterializationError, "refusing to overwrite"):
                atomic_write_json(path, {"terminal": "KILL"})

    def test_materializer_stops_before_runner_when_provenance_fails(self) -> None:
        plan, ledger = self.registered()
        runner = mock.Mock()
        with mock.patch(
            "scripts.analysis.materialize_independent_hw5k_expert_outputs_train_only.read_json",
            side_effect=[plan, ledger],
        ), mock.patch(
            "scripts.analysis.materialize_independent_hw5k_expert_outputs_train_only.validate_inputs",
            side_effect=MaterializationError("artifact sha256 changed"),
        ), mock.patch(
            "scripts.analysis.materialize_independent_hw5k_expert_outputs_train_only.cv2.imread"
        ) as decode:
            with self.assertRaisesRegex(MaterializationError, "artifact sha256 changed"):
                materialize(repo_root=ROOT, runner=runner)
        runner.assert_not_called()
        decode.assert_not_called()

    def test_page_features_are_exact_paired_rgb_with_current_only_ablation(self) -> None:
        current = np.asarray(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )
        expert = np.asarray(
            [
                [[110, 120, 130], [140, 150, 160]],
                [[170, 180, 190], [200, 210, 220]],
            ],
            dtype=np.uint8,
        )
        target = current.copy()
        target[0, 0] += 10
        target[1, 0] += 10
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            current_dir = root / "current"
            expert_dir = root / "expert"
            label_dir = root / "labels"
            write_rgb(current_dir / "page.png", current)
            write_rgb(expert_dir / "page.png", expert)
            write_rgb(label_dir / "page.jpg", target)
            page = build_page(
                file_name="page.jpg",
                label_dir=label_dir,
                current_primary_dir=current_dir,
                hw5k_expert_dir=expert_dir,
                expected_height=2,
                expected_width=2,
                margin_gray=2.0,
                sample_cap=1,
            )
        self.assertEqual(page["features"].shape, (2, 6))
        np.testing.assert_array_equal(
            page["ablation_features"], page["features"][:, :3]
        )
        selected_current = np.rint(page["features"][:, :3] * 255.0).astype(np.uint8)
        selected_expert = np.rint(page["features"][:, 3:] * 255.0).astype(np.uint8)
        positive_current = {(10, 20, 30), (70, 80, 90)}
        preserve_current = {(40, 50, 60), (100, 110, 120)}
        self.assertIn(tuple(selected_current[0]), positive_current)
        self.assertIn(tuple(selected_current[1]), preserve_current)
        for current_pixel, expert_pixel in zip(
            selected_current, selected_expert, strict=True
        ):
            np.testing.assert_array_equal(expert_pixel, current_pixel + 100)
        self.assertEqual(page["positive_pixel_count"], 2)

    def test_fold_evaluation_compares_six_channels_to_current_rgb(self) -> None:
        pages = []
        labels = np.asarray([-1, -1, 1, 1], dtype=np.int8)
        expert_signal = np.asarray([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
        for fold in range(5):
            for page_index in range(2):
                features = np.column_stack(
                    [
                        np.zeros((4, 3), dtype=np.float32),
                        expert_signal,
                        expert_signal * 0.5,
                        expert_signal * 0.25,
                    ]
                )
                pages.append(
                    {
                        "file": f"fold-{fold}-page-{page_index}.png",
                        "fold": fold,
                        "features": features,
                        "ablation_features": features[:, :3],
                        "labels": labels,
                        "samples_per_class": 2,
                    }
                )
        result = evaluate_fold(pages, 0, 1.0)
        self.assertEqual(result["test_page_count"], 2)
        self.assertGreater(result["full_auc"], result["ablation_auc"])


if __name__ == "__main__":
    unittest.main()
