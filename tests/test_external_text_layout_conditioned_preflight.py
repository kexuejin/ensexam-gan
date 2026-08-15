import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from scripts.analysis.validate_external_text_layout_conditioned_preflight import (
    LEDGER_PATH,
    PLAN_PATH,
    PreflightError,
    assert_exact_plan,
    conditioned_loss_terms,
    run_preflight,
    run_synthetic_preflight,
    stack_conditioned_input,
    validate_outputs_absent,
)
from scripts.infer.monotonic_residual_erase import (
    MonotonicResidualEraseCleanupNet,
)


ROOT = Path(__file__).resolve().parents[1]


class ExternalTextLayoutConditionedPreflightTest(unittest.TestCase):
    def write_json(self, path: Path, value: dict) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def registered_plan(self) -> dict:
        return json.loads((ROOT / PLAN_PATH).read_text(encoding="utf-8"))

    def registered_ledger(self) -> dict:
        return json.loads((ROOT / LEDGER_PATH).read_text(encoding="utf-8"))

    def test_registered_plan_contract_is_exact(self) -> None:
        assert_exact_plan(self.registered_plan())
        with self.assertRaisesRegex(PreflightError, "support requirements"):
            plan = self.registered_plan()
            plan["support_diagnostic_requirements"]["full_mean_fold_auc_min"] = 0.0
            assert_exact_plan(plan)
        with self.assertRaisesRegex(PreflightError, "planned outputs"):
            plan = self.registered_plan()
            plan["planned_outputs_must_be_absent"]["training_output_dir"] = (
                "artifacts/trials/other"
            )
            assert_exact_plan(plan)

    def test_conditioned_input_validation_fails_closed(self) -> None:
        rgb = torch.full((1, 3, 4, 4), 0.5)
        occupancy = torch.zeros((1, 1, 4, 4))
        confidence = torch.zeros((1, 1, 4, 4))
        stacked = stack_conditioned_input(rgb, occupancy, confidence)
        self.assertEqual(stacked.shape, (1, 5, 4, 4))

        with self.assertRaisesRegex(ValueError, "binary"):
            stack_conditioned_input(rgb, occupancy + 0.5, confidence)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            stack_conditioned_input(rgb, occupancy, confidence + 1.5)
        with self.assertRaisesRegex(ValueError, "height and width"):
            stack_conditioned_input(
                rgb,
                torch.zeros((1, 1, 2, 4)),
                confidence,
            )

    def test_synthetic_preflight_uses_layout_channels_and_rgb_output_only(self) -> None:
        result = run_synthetic_preflight(self.registered_plan())
        self.assertTrue(result["identity_exact"])
        self.assertEqual(result["output_channels"], 3)
        self.assertGreater(result["layout_encoder_gradient_abs"], 0.0)
        self.assertGreaterEqual(result["delta_min"], 0.0)

    def test_loss_rejects_model_feature_channel_drift(self) -> None:
        model = MonotonicResidualEraseCleanupNet(input_channels=5)
        features = torch.zeros((1, 3, 4, 4))
        target_rgb = torch.zeros((1, 3, 4, 4))
        with self.assertRaisesRegex(ValueError, "channel count"):
            conditioned_loss_terms(
                model,
                features,
                target_rgb,
                luminance_margin_gray=2.0,
            )

    def test_registered_preflight_passes_without_opening_quality_surface(self) -> None:
        result = run_preflight(repo_root=ROOT)
        self.assertEqual(result["terminal"], "PASS", result)
        self.assertTrue(result["runnable"])
        self.assertEqual(result["authority"]["support_diagnostic_status"], "passed")
        self.assertFalse(result["training_started"])
        self.assertFalse(result["checkpoint_generated"])
        self.assertFalse(result["candidate_inference_started"])
        self.assertFalse(result["quality_gate_started"])
        self.assertFalse(result["promotion_enabled"])
        self.assertFalse(result["target_decode"])
        self.assertFalse(result["real_image_decode"])

    def test_support_authority_must_remain_passed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger = self.registered_ledger()
            prerequisite = next(
                item
                for item in ledger["active_iteration"]["prerequisites"]
                if item["id"] == "external_text_layout_support_train_only_diagnostic"
            )
            prerequisite["status"] = "pending"
            ledger_path = root / "ledger.json"
            self.write_json(ledger_path, ledger)

            result = run_preflight(repo_root=ROOT, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("support diagnostic", result["reason"])

    def test_existing_planned_output_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            existing = root / "existing-output"
            existing.mkdir()
            plan = {
                "planned_outputs_must_be_absent": {
                    "training_output_dir": "existing-output"
                }
            }
            with self.assertRaisesRegex(PreflightError, "must be absent"):
                validate_outputs_absent(root, plan)


if __name__ == "__main__":
    unittest.main()
