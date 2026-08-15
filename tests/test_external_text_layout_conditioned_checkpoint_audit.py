import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.analysis.audit_external_text_layout_conditioned_monotonic_checkpoint import (
    CheckpointAuditError,
    KILL_OUTCOME,
    classify_patch_summary,
    summarize_history,
    validate_checkpoint_args,
)


class ExternalTextLayoutConditionedCheckpointAuditTest(unittest.TestCase):
    def plan(self) -> dict:
        return {
            "conditioned_input": {
                "layout_root": "outputs/external-text-layout-support-materialization-20260813/pages",
                "rgb_root": "outputs/archive/sign-separated-residual-repair-20260810/train275-frozen-pipeline/pred",
            },
            "trainer": {
                "batch_size": 1,
                "device": "cpu",
                "log_every": 10,
                "lr": 0.0001,
                "max_steps": 80,
                "output_dir": "artifacts/trials/external-text-layout-conditioned-monotonic-v1",
                "patch_index_file": "hardcase_lists/external-text-layout-conditioned-monotonic-train-patches-v1.csv",
                "residual_delta_bound": 0.08,
                "save_every": 0,
                "seed": 42,
                "tile_size": 256,
            },
        }

    def checkpoint(self) -> dict:
        return {
            "args": {
                "batch_size": 1,
                "data_root": "data-links/samples/SCUT-HW5K-mixed-20260729",
                "device": "cpu",
                "input_channels": 5,
                "input_dir": "outputs/archive/sign-separated-residual-repair-20260810/train275-frozen-pipeline/pred",
                "layout_dir": "outputs/external-text-layout-support-materialization-20260813/pages",
                "layout_source": "external_text_occupancy_confidence",
                "log_every": 10,
                "lr": 0.0001,
                "luminance_margin_gray": 2.0,
                "magnitude_weight": 1.0,
                "mask_source": "target_luma_delta",
                "max_steps": 80,
                "model_type": "monotonic_residual_erase",
                "output_dir": "artifacts/trials/external-text-layout-conditioned-monotonic-v1",
                "patch_index_file": "hardcase_lists/external-text-layout-conditioned-monotonic-train-patches-v1.csv",
                "preserve_delta_weight": 1.0,
                "residual_delta_bound": 0.08,
                "save_every": 0,
                "seed": 42,
                "split": "train",
                "support_positive_weight": 1.0,
                "support_preserve_weight": 1.0,
                "tile_size": 256,
                "validation_enabled": False,
            },
            "step": 80,
        }

    def write_history(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    "loss",
                    "support_positive_bce",
                    "support_preserve_bce",
                    "bright_magnitude_l1",
                    "preserve_delta_l1",
                ],
            )
            writer.writeheader()
            for step in range(1, 81):
                writer.writerow(
                    {
                        "step": step,
                        "loss": f"{1.5 - step / 1000:.8f}",
                        "support_positive_bce": "0.69314718",
                        "support_preserve_bce": "0.69000000",
                        "bright_magnitude_l1": "0.06000000",
                        "preserve_delta_l1": "0.00100000",
                    }
                )

    def test_checkpoint_args_are_registered_and_portable(self) -> None:
        validate_checkpoint_args(self.checkpoint(), self.plan())
        bad = self.checkpoint()
        bad["args"]["data_root"] = Path("data")
        with self.assertRaisesRegex(CheckpointAuditError, "Path"):
            validate_checkpoint_args(bad, self.plan())
        bad = self.checkpoint()
        bad["args"]["lr"] = 0.001
        with self.assertRaisesRegex(CheckpointAuditError, "lr"):
            validate_checkpoint_args(bad, self.plan())

    def test_history_summary_requires_registered_final_step(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "history.csv"
            self.write_history(path)
            summary = summarize_history(path, 80)
        self.assertEqual(summary["row_count"], 80)
        self.assertEqual(summary["final_step"], 80)
        self.assertLess(summary["loss_delta"], 0.0)

    def test_zero_gate_patch_summary_kills_before_candidate_inference(self) -> None:
        result = classify_patch_summary(
            {
                "max_delta_gray": 1.4,
                "negative_delta_pixel_count": 0,
                "patch_count": 256,
                "patches_with_gate": 0,
                "positive_gate_ratio": 0.0,
                "preserve_gate_ratio": 0.0,
            }
        )
        self.assertEqual(result["terminal"], "KILL")
        self.assertEqual(result["outcome"], KILL_OUTCOME)

    def test_nonzero_gate_requires_auditor_extension(self) -> None:
        with self.assertRaisesRegex(CheckpointAuditError, "candidate gate"):
            classify_patch_summary(
                {
                    "max_delta_gray": 12.5,
                    "negative_delta_pixel_count": 0,
                    "patch_count": 256,
                    "patches_with_gate": 1,
                    "positive_gate_ratio": 0.0001,
                    "preserve_gate_ratio": 0.0,
                }
            )


if __name__ == "__main__":
    unittest.main()
