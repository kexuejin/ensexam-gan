import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.analysis.audit_reconstruction_stage_disagreement import (
    AuditError,
    evaluate_fold,
    load_stage_npz,
    validate_authority as validate_audit_authority,
    validate_plan as validate_audit_plan,
)
from scripts.analysis.materialize_reconstruction_stage_disagreement_train_only import (
    CHANNELS,
    MaterializationError,
    assert_source_image_path,
    infer_stage_disagreement_full_page,
    read_manifest,
    stage_disagreement_channels,
    validate_authority,
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeStageGenerator(torch.nn.Module):
    def forward(self, source: torch.Tensor):
        batch, _channels, height, width = source.shape
        device = source.device
        zeros = torch.zeros((batch, 3, height, width), device=device)
        ic2 = torch.full((batch, 3, height // 2, width // 2), 0.25, device=device)
        ic4 = torch.full((batch, 3, height // 4, width // 4), -0.25, device=device)
        ire = torch.full_like(zeros, 0.5)
        mask = torch.zeros((batch, 1, height, width), device=device)
        return mask, mask, ic4, ic2, zeros, ire, zeros


class ReconstructionStageDisagreementPrerequisiteTest(unittest.TestCase):
    def registered(self) -> tuple[dict, dict]:
        plan = json.loads(
            (ROOT / "docs/reconstruction-stage-disagreement-prerequisite-v1.json")
            .read_text(encoding="utf-8")
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
        validate_audit_plan(plan)
        with self.assertRaisesRegex(AuditError, "diagnostic is not pending"):
            validate_audit_authority(ledger)
        with self.assertRaisesRegex(
            MaterializationError, "diagnostic is not pending"
        ):
            validate_authority(ledger)
        self.assertEqual(plan["representation"]["channels"], list(CHANNELS))
        self.assertFalse(plan["authorization"]["model_training"])
        self.assertFalse(plan["authorization"]["candidate_inference"])
        self.assertEqual(ledger["program"]["product_default"], "artifacts/current-primary")

    def test_stage_channels_follow_frozen_signed_and_absolute_definitions(self) -> None:
        ic1 = torch.zeros((1, 3, 4, 4))
        ire = torch.ones((1, 3, 4, 4))
        ic2 = torch.full((1, 3, 2, 2), 0.5)
        ic4 = torch.full((1, 3, 1, 1), -0.5)
        channels = stage_disagreement_channels(ic4, ic2, ic1, ire)
        self.assertEqual(set(channels), set(CHANNELS))
        torch.testing.assert_close(
            channels["coarse_refine_signed_luma"],
            torch.full((1, 4, 4), 127.5),
        )
        torch.testing.assert_close(
            channels["coarse_refine_abs_rgb"],
            torch.full((1, 4, 4), 127.5),
        )
        torch.testing.assert_close(
            channels["coarse_ic2_abs_rgb"],
            torch.full((1, 4, 4), 63.75),
        )
        torch.testing.assert_close(
            channels["coarse_ic4_abs_rgb"],
            torch.full((1, 4, 4), 63.75),
        )

    def test_full_page_stage_materialization_is_aligned_and_finite(self) -> None:
        rgb = np.zeros((5, 6, 3), dtype=np.uint8)
        result = infer_stage_disagreement_full_page(
            FakeStageGenerator(),
            rgb,
            torch.device("cpu"),
            patch_size=4,
            overlap=1,
            batch_size=2,
        )
        self.assertEqual(set(result), set(CHANNELS))
        for values in result.values():
            self.assertEqual(values.shape, (5, 6))
            self.assertEqual(values.dtype, np.float32)
            self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(result["coarse_refine_signed_luma"], 63.75)

    def test_npz_loader_requires_exact_float32_channels(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "page.npz"
            values = np.zeros((3, 4), dtype=np.float32)
            np.savez_compressed(path, **{name: values for name in CHANNELS})
            loaded = load_stage_npz(path, expected_shape=(3, 4))
            self.assertEqual(set(loaded), set(CHANNELS))
            np.savez_compressed(path, wrong=values)
            with self.assertRaisesRegex(AuditError, "channels changed"):
                load_stage_npz(path)

    def test_target_paths_and_incomplete_manifest_fail_closed(self) -> None:
        with self.assertRaisesRegex(MaterializationError, "target"):
            assert_source_image_path(Path("data/all_labels/page.jpg"))
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "page.jpg"
            source.write_bytes(b"source")
            manifest = root / "manifest.txt"
            manifest.write_text("page.jpg\n", encoding="utf-8")
            with self.assertRaisesRegex(MaterializationError, "expected 275"):
                read_manifest(root, manifest)

    def test_plan_drift_fails_closed(self) -> None:
        plan, _ledger = self.registered()
        plan["diagnostic"]["lambda"] = 0.1
        with self.assertRaisesRegex(AuditError, "diagnostic field changed"):
            validate_audit_plan(plan)

    def test_registered_fold_evaluation_covers_full_and_ablation(self) -> None:
        pages = []
        labels = np.asarray([-1, -1, 1, 1], dtype=np.int8)
        signal = np.asarray([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
        for fold in range(5):
            for page_index in range(2):
                pages.append(
                    {
                        "file": f"fold-{fold}-page-{page_index}.jpg",
                        "fold": fold,
                        "features": np.column_stack(
                            [signal, signal * 0.5, signal * 0.25, signal * 0.125]
                        ),
                        "ablation_features": np.zeros((4, 3), dtype=np.float32),
                        "labels": labels,
                        "samples_per_class": 2,
                    }
                )
        result = evaluate_fold(pages, 0, 1.0)
        self.assertEqual(result["test_page_count"], 2)
        self.assertGreater(result["full_auc"], result["ablation_auc"])
        self.assertEqual(set(result["stage_channel_strata"]), set(CHANNELS))


if __name__ == "__main__":
    unittest.main()
