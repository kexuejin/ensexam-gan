import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from scripts.analysis.audit_sign_separated_residual_checkpoint import (
    AuditError,
    parameter_movement,
    validate_history,
)
from scripts.infer.patch_cleanup_erasemap import (
    SignSeparatedResidualDeltaCleanupNet,
)
from scripts.train.train_sign_separated_residual_probe import LOSS_TERM_NAMES


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = Path(
    "outputs/archive/sign-separated-residual-repair-20260810/checkpoint-audit/audit.json"
)


class SignSeparatedResidualCheckpointAuditTest(unittest.TestCase):
    def test_recorded_checkpoint_is_killed_before_quality_gate(self) -> None:
        result = json.loads((ROOT / AUDIT_PATH).read_text(encoding="utf-8"))
        self.assertEqual(result["terminal"], "KILL")
        self.assertFalse(result["first_quality_gate_started"])
        behavior = result["real_patch_behavior"]
        self.assertEqual(behavior["applied_bright_pixel_count"], 0)
        self.assertGreater(behavior["applied_dark_pixel_count"], 0)
        self.assertEqual(
            behavior["route_argmax_pixel_counts"]["darken"],
            behavior["pixel_count"],
        )
        self.assertIn(
            "no_application_eligible_brighten_pixels",
            behavior["structural_failures"],
        )

    def test_history_requires_exact_five_term_rows(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "history.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["step", *LOSS_TERM_NAMES]
                )
                writer.writeheader()
                writer.writerow(
                    {"step": 1, **{name: 0.0 for name in LOSS_TERM_NAMES}}
                )
            result = validate_history(path, expected_steps=1)
            self.assertEqual(result["row_count"], 1)
            with path.open("a", encoding="utf-8") as handle:
                handle.write("2,nan,0,0,0,0\n")
            with self.assertRaisesRegex(AuditError, "invalid loss"):
                validate_history(path, expected_steps=2)

    def test_identity_initialized_state_is_not_checkpoint_movement(self) -> None:
        torch.manual_seed(42)
        model = SignSeparatedResidualDeltaCleanupNet(0.08)
        with self.assertRaisesRegex(AuditError, "identity-initialized no-op"):
            parameter_movement(model, seed=42)


if __name__ == "__main__":
    unittest.main()
