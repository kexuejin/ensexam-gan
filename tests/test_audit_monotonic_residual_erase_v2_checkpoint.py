import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.analysis.audit_monotonic_residual_erase_v2_checkpoint import (
    KILL_OUTCOME,
    LEDGER_PATH,
    run_audit,
)


ROOT = Path(__file__).resolve().parents[1]


class MonotonicResidualEraseV2CheckpointAuditTest(unittest.TestCase):
    def write_json(self, path: Path, value: dict) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_registered_checkpoint_is_killed_before_candidate_inference(self) -> None:
        result = run_audit(repo_root=ROOT)
        self.assertEqual(result["terminal"], "KILL", result)
        self.assertEqual(result["outcome"], KILL_OUTCOME)
        summary = result["patch_summary"]
        self.assertEqual(summary["patch_count"], 256)
        self.assertEqual(summary["patches_with_gate"], 0)
        self.assertEqual(summary["positive_gate_ratio"], 0.0)
        self.assertEqual(summary["preserve_gate_ratio"], 0.0)
        self.assertLess(summary["max_delta_gray"], 12.0)
        self.assertGreater(
            summary["preserve_delta_mean_gray"],
            summary["positive_delta_mean_gray"],
        )
        self.assertFalse(result["candidate_inference_started"])
        self.assertFalse(result["quality_gate_started"])
        self.assertFalse(result["promotion_enabled"])

    def test_application_authority_drift_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger = json.loads((ROOT / LEDGER_PATH).read_text(encoding="utf-8"))
            prerequisite = next(
                item
                for item in ledger["active_iteration"]["prerequisites"]
                if item["id"]
                == "monotonic_residual_erase_candidate_application_preflight"
            )
            prerequisite["status"] = "pending"
            path = root / "ledger.json"
            self.write_json(path, ledger)
            result = run_audit(repo_root=ROOT, ledger_path=path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("candidate application prerequisite", result["reason"])


if __name__ == "__main__":
    unittest.main()
