import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.analysis.validate_sign_separated_candidate_application_preflight import (
    PLAN_PATH,
    run_preflight,
)
from scripts.infer.run_sign_separated_residual_candidate import (
    apply_candidate_gate,
    build_parser,
    read_sample_paths,
)


ROOT = Path(__file__).resolve().parents[1]


class SignSeparatedCandidateApplicationPreflightTest(unittest.TestCase):
    def test_killed_family_cannot_reopen_application_preflight(self) -> None:
        result = run_preflight(repo_root=ROOT)
        self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED", result)
        self.assertFalse(result["runnable"])
        self.assertEqual(result["reason"], "active iteration changed")

    def test_inference_surface_has_no_legacy_gate_or_label_option(self) -> None:
        parser = build_parser()
        options = set(parser._option_string_actions)
        self.assertNotIn("--base-edit-threshold", options)
        self.assertNotIn("--second-delta-threshold", options)
        self.assertNotIn("--label-dir", options)
        self.assertNotIn("--target-dir", options)

    def test_gate_requires_probability_and_meaningful_delta(self) -> None:
        baseline = np.full((2, 2, 3), 100, dtype=np.uint8)
        candidate = np.full((2, 2, 3), 113, dtype=np.uint8)
        probability = np.array([[0.6, 0.4], [0.6, 0.6]], dtype=np.float32)
        candidate[1, 0] = 111
        merged, gate, _delta = apply_candidate_gate(
            baseline,
            candidate,
            probability,
            edit_probability_threshold=0.5,
            minimum_delta_threshold=12.0,
        )
        self.assertTrue(gate[0, 0])
        self.assertFalse(gate[0, 1])
        self.assertFalse(gate[1, 0])
        self.assertTrue(gate[1, 1])
        self.assertTrue(np.array_equal(merged[0, 1], baseline[0, 1]))

    def test_sample_list_rejects_label_paths(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "samples.txt"
            path.write_text("data/train/all_labels/a.jpg\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "target or label"):
                read_sample_paths(path)

    def test_plan_drift_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "plan.json"
            plan = json.loads((ROOT / PLAN_PATH).read_text(encoding="utf-8"))
            plan["trainer"]["lr"] = 0.001
            path.write_text(json.dumps(plan), encoding="utf-8")
            result = run_preflight(repo_root=ROOT, plan_path=path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("plan hash changed", result["reason"])


if __name__ == "__main__":
    unittest.main()
