import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
import yaml

from networks.discriminator import Discriminator
from networks.generator import Generator
from scripts.analysis.validate_universal_sidecar_d5_preflight import (
    D4_CONFIG,
    D5_CONFIG,
    EXPECTED_DIFFERENCES,
    run_preflight,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class D5RealPreflightTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> Path:
        d4_path = root / D4_CONFIG
        d5_path = root / D5_CONFIG
        d4_path.parent.mkdir(parents=True)
        d4_path.write_bytes((ROOT / D4_CONFIG).read_bytes())
        d5_path.write_bytes((ROOT / D5_CONFIG).read_bytes())

        hardcases = root / "hardcase_lists"
        hardcases.mkdir(parents=True)
        train_manifest = hardcases / "mixed_scut130_hw5k260_20260729.txt"
        train_manifest.write_text("001.jpg\n002.jpg\n", encoding="utf-8")
        inner_val = hardcases / "scut_train_hard_proxy_inner_val_15_20260726.txt"
        inner_val.write_text("other.jpg\n", encoding="utf-8")

        primary = root / "artifacts/current-primary"
        primary.mkdir(parents=True)
        baseline_config = primary / "config.yaml"
        baseline_cfg = {
            "model": {
                "coarse_in_channels": 3,
                "refine_in_channels": 7,
                "cbam_reduction": 16,
            }
        }
        baseline_config.write_text(
            yaml.safe_dump(baseline_cfg, sort_keys=False), encoding="utf-8"
        )
        generator = Generator(cfg=baseline_cfg["model"]).eval()
        discriminator = Discriminator().eval()
        baseline_weights = primary / "micro_region_probe_step0001.pth"
        torch.save(
            {
                "G_state_dict": generator.state_dict(),
                "D_state_dict": discriminator.state_dict(),
                "epoch": 1,
            },
            baseline_weights,
        )

        audit_path = (
            root
            / "outputs/primary-edit-direction-folded-sidecar-preflight-20260809"
            / "audit-final.json"
        )
        audit_path.parent.mkdir(parents=True)
        synthetic_audit = self.valid_synthetic_audit()
        synthetic_audit["current_primary_config_sha256"] = sha256_file(
            baseline_config
        )
        synthetic_audit["current_primary_checkpoint_sha256"] = sha256_file(
            baseline_weights
        )
        audit_path.write_text(
            json.dumps(synthetic_audit, indent=2) + "\n", encoding="utf-8"
        )
        ledger = {
            "baseline": {
                "config": {
                    "path": "artifacts/current-primary/config.yaml",
                    "sha256": sha256_file(baseline_config),
                },
                "checkpoint": {
                    "path": (
                        "artifacts/current-primary/"
                        "micro_region_probe_step0001.pth"
                    ),
                    "sha256": sha256_file(baseline_weights),
                },
            },
            "active_iteration": {
                "id": "universal-sidecar-d5-folded-direction-magnitude",
                "prerequisites": [
                    {
                        "id": "d5_folded_magnitude_synthetic_preflight",
                        "status": "passed",
                    },
                    {"id": "d5_real_preflight", "status": "pending"},
                ],
            },
            "records": [
                {
                    "id": (
                        "universal-sidecar-d5-folded-direction-magnitude-"
                        "synthetic-prerequisite"
                    ),
                    "terminal": "PASS",
                    "outcome": (
                        "synthetic_folded_direction_both_sign_gradient_"
                        "contract_passed"
                    ),
                    "evidence": [
                        {
                            "path": str(audit_path.relative_to(root)),
                            "sha256": sha256_file(audit_path),
                        }
                    ],
                }
            ],
        }
        ledger_path = root / "docs/current-primary-quality-loop-ledger.json"
        ledger_path.parent.mkdir(parents=True)
        ledger_path.write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return ledger_path

    def valid_synthetic_audit(self) -> dict:
        return {
            "terminal": "PASS",
            "mode": "primary_edit_direction_folded",
            "strict_current_primary_load": True,
            "exact_zero_init": True,
            "base_parameter_tensors": 226,
            "frozen_base_parameter_tensors": 226,
            "sidecar_only_missing_key_count": 17,
            "trainable_tensors": 17,
            "positive_negative_fold_equal": True,
            "opposed_channel_count": 0,
            "residual_abs_max": 0.019,
            "residual_bound": 0.02,
            "zero_primary_edit_noop": True,
            "public_interface_unchanged": True,
            "two_step_probes": [
                {
                    "raw_sign": 1,
                    "folded_support_count": 16,
                    "first_projection_gradient_min": 1.0e-6,
                    "second_projection_gradient_min": 1.0e-6,
                    "second_scale_gradient_abs": 1.0e-8,
                    "initial_global_residual_scale": 0.0010000000474974513,
                    "final_global_residual_scale": 0.001001,
                },
                {
                    "raw_sign": -1,
                    "folded_support_count": 16,
                    "first_projection_gradient_min": 1.0e-6,
                    "second_projection_gradient_min": 1.0e-6,
                    "second_scale_gradient_abs": 1.0e-8,
                    "initial_global_residual_scale": 0.0010000000474974513,
                    "final_global_residual_scale": 0.000999,
                },
            ],
        }

    def read_config(self, root: Path) -> dict:
        return yaml.safe_load((root / D5_CONFIG).read_text(encoding="utf-8"))

    def write_config(self, root: Path, config: dict) -> None:
        (root / D5_CONFIG).write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

    def rewrite_audit_and_hash(
        self, root: Path, ledger_path: Path, audit: dict
    ) -> None:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        audit["current_primary_config_sha256"] = ledger["baseline"]["config"][
            "sha256"
        ]
        audit["current_primary_checkpoint_sha256"] = ledger["baseline"][
            "checkpoint"
        ]["sha256"]
        evidence = ledger["records"][0]["evidence"][0]
        audit_path = root / evidence["path"]
        audit_path.write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        evidence["sha256"] = sha256_file(audit_path)
        ledger_path.write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_d5_semantic_delta_is_exactly_preregistered(self) -> None:
        d4 = yaml.safe_load((ROOT / D4_CONFIG).read_text(encoding="utf-8"))
        d5 = yaml.safe_load((ROOT / D5_CONFIG).read_text(encoding="utf-8"))

        def flatten(value, prefix=""):
            if not isinstance(value, dict):
                return {prefix: value}
            result = {}
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                result.update(flatten(child, path))
            return result

        left, right = flatten(d4), flatten(d5)
        changed = {
            key: right.get(key)
            for key in sorted(set(left) | set(right))
            if left.get(key) != right.get(key)
        }
        self.assertEqual(changed, EXPECTED_DIFFERENCES)

    def test_passes_with_synthetic_frozen_baseline(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PASS", result)
            self.assertTrue(result["runnable"])
            self.assertEqual(result["structure_audit"]["trainable_tensors"], 17)
            self.assertEqual(
                result["checkpoint_audit"]["sidecar_missing_keys"], 17
            )

    def test_passed_real_preflight_can_be_reverified(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["active_iteration"]["prerequisites"][1]["status"] = "passed"
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PASS", result)
            self.assertEqual(
                result["synthetic_prerequisite"][
                    "real_preflight_ledger_status"
                ],
                "passed",
            )

    def test_extra_config_delta_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            d5 = self.read_config(root)
            d5["train"]["lr"] = 1.0e-4
            self.write_config(root, d5)
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("semantic differences", result["reason"])

    def test_missing_synthetic_authority_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["records"] = []
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("synthetic prerequisite PASS record", result["reason"])

    def test_synthetic_audit_hash_failure_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["records"][0]["evidence"][0]["sha256"] = "0" * 64
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("evidence hash mismatch", result["reason"])

    def test_dead_synthetic_scale_evidence_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            audit = self.valid_synthetic_audit()
            audit["two_step_probes"][1]["second_scale_gradient_abs"] = 0.0
            self.rewrite_audit_and_hash(root, ledger_path, audit)
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("scale gradient", result["reason"])

    def test_unmoved_synthetic_scale_evidence_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            audit = self.valid_synthetic_audit()
            probe = audit["two_step_probes"][0]
            probe["final_global_residual_scale"] = probe[
                "initial_global_residual_scale"
            ]
            self.rewrite_audit_and_hash(root, ledger_path, audit)
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("global scale did not move", result["reason"])

    def test_train_inner_val_overlap_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            inner_val = (
                root
                / "hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt"
            )
            inner_val.write_text("002.jpg\n", encoding="utf-8")
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("overlaps inner-val15", result["reason"])

    def test_existing_save_dir_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            d5 = self.read_config(root)
            (root / d5["train"]["save_dir"]).mkdir(parents=True)
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("save_dir must not exist", result["reason"])

    def test_enabled_later_gate_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            d5 = self.read_config(root)
            d5["evaluation"]["final_test_mode"] = "paired"
            self.write_config(root, d5)
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("final test gate is enabled", result["reason"])


if __name__ == "__main__":
    unittest.main()
