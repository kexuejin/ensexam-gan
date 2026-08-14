import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
import yaml

from networks.discriminator import Discriminator
from networks.generator import Generator
from scripts.analysis.validate_universal_sidecar_d4_preflight import (
    D2_CONFIG,
    D4_CONFIG,
    EXPECTED_DIFFERENCES,
    run_preflight,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PreflightTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> Path:
        d2_path = root / D2_CONFIG
        d4_path = root / D4_CONFIG
        d2_path.parent.mkdir(parents=True)
        d2_path.write_bytes((ROOT / D2_CONFIG).read_bytes())
        d4_path.write_bytes((ROOT / D4_CONFIG).read_bytes())

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
        checkpoint = {
            "G_state_dict": generator.state_dict(),
            "D_state_dict": discriminator.state_dict(),
            "epoch": 1,
        }
        baseline_weights = primary / "micro_region_probe_step0001.pth"
        torch.save(checkpoint, baseline_weights)

        ledger = {
            "baseline": {
                "config": {
                    "path": "artifacts/current-primary/config.yaml",
                    "sha256": sha256_file(baseline_config),
                },
                "checkpoint": {
                    "path": "artifacts/current-primary/micro_region_probe_step0001.pth",
                    "sha256": sha256_file(baseline_weights),
                },
            }
        }
        ledger_path = root / "docs/current-primary-quality-loop-ledger.json"
        ledger_path.parent.mkdir(parents=True)
        ledger_path.write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return ledger_path

    def read_config(self, root: Path) -> dict:
        return yaml.safe_load((root / D4_CONFIG).read_text(encoding="utf-8"))

    def write_config(self, root: Path, config: dict) -> None:
        (root / D4_CONFIG).write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

    def test_d4_semantic_delta_is_exactly_preregistered(self) -> None:
        d2 = yaml.safe_load((ROOT / D2_CONFIG).read_text(encoding="utf-8"))
        d4 = yaml.safe_load((ROOT / D4_CONFIG).read_text(encoding="utf-8"))

        def flatten(value, prefix=""):
            if not isinstance(value, dict):
                return {prefix: value}
            result = {}
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                result.update(flatten(child, path))
            return result

        left, right = flatten(d2), flatten(d4)
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

    def test_extra_config_delta_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            d4 = self.read_config(root)
            d4["train"]["lr"] = 1.0e-4
            self.write_config(root, d4)
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("semantic differences", result["reason"])

    def test_forbidden_d3_field_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            d4 = self.read_config(root)
            d4["data"]["cached_baseline_tail_dir"] = "./artifacts/caches/forbidden"
            self.write_config(root, d4)
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("must be absent", result["reason"])

    def test_existing_save_dir_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            d4 = self.read_config(root)
            save_dir = root / d4["train"]["save_dir"]
            save_dir.mkdir(parents=True)
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("save_dir must not exist", result["reason"])

    def test_enabled_later_gate_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            d4 = self.read_config(root)
            d4["evaluation"]["final_test_mode"] = "paired"
            self.write_config(root, d4)
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("final test gate is enabled", result["reason"])

    def test_wrong_direction_mode_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            d4 = self.read_config(root)
            d4["model"]["universal_residual_adapter_sidecar"][
                "residual_parameterization"
            ] = "free_rgb"
            self.write_config(root, d4)
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("semantic differences", result["reason"])

    def test_frozen_artifact_hash_failure_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = self.make_fixture(root)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["baseline"]["checkpoint"]["sha256"] = "0" * 64
            ledger_path.write_text(
                json.dumps(ledger, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result = run_preflight(repo_root=root, ledger_path=ledger_path)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertIn("weights hash mismatch", result["reason"])


if __name__ == "__main__":
    unittest.main()
