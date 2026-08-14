import copy
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import torch
import yaml

from data.dataset import EnsExamRealDataset
from losses.losses import EnsExamLoss
from scripts.analysis.validate_universal_sidecar_d3_preflight import (
    D2_CONFIG,
    D3_CONFIG,
    EXPECTED_DIFFERENCES,
    run_preflight,
)
from train import validate_cached_baseline_tail_config


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rgb(path: Path, value: int, shape: tuple[int, int] = (8, 8)) -> None:
    rgb = np.full((shape[0], shape[1], 3), value, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def make_dataset_root(root: Path) -> None:
    write_rgb(root / "train/all_images/001.jpg", 220)
    write_rgb(root / "train/all_labels/001.jpg", 220)


class DatasetContractTest(unittest.TestCase):
    def test_cache_disabled_preserves_old_tuple_contract(self) -> None:
        with TemporaryDirectory() as raw:
            data_root = Path(raw) / "data"
            make_dataset_root(data_root)
            dataset = EnsExamRealDataset(
                str(data_root), img_size=8, is_train=True, file_list=["001.jpg"]
            )
            self.assertEqual(len(dataset[0]), 7)

    def test_cache_enabled_appends_aligned_two_channel_tensor(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            data_root = directory / "data"
            cache = directory / "cache"
            make_dataset_root(data_root)
            residual = np.zeros((8, 8), dtype=np.uint8)
            outside = np.zeros((8, 8), dtype=np.uint8)
            residual[:4, :4] = 255
            outside[4:, 4:] = 255
            (cache / "residual_safe").mkdir(parents=True)
            (cache / "outside_safe").mkdir(parents=True)
            self.assertTrue(
                cv2.imwrite(str(cache / "residual_safe/001.png"), residual)
            )
            self.assertTrue(
                cv2.imwrite(str(cache / "outside_safe/001.png"), outside)
            )
            dataset = EnsExamRealDataset(
                str(data_root),
                img_size=8,
                is_train=True,
                file_list=["001.jpg"],
                cached_baseline_tail_dir=str(cache),
            )
            sample = dataset[0]
            self.assertEqual(len(sample), 8)
            self.assertEqual(sample[-1].shape, (2, 8, 8))
            self.assertEqual(float(sample[-1][0].sum()), 16.0)
            self.assertEqual(float(sample[-1][1].sum()), 16.0)

    def test_cache_requires_both_masks_and_rejects_domain_augment(self) -> None:
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            data_root = directory / "data"
            cache = directory / "cache"
            make_dataset_root(data_root)
            (cache / "residual_safe").mkdir(parents=True)
            (cache / "outside_safe").mkdir(parents=True)
            with self.assertRaisesRegex(FileNotFoundError, "Missing cached"):
                EnsExamRealDataset(
                    str(data_root),
                    img_size=8,
                    is_train=True,
                    file_list=["001.jpg"],
                    cached_baseline_tail_dir=str(cache),
                )
            with self.assertRaisesRegex(ValueError, "domain_augment"):
                EnsExamRealDataset(
                    str(data_root),
                    img_size=8,
                    is_train=True,
                    file_list=["001.jpg"],
                    aug_cfg={"domain_augment": {"enabled": True}},
                    cached_baseline_tail_dir=str(cache),
                )


class CachedBaselineTailLossTest(unittest.TestCase):
    def test_loss_distinguishes_events_and_backpropagates(self) -> None:
        iin = torch.zeros((1, 3, 4, 4))
        igt = torch.zeros_like(iin)
        cache = torch.ones((1, 2, 4, 4))
        no_event = torch.zeros_like(iin)
        event = torch.full_like(iin, 13.0 / 127.5, requires_grad=True)
        no_event_loss = EnsExamLoss.cached_baseline_tail_nonregress_loss(
            no_event, iin, igt, cache, 12.0, 12.0, 0.25
        )
        event_loss = EnsExamLoss.cached_baseline_tail_nonregress_loss(
            event, iin, igt, cache, 12.0, 12.0, 0.25
        )
        self.assertLess(float(no_event_loss), 1e-6)
        self.assertGreater(float(event_loss.detach()), 1.9)
        event_loss.backward()
        self.assertIsNotNone(event.grad)
        self.assertGreater(float(event.grad.abs().sum()), 0.0)

    def test_tail_fraction_keeps_worst_sample(self) -> None:
        iin = torch.zeros((2, 3, 4, 4))
        igt = torch.zeros_like(iin)
        cache = torch.zeros((2, 2, 4, 4))
        cache[:, 0] = 1.0
        student = torch.zeros_like(iin)
        student[0] = 0.5
        full = EnsExamLoss.cached_baseline_tail_nonregress_loss(
            student, iin, igt, cache, 12.0, 12.0, 0.25,
            residual_alpha=1.0, overerase_alpha=0.0, tail_fraction=1.0,
        )
        tail = EnsExamLoss.cached_baseline_tail_nonregress_loss(
            student, iin, igt, cache, 12.0, 12.0, 0.25,
            residual_alpha=1.0, overerase_alpha=0.0, tail_fraction=0.5,
        )
        self.assertGreater(float(tail), 0.9)
        self.assertGreater(float(tail), float(full) * 1.5)


class ConfigContractTest(unittest.TestCase):
    def test_cache_and_positive_weight_are_coupled(self) -> None:
        validate_cached_baseline_tail_config({"loss": {}, "data": {}})
        with self.assertRaisesRegex(ValueError, "requires data"):
            validate_cached_baseline_tail_config({
                "loss": {"lambda_cached_baseline_tail_nonregress": 0.2},
                "data": {},
            })
        with self.assertRaisesRegex(ValueError, "requires positive"):
            validate_cached_baseline_tail_config({
                "loss": {}, "data": {"cached_baseline_tail_dir": "cache"}
            })

    def test_d3_semantic_delta_is_exactly_preregistered(self) -> None:
        d2 = yaml.safe_load((ROOT / D2_CONFIG).read_text(encoding="utf-8"))
        d3 = yaml.safe_load((ROOT / D3_CONFIG).read_text(encoding="utf-8"))

        def flatten(value, prefix=""):
            if not isinstance(value, dict):
                return {prefix: value}
            result = {}
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                result.update(flatten(child, path))
            return result

        left, right = flatten(d2), flatten(d3)
        changed = {
            key: right.get(key)
            for key in sorted(set(left) | set(right))
            if left.get(key) != right.get(key)
        }
        self.assertEqual(changed, EXPECTED_DIFFERENCES)


class PreflightTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> dict[str, str | int]:
        d2_path = root / D2_CONFIG
        d3_path = root / D3_CONFIG
        d2_path.parent.mkdir(parents=True)
        d2_path.write_bytes((ROOT / D2_CONFIG).read_bytes())
        d3_path.write_bytes((ROOT / D3_CONFIG).read_bytes())
        d3 = yaml.safe_load(d3_path.read_text(encoding="utf-8"))
        cache = root / d3["data"]["cached_baseline_tail_dir"]
        (cache / "residual_safe").mkdir(parents=True)
        (cache / "outside_safe").mkdir(parents=True)

        source = root / "data/source.jpg"
        label = root / "data/label.jpg"
        write_rgb(source, 220)
        write_rgb(label, 220)
        mask = np.full((8, 8), 255, dtype=np.uint8)
        residual = cache / "residual_safe/001.png"
        outside = cache / "outside_safe/001.png"
        self.assertTrue(cv2.imwrite(str(residual), mask))
        self.assertTrue(cv2.imwrite(str(outside), mask))
        rows = cache / "cache_rows.csv"
        row = {
            "file": "001.jpg",
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "label_path": str(label),
            "label_sha256": sha256_file(label),
            "residual_safe_sha256": sha256_file(residual),
            "outside_safe_sha256": sha256_file(outside),
        }
        with rows.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        rows_hash = sha256_file(rows)

        hardcases = root / "hardcase_lists"
        hardcases.mkdir(parents=True)
        train_manifest = hardcases / "mixed_scut130_hw5k260_20260729.txt"
        train_manifest.write_text("001.jpg\n", encoding="utf-8")
        (hardcases / "scut_train_hard_proxy_inner_val_15_20260726.txt").write_text(
            "other.jpg\n", encoding="utf-8"
        )
        primary = root / "artifacts/current-primary"
        primary.mkdir(parents=True)
        baseline_config = primary / "config.yaml"
        baseline_weights = primary / "micro_region_probe_step0001.pth"
        baseline_config.write_text("model: fixture\n", encoding="utf-8")
        baseline_weights.write_bytes(b"fixture-weights")
        manifest = {
            "protocol": "train_only_cached_baseline_tail_support",
            "sample_count": 1,
            "rows_csv_sha256": rows_hash,
            "train_file_list_sha256": sha256_file(train_manifest),
            "primary_config_sha256": sha256_file(baseline_config),
            "primary_weights_sha256": sha256_file(baseline_weights),
        }
        manifest_path = cache / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "expected_manifest_sha256": sha256_file(manifest_path),
            "expected_rows_sha256": rows_hash,
            "expected_train_manifest_sha256": sha256_file(train_manifest),
            "expected_baseline_config_sha256": sha256_file(baseline_config),
            "expected_baseline_weights_sha256": sha256_file(baseline_weights),
            "expected_count": 1,
        }

    def test_synthetic_exact_cache_passes(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            expected = self.make_fixture(root)
            result = run_preflight(repo_root=root, **expected)
            self.assertEqual(result["terminal"], "PASS", result)
            self.assertTrue(result["runnable"])

    def test_missing_or_tampered_support_never_reports_runnable(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            expected = self.make_fixture(root)
            d3 = yaml.safe_load((root / D3_CONFIG).read_text(encoding="utf-8"))
            cache = root / d3["data"]["cached_baseline_tail_dir"]
            (cache / "outside_safe/001.png").write_bytes(b"tampered")
            result = run_preflight(repo_root=root, **expected)
            self.assertEqual(result["terminal"], "PREREQUISITE_NEEDED")
            self.assertFalse(result["runnable"])


if __name__ == "__main__":
    unittest.main()
