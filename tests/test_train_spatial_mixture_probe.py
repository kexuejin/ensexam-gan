"""Focused CPU tests for the spatial mixture Phase 0 trainer skeleton.

These tests exercise the trainer's owned logic without the not-yet-implemented
plan modules (networks.spatial_reconstruction_mixture / losses.spatial_mixture_losses)
by injecting seams through ``run_probe``, and without requiring MPS.

The single end-to-end test is the CPU one-step smoke whose output directory
contains the mandatory 'cpu-smoke-not-a-model-result' marker and whose result is
explicitly NOT a model result.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train" / "train_spatial_mixture_probe.py"

from scripts.train.train_spatial_mixture_probe import (  # noqa: E402
    FROZEN_BETAS,
    FROZEN_FOLDS,
    FROZEN_LR,
    FROZEN_MAX_STEPS,
    LEARNED_CONTROLS,
    RunSpec,
    Phase0PreconditionError,
    Phase0RunError,
    read_master_manifest,
    run_probe,
    sha256_file,
    split_train_holdout,
)

SMOKE_DIRNAME = "out-cpu-smoke-not-a-model-result"


def _noop_builder(control, base, device):
    raise AssertionError("mixture builder should not be reached")


def _noop_loss(*_args, **_kwargs):
    raise AssertionError("loss should not be reached")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _valid_master(tmp: Path) -> Path:
    folds = {
        "0": ["scut/train/a.jpg"],
        "1": ["scut/train/b.jpg"],
        "2": ["scut/train/c.jpg"],
        "3": ["hw5k/train/d.jpg"],
        "4": ["hw5k/train/e.jpg"],
        "5": ["hw5k/train/f.jpg"],
    }
    master = {
        "schema_version": 1,
        "program": "spatial-mixture-phase0",
        "variant": "v1",
        "train_only": True,
        "prohibited": [
            "inner_val15",
            "dev40",
            "scut115",
            "holdout40",
            "dev232",
            "reserved_blind",
        ],
        "fold_counts": {key: len(value) for key, value in folds.items()},
        "folds": folds,
    }
    path = tmp / "master.json"
    _write_json(path, master)
    return path


def _valid_config(tmp: Path, data_root: str = "unused") -> Path:
    cfg = {
        "model": {
            "coarse_in_channels": 3,
            "refine_in_channels": 7,
            "cbam_reduction": 16,
        },
        "data": {
            "data_root": data_root,
            "img_size": 64,
            "overlap": 0,
            "mask_threshold": 12,
            "augmentation": None,
        },
        "train": {
            "device": "cpu",
            "lr": FROZEN_LR,
            "adam_betas": list(FROZEN_BETAS),
            "batch_size": 1,
            "max_steps_per_epoch": 1,
            "seed": 42,
            "scheduler": {"enabled": False},
        },
        "early_stopping": {"enabled": False},
    }
    path = tmp / "control.yaml"
    _write_yaml(path, cfg)
    return path


class TrainerBudgetValidationTest(unittest.TestCase):
    def base_spec(self, **overrides) -> RunSpec:
        values = {
            "control": "spatial_mixture",
            "fold": 0,
            "seed": 42,
            "config_path": "cfg.yaml",
            "master_manifest_path": "master.json",
            "output_dir": "/tmp/whatever",
            "smoke": False,
        }
        values.update(overrides)
        return RunSpec(**values)

    def test_non_learned_control_rejected(self) -> None:
        with self.assertRaisesRegex(Phase0RunError, "not a learned control"):
            run_probe(self.base_spec(control="baseline"))

    def test_fold_out_of_range_rejected(self) -> None:
        with self.assertRaisesRegex(Phase0RunError, "outside frozen folds"):
            run_probe(self.base_spec(fold=6))

    def test_seed_outside_frozen_family_rejected(self) -> None:
        with self.assertRaisesRegex(Phase0RunError, "frozen seed family"):
            run_probe(self.base_spec(seed=7))

    def test_unfrozen_lr_rejected(self) -> None:
        with self.assertRaisesRegex(Phase0RunError, "violates frozen budget"):
            run_probe(self.base_spec(lr=1e-3))

    def test_unfrozen_betas_rejected(self) -> None:
        with self.assertRaisesRegex(Phase0RunError, "violates frozen budget"):
            run_probe(self.base_spec(betas=(0.9, 0.999)))

    def test_unfrozen_max_steps_rejected(self) -> None:
        with self.assertRaisesRegex(Phase0RunError, "violates frozen budget"):
            run_probe(self.base_spec(max_steps=8))

    def test_unfrozen_batch_size_rejected(self) -> None:
        with self.assertRaisesRegex(Phase0RunError, "violates frozen budget"):
            run_probe(self.base_spec(batch_size=2))

    def test_smoke_requires_marker_output_dir(self) -> None:
        with self.assertRaisesRegex(Phase0RunError, "cpu-smoke-not-a-model-result"):
            run_probe(self.base_spec(smoke=True, output_dir="/tmp/plain-name"))

    def test_mps_frozen_budget_constants(self) -> None:
        self.assertEqual(FROZEN_MAX_STEPS, 640)
        self.assertEqual(FROZEN_LR, 5e-5)
        self.assertEqual(tuple(FROZEN_BETAS), (0.3037, 0.9))
        self.assertEqual(LEARNED_CONTROLS, ("single_head", "uniform_two_expert", "spatial_mixture"))
        self.assertEqual(tuple(FROZEN_FOLDS), (0, 1, 2, 3, 4, 5))


class TrainerFailClosedDeviceTest(unittest.TestCase):
    def test_real_run_requires_mps_no_silent_cpu_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg = _valid_config(tmp)
            master = _valid_master(tmp)
            spec = RunSpec(
                control="single_head",
                fold=0,
                seed=42,
                config_path=str(cfg),
                master_manifest_path=str(master),
                output_dir=str(tmp / "real-run"),
                device="auto",
                smoke=False,
            )
            with self.assertRaisesRegex(
                Phase0PreconditionError, "PREREQUISITE_NEEDED.*no silent CPU fallback"
            ):
                run_probe(
                    spec,
                    mixture_builder=_noop_builder,
                    loss_fn=_noop_loss,
                    mps_probe=lambda: (False, "mps unavailable in test"),
                )

    def test_real_run_rejects_explicit_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg = _valid_config(tmp)
            master = _valid_master(tmp)
            spec = RunSpec(
                control="single_head",
                fold=0,
                seed=42,
                config_path=str(cfg),
                master_manifest_path=str(master),
                output_dir=str(tmp / "real-run"),
                device="cpu",
                smoke=False,
            )
            with self.assertRaisesRegex(
                Phase0RunError, "--device cpu is prohibited"
            ):
                run_probe(
                    spec,
                    mixture_builder=_noop_builder,
                    loss_fn=_noop_loss,
                    mps_probe=lambda: (True, "probe"),
                )


class TrainerManifestCustodyTest(unittest.TestCase):
    def test_valid_master_splits_train_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            master_path = _valid_master(tmp)
            master = read_master_manifest(master_path)
            self.assertTrue(master["train_only"])
            train_ids, holdout_ids = split_train_holdout(master, holdout_fold=0)
            self.assertEqual(len(train_ids), 5)
            self.assertEqual(len(holdout_ids), 1)
            self.assertEqual(set(train_ids) & set(holdout_ids), set())
            self.assertEqual(len(master["folds"]), 6)

    def test_train_only_false_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            master = json.loads(_valid_master(tmp).read_text(encoding="utf-8"))
            master["train_only"] = False
            path = tmp / "bad-master.json"
            _write_json(path, master)
            with self.assertRaisesRegex(Phase0PreconditionError, "train_only"):
                read_master_manifest(path)

    def test_missing_fold_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            master = json.loads(_valid_master(tmp).read_text(encoding="utf-8"))
            del master["folds"]["5"]
            del master["fold_counts"]["5"]
            path = tmp / "bad-master.json"
            _write_json(path, master)
            with self.assertRaisesRegex(Phase0PreconditionError, "missing fold 5"):
                read_master_manifest(path)

    def test_duplicate_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            master = json.loads(_valid_master(tmp).read_text(encoding="utf-8"))
            master["folds"]["1"] = list(master["folds"]["0"])
            master["fold_counts"]["1"] = 1
            path = tmp / "bad-master.json"
            _write_json(path, master)
            with self.assertRaisesRegex(Phase0PreconditionError, "appears in folds"):
                read_master_manifest(path)

    def test_prohibited_token_in_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            master = json.loads(_valid_master(tmp).read_text(encoding="utf-8"))
            master["folds"]["0"] = ["scut/train/inner_val15_leak.jpg"]
            path = tmp / "bad-master.json"
            _write_json(path, master)
            with self.assertRaisesRegex(Phase0PreconditionError, "prohibited surface"):
                read_master_manifest(path)

    def test_non_train_split_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            master = json.loads(_valid_master(tmp).read_text(encoding="utf-8"))
            master["folds"]["0"] = ["scut/test/a.jpg"]
            path = tmp / "bad-master.json"
            _write_json(path, master)
            with self.assertRaisesRegex(Phase0PreconditionError, "must be"):
                read_master_manifest(path)

    def test_fold_count_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            master = json.loads(_valid_master(tmp).read_text(encoding="utf-8"))
            # Two ids in fold 0 while fold_counts still declares 1.
            master["folds"]["0"] = ["scut/train/a.jpg", "hw5k/train/z.jpg"]
            path = tmp / "bad-master.json"
            _write_json(path, master)
            with self.assertRaisesRegex(Phase0PreconditionError, "expected 1"):
                read_master_manifest(path)


class TrainerModulePreconditionTest(unittest.TestCase):
    def test_default_modules_present_proceed_past_module_gate(self) -> None:
        """Now that build_control / compute_spatial_mixture_loss exist, the trainer
        must NOT fail at the module-absence precondition. It proceeds past module
        loading and fails closed at a later, well-defined data precondition."""
        # The plan-owned modules must expose the symbols the trainer consumes.
        from networks.spatial_reconstruction_mixture import build_control  # noqa: F401,E402
        from losses.spatial_mixture_losses import compute_spatial_mixture_loss  # noqa: F401,E402
        self.assertTrue(callable(build_control))
        self.assertTrue(callable(compute_spatial_mixture_loss))

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg = _valid_config(tmp)
            master = _valid_master(tmp)
            spec = RunSpec(
                control="single_head",
                fold=0,
                seed=42,
                config_path=str(cfg),
                master_manifest_path=str(master),
                output_dir=str(tmp / SMOKE_DIRNAME),
                smoke=True,
            )
            with self.assertRaisesRegex(AssertionError, "未找到有效样本"):
                # The empty data-root precondition now rejects before mixed
                # training, proving the module-absence gate is no longer hit.
                run_probe(spec)

    def test_cli_fails_closed_at_data_precondition_when_modules_present(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cfg = _valid_config(tmp)
            master = _valid_master(tmp)
            out = tmp / SMOKE_DIRNAME
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--control",
                    "single_head",
                    "--fold",
                    "0",
                    "--seed",
                    "42",
                    "--config",
                    str(cfg),
                    "--master-manifest",
                    str(master),
                    "--output-dir",
                    str(out),
                    "--smoke",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
            combined = result.stdout + result.stderr
            # Module gate no longer triggers; the empty data root is the first
            # failing precondition now.
            self.assertNotIn("PREREQUISITE_NEEDED", combined)
            self.assertIn("未找到有效样本", combined)


class _FakeMixture(torch.nn.Module):
    """Tiny mixture stand-in honoring the trainer's forward contract."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv2d(27, 4, 3, padding=1, bias=False)
        self.head = torch.nn.Conv2d(4, 3, 1, bias=True)

    def forward(self, x, y0, Ms, Mb, Ic1, feature):
        bundle = torch.cat([x, y0, Ms, Mb, Ic1, feature], dim=1)
        correction = self.head(torch.relu(self.proj(bundle)))
        y = torch.clamp(y0 + 0.05 * torch.tanh(correction), -1.0, 1.0)
        telemetry = {
            "anchor_share": torch.full((), 1 / 3, device=y0.device),
            "gate_entropy": y0.mean().detach() * 0.0,
        }
        return y, telemetry


def _fake_loss(y, y0, source, target, telemetry):
    total = F.mse_loss(y, target)
    return total, {"mse": total.detach()}


class TrainerCpuOneStepSmokeTest(unittest.TestCase):
    def _make_data_root(self, root: Path) -> Path:
        imgs = root / "train" / "all_images"
        labels = root / "train" / "all_labels"
        imgs.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(7)
        for name in ("b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg"):
            source = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
            delta = rng.integers(-40, 40, size=(64, 64, 3))
            target = np.clip(
                source.astype(np.int16) + delta, 0, 255
            ).astype(np.uint8)
            import cv2

            cv2.imwrite(str(imgs / name), source[:, :, ::-1])
            cv2.imwrite(str(labels / name), target[:, :, ::-1])
        return root

    def _make_base_checkpoint(self, tmp: Path) -> Path:
        from networks.generator import Generator

        G = Generator()
        ckpt = tmp / "fake-current-primary.pth"
        torch.save({"G_state_dict": G.state_dict()}, ckpt)
        return ckpt

    def test_cpu_one_step_smoke_is_explicitly_non_result(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            data_root = self._make_data_root(tmp / "data")
            cfg = _valid_config(tmp, data_root=str(data_root))
            master = _valid_master(tmp)
            ckpt = self._make_base_checkpoint(tmp)
            out = tmp / SMOKE_DIRNAME

            spec = RunSpec(
                control="spatial_mixture",
                fold=0,
                seed=42,
                config_path=str(cfg),
                master_manifest_path=str(master),
                output_dir=str(out),
                checkpoint_path=str(ckpt),
                device="auto",
                smoke=True,
            )
            result = run_probe(
                spec,
                mixture_builder=lambda control, base, device: _FakeMixture(),
                loss_fn=_fake_loss,
            )

            self.assertEqual(result["terminal"], "SMOKE")
            self.assertFalse(result["is_model_result"])
            self.assertEqual(result["device"], "cpu")
            self.assertEqual(result["micro_steps"], 1)
            self.assertEqual(result["train_fold_count"], 5)
            self.assertEqual(result["holdout_fold_count"], 1)
            self.assertEqual(result["base_trainable_params"], 0)
            self.assertFalse(result["prohibited_surfaces_accessed"])
            self.assertTrue(result["base_frozen_and_buffers_unchanged"])
            self.assertTrue(result["no_checkpoint_selection"])
            self.assertIn("nonresult", Path(result["final_checkpoint"]).name)

            self.assertTrue((out / "sealed-config.yaml").is_file())
            self.assertTrue((out / "master-manifest.json").is_file())
            self.assertTrue((out / "result.json").is_file())

            import csv as _csv
            import io as _io

            trace_text = (out / "step_trace.csv").read_text(encoding="utf-8")
            rows = list(_csv.reader(_io.StringIO(trace_text)))
            self.assertEqual(len(rows), 2)  # header + one step
            self.assertEqual(rows[1][0], "1")
            self.assertEqual(rows[1][2], "false")

            final = torch.load(
                Path(result["final_checkpoint"]), map_location="cpu", weights_only=False
            )
            self.assertFalse(final["is_model_result"])
            self.assertTrue(final["smoke"])
            self.assertEqual(final["micro_steps"], 1)

            # Optimizer ownership: exactly the fake mixture's trainable params.
            model = _FakeMixture()
            expected = sorted(
                name for name, p in model.named_parameters() if p.requires_grad
            )
            self.assertEqual(sorted(result["optimizer_owned_param_names"]), expected)

            # Sealed-input hashes recorded.
            self.assertEqual(result["master_manifest_sha256"], sha256_file(master))
            self.assertEqual(result["config_sha256"], sha256_file(cfg))


if __name__ == "__main__":
    unittest.main()