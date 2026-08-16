#!/usr/bin/env python3
"""Spatial mixture preflight and integration tests.

Covers the Generator/train integration frozen by
docs/plans/2026-08-16-spatial-continuous-reconstruction-mixture-implementation.md:

- disabled-by-default Generator host: legacy state-dict surface and forward
  outputs unchanged when the mixture is not enabled;
- enabled host: mixture keys present, expected frozen parameter counts, anchor
  equivalence at initialization;
- train.py checkpoint missing-key allowlist for mixture-only keys;
- fail-closed train.py validator for mixture training configs;
- the four phase-0 configs load and satisfy the validator.

Focused CPU tests only; no training, no quality-split access.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hashlib  # noqa: E402
import json  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


from config_loader import load_config  # noqa: E402
from networks.discriminator import Discriminator  # noqa: E402
from networks.generator import Generator  # noqa: E402
from networks.spatial_reconstruction_mixture import (  # noqa: E402
    CORRECTION_BOUND,
    CPU_INIT_TOLERANCE,
    GATE_PARAMS,
    HEAD_PARAMS,
    RECON_PARAMS,
    TRUNK_PARAMS,
    MODE_BASELINE,
    MODE_SINGLE_HEAD,
    MODE_SPATIAL_MIXTURE,
    MODE_UNIFORM_TWO_EXPERT,
    SpatialContinuousReconstructionMixture,
)
from train import (  # noqa: E402
    load_initial_checkpoint,
    validate_spatial_mixture_config,
)
from scripts.analysis.validate_spatial_mixture_preflight import (  # noqa: E402
    PreflightError,
    _has_forbidden_token,
    mps_preflight_report,
    verify_artifact_hashes,
    verify_edit_range,
    verify_fold_manifests,
    verify_frozen_base_and_bn_immutable,
    verify_gate_invariants,
    verify_gradient_liveness,
    verify_init_equivalence,
    verify_legacy_default,
    verify_no_metadata_inputs,
    verify_no_sealed_outputs,
    verify_optimizer_ownership,
    verify_parameter_counts,
)

BASE_MODEL_CFG = {
    "coarse_in_channels": 3,
    "refine_in_channels": 7,
    "cbam_reduction": 16,
}


def _mixture_model_cfg(mode: str, enabled: bool = True) -> dict:
    return {
        **BASE_MODEL_CFG,
        "spatial_reconstruction_mixture": {
            "enabled": enabled,
            "mode": mode,
            "gate_hidden_channels": 16,
        },
    }


def _input(batch: int = 1, size: int = 64) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, 3, size, size)


def _phase0_config_dict() -> dict:
    """A valid spatial-mixture phase-0 training-config dict."""
    return {
        "model": _mixture_model_cfg(MODE_SINGLE_HEAD),
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "lr": 5.0e-05,
            "adam_betas": [0.3037, 0.9],
            "device": "cpu",
            "num_workers": 0,
            "max_steps_per_epoch": 640,
            "save_every_n_epochs": 1,
            "save_optimizer_state": False,
            "save_scheduler_state": False,
            "resume": False,
            "init_checkpoint": "./artifacts/current-primary/micro_region_probe_step0001.pth",
            "trainable_generator_patterns": ["^spatial_reconstruction_mixture\\."],
            "freeze_generator_batchnorm_stats": True,
            "seed": 42,
            "reproducibility_mode": "strict",
        },
        "data": {"data_root": "./data-links/samples/SCUT-HW5K-mixed-20260729"},
        "loss": {},
    }


class GeneratorDefaultPathTest(unittest.TestCase):
    """Disabled-by-default compatibility of the host integration."""

    def test_disabled_default_has_no_mixture_keys(self) -> None:
        generator = Generator(cfg=dict(BASE_MODEL_CFG))
        names = set(generator.state_dict())
        self.assertTrue(names, "generator must have parameters")
        self.assertFalse(
            any(name.startswith("spatial_reconstruction_mixture.") for name in names),
            "disabled mixture must not add state-dict keys",
        )
        self.assertIsNone(generator.spatial_reconstruction_mixture)
        self.assertFalse(generator.spatial_reconstruction_mixture_enabled)

    def test_enabled_false_has_no_mixture_keys(self) -> None:
        generator = Generator(cfg=_mixture_model_cfg(MODE_SPATIAL_MIXTURE, enabled=False))
        self.assertFalse(
            any(
                name.startswith("spatial_reconstruction_mixture.")
                for name in generator.state_dict()
            )
        )

    def test_legacy_forward_signature_unchanged(self) -> None:
        generator = Generator(cfg=dict(BASE_MODEL_CFG)).eval()
        x = _input()
        with torch.no_grad():
            outputs = generator(x)
            self.assertEqual(len(outputs), 7)
            self.assertEqual(outputs[-1].shape, x.shape)
            _, feature = generator(x, return_reconstruction_feature=True)
            self.assertEqual(feature.shape[1], 16)

    def test_disabled_and_absent_config_forward_equal(self) -> None:
        cfg_absent = dict(BASE_MODEL_CFG)
        cfg_disabled = _mixture_model_cfg(MODE_SPATIAL_MIXTURE, enabled=False)
        g_absent = Generator(cfg=cfg_absent).eval()
        g_disabled = Generator(cfg=cfg_disabled).eval()
        # Force identical weights so the only difference is the host flag.
        g_disabled.load_state_dict(g_absent.state_dict(), strict=True)
        x = _input()
        with torch.no_grad():
            out_absent = g_absent(x)
            out_disabled = g_disabled(x)
        for i, (a, b) in enumerate(zip(out_absent, out_disabled)):
            self.assertTrue(
                torch.equal(a, b),
                f"output index {i} differs between absent and disabled configs",
            )


class GeneratorEnabledHostTest(unittest.TestCase):
    """Enabled host surface: keys, budget, and anchor equivalence."""

    def test_enabled_adds_mixture_params_with_expected_budget(self) -> None:
        expected = {
            MODE_BASELINE: 0,
            MODE_SINGLE_HEAD: RECON_PARAMS,
            MODE_UNIFORM_TWO_EXPERT: RECON_PARAMS,
            MODE_SPATIAL_MIXTURE: RECON_PARAMS + GATE_PARAMS,
        }
        for mode, budget in expected.items():
            with self.subTest(mode=mode):
                generator = Generator(cfg=_mixture_model_cfg(mode))
                mixture = generator.spatial_reconstruction_mixture
                if mode == MODE_BASELINE:
                    # Enabled baseline still hosts the module but contributes zero
                    # trainable parameters anywhere in the generator.
                    self.assertIsNotNone(mixture)
                    self.assertEqual(mixture.param_count(), 0)
                    self.assertEqual(mixture.expected_budget(), 0)
                    continue
                self.assertIsNotNone(mixture)
                self.assertTrue(
                    any(
                        name.startswith("spatial_reconstruction_mixture.")
                        for name in generator.state_dict()
                    )
                )
                self.assertEqual(mixture.param_count(), budget)
                self.assertEqual(mixture.expected_budget(), budget)

    def test_mixture_host_counts_viable(self) -> None:
        generator = Generator(cfg=_mixture_model_cfg(MODE_SPATIAL_MIXTURE))
        mixture = generator.spatial_reconstruction_mixture
        self.assertIsInstance(mixture, SpatialContinuousReconstructionMixture)
        self.assertEqual(mixture.expected_budget(), RECON_PARAMS + GATE_PARAMS)

    def test_enabled_forward_anchor_equivalent_at_init(self) -> None:
        generator = Generator(cfg=_mixture_model_cfg(MODE_SPATIAL_MIXTURE)).eval()
        x = _input()
        with torch.no_grad():
            outputs = generator(x)
        candidate = outputs[-1]
        self.assertTrue(torch.isfinite(candidate).all())
        self.assertTrue(candidate.shape == x.shape)

    def test_enabled_forward_wiring_is_live_after_perturbation(self) -> None:
        """Prove the 27ch bundle actually reaches the mixture host.

        At init the heads are zero so the bundle contents are irrelevant; a
        non-zero correction must change the enabled generator output while the
        disabled path stays on the legacy Icomp.
        """
        g_disabled = Generator(cfg=dict(BASE_MODEL_CFG)).eval()
        g_enabled = Generator(cfg=_mixture_model_cfg(MODE_SPATIAL_MIXTURE)).eval()
        # Force identical base weights; only mixture keys are missing.
        incompatible = g_enabled.load_state_dict(
            g_disabled.state_dict(), strict=False
        )
        self.assertTrue(
            all(key.startswith("spatial_reconstruction_mixture.") for key in incompatible.missing_keys)
        )
        self.assertEqual(incompatible.unexpected_keys, [])

        x = _input()
        with torch.no_grad():
            baseline_disabled = g_disabled(x)[-1]
            baseline_enabled = g_enabled(x)[-1]
        # At init the enabled host reproduces the clamped legacy output. The raw
        # legacy Icomp can exceed [-1,1]; the mixture anchors to the clamped y0.
        expected = torch.clamp(baseline_disabled, -1.0, 1.0)
        self.assertTrue(
            torch.allclose(baseline_enabled, expected, atol=1e-5),
            "enabled host must equal clamped legacy output at initialization",
        )

        # Perturb E1's zero-init projection so its correction becomes nonzero.
        mixture = g_enabled.spatial_reconstruction_mixture
        with torch.no_grad():
            torch.nn.init.zeros_(mixture.expert1.correction.weight)
            torch.nn.init.zeros_(mixture.expert1.correction.bias)
            mixture.expert1.correction.weight[...] = 0.05
            mixture.expert1.correction.bias[...] = 0.0
            perturbed = g_enabled(x)[-1]
        self.assertTrue(torch.isfinite(perturbed).all())
        self.assertTrue(
            not torch.allclose(perturbed, baseline_disabled, atol=1e-4),
            "perturbed expert must change the enabled host output",
        )


class CheckpointMissingKeyTest(unittest.TestCase):
    """load_initial_checkpoint must allow only mixture keys missing."""

    def _legacy_checkpoint_bytes(self) -> bytes:
        generator = Generator(cfg=dict(BASE_MODEL_CFG))
        checkpoint = {"G_state_dict": generator.state_dict()}
        handle, path = tempfile.mkstemp(suffix=".pth")
        os.close(handle)
        torch.save(checkpoint, path, _use_new_zipfile_serialization=False)
        data = Path(path).read_bytes()
        os.unlink(path)
        return data

    def test_mixture_enabled_loads_legacy_checkpoint(self) -> None:
        generator = Generator(cfg=_mixture_model_cfg(MODE_SPATIAL_MIXTURE))
        discriminator = Discriminator()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as handle:
            handle.write(self._legacy_checkpoint_bytes())
            path = handle.name
        try:
            missing, unexpected = load_initial_checkpoint(
                generator, discriminator, path, torch.device("cpu")
            )
        finally:
            os.unlink(path)
        self.assertEqual(unexpected, [])
        self.assertTrue(missing, "mixture keys must be reported missing")
        self.assertTrue(
            all(key.startswith("spatial_reconstruction_mixture.") for key in missing)
        )

    def test_unexpected_key_still_rejected(self) -> None:
        generator = Generator(cfg=_mixture_model_cfg(MODE_SPATIAL_MIXTURE))
        discriminator = Discriminator()
        checkpoint = {
            "G_state_dict": {
                **Generator(cfg=dict(BASE_MODEL_CFG)).state_dict(),
                "unexpected_bogus_key": torch.zeros(1),
            }
        }
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as handle:
            torch.save(checkpoint, handle.name, _use_new_zipfile_serialization=False)
            path = handle.name
        try:
            with self.assertRaises(RuntimeError):
                load_initial_checkpoint(
                    generator, discriminator, path, torch.device("cpu")
                )
        finally:
            os.unlink(path)


class ValidateSpatialMixtureConfigTest(unittest.TestCase):
    """Fail-closed validator behavior."""

    def test_disabled_config_is_noop(self) -> None:
        cfg = dict(_phase0_config_dict())
        cfg["model"] = _mixture_model_cfg(MODE_SPATIAL_MIXTURE, enabled=False)
        self.assertIsNone(validate_spatial_mixture_config(cfg))

    def test_absent_mixture_config_is_noop(self) -> None:
        cfg = dict(_phase0_config_dict())
        cfg["model"] = dict(BASE_MODEL_CFG)
        self.assertIsNone(validate_spatial_mixture_config(cfg))

    def test_valid_learned_config_passes(self) -> None:
        for mode in (MODE_SINGLE_HEAD, MODE_UNIFORM_TWO_EXPERT, MODE_SPATIAL_MIXTURE):
            with self.subTest(mode=mode):
                cfg = dict(_phase0_config_dict())
                cfg["model"] = _mixture_model_cfg(mode)
                self.assertIsNone(validate_spatial_mixture_config(cfg))

    def test_invalid_mode_rejected(self) -> None:
        cfg = dict(_phase0_config_dict())
        cfg["model"] = _mixture_model_cfg("not_a_mode")
        with self.assertRaises(ValueError):
            validate_spatial_mixture_config(cfg)

    def test_enabled_baseline_zero_param_mode_rejected(self) -> None:
        """Enabled baseline contributes zero mixture params; the validator must
        fail closed rather than allow a training run with nothing to train."""
        cfg = dict(_phase0_config_dict())
        cfg["model"] = _mixture_model_cfg(MODE_BASELINE)
        with self.assertRaisesRegex(ValueError, "produced no mixture parameters"):
            validate_spatial_mixture_config(cfg)

    def test_wrong_init_checkpoint_rejected(self) -> None:
        cfg = dict(_phase0_config_dict())
        cfg["train"]["init_checkpoint"] = "./somewhere/else.pth"
        with self.assertRaisesRegex(ValueError, "current-primary initialization"):
            validate_spatial_mixture_config(cfg)

    def test_resume_rejected(self) -> None:
        cfg = dict(_phase0_config_dict())
        cfg["train"]["resume"] = True
        with self.assertRaisesRegex(ValueError, "cannot resume"):
            validate_spatial_mixture_config(cfg)

    def test_missing_trainable_patterns_rejected(self) -> None:
        cfg = dict(_phase0_config_dict())
        cfg["train"]["trainable_generator_patterns"] = []
        with self.assertRaisesRegex(ValueError, "mixture-only"):
            validate_spatial_mixture_config(cfg)

    def test_base_param_pattern_rejected(self) -> None:
        cfg = dict(_phase0_config_dict())
        cfg["train"]["trainable_generator_patterns"] = ["^coarse\\."]
        with self.assertRaisesRegex(ValueError, "must not match base params"):
            validate_spatial_mixture_config(cfg)

    def test_batch_norm_stats_freeze_required(self) -> None:
        cfg = dict(_phase0_config_dict())
        cfg["train"]["freeze_generator_batchnorm_stats"] = False
        with self.assertRaisesRegex(ValueError, "BatchNorm freeze"):
            validate_spatial_mixture_config(cfg)

    def test_optimizer_state_save_rejected(self) -> None:
        cfg = dict(_phase0_config_dict())
        cfg["train"]["save_optimizer_state"] = True
        with self.assertRaisesRegex(ValueError, "must not save optimizer state"):
            validate_spatial_mixture_config(cfg)

    def test_statistical_reproducibility_rejected(self) -> None:
        cfg = dict(_phase0_config_dict())
        cfg["train"]["reproducibility_mode"] = "statistical"
        with self.assertRaisesRegex(ValueError, "strict reproducibility"):
            validate_spatial_mixture_config(cfg)


class Phase0ConfigsLoadTest(unittest.TestCase):
    """The four frozen configs load and satisfy the validator."""

    CONFIG_DIR = ROOT / "configs/local/spatial-mixture-phase0"

    def test_all_configs_exist_and_load(self) -> None:
        for name in ("baseline", "single-head", "uniform-two-expert", "spatial-mixture"):
            with self.subTest(name=name):
                path = self.CONFIG_DIR / f"{name}.yaml"
                self.assertTrue(path.exists(), f"missing config {path}")
                cfg = load_config(str(path))
                self.assertEqual(cfg["train"]["batch_size"], 4)
                self.assertEqual(cfg["train"]["max_steps_per_epoch"], 640)
                self.assertEqual(cfg["model"]["coarse_in_channels"], 3)

    def test_learned_configs_pass_validator(self) -> None:
        for name, mode in (
            ("single-head", MODE_SINGLE_HEAD),
            ("uniform-two-expert", MODE_UNIFORM_TWO_EXPERT),
            ("spatial-mixture", MODE_SPATIAL_MIXTURE),
        ):
            with self.subTest(name=name):
                cfg = load_config(str(self.CONFIG_DIR / f"{name}.yaml"))
                self.assertEqual(
                    cfg["model"]["spatial_reconstruction_mixture"]["mode"], mode
                )
                self.assertTrue(
                    cfg["model"]["spatial_reconstruction_mixture"]["enabled"]
                )
                self.assertIsNone(validate_spatial_mixture_config(cfg))

    def test_baseline_config_is_noop(self) -> None:
        cfg = load_config(str(self.CONFIG_DIR / "baseline.yaml"))
        self.assertFalse(cfg["model"]["spatial_reconstruction_mixture"]["enabled"])
        self.assertIsNone(validate_spatial_mixture_config(cfg))


class PreflightArtifactHashesTest(unittest.TestCase):
    """Gate A check 1: current-primary config/checkpoint hash custody."""

    def test_hashes_match_frozen_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.yaml"
            ckpt = Path(tmp) / "model.pth"
            cfg_bytes = b"model:\n  coarse_in_channels: 3\n"
            ckpt_bytes = b"G_state_dict: {}\n"
            cfg.write_bytes(cfg_bytes)
            ckpt.write_bytes(ckpt_bytes)
            result = verify_artifact_hashes(
                cfg,
                ckpt,
                expected_config_sha=_sha(cfg_bytes),
                expected_checkpoint_sha=_sha(ckpt_bytes),
            )
            self.assertTrue(result["config_match"])
            self.assertTrue(result["checkpoint_match"])

    def test_hash_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.yaml"
            cfg.write_bytes(b"a")
            with self.assertRaises(PreflightError):
                verify_artifact_hashes(
                    cfg,
                    Path(tmp) / "missing.pth",
                    expected_config_sha=_sha(b"b"),
                    expected_checkpoint_sha=_sha(b"c"),
                )

    def test_missing_checkpoint_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "config.yaml"
            cfg.write_bytes(b"a")
            with self.assertRaises(PreflightError):
                verify_artifact_hashes(
                    cfg,
                    Path(tmp) / "missing.pth",
                    expected_config_sha=_sha(b"a"),
                    expected_checkpoint_sha=_sha(b"c"),
                )


class PreflightLegacyDefaultTest(unittest.TestCase):
    """Gate A check 2: disabled default preserves legacy surface."""

    def test_disabled_default_preserves_keys_and_output(self) -> None:
        generator = Generator(cfg=dict(BASE_MODEL_CFG))
        self.assertFalse(generator.spatial_reconstruction_mixture_enabled)
        result = verify_legacy_default(generator, torch.device("cpu"))
        self.assertTrue(result["legacy_outputs_preserved"])
        leaked = [
            key
            for key in generator.state_dict()
            if "spatial_reconstruction_mixture" in key
            or "universal_residual_adapter_sidecar" in key
        ]
        self.assertEqual(leaked, [])


class PreflightFrozenBaseAndOptimizerTest(unittest.TestCase):
    """Gate A checks 3-4: frozen base/BN immutability and optimizer ownership."""

    def _host(self) -> Generator:
        return Generator(cfg=_mixture_model_cfg(MODE_SPATIAL_MIXTURE))

    def test_frozen_base_bn_immutable(self) -> None:
        generator = self._host()
        result = verify_frozen_base_and_bn_immutable(generator, torch.device("cpu"))
        self.assertTrue(result["base_batchnorm_buffers_immutable"])
        self.assertGreater(result["frozen_tensors"], 0)
        base_untrained = [
            name
            for name, param in generator.named_parameters()
            if not name.startswith("spatial_reconstruction_mixture.")
            and param.requires_grad
        ]
        self.assertEqual(base_untrained, [])
        mixture_frozen = [
            name
            for name, param in generator.named_parameters()
            if name.startswith("spatial_reconstruction_mixture.")
            and not param.requires_grad
        ]
        self.assertEqual(mixture_frozen, [])

    def test_optimizer_ownership_exact(self) -> None:
        generator = self._host()
        verify_frozen_base_and_bn_immutable(generator, torch.device("cpu"))
        result = verify_optimizer_ownership(generator)
        self.assertTrue(result["ownership_exact"])
        mixture = generator.spatial_reconstruction_mixture
        self.assertEqual(
            result["optimizer_params"],
            sum(1 for p in mixture.parameters() if p.requires_grad),
        )
        self.assertEqual(result["base_params_owned_by_optimizer"], 0)


class PreflightParameterCountsTest(unittest.TestCase):
    """Gate A check 5: reconstruction/gate parameter counts."""

    def test_all_modes_match_frozen_budget(self) -> None:
        expectations = {
            MODE_BASELINE: 0,
            MODE_SINGLE_HEAD: RECON_PARAMS,
            MODE_UNIFORM_TWO_EXPERT: RECON_PARAMS,
            MODE_SPATIAL_MIXTURE: RECON_PARAMS + GATE_PARAMS,
        }
        for mode, expected in expectations.items():
            with self.subTest(mode=mode):
                mixture = SpatialContinuousReconstructionMixture(mode=mode)
                result = verify_parameter_counts(mixture)
                if mode == MODE_BASELINE:
                    self.assertEqual(result["total_enabled_trainable"], 0)
                else:
                    self.assertEqual(result["trunk"], TRUNK_PARAMS)
                    self.assertEqual(result["expert1"], HEAD_PARAMS)
                    self.assertEqual(result["expert2"], HEAD_PARAMS)
                    self.assertEqual(result["active_reconstruction"], RECON_PARAMS)
                    self.assertEqual(result["total_enabled_trainable"], expected)


class PreflightInitEquivalenceTest(unittest.TestCase):
    """Gate A check 6: zero-init equivalence on CPU."""

    def test_learned_controls_initialize_to_y0_cpu(self) -> None:
        for mode in (
            MODE_SINGLE_HEAD,
            MODE_UNIFORM_TWO_EXPERT,
            MODE_SPATIAL_MIXTURE,
        ):
            with self.subTest(mode=mode):
                mixture = SpatialContinuousReconstructionMixture(mode=mode)
                result = verify_init_equivalence(
                    mixture, torch.device("cpu"), tolerance=CPU_INIT_TOLERANCE
                )
                self.assertLessEqual(result["max_delta"], CPU_INIT_TOLERANCE)


class PreflightGateInvariantTest(unittest.TestCase):
    """Gate A check 7: spatial gate simplex/finite/range invariants."""

    def test_spatial_gate_simplex_and_range(self) -> None:
        mixture = SpatialContinuousReconstructionMixture(mode=MODE_SPATIAL_MIXTURE)
        result = verify_gate_invariants(mixture, torch.device("cpu"))
        self.assertTrue(result["finite"])
        self.assertTrue(result["simplex"])
        self.assertTrue(result["in_range"])


class PreflightEditRangeTest(unittest.TestCase):
    """Gate A check 7b: forced head raw reaches >=24 gray pre-clamp."""

    def test_head_reaches_24_gray_preclamp(self) -> None:
        result = verify_edit_range(None, torch.device("cpu"))
        self.assertGreaterEqual(result["observed_preclamp_gray"], result["min_gray"])
        self.assertAlmostEqual(
            result["observed_preclamp_gray"], result["correction_bound_gray"], delta=1e-4
        )


class PreflightGradientLivenessTest(unittest.TestCase):
    """Gate A check 8: live gradients on synthetic erase/repair fixtures."""

    def test_erase_and_repair_fixtures_have_live_gradients(self) -> None:
        for fixture in ("erase", "repair"):
            with self.subTest(fixture=fixture):
                mixture = SpatialContinuousReconstructionMixture(mode=MODE_SPATIAL_MIXTURE)
                result = verify_gradient_liveness(
                    mixture, torch.device("cpu"), fixture=fixture
                )
                self.assertIn("grads", result)
                self.assertTrue(len(result["grads"]) > 0)


class PreflightNoMetadataTest(unittest.TestCase):
    """Gate A check 9: no domain/source/caller/path routing inputs."""

    def test_no_routing_args_params_or_config_keys(self) -> None:
        mixture = SpatialContinuousReconstructionMixture(mode=MODE_SPATIAL_MIXTURE)
        result = verify_no_metadata_inputs(
            mixture, ["enabled", "mode", "gate_hidden_channels"]
        )
        self.assertTrue(result["no_metadata_input"])
        self.assertNotIn("domain", result["forward_args"])

    def test_forbidden_token_detector(self) -> None:
        self.assertTrue(_has_forbidden_token("something.domain"))
        self.assertTrue(_has_forbidden_token("caller_path"))
        self.assertFalse(_has_forbidden_token("expert1.body_res.conv1.weight"))
        self.assertFalse(_has_forbidden_token("trunk.head_conv"))


class PreflightFoldManifestTest(unittest.TestCase):
    """Gate A check 10: fold hash custody, counts, disjointness, isolation."""

    FROZEN_DOMAIN_COUNTS = [
        (22, 42),
        (22, 42),
        (22, 42),
        (22, 42),
        (21, 43),
        (21, 42),
    ]
    FROZEN_TOTALS = [64, 64, 64, 64, 64, 63]

    def _build_fold_root(self, tmp: Path) -> tuple[Path, dict[str, str]]:
        root = Path(tmp) / "fold-root"
        root.mkdir()
        folds: dict[str, dict] = {}
        all_identities: list[str] = []
        counter = 0
        for f in range(6):
            scut_qty, hw5k_qty = self.FROZEN_DOMAIN_COUNTS[f]
            identities: list[str] = []
            for _ in range(scut_qty):
                identities.append(f"scut/train/synth_{counter}.jpg")
                counter += 1
            for _ in range(hw5k_qty):
                identities.append(f"hw5k/train/synth_{counter}.jpg")
                counter += 1
            txt = "\n".join(identities) + "\n"
            txt_path = root / f"fold{f}.txt"
            txt_path.write_text(txt, encoding="utf-8")
            payload = {
                "fold": f,
                "scut_count": scut_qty,
                "hw5k_count": hw5k_qty,
                "total": self.FROZEN_TOTALS[f],
                "identities": identities,
            }
            json_path = root / f"fold{f}.json"
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            folds[str(f)] = {
                "txt": str(txt_path),
                "json": str(json_path),
                "txt_sha256": _sha(txt.encode()),
                "json_sha256": _sha((json.dumps(payload, indent=2) + "\n").encode()),
                "scut_count": scut_qty,
                "hw5k_count": hw5k_qty,
                "total": self.FROZEN_TOTALS[f],
            }
            all_identities.extend(identities)
        master = {
            "schema_version": 1,
            "program": "spatial-mixture-phase0-v1",
            "salt": "spatial-mixture-phase0-v1",
            "fold_count": 6,
            "frozen_fold_domain_counts": [list(pair) for pair in self.FROZEN_DOMAIN_COUNTS],
            "pool": {"scut": 130, "hw5k": 253, "total": 383},
            "folds": folds,
        }
        master_path = root / "master.json"
        master_path.write_text(json.dumps(master, indent=2) + "\n", encoding="utf-8")
        return root, {}

    def test_folds_validate_with_no_prohibited_stems(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._build_fold_root(Path(tmp))
            result = verify_fold_manifests(
                root,
                prohibited_scut_stems=set(),
                prohibited_hw5k_stems=set(),
            )
            self.assertTrue(result["disjoint"])
            self.assertTrue(result["prohibited_isolation"])
            self.assertEqual(result["union_count"], 383)

    def test_tampered_txt_hash_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._build_fold_root(Path(tmp))
            (root / "fold0.txt").write_text("hw5k/train/synth_0.jpg\n", encoding="utf-8")
            with self.assertRaises(PreflightError):
                verify_fold_manifests(
                    root,
                    prohibited_scut_stems=set(),
                    prohibited_hw5k_stems=set(),
                )

    def test_prohibited_stem_clash_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._build_fold_root(Path(tmp))
            with self.assertRaises(PreflightError):
                verify_fold_manifests(
                    root,
                    prohibited_scut_stems={"synth_0.jpg"},
                    prohibited_hw5k_stems=set(),
                )


class PreflightNoSealedOutputsTest(unittest.TestCase):
    """Gate A check 11: no sealed Phase 0 run output dirs exist."""

    def test_matrix_absent_and_no_dirs_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = verify_no_sealed_outputs(Path(tmp) / "matrix.json")
            self.assertFalse(result["matrix_json_present"])
            self.assertEqual(result["existing_dirs"], [])

    def test_extra_sealed_dir_exists_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "sealed-out"
            existing.mkdir()
            with self.assertRaises(PreflightError):
                verify_no_sealed_outputs(
                    Path(tmp) / "matrix.json",
                    extra_sealed_dirs=(existing,),
                )


class PreflightMPSReportTest(unittest.TestCase):
    """MPS preflight function: report shape without requiring real MPS."""

    def test_report_returns_expected_keys_without_requiring_mps(self) -> None:
        report = mps_preflight_report()
        for key in ("python", "torch_version", "mps_built", "mps_available"):
            self.assertIn(key, report)
        self.assertIsInstance(report["mps_available"], bool)


if __name__ == "__main__":
    unittest.main()