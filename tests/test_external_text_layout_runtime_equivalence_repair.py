from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import scripts.analysis.materialize_external_text_layout_support_train_only as materializer
from scripts.analysis import external_text_layout_transformers_runtime_repair as repair
from scripts.analysis import external_text_layout_tiled_9x9_runtime_repair as tiled_repair


ROOT = Path(__file__).resolve().parents[1]


MODEL_SOURCE = """class PPOCRV6MediumDetNeck:
    def forward(self, backbone_stage_feature_maps, **kwargs):
        channel_adjusted = []
        for i, feature_map in enumerate(backbone_stage_feature_maps):
            hidden_states = self.input_channel_adjustment_convolution[i](feature_map)
            channel_adjusted.append(hidden_states)

        top_down = [None] * self.num_backbone_stages
        top_down[3] = channel_adjusted[3]
        for i in range(self.num_backbone_stages - 2, -1, -1):
            top_down[i] = channel_adjusted[i] + F.interpolate(
                top_down[i + 1], scale_factor=2, mode=self.interpolate_mode
            )

        projected = []
        for i in range(self.num_backbone_stages):
            hidden_states = top_down[i] if i < self.num_backbone_stages - 1 else channel_adjusted[-1]
            hidden_states = self.input_feature_projection_convolution[i](hidden_states)
            projected.append(hidden_states)

        bottom_up = [None] * self.num_backbone_stages
        bottom_up[0] = projected[0]
        for i in range(1, self.num_backbone_stages):
            bottom_up[i] = projected[i] + self.path_aggregation_head_convolution[i - 1](bottom_up[i - 1])

        lateral_refined = []
        for i in range(self.num_backbone_stages):
            hidden_states = projected[0] if i == 0 else bottom_up[i]
            hidden_states = self.path_aggregation_lateral_convolution[i](hidden_states)
            lateral_refined.append(hidden_states)

        intraclass_refined = [block(feature) for block, feature in zip(self.intraclass_blocks, lateral_refined)]

        upsampled = []
        for feature, scale_factor in zip(intraclass_refined, self.scale_factor_list):
            if scale_factor > 1:
                hidden_states = F.interpolate(feature, scale_factor=scale_factor, mode=self.interpolate_mode)
            else:
                hidden_states = feature
            upsampled.append(hidden_states)

        upsampled = [
            F.interpolate(feature, scale_factor=scale_factor, mode=self.interpolate_mode)
            if scale_factor > 1
            else feature
            for feature, scale_factor in zip(intraclass_refined, self.scale_factor_list)
        ]

        return torch.cat(upsampled[::-1], dim=1)
"""


class FakeDistribution:
    def __init__(self, root: Path, *, version: str = "5.12.1") -> None:
        self.root = root
        self.version = version

    def locate_file(self, relative: Path) -> Path:
        return self.root / relative


class FakeTensor:
    def __init__(self, label: str) -> None:
        self.label = label

    def __add__(self, other: object) -> "FakeTensor":
        assert isinstance(other, FakeTensor)
        return FakeTensor(f"({self.label}+{other.label})")


class FakeFunctional:
    def __init__(self) -> None:
        self.calls: list[tuple[FakeTensor, int, str]] = []

    def interpolate(
        self, feature: FakeTensor, *, scale_factor: int, mode: str
    ) -> FakeTensor:
        self.calls.append((feature, scale_factor, mode))
        return FakeTensor(f"resize{scale_factor}({feature.label})")


class FakeTorch:
    def __init__(self) -> None:
        self.cat_inputs: list[FakeTensor] = []
        self.cat_dim: int | None = None

    def cat(self, values: list[FakeTensor], *, dim: int) -> tuple[str, ...]:
        self.cat_inputs = list(values)
        self.cat_dim = dim
        return tuple(value.label for value in values)


class RecordingIdentity:
    def __init__(self) -> None:
        self.outputs: list[FakeTensor] = []

    def __call__(self, value: FakeTensor) -> FakeTensor:
        self.outputs.append(value)
        return value


def tagged_layer(tag: str):
    def apply(value: FakeTensor) -> FakeTensor:
        return FakeTensor(f"{tag}({value.label})")

    return apply


def build_fake_module(source_path: Path, source: str) -> ModuleType:
    module = ModuleType(repair.MODEL_MODULE_NAME)
    module.__file__ = str(source_path)
    module.F = FakeFunctional()
    module.torch = FakeTorch()
    exec(compile(source, str(source_path), "exec"), vars(module))
    return module


def configure_neck(neck: object) -> list[RecordingIdentity]:
    neck.num_backbone_stages = 4
    neck.interpolate_mode = "nearest"
    neck.scale_factor_list = [1, 2, 4, 8]
    neck.input_channel_adjustment_convolution = [
        tagged_layer(f"adjust{index}") for index in range(4)
    ]
    neck.input_feature_projection_convolution = [
        tagged_layer(f"project{index}") for index in range(4)
    ]
    neck.path_aggregation_head_convolution = [
        tagged_layer(f"head{index}") for index in range(3)
    ]
    neck.path_aggregation_lateral_convolution = [
        tagged_layer(f"lateral{index}") for index in range(4)
    ]
    blocks = [RecordingIdentity() for _ in range(4)]
    neck.intraclass_blocks = blocks
    return blocks


def run_forward(
    module: ModuleType, forward
) -> tuple[tuple[str, ...], list[int], bool, int | None]:
    functional = module.F
    torch = module.torch
    assert isinstance(functional, FakeFunctional)
    assert isinstance(torch, FakeTorch)
    functional.calls.clear()
    torch.cat_inputs.clear()
    torch.cat_dim = None
    neck = module.PPOCRV6MediumDetNeck()
    blocks = configure_neck(neck)
    result = forward(
        neck,
        [FakeTensor(f"source{index}") for index in range(4)],
    )
    intraclass_outputs = [block.outputs[-1] for block in blocks]
    upsample_scales = [
        scale
        for feature, scale, _mode in functional.calls
        if any(feature is output for output in intraclass_outputs)
    ]
    scale_one_reused = torch.cat_inputs[-1] is intraclass_outputs[0]
    return result, upsample_scales, scale_one_reused, torch.cat_dim


class ExternalTextLayoutRuntimeEquivalenceRepairTest(unittest.TestCase):
    def write_source(self, root: Path, source: str = MODEL_SOURCE) -> Path:
        path = root / repair.MODEL_SOURCE_RELATIVE_PATH
        path.parent.mkdir(parents=True)
        path.write_text(source, encoding="utf-8")
        return path

    def test_fake_feature_outputs_match_and_duplicate_interpolation_is_removed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_path = self.write_source(root)
            source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
            module = build_fake_module(source_path, MODEL_SOURCE)
            original_forward = module.PPOCRV6MediumDetNeck.forward
            original_class_surface = dict(vars(module.PPOCRV6MediumDetNeck))
            original = run_forward(module, original_forward)
            distribution = FakeDistribution(root)

            with mock.patch.object(
                repair, "EXPECTED_MODEL_SOURCE_SHA256", source_sha256
            ):
                applied = repair.apply_runtime_equivalence_repair(
                    _distribution_getter=lambda _name: distribution,
                    _module_loader=lambda _name: module,
                )
                repaired_forward = module.PPOCRV6MediumDetNeck.forward
                repaired = run_forward(module, repaired_forward)
                repeated = repair.apply_runtime_equivalence_repair(
                    _distribution_getter=lambda _name: distribution,
                    _module_loader=lambda _name: module,
                )
                setattr(repaired_forward, repair.PATCH_SOURCE_MARKER, "drifted")
                with self.assertRaisesRegex(
                    repair.RuntimeEquivalenceRepairError,
                    "inconsistent runtime repair markers",
                ):
                    repair.apply_runtime_equivalence_repair(
                        _distribution_getter=lambda _name: distribution,
                        _module_loader=lambda _name: module,
                    )

            self.assertEqual(applied.status, "applied")
            self.assertEqual(repeated.status, "already_applied")
            self.assertIs(module.PPOCRV6MediumDetNeck.forward, repaired_forward)
            self.assertEqual(
                set(vars(module.PPOCRV6MediumDetNeck)), set(original_class_surface)
            )
            for name, value in original_class_surface.items():
                if name != "forward":
                    self.assertIs(vars(module.PPOCRV6MediumDetNeck)[name], value)
            self.assertEqual(original[0], repaired[0])
            self.assertEqual(original[1], [2, 4, 8, 2, 4, 8])
            self.assertEqual(repaired[1], [2, 4, 8])
            self.assertTrue(original[2])
            self.assertTrue(repaired[2])
            self.assertEqual(original[3], 1)
            self.assertEqual(repaired[3], 1)

    def test_preregistered_contract_matches_runtime_repair_constants(self) -> None:
        contract = json.loads(
            (
                ROOT / "docs/external-text-layout-runtime-equivalence-repair-v1.json"
            ).read_text(encoding="utf-8")
        )
        binding = contract["runtime_binding"]
        self.assertEqual(
            binding["transformers_version"], repair.EXPECTED_TRANSFORMERS_VERSION
        )
        self.assertEqual(
            binding["model_source_sha256"], repair.EXPECTED_MODEL_SOURCE_SHA256
        )
        self.assertEqual(binding["module"], repair.MODEL_MODULE_NAME)
        self.assertEqual(binding["class"], repair.NECK_CLASS_NAME)
        self.assertEqual(contract["repair"]["id"], repair.REPAIR_ID)
        self.assertTrue(contract["repair"]["in_memory_only"])
        self.assertFalse(contract["repair"]["site_packages_write"])
        self.assertFalse(contract["verification_contract"]["model_execution"])
        original = contract["original_preregistration"]
        original_path = ROOT / original["path"]
        self.assertEqual(
            hashlib.sha256(original_path.read_bytes()).hexdigest(), original["sha256"]
        )

    def test_version_drift_fails_before_source_or_module_access(self) -> None:
        loader = mock.Mock(side_effect=AssertionError("module import must not run"))
        with tempfile.TemporaryDirectory() as raw:
            distribution = FakeDistribution(Path(raw), version="5.12.2")
            with self.assertRaisesRegex(
                repair.RuntimeEquivalenceRepairError,
                "Transformers version changed",
            ):
                repair.apply_runtime_equivalence_repair(
                    _distribution_getter=lambda _name: distribution,
                    _module_loader=loader,
                )
        loader.assert_not_called()

    def test_source_hash_drift_fails_before_module_import(self) -> None:
        loader = mock.Mock(side_effect=AssertionError("module import must not run"))
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.write_source(root)
            distribution = FakeDistribution(root)
            with self.assertRaisesRegex(
                repair.RuntimeEquivalenceRepairError,
                "model source changed",
            ):
                repair.apply_runtime_equivalence_repair(
                    _distribution_getter=lambda _name: distribution,
                    _module_loader=loader,
                )
        loader.assert_not_called()

    def test_ast_drift_fails_before_module_import(self) -> None:
        changed = MODEL_SOURCE.replace(
            (
                "hidden_states = F.interpolate(feature, scale_factor=scale_factor, "
                "mode=self.interpolate_mode)"
            ),
            (
                "hidden_states = F.changed(feature, scale_factor=scale_factor, "
                "mode=self.interpolate_mode)"
            ),
            1,
        )
        loader = mock.Mock(side_effect=AssertionError("module import must not run"))
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_path = self.write_source(root, changed)
            source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
            distribution = FakeDistribution(root)
            with (
                mock.patch.object(
                    repair, "EXPECTED_MODEL_SOURCE_SHA256", source_sha256
                ),
                self.assertRaisesRegex(
                    repair.RuntimeEquivalenceRepairError,
                    "AST no longer matches",
                ),
            ):
                repair.apply_runtime_equivalence_repair(
                    _distribution_getter=lambda _name: distribution,
                    _module_loader=loader,
                )
        loader.assert_not_called()

    def test_detector_applies_repair_before_paddleocr_construction(self) -> None:
        events: list[str] = []
        paddleocr = ModuleType("paddleocr")

        class FakeTextDetection:
            def __init__(self, **kwargs) -> None:
                events.append("construct")
                self.kwargs = kwargs

        paddleocr.TextDetection = FakeTextDetection

        def apply_repair():
            events.append("repair")

        spec = {
            "device": "cpu",
            "engine": "transformers",
            "model_dir": "/tmp/frozen-model",
            "model_name": "PP-OCRv6_medium_det",
        }
        with (
            mock.patch.object(
                materializer,
                "apply_tiled_9x9_runtime_repair",
                side_effect=apply_repair,
            ),
            mock.patch.dict(sys.modules, {"paddleocr": paddleocr}),
        ):
            detector = materializer.create_detector(spec)

        self.assertEqual(events, ["repair", "construct"])
        self.assertEqual(detector.kwargs, spec)

    def test_detector_does_not_import_paddleocr_when_repair_fails(self) -> None:
        paddleocr = ModuleType("paddleocr")
        paddleocr.TextDetection = mock.Mock(
            side_effect=AssertionError("detector construction must not run")
        )
        with (
            mock.patch.object(
                materializer,
                "apply_tiled_9x9_runtime_repair",
                side_effect=tiled_repair.Tiled9x9RuntimeRepairError("source drift"),
            ),
            mock.patch.dict(sys.modules, {"paddleocr": paddleocr}),
            self.assertRaisesRegex(
                materializer.MaterializationError,
                "tiled 9x9 runtime repair failed",
            ),
        ):
            materializer.create_detector(
                {
                    "device": "cpu",
                    "engine": "transformers",
                    "model_dir": "/tmp/frozen-model",
                    "model_name": "PP-OCRv6_medium_det",
                }
            )
        paddleocr.TextDetection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
