from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import torch
import torch.nn.functional as F


from scripts.analysis import external_text_layout_tiled_9x9_runtime_repair as repair


ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = """
class PPOCRV6MediumDetNeck:
    def __init__(self, config):
        feature_projection_convolution = nn.Conv2d(
            in_channels=config.neck_out_channels,
            out_channels=config.neck_out_channels // 4,
            kernel_size=9,
            padding=4,
            bias=True,
        )
        self.input_feature_projection_convolution.append(feature_projection_convolution)
        pan_lateral_convolution = nn.Conv2d(
            in_channels=config.neck_out_channels // 4,
            out_channels=config.neck_out_channels // 4,
            kernel_size=9,
            padding=4,
            bias=True,
        )
        self.path_aggregation_lateral_convolution.append(pan_lateral_convolution)

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
    def __init__(self, root: Path, version: str = "5.12.1") -> None:
        self.root = root
        self.version = version

    def locate_file(self, relative: Path) -> Path:
        return self.root / relative


class RecordingFunctional:
    def __init__(self) -> None:
        self.padded_shapes: list[tuple[int, ...]] = []
        self.output_shapes: list[tuple[int, ...]] = []

    def pad(self, *args, **kwargs):
        value = F.pad(*args, **kwargs)
        self.padded_shapes.append(tuple(value.shape))
        return value

    def conv2d(self, *args, **kwargs):
        value = F.conv2d(*args, **kwargs)
        self.output_shapes.append(tuple(value.shape))
        return value


class ExternalTextLayoutTiled9x9RuntimeRepairTest(unittest.TestCase):
    def test_cpu_float32_projection_and_lateral_are_bitwise_equal(self) -> None:
        cases = (
            (256, 64, 9, 13),
            (64, 64, 10, 11),
            (64, 64, 3, 7),
            (64, 64, 17, 19),
        )
        for in_channels, out_channels, height, width in cases:
            with self.subTest(shape=(in_channels, out_channels, height, width)):
                torch.manual_seed(in_channels + out_channels + height + width)
                convolution = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=9,
                    padding=4,
                    bias=True,
                ).eval()
                hidden_states = torch.randn(1, in_channels, height, width)
                with torch.inference_mode():
                    expected = convolution(hidden_states)
                    actual = repair.tiled_conv2d_rows(
                        convolution,
                        hidden_states,
                        functional=F,
                        torch_module=torch,
                    )
                self.assertTrue(torch.equal(expected, actual))
                self.assertEqual(
                    torch.max(torch.abs(expected - actual)).item(), 0.0
                )
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.device, expected.device)
                self.assertEqual(actual.is_contiguous(), expected.is_contiguous())

    def test_tile_calls_never_exceed_four_output_rows(self) -> None:
        torch.manual_seed(1)
        convolution = torch.nn.Conv2d(64, 64, 9, padding=4, bias=True).eval()
        hidden_states = torch.randn(1, 64, 10, 11)
        functional = RecordingFunctional()
        with torch.inference_mode():
            result = repair.tiled_conv2d_rows(
                convolution,
                hidden_states,
                functional=functional,
                torch_module=torch,
            )
        self.assertEqual(result.shape, (1, 64, 10, 11))
        self.assertEqual(
            [shape[-2] for shape in functional.output_shapes], [4, 4, 2]
        )
        self.assertEqual(
            [shape[-2] for shape in functional.padded_shapes], [12, 12, 10]
        )

    def test_gradient_dtype_and_convolution_drift_fail_closed(self) -> None:
        convolution = torch.nn.Conv2d(64, 64, 9, padding=4, bias=True).eval()
        hidden_states = torch.randn(1, 64, 5, 7)
        with self.assertRaisesRegex(
            repair.Tiled9x9RuntimeRepairError, "gradient-disabled"
        ):
            repair.tiled_conv2d_rows(
                convolution,
                hidden_states,
                functional=F,
                torch_module=torch,
            )
        with torch.inference_mode(), self.assertRaisesRegex(
            repair.Tiled9x9RuntimeRepairError, "float32"
        ):
            repair.tiled_conv2d_rows(
                convolution.double(),
                hidden_states.double(),
                functional=F,
                torch_module=torch,
            )
        wrong_stride = torch.nn.Conv2d(
            64, 64, 9, stride=2, padding=4, bias=True
        ).eval()
        with torch.inference_mode(), self.assertRaisesRegex(
            repair.Tiled9x9RuntimeRepairError, "stride changed"
        ):
            repair.tiled_conv2d_rows(
                wrong_stride,
                hidden_states,
                functional=F,
                torch_module=torch,
            )

    def test_combined_ast_removes_duplicate_and_replaces_two_calls(self) -> None:
        repaired, target = repair.analyze_tiled_forward_target(MODEL_SOURCE)
        dump = ast.dump(repaired, include_attributes=False)
        self.assertEqual(dump.count(f"id='{repair.TILED_HELPER_GLOBAL}'"), 2)
        assignments = [
            node
            for node in ast.walk(repaired)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(value, ast.Name) and value.id == "upsampled"
                for value in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        self.assertLess(target.projection_call_line, target.lateral_call_line)

    def test_application_is_idempotent_and_rejects_partial_markers(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_path = root / repair.MODEL_SOURCE_RELATIVE_PATH
            source_path.parent.mkdir(parents=True)
            source_path.write_text(MODEL_SOURCE, encoding="utf-8")
            source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
            module = ModuleType(repair.MODEL_MODULE_NAME)
            module.__file__ = str(source_path)
            module.F = object()
            module.torch = object()
            exec(compile(MODEL_SOURCE, str(source_path), "exec"), vars(module))
            distribution = FakeDistribution(root)

            with mock.patch.object(
                repair, "EXPECTED_MODEL_SOURCE_SHA256", source_sha256
            ):
                applied = repair.apply_tiled_9x9_runtime_repair(
                    _distribution_getter=lambda _name: distribution,
                    _module_loader=lambda _name: module,
                )
                repeated = repair.apply_tiled_9x9_runtime_repair(
                    _distribution_getter=lambda _name: distribution,
                    _module_loader=lambda _name: module,
                )
                forward = module.PPOCRV6MediumDetNeck.forward
                setattr(forward, repair.PATCH_SOURCE_MARKER, "drifted")
                with self.assertRaisesRegex(
                    repair.Tiled9x9RuntimeRepairError,
                    "partial or conflicting repair markers",
                ):
                    repair.apply_tiled_9x9_runtime_repair(
                        _distribution_getter=lambda _name: distribution,
                        _module_loader=lambda _name: module,
                    )

            self.assertEqual(applied.status, "applied")
            self.assertEqual(repeated.status, "already_applied")
            self.assertIn(repair.TILED_HELPER_GLOBAL, vars(module))

    def test_contract_keeps_real_model_execution_closed(self) -> None:
        contract_path = ROOT / repair.CONTRACT_PATH
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(contract_path.read_bytes()).hexdigest(),
            repair.EXPECTED_CONTRACT_SHA256,
        )
        self.assertEqual(contract["candidate"]["tile_output_rows"], 4)
        self.assertFalse(contract["authority"]["model_execution"])
        self.assertFalse(
            contract["verification_contract"]["real_detector_probe_authorized"]
        )
        self.assertTrue(contract["verification_contract"]["fake_feature_only"])


if __name__ == "__main__":
    unittest.main()
