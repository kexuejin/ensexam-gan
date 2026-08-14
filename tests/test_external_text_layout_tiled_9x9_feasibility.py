from __future__ import annotations

import unittest


from scripts.analysis import report_external_text_layout_tiled_9x9_feasibility as report


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
        pan_lateral_convolution = nn.Conv2d(
            in_channels=config.neck_out_channels // 4,
            out_channels=config.neck_out_channels // 4,
            kernel_size=9,
            padding=4,
            bias=True,
        )

    def forward(self, hidden_states, i):
        hidden_states = self.input_feature_projection_convolution[i](hidden_states)
        return self.path_aggregation_lateral_convolution[i](hidden_states)
"""


def static_report() -> dict[str, object]:
    return {
        "model": {
            "feature_maps": [
                {"stage": 1, "height": 432, "width": 608},
                {"stage": 2, "height": 216, "width": 304},
                {"stage": 3, "height": 108, "width": 152},
                {"stage": 4, "height": 54, "width": 76},
            ],
            "neck_out_channels": 256,
        }
    }


class ExternalTextLayoutTiled9x9FeasibilityTest(unittest.TestCase):
    def test_frozen_convolution_surface_is_exact_and_drift_fails_closed(self) -> None:
        observations = report.frozen_9x9_convolutions(MODEL_SOURCE)
        for name in ("projection", "lateral"):
            self.assertEqual(observations[name]["kernel_size"], 9)
            self.assertEqual(observations[name]["padding"], 4)
            self.assertEqual(observations[name]["stride"], 1)
            self.assertEqual(observations[name]["groups"], 1)
            self.assertTrue(observations[name]["bias"])

        with self.assertRaisesRegex(
            report.TiledConvolutionFeasibilityError, "bias changed"
        ):
            report.frozen_9x9_convolutions(
                MODEL_SOURCE.replace("bias=True", "bias=False", 1)
            )

    def test_row_tiles_cover_each_output_once_with_exact_halo(self) -> None:
        tiles = report.row_tiles(10, 4)
        self.assertEqual(
            [
                (
                    tile.output_start,
                    tile.output_end,
                    tile.source_start,
                    tile.source_end,
                    tile.pad_top,
                    tile.pad_bottom,
                )
                for tile in tiles
            ],
            [
                (0, 4, 0, 8, 4, 0),
                (4, 8, 0, 10, 0, 2),
                (8, 10, 4, 10, 0, 4),
            ],
        )
        self.assertEqual(
            [row for tile in tiles for row in range(tile.output_start, tile.output_end)],
            list(range(10)),
        )
        for tile in tiles:
            self.assertEqual(
                tile.padded_input_rows - report.KERNEL_SIZE + 1,
                tile.output_rows,
            )

    def test_four_row_tile_reduces_exact_shape_unfold_bounds(self) -> None:
        observations = report.frozen_9x9_convolutions(MODEL_SOURCE)
        result = report.build_report(
            source_sha256=report.EXPECTED_MODEL_SOURCE_SHA256,
            source_observations=observations,
            static_report=static_report(),
            static_report_sha256=report.EXPECTED_STATIC_REPORT_SHA256,
            probe_result={"terminal": "KILL"},
            probe_result_sha256=report.EXPECTED_PROBE_RESULT_SHA256,
        )

        self.assertFalse(result["analysis"]["torch_imported"])
        self.assertFalse(result["analysis"]["model_imported"])
        self.assertFalse(result["analysis"]["model_executed"])
        self.assertFalse(result["feasibility"]["implementation_authorized"])
        self.assertFalse(result["feasibility"]["model_execution_authorized"])
        self.assertEqual(result["feasibility"]["terminal"], "PREREQUISITE_NEEDED")
        self.assertEqual(result["candidate"]["tile_output_rows"], 4)

        projection = result["memory_estimates"]["highest_resolution_projection"]
        self.assertEqual(
            projection["full_spatial_explicit_unfold_equivalent_bytes_float32"],
            21_785_739_264,
        )
        self.assertEqual(
            projection["maximum_tiled_explicit_unfold_equivalent_bytes_float32"],
            201_719_808,
        )
        self.assertEqual(
            projection["maximum_padded_input_tile_bytes_float32"], 7_569_408
        )
        self.assertEqual(projection["unfold_bound_reduction_factor"], 108)

        lateral = result["memory_estimates"]["highest_resolution_lateral"]
        self.assertEqual(
            lateral["full_spatial_explicit_unfold_equivalent_bytes_float32"],
            5_446_434_816,
        )
        self.assertEqual(
            lateral["maximum_tiled_explicit_unfold_equivalent_bytes_float32"],
            50_429_952,
        )
        self.assertEqual(
            result["memory_estimates"]["total_tiled_convolution_calls_per_page"],
            406,
        )

    def test_bound_inputs_fail_closed(self) -> None:
        observations = report.frozen_9x9_convolutions(MODEL_SOURCE)
        with self.assertRaisesRegex(
            report.TiledConvolutionFeasibilityError, "static memory report hash changed"
        ):
            report.build_report(
                source_sha256=report.EXPECTED_MODEL_SOURCE_SHA256,
                source_observations=observations,
                static_report=static_report(),
                static_report_sha256="drifted",
                probe_result={"terminal": "KILL"},
                probe_result_sha256=report.EXPECTED_PROBE_RESULT_SHA256,
            )


if __name__ == "__main__":
    unittest.main()
