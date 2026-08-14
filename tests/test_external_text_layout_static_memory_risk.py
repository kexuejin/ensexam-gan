from __future__ import annotations

import unittest


from scripts.analysis import report_external_text_layout_static_memory_risk as report


MODEL_SOURCE = """
class PPOCRV6MediumDetNeck:
    def __init__(self):
        self.project = nn.Conv2d(256, 64, kernel_size=9)
        self.lateral = nn.Conv2d(64, 64, kernel_size=9)

    def forward(self, features):
        upsampled = []
        for feature in features:
            upsampled.append(feature)
        upsampled = [feature for feature in features]
        return upsampled
"""

BACKBONE_CONFIG_SOURCE = """
class PPLCNetV4Config:
    stem_strides: tuple[int, ...] = (2, 1, 1, 2, 1)
"""


def model_config() -> dict[str, object]:
    return {
        "model_type": "pp_ocrv6_medium_det",
        "backbone_config": {
            "stem_type": "large",
            "block_configs": [
                [[3, 128, 128, 1, True], [3, 128, 128, 1, False]],
                [[3, 128, 256, 2, False], [3, 256, 256, 1, False]],
                [[3, 256, 512, 2, False], [3, 512, 512, 1, False]],
                [[3, 512, 896, 2, False], [3, 896, 896, 1, False]],
            ],
        },
        "neck_out_channels": 256,
        "scale_factor_list": [1, 2, 4, 8],
    }


class ExternalTextLayoutStaticMemoryRiskTest(unittest.TestCase):
    def test_min_limit_keeps_large_page_before_rounding(self) -> None:
        self.assertEqual(
            report.resized_dimensions(
                width=2436,
                height=1719,
                limit_side_len=736,
                limit_type="min",
                max_side_limit=4000,
            ),
            (2432, 1728),
        )
        self.assertEqual(
            report.resize_ratio(
                width=2436,
                height=1719,
                limit_side_len=736,
                limit_type="min",
                max_side_limit=4000,
            ),
            1.0,
        )

    def test_source_observations_require_duplicate_upsample_and_9x9_convs(
        self,
    ) -> None:
        observations = report.neck_source_observations(MODEL_SOURCE)
        self.assertEqual(observations["kernel_size_9_convolution_count"], 2)
        self.assertEqual(observations["upsampled_assignment_count"], 2)

    def test_static_report_uses_exact_high_resolution_shapes(self) -> None:
        result = report.build_static_report(
            model_config=model_config(),
            preprocessor_config={
                "limit_side_len": 736,
                "limit_type": "min",
                "max_side_limit": 4000,
            },
            model_source=MODEL_SOURCE,
            backbone_config_source=BACKBONE_CONFIG_SOURCE,
            source_width=2436,
            source_height=1719,
            source_path="raw/hw5k_1011.jpg",
            source_sha256="source",
            plan_path="docs/plan.json",
            plan_sha256="plan",
            artifacts={},
        )

        self.assertFalse(result["analysis"]["model_imported"])
        self.assertFalse(result["analysis"]["model_executed"])
        self.assertEqual(result["preprocessing"]["processed_width"], 2432)
        self.assertEqual(result["preprocessing"]["processed_height"], 1728)
        self.assertFalse(result["preprocessing"]["source_was_downscaled_by_limit"])
        self.assertTrue(
            result["preprocessing"]["multiple_of_32_rounding_changed_dimensions"]
        )
        self.assertEqual(
            [
                (stage["channels"], stage["height"], stage["width"])
                for stage in result["model"]["feature_maps"]
            ],
            [
                (128, 432, 608),
                (256, 216, 304),
                (512, 108, 152),
                (896, 54, 76),
            ],
        )
        estimates = result["memory_estimates"]
        self.assertEqual(
            estimates["single_upsampled_list_payload_bytes_float32"],
            268_959_744,
        )
        self.assertEqual(
            estimates[
                "duplicate_upsample_additional_interpolated_payload_bytes_float32"
            ],
            201_719_808,
        )
        self.assertEqual(
            estimates["duplicate_upsampled_lists_distinct_payload_bytes_float32"],
            470_679_552,
        )
        self.assertEqual(
            estimates["concatenation_input_plus_output_payload_bytes_float32"],
            537_919_488,
        )
        self.assertEqual(
            estimates[
                "high_resolution_9x9_projection_explicit_unfold_equivalent_bytes_float32"
            ],
            21_785_739_264,
        )
        self.assertEqual(result["risk_assessment"]["terminal"], "PREREQUISITE_NEEDED")


if __name__ == "__main__":
    unittest.main()
