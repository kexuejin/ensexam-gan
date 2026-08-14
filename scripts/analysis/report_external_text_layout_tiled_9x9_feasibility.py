#!/usr/bin/env python3
"""Assess row-tiled 9x9 neck convolution feasibility without importing Torch."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODEL_SOURCE = Path(
    "/Users/kexuejin/.pyenv/versions/3.13.1/lib/python3.13/site-packages/"
    "transformers/models/pp_ocrv6_medium_det/modeling_pp_ocrv6_medium_det.py"
)
STATIC_REPORT_PATH = Path("docs/external-text-layout-static-memory-risk-20260814.json")
PROBE_RESULT_PATH = Path(
    "docs/external-text-layout-runtime-equivalence-repair-probe-20260814.json"
)
OUTPUT_PATH = Path("docs/external-text-layout-tiled-9x9-feasibility-20260814.json")
EXPECTED_MODEL_SOURCE_SHA256 = (
    "4bb27b16b04056ee00779391a4943efa5b5c2745e9431e4e9aa652423b271210"
)
EXPECTED_STATIC_REPORT_SHA256 = (
    "144bf4b1c55cfa8b4b19ec125b603dfb5daaf0fa02d0f92a9725adf60ab1651f"
)
EXPECTED_PROBE_RESULT_SHA256 = (
    "d56d49f9ac37d127b0a17359af9f2b53a2fbcee0fcff3beabf638487a29ae5b9"
)
FLOAT32_BYTES = 4
KERNEL_SIZE = 9
PADDING = 4
TILE_OUTPUT_ROWS = 4


class TiledConvolutionFeasibilityError(ValueError):
    pass


@dataclass(frozen=True)
class RowTile:
    output_start: int
    output_end: int
    source_start: int
    source_end: int
    pad_top: int
    pad_bottom: int

    @property
    def output_rows(self) -> int:
        return self.output_end - self.output_start

    @property
    def padded_input_rows(self) -> int:
        return self.source_end - self.source_start + self.pad_top + self.pad_bottom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--model-source", type=Path, default=MODEL_SOURCE)
    parser.add_argument("--static-report", type=Path, default=STATIC_REPORT_PATH)
    parser.add_argument("--probe-result", type=Path, default=PROBE_RESULT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TiledConvolutionFeasibilityError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _single_class(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(matches) != 1:
        raise TiledConvolutionFeasibilityError(
            f"expected one frozen class named {name}"
        )
    return matches[0]


def _single_method(node: ast.ClassDef, name: str) -> ast.FunctionDef:
    matches = [
        child
        for child in node.body
        if isinstance(child, ast.FunctionDef) and child.name == name
    ]
    if len(matches) != 1:
        raise TiledConvolutionFeasibilityError(
            f"expected one frozen method named {node.name}.{name}"
        )
    return matches[0]


def _expression_dump(source: str) -> str:
    return ast.dump(ast.parse(source, mode="eval").body, include_attributes=False)


def _keyword_map(call: ast.Call) -> dict[str, ast.expr]:
    if any(keyword.arg is None for keyword in call.keywords):
        raise TiledConvolutionFeasibilityError("9x9 Conv2d gained kwargs expansion")
    return {str(keyword.arg): keyword.value for keyword in call.keywords}


def _conv_assignment(initializer: ast.FunctionDef, name: str) -> ast.Call:
    matches: list[ast.Call] = []
    for node in ast.walk(initializer):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != name:
            continue
        if not isinstance(node.value, ast.Call):
            raise TiledConvolutionFeasibilityError(f"{name} is no longer a call")
        matches.append(node.value)
    if len(matches) != 1:
        raise TiledConvolutionFeasibilityError(
            f"expected one frozen assignment named {name}"
        )
    return matches[0]


def _validate_conv(
    call: ast.Call,
    *,
    name: str,
    expected_in_channels: str,
    expected_out_channels: str,
) -> dict[str, Any]:
    if (
        not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "nn"
        or call.func.attr != "Conv2d"
        or call.args
    ):
        raise TiledConvolutionFeasibilityError(f"{name} is no longer keyword nn.Conv2d")
    keywords = _keyword_map(call)
    expected_keys = {"in_channels", "out_channels", "kernel_size", "padding", "bias"}
    if set(keywords) != expected_keys:
        raise TiledConvolutionFeasibilityError(f"{name} Conv2d surface changed")
    if ast.dump(keywords["in_channels"], include_attributes=False) != _expression_dump(
        expected_in_channels
    ):
        raise TiledConvolutionFeasibilityError(f"{name} input channels changed")
    if ast.dump(keywords["out_channels"], include_attributes=False) != _expression_dump(
        expected_out_channels
    ):
        raise TiledConvolutionFeasibilityError(f"{name} output channels changed")
    if ast.literal_eval(keywords["kernel_size"]) != KERNEL_SIZE:
        raise TiledConvolutionFeasibilityError(f"{name} kernel changed")
    if ast.literal_eval(keywords["padding"]) != PADDING:
        raise TiledConvolutionFeasibilityError(f"{name} padding changed")
    if ast.literal_eval(keywords["bias"]) is not True:
        raise TiledConvolutionFeasibilityError(f"{name} bias changed")
    return {
        "assignment_line": call.lineno,
        "bias": True,
        "dilation": 1,
        "groups": 1,
        "kernel_size": KERNEL_SIZE,
        "padding": PADDING,
        "padding_mode": "zeros",
        "stride": 1,
    }


def frozen_9x9_convolutions(source: str) -> dict[str, dict[str, Any]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise TiledConvolutionFeasibilityError("frozen model source is invalid") from error
    neck = _single_class(tree, "PPOCRV6MediumDetNeck")
    initializer = _single_method(neck, "__init__")
    forward = _single_method(neck, "forward")
    projection = _validate_conv(
        _conv_assignment(initializer, "feature_projection_convolution"),
        name="feature_projection_convolution",
        expected_in_channels="config.neck_out_channels",
        expected_out_channels="config.neck_out_channels // 4",
    )
    lateral = _validate_conv(
        _conv_assignment(initializer, "pan_lateral_convolution"),
        name="pan_lateral_convolution",
        expected_in_channels="config.neck_out_channels // 4",
        expected_out_channels="config.neck_out_channels // 4",
    )
    forward_source = ast.dump(forward, include_attributes=False)
    for attribute in (
        "input_feature_projection_convolution",
        "path_aggregation_lateral_convolution",
    ):
        if attribute not in forward_source:
            raise TiledConvolutionFeasibilityError(
                f"frozen forward no longer calls {attribute}"
            )
    return {"projection": projection, "lateral": lateral}


def row_tiles(height: int, tile_output_rows: int = TILE_OUTPUT_ROWS) -> list[RowTile]:
    if height <= 0 or tile_output_rows <= 0:
        raise TiledConvolutionFeasibilityError("tile dimensions must be positive")
    tiles: list[RowTile] = []
    for output_start in range(0, height, tile_output_rows):
        output_end = min(height, output_start + tile_output_rows)
        source_start = max(0, output_start - PADDING)
        source_end = min(height, output_end + PADDING)
        tile = RowTile(
            output_start=output_start,
            output_end=output_end,
            source_start=source_start,
            source_end=source_end,
            pad_top=max(0, PADDING - output_start),
            pad_bottom=max(0, output_end + PADDING - height),
        )
        if tile.padded_input_rows - KERNEL_SIZE + 1 != tile.output_rows:
            raise TiledConvolutionFeasibilityError("tile halo does not preserve output rows")
        tiles.append(tile)
    covered = [row for tile in tiles for row in range(tile.output_start, tile.output_end)]
    if covered != list(range(height)):
        raise TiledConvolutionFeasibilityError("tiles do not cover each output row once")
    return tiles


def explicit_unfold_bytes(
    *, in_channels: int, output_rows: int, output_width: int
) -> int:
    return (
        in_channels
        * KERNEL_SIZE
        * KERNEL_SIZE
        * output_rows
        * output_width
        * FLOAT32_BYTES
    )


def convolution_stage_estimate(
    *,
    stage: int,
    height: int,
    width: int,
    in_channels: int,
    out_channels: int,
    tile_output_rows: int,
) -> dict[str, Any]:
    tiles = row_tiles(height, tile_output_rows)
    maximum_rows = max(tile.output_rows for tile in tiles)
    maximum_padded_rows = max(tile.padded_input_rows for tile in tiles)
    full_unfold = explicit_unfold_bytes(
        in_channels=in_channels, output_rows=height, output_width=width
    )
    tiled_unfold = explicit_unfold_bytes(
        in_channels=in_channels,
        output_rows=maximum_rows,
        output_width=width,
    )
    return {
        "full_output_payload_bytes_float32": (
            out_channels * height * width * FLOAT32_BYTES
        ),
        "full_spatial_explicit_unfold_equivalent_bytes_float32": full_unfold,
        "height": height,
        "in_channels": in_channels,
        "maximum_padded_input_tile_bytes_float32": (
            in_channels
            * maximum_padded_rows
            * (width + 2 * PADDING)
            * FLOAT32_BYTES
        ),
        "maximum_tiled_explicit_unfold_equivalent_bytes_float32": tiled_unfold,
        "maximum_tiled_output_payload_bytes_float32": (
            out_channels * maximum_rows * width * FLOAT32_BYTES
        ),
        "out_channels": out_channels,
        "stage": stage,
        "tile_count": len(tiles),
        "unfold_bound_reduction_factor": full_unfold // tiled_unfold,
        "width": width,
    }


def build_report(
    *,
    source_sha256: str,
    source_observations: dict[str, dict[str, Any]],
    static_report: dict[str, Any],
    static_report_sha256: str,
    probe_result: dict[str, Any],
    probe_result_sha256: str,
    tile_output_rows: int = TILE_OUTPUT_ROWS,
) -> dict[str, Any]:
    if source_sha256 != EXPECTED_MODEL_SOURCE_SHA256:
        raise TiledConvolutionFeasibilityError("frozen model source hash changed")
    if static_report_sha256 != EXPECTED_STATIC_REPORT_SHA256:
        raise TiledConvolutionFeasibilityError("static memory report hash changed")
    if probe_result_sha256 != EXPECTED_PROBE_RESULT_SHA256:
        raise TiledConvolutionFeasibilityError("repaired probe result hash changed")
    if probe_result.get("terminal") != "KILL":
        raise TiledConvolutionFeasibilityError("repaired probe terminal changed")
    feature_maps = static_report.get("model", {}).get("feature_maps")
    if not isinstance(feature_maps, list) or len(feature_maps) != 4:
        raise TiledConvolutionFeasibilityError("expected four frozen feature maps")
    neck_channels = int(static_report["model"]["neck_out_channels"])
    projected_channels = neck_channels // 4
    projection = [
        convolution_stage_estimate(
            stage=int(feature["stage"]),
            height=int(feature["height"]),
            width=int(feature["width"]),
            in_channels=neck_channels,
            out_channels=projected_channels,
            tile_output_rows=tile_output_rows,
        )
        for feature in feature_maps
    ]
    lateral = [
        convolution_stage_estimate(
            stage=int(feature["stage"]),
            height=int(feature["height"]),
            width=int(feature["width"]),
            in_channels=projected_channels,
            out_channels=projected_channels,
            tile_output_rows=tile_output_rows,
        )
        for feature in feature_maps
    ]
    highest_projection = projection[0]
    highest_lateral = lateral[0]
    return {
        "analysis": {
            "cv2_imported": False,
            "model_executed": False,
            "model_imported": False,
            "paddle_imported": False,
            "scope": "static_ast_tile_coverage_and_shape_arithmetic_only",
            "torch_imported": False,
        },
        "candidate": {
            "allocation_strategy": "preallocate_full_output_then_copy_each_output_row_tile",
            "horizontal_padding": [PADDING, PADDING],
            "input_or_output_values_changed": False,
            "kernel_bias_stride_dilation_groups_changed": False,
            "numerical_equivalence_status": "requires_preregistered_fake_feature_verification",
            "output_coordinate_coverage": "each_output_row_exactly_once",
            "row_halo": PADDING,
            "tile_output_rows": tile_output_rows,
            "vertical_boundary_padding": "per_tile_zero_padding_matching_original_padding_4",
        },
        "feasibility": {
            "implementation_authorized": False,
            "interpretation": [
                "spatial row tiling preserves the exact receptive-field coordinates for stride-one zero-padded 9x9 convolution",
                "the static explicit-unfold comparison is a bound, not a measured PyTorch allocation",
                "backend algorithm selection can depend on tile shape, so floating-point equivalence is not yet proven",
                "a runtime implementation requires separate preregistration and fake-feature verification before any detector import",
            ],
            "model_execution_authorized": False,
            "reason_code": "tiled_9x9_static_feasibility_supported_equivalence_unproven",
            "terminal": "PREREQUISITE_NEEDED",
        },
        "memory_estimates": {
            "highest_resolution_lateral": highest_lateral,
            "highest_resolution_projection": highest_projection,
            "lateral_all_stages": lateral,
            "projection_all_stages": projection,
            "total_tiled_convolution_calls_per_page": sum(
                item["tile_count"] for item in projection + lateral
            ),
        },
        "probe_result": {
            "path": str(PROBE_RESULT_PATH),
            "sha256": probe_result_sha256,
            "terminal": probe_result["terminal"],
        },
        "schema_version": 1,
        "source": {
            "convolutions": source_observations,
            "model_source_path": str(MODEL_SOURCE),
            "model_source_sha256": source_sha256,
        },
        "static_report": {
            "path": str(STATIC_REPORT_PATH),
            "sha256": static_report_sha256,
        },
        "tile_examples": {
            str(int(feature["stage"])): [
                asdict(tile)
                for tile in (
                    row_tiles(int(feature["height"]), tile_output_rows)[:1]
                    + row_tiles(int(feature["height"]), tile_output_rows)[-1:]
                )
            ]
            for feature in feature_maps
        },
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    model_source = args.model_source.resolve()
    static_report_path = resolve_repo_path(repo_root, args.static_report)
    probe_result_path = resolve_repo_path(repo_root, args.probe_result)
    output_path = resolve_repo_path(repo_root, args.output)
    source = model_source.read_text(encoding="utf-8")
    report = build_report(
        source_sha256=sha256_file(model_source),
        source_observations=frozen_9x9_convolutions(source),
        static_report=read_json(static_report_path),
        static_report_sha256=sha256_file(static_report_path),
        probe_result=read_json(probe_result_path),
        probe_result_sha256=sha256_file(probe_result_path),
    )
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
