#!/usr/bin/env python3
"""Report static PP-OCRv6 detector memory risk without importing the model."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = Path("docs/external-text-layout-support-prerequisite-v1.json")
OUTPUT_PATH = Path("docs/external-text-layout-static-memory-risk-20260814.json")
MODEL_SOURCE = Path(
    "/Users/kexuejin/.pyenv/versions/3.13.1/lib/python3.13/site-packages/"
    "transformers/models/pp_ocrv6_medium_det/modeling_pp_ocrv6_medium_det.py"
)
BACKBONE_CONFIG_SOURCE = Path(
    "/Users/kexuejin/.pyenv/versions/3.13.1/lib/python3.13/site-packages/"
    "transformers/models/pp_lcnet_v4/configuration_pp_lcnet_v4.py"
)
FLOAT32_BYTES = 4
SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--model-source", type=Path, default=MODEL_SOURCE)
    parser.add_argument(
        "--backbone-config-source", type=Path, default=BACKBONE_CONFIG_SOURCE
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected an object in {path}")
    return data


def jpeg_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ValueError(f"source is not a JPEG: {path}")
    offset = 2
    while offset + 3 < len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            break
        segment_length = int.from_bytes(data[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(data):
            break
        if marker in SOF_MARKERS:
            if segment_length < 7:
                break
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            if width <= 0 or height <= 0:
                break
            return width, height
        offset += segment_length
    raise ValueError(f"JPEG dimensions were not found: {path}")


def unrounded_resize(
    *,
    width: int,
    height: int,
    limit_side_len: int,
    limit_type: str,
    max_side_limit: int,
) -> tuple[int, int, float]:
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    ratio = 1.0
    if limit_type == "max":
        if max(height, width) > limit_side_len:
            ratio = float(limit_side_len) / max(height, width)
    elif limit_type == "min":
        if min(height, width) < limit_side_len:
            ratio = float(limit_side_len) / min(height, width)
    elif limit_type == "resize_long":
        ratio = float(limit_side_len) / max(height, width)
    else:
        raise ValueError(f"unsupported limit type: {limit_type}")

    resized_height = int(height * ratio)
    resized_width = int(width * ratio)
    if max(resized_height, resized_width) > max_side_limit:
        cap_ratio = float(max_side_limit) / max(resized_height, resized_width)
        resized_height = int(resized_height * cap_ratio)
        resized_width = int(resized_width * cap_ratio)
        ratio *= cap_ratio
    return resized_width, resized_height, ratio


def resize_ratio(
    *,
    width: int,
    height: int,
    limit_side_len: int,
    limit_type: str,
    max_side_limit: int,
) -> float:
    return unrounded_resize(
        width=width,
        height=height,
        limit_side_len=limit_side_len,
        limit_type=limit_type,
        max_side_limit=max_side_limit,
    )[2]


def resized_dimensions(
    *,
    width: int,
    height: int,
    limit_side_len: int,
    limit_type: str,
    max_side_limit: int,
) -> tuple[int, int]:
    resized_width, resized_height, _ratio = unrounded_resize(
        width=width,
        height=height,
        limit_side_len=limit_side_len,
        limit_type=limit_type,
        max_side_limit=max_side_limit,
    )

    resized_height = max(int(round(resized_height / 32) * 32), 32)
    resized_width = max(int(round(resized_width / 32) * 32), 32)
    return resized_width, resized_height


def class_node(tree: ast.AST, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise ValueError(f"class not found in frozen source: {name}")


def function_node(node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for child in node.body:
        if isinstance(child, ast.FunctionDef) and child.name == name:
            return child
    raise ValueError(f"method not found in frozen source: {node.name}.{name}")


def neck_source_observations(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    neck = class_node(tree, "PPOCRV6MediumDetNeck")
    initializer = function_node(neck, "__init__")
    forward = function_node(neck, "forward")

    kernel_9_lines: list[int] = []
    for node in ast.walk(initializer):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "Conv2d":
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "kernel_size"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == 9
            ):
                kernel_9_lines.append(node.lineno)

    upsampled_assignment_lines: list[int] = []
    for node in ast.walk(forward):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "upsampled"
            for target in node.targets
        ):
            upsampled_assignment_lines.append(node.lineno)

    if len(kernel_9_lines) != 2:
        raise ValueError("expected exactly two 9x9 neck convolutions")
    if len(upsampled_assignment_lines) != 2:
        raise ValueError("expected exactly two upsampled assignments")
    return {
        "kernel_size_9_convolution_count": len(kernel_9_lines),
        "kernel_size_9_convolution_lines": sorted(kernel_9_lines),
        "upsampled_assignment_count": len(upsampled_assignment_lines),
        "upsampled_assignment_lines": sorted(upsampled_assignment_lines),
    }


def backbone_stem_stride(source: str) -> int:
    tree = ast.parse(source)
    config = class_node(tree, "PPLCNetV4Config")
    for node in config.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "stem_strides"
        ):
            values = ast.literal_eval(node.value)
            stride = 1
            for value in values:
                if not isinstance(value, int):
                    raise ValueError("non-integer PPLCNetV4 stem stride")
                stride *= value
            return stride
    raise ValueError("PPLCNetV4 stem_strides default was not found")


def tensor_bytes(channels: int, height: int, width: int) -> int:
    return channels * height * width * FLOAT32_BYTES


def stage_shapes(
    model_config: dict[str, Any], *, width: int, height: int, stem_stride: int
) -> list[dict[str, int]]:
    backbone = model_config["backbone_config"]
    if backbone.get("stem_type") != "large":
        raise ValueError("expected the frozen large PPLCNetV4 stem")
    cumulative_stride = stem_stride
    stages: list[dict[str, int]] = []
    for index, blocks in enumerate(backbone["block_configs"], start=1):
        for block in blocks:
            cumulative_stride *= int(block[3])
        if width % cumulative_stride or height % cumulative_stride:
            raise ValueError("processed dimensions are not divisible by stage stride")
        channels = int(blocks[-1][2])
        stage_height = height // cumulative_stride
        stage_width = width // cumulative_stride
        stages.append(
            {
                "channels": channels,
                "height": stage_height,
                "payload_bytes_float32": tensor_bytes(
                    channels, stage_height, stage_width
                ),
                "stage": index,
                "stride": cumulative_stride,
                "width": stage_width,
            }
        )
    if len(stages) != 4:
        raise ValueError("expected four frozen backbone stages")
    return stages


def build_static_report(
    *,
    model_config: dict[str, Any],
    preprocessor_config: dict[str, Any],
    model_source: str,
    backbone_config_source: str,
    source_width: int,
    source_height: int,
    source_path: str,
    source_sha256: str,
    plan_path: str,
    plan_sha256: str,
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    if model_config.get("model_type") != "pp_ocrv6_medium_det":
        raise ValueError("frozen model type changed")
    limit_side_len = int(preprocessor_config["limit_side_len"])
    limit_type = str(preprocessor_config["limit_type"])
    max_side_limit = int(preprocessor_config["max_side_limit"])
    processed_width, processed_height = resized_dimensions(
        width=source_width,
        height=source_height,
        limit_side_len=limit_side_len,
        limit_type=limit_type,
        max_side_limit=max_side_limit,
    )
    applied_resize_ratio = resize_ratio(
        width=source_width,
        height=source_height,
        limit_side_len=limit_side_len,
        limit_type=limit_type,
        max_side_limit=max_side_limit,
    )
    stem_stride = backbone_stem_stride(backbone_config_source)
    stages = stage_shapes(
        model_config,
        width=processed_width,
        height=processed_height,
        stem_stride=stem_stride,
    )
    observations = neck_source_observations(model_source)

    neck_channels = int(model_config["neck_out_channels"])
    projected_channels = neck_channels // 4
    high_resolution = stages[0]
    high_height = high_resolution["height"]
    high_width = high_resolution["width"]
    scales = [int(value) for value in model_config["scale_factor_list"]]
    if scales != [1, 2, 4, 8]:
        raise ValueError("frozen neck scale factors changed")
    for stage, scale in zip(stages, scales):
        if stage["height"] * scale != high_height:
            raise ValueError("neck scales do not align feature-map heights")
        if stage["width"] * scale != high_width:
            raise ValueError("neck scales do not align feature-map widths")

    input_payload = tensor_bytes(3, processed_height, processed_width)
    high_neck_payload = tensor_bytes(neck_channels, high_height, high_width)
    high_projected_payload = tensor_bytes(
        projected_channels, high_height, high_width
    )
    upsampled_payload = high_projected_payload * len(scales)
    duplicate_interpolated_payload = high_projected_payload * sum(
        scale > 1 for scale in scales
    )
    projection_unfold = (
        neck_channels * 9 * 9 * high_height * high_width * FLOAT32_BYTES
    )
    lateral_unfold = (
        projected_channels * 9 * 9 * high_height * high_width * FLOAT32_BYTES
    )

    return {
        "analysis": {
            "model_executed": False,
            "model_imported": False,
            "scope": "static_source_and_shape_arithmetic_only",
        },
        "artifacts": artifacts,
        "input": {
            "height": source_height,
            "path": source_path,
            "sha256": source_sha256,
            "width": source_width,
        },
        "memory_estimates": {
            "backbone_output_payload_bytes_float32": sum(
                stage["payload_bytes_float32"] for stage in stages
            ),
            "concatenation_input_plus_output_payload_bytes_float32": (
                upsampled_payload * 2
            ),
            "duplicate_upsample_additional_interpolated_payload_bytes_float32": (
                duplicate_interpolated_payload
            ),
            "duplicate_upsampled_lists_distinct_payload_bytes_float32": (
                upsampled_payload + duplicate_interpolated_payload
            ),
            "high_resolution_9x9_lateral_explicit_unfold_equivalent_bytes_float32": (
                lateral_unfold
            ),
            "high_resolution_9x9_projection_explicit_unfold_equivalent_bytes_float32": (
                projection_unfold
            ),
            "high_resolution_neck_map_payload_bytes_float32": high_neck_payload,
            "input_tensor_payload_bytes_float32": input_payload,
            "single_upsampled_list_payload_bytes_float32": upsampled_payload,
        },
        "model": {
            "backbone_stem_stride": stem_stride,
            "explicit_lower_precision_in_model_config": (
                "dtype" in model_config or "torch_dtype" in model_config
            ),
            "feature_maps": stages,
            "model_type": model_config["model_type"],
            "neck_out_channels": neck_channels,
            "nominal_static_estimate_dtype": "float32",
            "scale_factor_list": scales,
            "source_observations": observations,
        },
        "plan": {"path": plan_path, "sha256": plan_sha256},
        "preprocessing": {
            "limit_side_len": limit_side_len,
            "limit_type": limit_type,
            "max_side_limit": max_side_limit,
            "multiple_of_32_rounding_changed_dimensions": (
                processed_width != source_width or processed_height != source_height
            ),
            "processed_height": processed_height,
            "processed_width": processed_width,
            "rounding_multiple": 32,
            "source_was_downscaled_by_limit": applied_resize_ratio < 1.0,
        },
        "risk_assessment": {
            "interpretation": [
                "limit_type=min does not shrink this page before multiple-of-32 rounding",
                "the duplicate upsampled construction adds three interpolated high-resolution outputs while the scale-1 entry reuses its existing feature",
                "explicit-unfold equivalents are risk bounds for shape comparison, not measured PyTorch allocations or proof of the runtime root cause",
            ],
            "reason_code": "static_full_resolution_detector_memory_risk",
            "terminal": "PREREQUISITE_NEEDED",
        },
        "schema_version": 1,
    }


def validate_external_artifact(
    official_detector: dict[str, Any], key: str
) -> tuple[Path, dict[str, Any]]:
    evidence = official_detector[key]
    path = Path(evidence["external_path"])
    if not path.is_file():
        raise ValueError(f"missing frozen detector artifact: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != evidence["sha256"]:
        raise ValueError(f"frozen detector artifact changed: {path}")
    return path, {
        "bytes": path.stat().st_size,
        "path": str(path),
        "sha256": actual_sha256,
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else repo_root / args.plan
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    plan = read_json(plan_path)
    materialization = plan["external_text_layout_materialization"]
    official_detector = plan["evidence"]["official_text_detector"]

    config_path, config_evidence = validate_external_artifact(
        official_detector, "config_json"
    )
    preprocessor_path, preprocessor_evidence = validate_external_artifact(
        official_detector, "preprocessor_config_json"
    )
    weights_path, weights_evidence = validate_external_artifact(
        official_detector, "model_safetensors"
    )
    preprocessor_config = read_json(preprocessor_path)
    for key in ("limit_side_len", "limit_type", "max_side_limit"):
        if preprocessor_config[key] != materialization[key]:
            raise ValueError(f"plan and frozen preprocessor disagree on {key}")

    manifest_path = repo_root / plan["data"]["manifest"]["path"]
    if sha256_file(manifest_path) != plan["data"]["manifest"]["sha256"]:
        raise ValueError("frozen source manifest changed")
    source_rows = [
        row.strip()
        for row in manifest_path.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    if not source_rows:
        raise ValueError("frozen source manifest is empty")
    source_path = repo_root / source_rows[0]
    source_width, source_height = jpeg_dimensions(source_path)

    model_source = args.model_source.resolve()
    backbone_source = args.backbone_config_source.resolve()
    artifacts = {
        "backbone_config_source": {
            "bytes": backbone_source.stat().st_size,
            "path": str(backbone_source),
            "sha256": sha256_file(backbone_source),
        },
        "model_config": config_evidence,
        "model_source": {
            "bytes": model_source.stat().st_size,
            "path": str(model_source),
            "sha256": sha256_file(model_source),
        },
        "preprocessor_config": preprocessor_evidence,
        "weights": weights_evidence,
    }
    report = build_static_report(
        model_config=read_json(config_path),
        preprocessor_config=preprocessor_config,
        model_source=model_source.read_text(encoding="utf-8"),
        backbone_config_source=backbone_source.read_text(encoding="utf-8"),
        source_width=source_width,
        source_height=source_height,
        source_path=source_rows[0],
        source_sha256=sha256_file(source_path),
        plan_path=str(plan_path.relative_to(repo_root)),
        plan_sha256=sha256_file(plan_path),
        artifacts=artifacts,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
