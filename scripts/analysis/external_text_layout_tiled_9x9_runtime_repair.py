#!/usr/bin/env python3
"""Apply the preregistered in-memory tiled 9x9 detector neck repair."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import importlib
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


from scripts.analysis import external_text_layout_transformers_runtime_repair as duplicate_repair
from scripts.analysis import report_external_text_layout_tiled_9x9_feasibility as feasibility


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = Path("docs/external-text-layout-tiled-9x9-runtime-repair-v1.json")
EXPECTED_CONTRACT_SHA256 = (
    "203e67ac2d12557034f546d9e25e475d083bb2086d2f0ed7bfc1a3244ba3b250"
)
REPAIR_ID = "external_text_layout_transformers_neck_tiled_9x9_v1"
EXPECTED_TRANSFORMERS_VERSION = "5.12.1"
EXPECTED_MODEL_SOURCE_SHA256 = feasibility.EXPECTED_MODEL_SOURCE_SHA256
MODEL_MODULE_NAME = duplicate_repair.MODEL_MODULE_NAME
MODEL_SOURCE_RELATIVE_PATH = duplicate_repair.MODEL_SOURCE_RELATIVE_PATH
NECK_CLASS_NAME = duplicate_repair.NECK_CLASS_NAME
TILE_OUTPUT_ROWS = 4
TILED_HELPER_GLOBAL = "__ensexam_tiled_conv2d_rows_v1"
PATCH_MARKER = "__ensexam_tiled_9x9_runtime_repair_id__"
PATCH_SOURCE_MARKER = "__ensexam_tiled_9x9_runtime_repair_source_sha256__"
PATCH_VERSION_MARKER = "__ensexam_tiled_9x9_runtime_repair_transformers_version__"


class Tiled9x9RuntimeRepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class Tiled9x9RepairResult:
    status: str
    repair_id: str
    transformers_version: str
    model_source_path: str
    model_source_sha256: str
    tile_output_rows: int
    removed_assignment_line: int
    projection_call_line: int
    lateral_call_line: int


@dataclass(frozen=True)
class TiledForwardTarget:
    forward: ast.FunctionDef
    removed_assignment_line: int
    projection_call_line: int
    lateral_call_line: int


def _pair(value: Any, label: str) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, tuple) and len(value) == 2 and all(
        isinstance(item, int) for item in value
    ):
        return value
    raise Tiled9x9RuntimeRepairError(f"tiled convolution {label} changed")


def _validate_tiled_inputs(
    convolution: Any, hidden_states: Any, *, torch_module: Any
) -> tuple[int, int, int, int]:
    if torch_module.is_grad_enabled():
        raise Tiled9x9RuntimeRepairError(
            "tiled convolution requires gradient-disabled inference"
        )
    if not isinstance(convolution, torch_module.nn.Conv2d):
        raise Tiled9x9RuntimeRepairError("tiled target is not Conv2d")
    if _pair(convolution.kernel_size, "kernel") != (9, 9):
        raise Tiled9x9RuntimeRepairError("tiled convolution kernel changed")
    if _pair(convolution.stride, "stride") != (1, 1):
        raise Tiled9x9RuntimeRepairError("tiled convolution stride changed")
    if _pair(convolution.padding, "padding") != (4, 4):
        raise Tiled9x9RuntimeRepairError("tiled convolution padding changed")
    if _pair(convolution.dilation, "dilation") != (1, 1):
        raise Tiled9x9RuntimeRepairError("tiled convolution dilation changed")
    if convolution.groups != 1:
        raise Tiled9x9RuntimeRepairError("tiled convolution groups changed")
    if convolution.padding_mode != "zeros":
        raise Tiled9x9RuntimeRepairError("tiled convolution padding mode changed")
    if convolution.bias is None:
        raise Tiled9x9RuntimeRepairError("tiled convolution bias is missing")
    if not isinstance(hidden_states, torch_module.Tensor) or hidden_states.ndim != 4:
        raise Tiled9x9RuntimeRepairError("tiled input must be a rank-four tensor")
    if hidden_states.device.type != "cpu":
        raise Tiled9x9RuntimeRepairError("tiled input must stay on CPU")
    if hidden_states.dtype != torch_module.float32:
        raise Tiled9x9RuntimeRepairError("tiled input must stay float32")
    if not hidden_states.is_contiguous():
        raise Tiled9x9RuntimeRepairError("tiled input must be contiguous")
    batch, in_channels, height, width = hidden_states.shape
    if batch != 1 or height <= 0 or width <= 0:
        raise Tiled9x9RuntimeRepairError("tiled input shape changed")
    weight = convolution.weight
    if (
        weight.device != hidden_states.device
        or weight.dtype != hidden_states.dtype
        or tuple(weight.shape[1:]) != (in_channels, 9, 9)
        or convolution.bias.device != hidden_states.device
        or convolution.bias.dtype != hidden_states.dtype
    ):
        raise Tiled9x9RuntimeRepairError("tiled convolution parameter surface changed")
    return batch, int(weight.shape[0]), height, width


def tiled_conv2d_rows(
    convolution: Any,
    hidden_states: Any,
    *,
    functional: Any,
    torch_module: Any,
    tile_output_rows: int = TILE_OUTPUT_ROWS,
) -> Any:
    batch, out_channels, height, width = _validate_tiled_inputs(
        convolution, hidden_states, torch_module=torch_module
    )
    if tile_output_rows != TILE_OUTPUT_ROWS:
        raise Tiled9x9RuntimeRepairError("tiled output row limit changed")
    output = hidden_states.new_empty((batch, out_channels, height, width))
    for tile in feasibility.row_tiles(height, tile_output_rows):
        source = hidden_states[
            ..., tile.source_start : tile.source_end, :
        ]
        padded = functional.pad(
            source,
            (4, 4, tile.pad_top, tile.pad_bottom),
            mode="constant",
            value=0.0,
        )
        tile_output = functional.conv2d(
            padded,
            convolution.weight,
            convolution.bias,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
        expected_shape = (
            batch,
            out_channels,
            tile.output_rows,
            width,
        )
        if tuple(tile_output.shape) != expected_shape:
            raise Tiled9x9RuntimeRepairError("tiled convolution output shape changed")
        output[..., tile.output_start : tile.output_end, :].copy_(tile_output)
    if not output.is_contiguous():
        raise Tiled9x9RuntimeRepairError("tiled convolution output is not contiguous")
    return output


def _is_frozen_module_list_call(node: ast.Call, attribute: str) -> bool:
    function = node.func
    return (
        isinstance(function, ast.Subscript)
        and isinstance(function.value, ast.Attribute)
        and isinstance(function.value.value, ast.Name)
        and function.value.value.id == "self"
        and function.value.attr == attribute
        and isinstance(function.slice, ast.Name)
        and function.slice.id == "i"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "hidden_states"
        and not node.keywords
    )


class _TiledCallTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.lines: dict[str, list[int]] = {"projection": [], "lateral": []}

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        assert isinstance(node, ast.Call)
        labels = {
            "projection": "input_feature_projection_convolution",
            "lateral": "path_aggregation_lateral_convolution",
        }
        for label, attribute in labels.items():
            if _is_frozen_module_list_call(node, attribute):
                self.lines[label].append(node.lineno)
                replacement = ast.Call(
                    func=ast.Name(id=TILED_HELPER_GLOBAL, ctx=ast.Load()),
                    args=[copy.deepcopy(node.func), copy.deepcopy(node.args[0])],
                    keywords=[],
                )
                return ast.copy_location(replacement, node)
        return node


def analyze_tiled_forward_target(
    source: str,
) -> tuple[ast.FunctionDef, TiledForwardTarget]:
    feasibility.frozen_9x9_convolutions(source)
    duplicate_target = duplicate_repair.analyze_repair_target(source)
    repaired = copy.deepcopy(duplicate_target.forward)
    del repaired.body[
        duplicate_target.first_assignment_index : duplicate_target.replacement_assignment_index
    ]
    transformer = _TiledCallTransformer()
    repaired = transformer.visit(repaired)
    assert isinstance(repaired, ast.FunctionDef)
    if any(len(lines) != 1 for lines in transformer.lines.values()):
        raise Tiled9x9RuntimeRepairError(
            "frozen neck forward no longer has one projection and lateral call"
        )
    ast.fix_missing_locations(repaired)
    target = TiledForwardTarget(
        forward=repaired,
        removed_assignment_line=duplicate_target.removed_assignment_line,
        projection_call_line=transformer.lines["projection"][0],
        lateral_call_line=transformer.lines["lateral"][0],
    )
    return repaired, target


def build_tiled_forward(
    source: str,
    *,
    module_globals: dict[str, Any],
    filename: str,
) -> tuple[Callable[..., Any], TiledForwardTarget]:
    repaired, target = analyze_tiled_forward_target(source)
    namespace: dict[str, Any] = {}
    try:
        code = compile(
            ast.Module(body=[repaired], type_ignores=[]),
            filename=filename,
            mode="exec",
            dont_inherit=True,
            optimize=0,
        )
        exec(code, module_globals, namespace)
    except Exception as error:
        raise Tiled9x9RuntimeRepairError("could not compile tiled neck forward") from error
    forward = namespace.get("forward")
    if not callable(forward):
        raise Tiled9x9RuntimeRepairError("tiled repair did not produce a forward")
    return forward, target


def _load_registered_source(
    distribution_getter: Callable[[str], Any],
) -> tuple[str, Path, str, str]:
    contract = ROOT / CONTRACT_PATH
    if feasibility.sha256_file(contract) != EXPECTED_CONTRACT_SHA256:
        raise Tiled9x9RuntimeRepairError("tiled repair contract changed")
    try:
        distribution = distribution_getter("transformers")
    except metadata.PackageNotFoundError as error:
        raise Tiled9x9RuntimeRepairError("Transformers is not installed") from error
    version = str(distribution.version)
    if version != EXPECTED_TRANSFORMERS_VERSION:
        raise Tiled9x9RuntimeRepairError(
            f"Transformers version changed: expected {EXPECTED_TRANSFORMERS_VERSION}, got {version}"
        )
    source_path = Path(distribution.locate_file(MODEL_SOURCE_RELATIVE_PATH)).resolve()
    try:
        source_bytes = source_path.read_bytes()
    except OSError as error:
        raise Tiled9x9RuntimeRepairError("could not read frozen model source") from error
    source_sha256 = feasibility.sha256_file(source_path)
    if source_sha256 != EXPECTED_MODEL_SOURCE_SHA256:
        raise Tiled9x9RuntimeRepairError("frozen model source hash changed")
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Tiled9x9RuntimeRepairError("frozen model source is not UTF-8") from error
    feasibility.frozen_9x9_convolutions(source)
    duplicate_repair.analyze_repair_target(source)
    return source, source_path, source_sha256, version


def _validate_loaded_neck(module: ModuleType, source_path: Path) -> type[Any]:
    module_path = getattr(module, "__file__", None)
    if module_path is None or Path(module_path).resolve() != source_path:
        raise Tiled9x9RuntimeRepairError("loaded module does not match frozen source")
    neck = getattr(module, NECK_CLASS_NAME, None)
    if not isinstance(neck, type):
        raise Tiled9x9RuntimeRepairError("loaded module lacks frozen neck class")
    return neck


def _markers_match(forward: Callable[..., Any], source_sha256: str, version: str) -> bool:
    return (
        getattr(forward, PATCH_MARKER, None) == REPAIR_ID
        and getattr(forward, PATCH_SOURCE_MARKER, None) == source_sha256
        and getattr(forward, PATCH_VERSION_MARKER, None) == version
        and getattr(forward, duplicate_repair.PATCH_MARKER, None)
        == duplicate_repair.REPAIR_ID
        and getattr(forward, duplicate_repair.PATCH_SOURCE_MARKER, None)
        == source_sha256
        and getattr(forward, duplicate_repair.PATCH_VERSION_MARKER, None) == version
    )


def _result(
    *,
    status: str,
    source_path: Path,
    source_sha256: str,
    version: str,
    target: TiledForwardTarget,
) -> Tiled9x9RepairResult:
    return Tiled9x9RepairResult(
        status=status,
        repair_id=REPAIR_ID,
        transformers_version=version,
        model_source_path=str(source_path),
        model_source_sha256=source_sha256,
        tile_output_rows=TILE_OUTPUT_ROWS,
        removed_assignment_line=target.removed_assignment_line,
        projection_call_line=target.projection_call_line,
        lateral_call_line=target.lateral_call_line,
    )


def apply_tiled_9x9_runtime_repair(
    *,
    _distribution_getter: Callable[[str], Any] = metadata.distribution,
    _module_loader: Callable[[str], ModuleType] = importlib.import_module,
) -> Tiled9x9RepairResult:
    source, source_path, source_sha256, version = _load_registered_source(
        _distribution_getter
    )
    _unused_forward_ast, target = analyze_tiled_forward_target(source)
    try:
        module = _module_loader(MODEL_MODULE_NAME)
    except Exception as error:
        raise Tiled9x9RuntimeRepairError("could not import frozen model module") from error
    neck = _validate_loaded_neck(module, source_path)
    current_forward = getattr(neck, "forward", None)
    if not callable(current_forward):
        raise Tiled9x9RuntimeRepairError("loaded neck forward is not callable")

    marker_values = (
        getattr(current_forward, PATCH_MARKER, None),
        getattr(current_forward, duplicate_repair.PATCH_MARKER, None),
    )
    if any(value is not None for value in marker_values):
        if not _markers_match(current_forward, source_sha256, version):
            raise Tiled9x9RuntimeRepairError(
                "loaded neck forward carries partial or conflicting repair markers"
            )
        return _result(
            status="already_applied",
            source_path=source_path,
            source_sha256=source_sha256,
            version=version,
            target=target,
        )

    code = getattr(current_forward, "__code__", None)
    if (
        code is None
        or Path(code.co_filename).resolve() != source_path
        or code.co_firstlineno != target.forward.lineno
        or getattr(current_forward, "__module__", None) != MODEL_MODULE_NAME
        or getattr(current_forward, "__qualname__", None)
        != f"{NECK_CLASS_NAME}.forward"
    ):
        raise Tiled9x9RuntimeRepairError(
            "loaded neck forward does not match frozen source definition"
        )
    existing_helper = vars(module).get(TILED_HELPER_GLOBAL)
    if existing_helper is not None:
        raise Tiled9x9RuntimeRepairError("tiled helper global already exists")

    def bound_helper(convolution: Any, hidden_states: Any) -> Any:
        return tiled_conv2d_rows(
            convolution,
            hidden_states,
            functional=module.F,
            torch_module=module.torch,
        )

    module_globals = vars(module)
    module_globals[TILED_HELPER_GLOBAL] = bound_helper
    try:
        repaired_forward, compiled_target = build_tiled_forward(
            source,
            module_globals=module_globals,
            filename=str(source_path),
        )
    except Exception:
        module_globals.pop(TILED_HELPER_GLOBAL, None)
        raise
    repaired_forward.__name__ = current_forward.__name__
    repaired_forward.__qualname__ = current_forward.__qualname__
    repaired_forward.__module__ = current_forward.__module__
    repaired_forward.__doc__ = current_forward.__doc__
    repaired_forward.__annotations__ = dict(current_forward.__annotations__)
    repaired_forward.__dict__.update(current_forward.__dict__)
    setattr(repaired_forward, PATCH_MARKER, REPAIR_ID)
    setattr(repaired_forward, PATCH_SOURCE_MARKER, source_sha256)
    setattr(repaired_forward, PATCH_VERSION_MARKER, version)
    setattr(repaired_forward, duplicate_repair.PATCH_MARKER, duplicate_repair.REPAIR_ID)
    setattr(repaired_forward, duplicate_repair.PATCH_SOURCE_MARKER, source_sha256)
    setattr(repaired_forward, duplicate_repair.PATCH_VERSION_MARKER, version)
    setattr(neck, "forward", repaired_forward)
    return _result(
        status="applied",
        source_path=source_path,
        source_sha256=source_sha256,
        version=version,
        target=compiled_target,
    )
