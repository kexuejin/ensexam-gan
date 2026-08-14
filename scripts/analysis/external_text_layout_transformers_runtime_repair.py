#!/usr/bin/env python3
"""Apply the hash-bound PP-OCRv6 duplicate-upsample runtime repair."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import importlib
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


REPAIR_ID = "external_text_layout_transformers_neck_duplicate_upsample_v1"
EXPECTED_TRANSFORMERS_VERSION = "5.12.1"
EXPECTED_MODEL_SOURCE_SHA256 = (
    "4bb27b16b04056ee00779391a4943efa5b5c2745e9431e4e9aa652423b271210"
)
MODEL_MODULE_NAME = (
    "transformers.models.pp_ocrv6_medium_det.modeling_pp_ocrv6_medium_det"
)
MODEL_SOURCE_RELATIVE_PATH = Path(
    "transformers/models/pp_ocrv6_medium_det/modeling_pp_ocrv6_medium_det.py"
)
NECK_CLASS_NAME = "PPOCRV6MediumDetNeck"
PATCH_MARKER = "__ensexam_runtime_equivalence_repair_id__"
PATCH_SOURCE_MARKER = "__ensexam_runtime_equivalence_repair_source_sha256__"
PATCH_VERSION_MARKER = "__ensexam_runtime_equivalence_repair_transformers_version__"

_EXPECTED_REPAIR_TAIL = """
def expected(self):
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


class RuntimeEquivalenceRepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class RepairTarget:
    forward: ast.FunctionDef
    first_assignment_index: int
    replacement_assignment_index: int

    @property
    def removed_assignment_line(self) -> int:
        return self.forward.body[self.first_assignment_index].lineno

    @property
    def replacement_assignment_line(self) -> int:
        return self.forward.body[self.replacement_assignment_index].lineno


@dataclass(frozen=True)
class RuntimeRepairResult:
    status: str
    repair_id: str
    transformers_version: str
    model_source_path: str
    model_source_sha256: str
    removed_assignment_line: int
    replacement_assignment_line: int


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _single_class(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeEquivalenceRepairError(
            f"expected exactly one frozen class named {name}"
        )
    return matches[0]


def _single_method(node: ast.ClassDef, name: str) -> ast.FunctionDef:
    matches = [
        child
        for child in node.body
        if isinstance(child, ast.FunctionDef) and child.name == name
    ]
    if len(matches) != 1:
        raise RuntimeEquivalenceRepairError(
            f"expected exactly one frozen method named {node.name}.{name}"
        )
    return matches[0]


def _upsampled_assignment_indices(forward: ast.FunctionDef) -> list[int]:
    indices: list[int] = []
    for index, node in enumerate(forward.body):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "upsampled"
            for target in node.targets
        ):
            indices.append(index)
    return indices


def _node_sequence_dump(nodes: list[ast.stmt]) -> str:
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)


def analyze_repair_target(source: str) -> RepairTarget:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise RuntimeEquivalenceRepairError(
            "frozen Transformers model source is not valid Python"
        ) from error
    neck = _single_class(tree, NECK_CLASS_NAME)
    forward = _single_method(neck, "forward")
    if forward.decorator_list:
        raise RuntimeEquivalenceRepairError("frozen neck forward gained decorators")

    indices = _upsampled_assignment_indices(forward)
    if len(indices) != 2:
        raise RuntimeEquivalenceRepairError(
            "frozen neck forward no longer has exactly two direct upsampled assignments"
        )
    first_index, replacement_index = indices
    if replacement_index != first_index + 2:
        raise RuntimeEquivalenceRepairError(
            "frozen neck duplicate-upsample statements are no longer contiguous"
        )
    actual_tail = forward.body[first_index : replacement_index + 2]
    expected_function = ast.parse(_EXPECTED_REPAIR_TAIL).body[0]
    assert isinstance(expected_function, ast.FunctionDef)
    if _node_sequence_dump(actual_tail) != _node_sequence_dump(
        expected_function.body
    ):
        raise RuntimeEquivalenceRepairError(
            "frozen neck duplicate-upsample AST no longer matches the registered repair"
        )
    return RepairTarget(
        forward=forward,
        first_assignment_index=first_index,
        replacement_assignment_index=replacement_index,
    )


def build_repaired_forward(
    source: str,
    *,
    module_globals: dict[str, Any],
    filename: str,
) -> tuple[Callable[..., Any], RepairTarget]:
    target = analyze_repair_target(source)
    repaired = copy.deepcopy(target.forward)
    del repaired.body[
        target.first_assignment_index : target.replacement_assignment_index
    ]
    ast.fix_missing_locations(repaired)
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
        raise RuntimeEquivalenceRepairError(
            "could not compile the registered neck forward repair"
        ) from error
    repaired_forward = namespace.get("forward")
    if not callable(repaired_forward):
        raise RuntimeEquivalenceRepairError(
            "registered neck forward repair did not produce a callable"
        )
    return repaired_forward, target


def _load_registered_source(
    distribution_getter: Callable[[str], Any],
) -> tuple[str, Path, str, str]:
    try:
        distribution = distribution_getter("transformers")
    except metadata.PackageNotFoundError as error:
        raise RuntimeEquivalenceRepairError("Transformers is not installed") from error
    actual_version = str(distribution.version)
    if actual_version != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeEquivalenceRepairError(
            "Transformers version changed: "
            f"expected {EXPECTED_TRANSFORMERS_VERSION}, got {actual_version}"
        )
    source_path = Path(distribution.locate_file(MODEL_SOURCE_RELATIVE_PATH)).resolve()
    try:
        source_bytes = source_path.read_bytes()
    except OSError as error:
        raise RuntimeEquivalenceRepairError(
            f"could not read frozen Transformers model source: {source_path}"
        ) from error
    actual_sha256 = sha256_bytes(source_bytes)
    if actual_sha256 != EXPECTED_MODEL_SOURCE_SHA256:
        raise RuntimeEquivalenceRepairError(
            "Transformers model source changed: "
            f"expected {EXPECTED_MODEL_SOURCE_SHA256}, got {actual_sha256}"
        )
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeEquivalenceRepairError(
            "frozen Transformers model source is not UTF-8"
        ) from error
    analyze_repair_target(source)
    return source, source_path, actual_sha256, actual_version


def _validate_loaded_module(module: ModuleType, source_path: Path) -> type[Any]:
    module_path_value = getattr(module, "__file__", None)
    if module_path_value is None or Path(module_path_value).resolve() != source_path:
        raise RuntimeEquivalenceRepairError(
            "loaded Transformers model module does not match the hashed source path"
        )
    neck = getattr(module, NECK_CLASS_NAME, None)
    if not isinstance(neck, type):
        raise RuntimeEquivalenceRepairError(
            f"loaded Transformers module lacks {NECK_CLASS_NAME}"
        )
    return neck


def _result(
    *,
    status: str,
    source_path: Path,
    source_sha256: str,
    transformers_version: str,
    target: RepairTarget,
) -> RuntimeRepairResult:
    return RuntimeRepairResult(
        status=status,
        repair_id=REPAIR_ID,
        transformers_version=transformers_version,
        model_source_path=str(source_path),
        model_source_sha256=source_sha256,
        removed_assignment_line=target.removed_assignment_line,
        replacement_assignment_line=target.replacement_assignment_line,
    )


def apply_runtime_equivalence_repair(
    *,
    _distribution_getter: Callable[[str], Any] = metadata.distribution,
    _module_loader: Callable[[str], ModuleType] = importlib.import_module,
) -> RuntimeRepairResult:
    source, source_path, source_sha256, transformers_version = (
        _load_registered_source(_distribution_getter)
    )
    target = analyze_repair_target(source)
    try:
        module = _module_loader(MODEL_MODULE_NAME)
    except Exception as error:
        raise RuntimeEquivalenceRepairError(
            "could not import the frozen Transformers detector module"
        ) from error
    neck = _validate_loaded_module(module, source_path)
    current_forward = getattr(neck, "forward", None)
    if not callable(current_forward):
        raise RuntimeEquivalenceRepairError("loaded neck forward is not callable")

    existing_marker = getattr(current_forward, PATCH_MARKER, None)
    if existing_marker is not None:
        if (
            existing_marker != REPAIR_ID
            or getattr(current_forward, PATCH_SOURCE_MARKER, None) != source_sha256
            or getattr(current_forward, PATCH_VERSION_MARKER, None)
            != transformers_version
        ):
            raise RuntimeEquivalenceRepairError(
                "loaded neck forward carries inconsistent runtime repair markers"
            )
        return _result(
            status="already_applied",
            source_path=source_path,
            source_sha256=source_sha256,
            transformers_version=transformers_version,
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
        raise RuntimeEquivalenceRepairError(
            "loaded neck forward does not match the registered source definition"
        )

    repaired_forward, compiled_target = build_repaired_forward(
        source,
        module_globals=vars(module),
        filename=str(source_path),
    )
    repaired_forward.__name__ = current_forward.__name__
    repaired_forward.__qualname__ = current_forward.__qualname__
    repaired_forward.__module__ = current_forward.__module__
    repaired_forward.__doc__ = current_forward.__doc__
    repaired_forward.__annotations__ = dict(current_forward.__annotations__)
    repaired_forward.__dict__.update(current_forward.__dict__)
    setattr(repaired_forward, PATCH_MARKER, REPAIR_ID)
    setattr(repaired_forward, PATCH_SOURCE_MARKER, source_sha256)
    setattr(repaired_forward, PATCH_VERSION_MARKER, transformers_version)
    setattr(neck, "forward", repaired_forward)

    return _result(
        status="applied",
        source_path=source_path,
        source_sha256=source_sha256,
        transformers_version=transformers_version,
        target=compiled_target,
    )
