#!/usr/bin/env python3
"""Audit semantic identity of the retained second-stage temporary cache."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.eval_hardcase_worst_pages import (  # noqa: E402
    compute_residual_metrics,
    ensure_same_size,
    label_path_for,
    read_bgr,
)


CONTRACT_PATH = Path(
    "docs/external-text-layout-second-stage-cache-salvage-audit-v1.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "b53a6a21882918a1aa6160763a199d023352b46678310abd9abba1746b46b906"
)
OUTPUT_PATH = Path(
    "outputs/external-text-layout-second-stage-cache-salvage-audit-20260815/"
    "audit.json"
)
PRIMARY_CACHE = Path("outputs/sign-separated-residual-repair-train275-primary-v1")
FINAL_CACHE = Path(
    "outputs/sign-separated-residual-repair-train275-frozen-pipeline-v1"
)
ARCHIVE_PATHS = (
    Path("outputs/archive/sign-separated-residual-repair-20260810/train275-primary"),
    Path(
        "outputs/archive/sign-separated-residual-repair-20260810/"
        "train275-frozen-pipeline"
    ),
)
EXPECTED_FIELDS = [
    "base_edit_threshold",
    "changed_px",
    "dark_threshold",
    "file",
    "gate_ratio",
    "image_path",
    "max_brighten_delta",
    "mean_over_delta",
    "mean_residual_delta",
    "outside_px",
    "over_px",
    "overerase_ratio",
    "pred_path",
    "residual_px",
    "residual_ratio",
    "second_delta_threshold",
]
RECOMPUTED_INTEGER_FIELDS = {"changed_px", "outside_px", "over_px", "residual_px"}


class PrerequisiteError(RuntimeError):
    pass


class SemanticMismatch(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_rows(rows: list[str]) -> str:
    payload = "\n".join(sorted(rows)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PrerequisiteError(f"cannot read JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise PrerequisiteError(f"JSON evidence is not an object: {path}")
    return value


def repo_path(repo_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise PrerequisiteError(f"registered path escaped repository: {value}")
    return repo_root / relative


def is_registered_repository_root(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if "\x00" in value or "\n" in value or "\r" in value:
        return False
    path = Path(value)
    return path.is_absolute() and path != path.parent


def validate_artifact(repo_root: Path, artifact: Any, label: str) -> Path:
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise PrerequisiteError(f"{label} artifact contract changed")
    path = repo_path(repo_root, str(artifact["path"]))
    if not path.is_file() or path.is_symlink():
        raise PrerequisiteError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != artifact["sha256"]:
        raise PrerequisiteError(
            f"{label} sha256 changed: expected {artifact['sha256']}, got {actual}"
        )
    return path


def validate_source_provenance(
    repo_root: Path, artifact: Any, label: str
) -> dict[str, Any]:
    if (
        not isinstance(artifact, dict)
        or "path" not in artifact
        or "sha256" not in artifact
    ):
        raise PrerequisiteError(f"{label} provenance contract changed")
    path = repo_path(repo_root, str(artifact["path"]))
    if not path.is_file() or path.is_symlink():
        raise PrerequisiteError(f"missing {label}: {path}")
    actual = sha256_file(path)
    return {
        "actual_sha256": actual,
        "expected_sha256": str(artifact["sha256"]),
        "path": str(path),
        "status": "current" if actual == artifact["sha256"] else "changed",
    }


def validate_repository_contract(repo_root: Path) -> dict[str, Any]:
    path = repo_root / CONTRACT_PATH
    if sha256_file(path) != EXPECTED_CONTRACT_SHA256:
        raise PrerequisiteError("salvage audit contract sha256 changed")
    contract = read_json(path)
    if (
        contract.get("schema_version") != 1
        or contract.get("state")
        != "preregistered_second_stage_temporary_cache_salvage_audit"
        or contract.get("terminal") != "PREREQUISITE_NEEDED"
        or not is_registered_repository_root(
            contract.get("canonical_candidate", {}).get("current_repository_root")
        )
        or contract.get("implementation", {}).get("allowed_files")
        != [
            "scripts/analysis/"
            "audit_external_text_layout_second_stage_cache_salvage.py",
            "tests/test_external_text_layout_second_stage_cache_salvage.py",
        ]
        or contract.get("implementation", {}).get("new_dependency") is not False
        or contract.get("implementation", {}).get("site_packages_write") is not False
        or contract.get("implementation", {}).get("output") != str(OUTPUT_PATH)
    ):
        raise PrerequisiteError("salvage audit contract changed")
    for label, artifact in contract.get("evidence", {}).items():
        validate_artifact(repo_root, artifact, f"salvage {label}")
    for label, artifact in contract.get("frozen_functions", {}).items():
        validate_source_provenance(repo_root, artifact, f"salvage function {label}")
    return contract


def read_metrics(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    fields, rows = read_csv_rows(path)
    if fields != EXPECTED_FIELDS:
        raise SemanticMismatch(f"metrics fields changed: {fields}")
    return fields, rows


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file() or path.is_symlink():
        raise PrerequisiteError(f"metrics evidence is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields:
        raise SemanticMismatch(f"metrics fields are missing: {path}")
    if any(
        None in row
        or set(row) != set(fields)
        or any(value is None for value in row.values())
        for row in rows
    ):
        raise SemanticMismatch("metrics row shape changed")
    return fields, rows


def _numeric(value: str, field: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise SemanticMismatch(f"metrics field {field} is not numeric") from error


def validate_metric_row(
    row: dict[str, str],
    *,
    expected_file: str,
    expected_image_path: str,
    expected_pred_path: str,
    expected_gate_ratio: float,
    constants: dict[str, float | int],
    recomputed: dict[str, float | int],
) -> None:
    if set(row) != set(EXPECTED_FIELDS):
        raise SemanticMismatch(f"metrics fields changed for {expected_file}")
    expected_strings = {
        "file": expected_file,
        "image_path": expected_image_path,
        "pred_path": expected_pred_path,
    }
    for field, expected in expected_strings.items():
        if row[field] != expected:
            raise SemanticMismatch(f"metrics {field} changed for {expected_file}")
    if _numeric(row["gate_ratio"], "gate_ratio") != expected_gate_ratio:
        raise SemanticMismatch(f"metrics gate_ratio changed for {expected_file}")
    for field, expected in constants.items():
        if _numeric(row[field], field) != float(expected):
            raise SemanticMismatch(f"metrics {field} changed for {expected_file}")
    for field, expected in recomputed.items():
        if field in RECOMPUTED_INTEGER_FIELDS:
            try:
                actual: float | int = int(row[field])
            except ValueError as error:
                raise SemanticMismatch(
                    f"metrics {field} is not an integer for {expected_file}"
                ) from error
        else:
            actual = _numeric(row[field], field)
        if actual != expected:
            raise SemanticMismatch(f"metrics {field} changed for {expected_file}")


def validate_canonical_candidate(
    metrics_path: Path,
    *,
    current_repository_root: str,
    frozen_historical_repository_root: str,
    expected_replacement_count: int,
    expected_sha256: str,
) -> dict[str, int | str]:
    before = metrics_path.read_bytes()
    current = current_repository_root.encode("utf-8")
    historical = frozen_historical_repository_root.encode("utf-8")
    replacement_count = before.count(current)
    if replacement_count != expected_replacement_count or historical in before:
        raise SemanticMismatch(
            "canonical candidate repository-root precondition changed"
        )
    candidate = before.replace(current, historical)
    candidate_sha256 = hashlib.sha256(candidate).hexdigest()
    if (
        current in candidate
        or candidate.count(historical) != expected_replacement_count
        or candidate_sha256 != expected_sha256
    ):
        raise SemanticMismatch(
            f"canonical candidate identity changed: {candidate_sha256}"
        )
    return {
        "candidate_sha256": candidate_sha256,
        "replacement_count": replacement_count,
        "source_sha256": hashlib.sha256(before).hexdigest(),
    }


def validate_prediction_set(
    prediction_dir: Path,
    expected_names: list[str],
    expected: dict[str, Any],
) -> dict[str, Any]:
    if not prediction_dir.is_dir() or prediction_dir.is_symlink():
        raise SemanticMismatch("temporary prediction directory changed")
    paths = list(prediction_dir.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise SemanticMismatch("temporary prediction surface changed")
    actual_names = sorted(path.name for path in paths)
    rows = [f"{name} {sha256_file(prediction_dir / name)}" for name in actual_names]
    actual = {
        "content_sha256": sha256_rows(rows),
        "count": len(actual_names),
        "filename_sha256": sha256_rows(actual_names),
    }
    if actual_names != expected_names or actual != expected:
        raise SemanticMismatch("temporary prediction identity changed")
    return actual


def _assert_closed_paths(repo_root: Path) -> None:
    for relative in (FINAL_CACHE, *ARCHIVE_PATHS):
        path = repo_root / relative
        if path.exists() or path.is_symlink():
            raise PrerequisiteError(f"salvage downstream path is present: {path}")


def _static_state(repo_root: Path) -> dict[str, Any]:
    contract = validate_repository_contract(repo_root)
    output_path = repo_root / contract["implementation"]["output"]
    if output_path.exists() or output_path.is_symlink():
        raise PrerequisiteError(f"salvage audit output already exists: {output_path}")
    _assert_closed_paths(repo_root)
    temporary = repo_root / contract["temporary_cache"]["path"]
    if temporary.is_symlink() or not temporary.is_dir():
        raise PrerequisiteError("retained second-stage temporary cache is missing")
    if {path.name for path in temporary.iterdir()} != set(
        contract["temporary_cache"]["required_surface"]
    ):
        raise SemanticMismatch("retained temporary cache surface changed")
    metrics_path = temporary / "metrics.csv"
    actual_metrics_sha256 = sha256_file(metrics_path)
    if actual_metrics_sha256 != contract["temporary_cache"][
        "expected_metrics_sha256"
    ]:
        raise SemanticMismatch("retained temporary metrics sha256 changed")
    result = read_json(
        repo_root / contract["evidence"]["second_stage_result"]["path"]
    )
    historical = read_json(
        repo_root
        / contract["evidence"]["historical_materialization_audit"]["path"]
    )
    gates = read_json(
        repo_root
        / contract["evidence"]["dual_input_historical_gate_audit"]["path"]
    )
    manifest_path = repo_root / contract["evidence"]["sample_manifest"]["path"]
    manifest = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        result.get("terminal") != "PREREQUISITE_NEEDED"
        or historical.get("terminal") != "PASS"
        or len(manifest) != 275
        or len(set(manifest)) != 275
        or historical.get("second_stage_metrics", {}).get("metrics_sha256")
        != contract["historical_identity"]["metrics_sha256"]
    ):
        raise PrerequisiteError("salvage historical evidence changed")
    return {
        "contract": contract,
        "gates": gates,
        "historical": historical,
        "manifest": manifest,
        "metrics_path": metrics_path,
        "output_path": output_path,
        "result": result,
        "temporary": temporary,
    }


def run_audit(repo_root: Path) -> dict[str, Any]:
    state = _static_state(repo_root)
    contract = state["contract"]
    manifest = state["manifest"]
    metrics_path = state["metrics_path"]
    _fields, rows = read_metrics(metrics_path)
    if len(rows) != len(manifest):
        raise SemanticMismatch("temporary metrics population changed")
    expected_files = [Path(value).name for value in manifest]
    if [row["file"] for row in rows] != expected_files:
        raise SemanticMismatch("temporary metrics row order changed")

    names = [f"{Path(value).stem}.png" for value in manifest]
    prediction_set = validate_prediction_set(
        state["temporary"] / "pred",
        sorted(names),
        contract["historical_identity"]["prediction_set"],
    )
    primary_metrics_path = repo_root / PRIMARY_CACHE / "metrics.csv"
    expected_primary_sha256 = state["historical"]["primary_metrics"][
        "metrics_sha256"
    ]
    if sha256_file(primary_metrics_path) != expected_primary_sha256:
        raise PrerequisiteError("frozen primary metrics identity changed")
    primary_fields, primary_rows = read_csv_rows(primary_metrics_path)
    if not {"file", "image_path", "image_sha256"}.issubset(primary_fields):
        raise PrerequisiteError("frozen primary metrics source fields changed")
    primary_by_file = {row["file"]: row for row in primary_rows}
    if len(primary_by_file) != len(manifest):
        raise PrerequisiteError("frozen primary metrics population changed")

    gate_rows = state["gates"].get("page_samples", [])
    gates_by_file = {
        row["file"]: row["gate_features"]["second_stage_gate_ratio"]
        for row in gate_rows
    }
    if len(gates_by_file) != len(manifest):
        raise PrerequisiteError("historical gate audit population changed")

    constants = contract["semantic_field_sources"]["constants"]
    label_hashes: list[str] = []
    source_hashes: list[str] = []
    for manifest_value, row in zip(manifest, rows):
        image_path = repo_root / manifest_value
        file_name = image_path.name
        primary_row = primary_by_file.get(file_name)
        if (
            primary_row is None
            or primary_row.get("image_path") != manifest_value
            or primary_row.get("image_sha256") != sha256_file(image_path)
        ):
            raise SemanticMismatch(f"source identity changed for {file_name}")
        source_hashes.append(f"{file_name} {primary_row['image_sha256']}")
        label_path = label_path_for(Path(manifest_value))
        absolute_label_path = repo_root / label_path
        label_hashes.append(f"{file_name} {sha256_file(absolute_label_path)}")
        prediction_path = state["temporary"] / "pred" / f"{image_path.stem}.png"
        input_bgr = read_bgr(image_path)
        label_bgr = ensure_same_size(read_bgr(absolute_label_path), input_bgr)
        prediction_bgr = ensure_same_size(read_bgr(prediction_path), input_bgr)
        recomputed = compute_residual_metrics(
            input_bgr,
            label_bgr,
            prediction_bgr,
            change_threshold=12,
            eval_threshold=12,
        )
        expected_pred_path = str(
            repo_root / FINAL_CACHE / "pred" / f"{image_path.stem}.png"
        )
        if file_name not in gates_by_file:
            raise PrerequisiteError(f"historical gate is missing for {file_name}")
        validate_metric_row(
            row,
            expected_file=file_name,
            expected_image_path=manifest_value,
            expected_pred_path=expected_pred_path,
            expected_gate_ratio=float(gates_by_file[file_name]),
            constants=constants,
            recomputed=recomputed,
        )

    observed_label_sha256 = sha256_rows(label_hashes)
    expected_label_sha256 = state["historical"]["patch_summary"][
        "label_content_sha256"
    ]
    if observed_label_sha256 != expected_label_sha256:
        raise SemanticMismatch("train275 label content identity changed")
    candidate_contract = contract["canonical_candidate"]
    candidate = validate_canonical_candidate(
        metrics_path,
        current_repository_root=candidate_contract["current_repository_root"],
        frozen_historical_repository_root=candidate_contract[
            "frozen_historical_repository_root"
        ],
        expected_replacement_count=candidate_contract[
            "expected_replacement_count"
        ],
        expected_sha256=candidate_contract["metrics_sha256"],
    )
    if sha256_file(metrics_path) != contract["temporary_cache"][
        "expected_metrics_sha256"
    ]:
        raise SemanticMismatch("temporary metrics changed during salvage audit")
    _assert_closed_paths(repo_root)
    return {
        "authority": {
            "cache_mutation": False,
            "model_execution": False,
            "result_authority": "semantic_equivalence_audit_only",
            "terminal_successor": (
                "separate_hash_bound_rebaseline_preregistration_only"
            ),
        },
        "canonical_candidate": candidate,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "historical_identity": {
            "metrics_payload_present": False,
            "metrics_sha256": contract["historical_identity"]["metrics_sha256"],
            "status": "not_reproduced",
        },
        "prediction_set": prediction_set,
        "schema_version": 1,
        "semantic_validation": {
            "constant_field_matches": 4 * len(rows),
            "gate_ratio_matches": len(rows),
            "label_content_sha256": observed_label_sha256,
            "path_field_matches": 3 * len(rows),
            "recomputed_field_matches": 8 * len(rows),
            "row_count": len(rows),
            "source_content_sha256": sha256_rows(source_hashes),
        },
        "temporary_cache": {
            "metrics_sha256": sha256_file(metrics_path),
            "path": str(state["temporary"].relative_to(repo_root)),
            "unchanged": True,
        },
        "terminal": "PASS",
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    if path.exists() or path.is_symlink() or temporary.exists():
        raise FileExistsError(path)
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    output_path = ROOT / OUTPUT_PATH
    try:
        result = run_audit(ROOT)
    except PrerequisiteError as error:
        result = {"reason": str(error), "schema_version": 1, "terminal": "PREREQUISITE_NEEDED"}
        write_result(output_path, result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 2
    except SemanticMismatch as error:
        result = {"reason": str(error), "schema_version": 1, "terminal": "KILL"}
        write_result(output_path, result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 3
    write_result(output_path, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
