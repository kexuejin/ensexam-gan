#!/usr/bin/env python3
"""Launch frozen external-layout materialization with recovered cache evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import materialize_external_text_layout_support_train_only as materializer  # noqa: E402


CONTRACT_PATH = Path("docs/external-text-layout-recovered-materializer-launch-v2.json")
EXPECTED_CONTRACT_SHA256 = (
    "26eec02ad8c7dcbbb7ebfea2e03b2b431c35cacff0e78f5c2029f459b00d6978"
)
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
DERIVED_PLAN_PATH = Path(
    "outputs/external-text-layout-recovered-materializer-input-20260815/"
    "effective-plan.json"
)
RESULT_PATH = Path(
    "outputs/external-text-layout-recovered-materializer-input-20260815/result.json"
)
EXPECTED_DERIVED_PLAN_SHA256 = (
    "39d5d801c0507dc965c970927b2d2ea6a2e7d9a2f3f04b27564956093fbca5d4"
)
EXPECTED_SHARED_MATERIALIZER_SHA256 = (
    "2a87d2a21b9141c9ca16e5f11f7ab1f523d59ffacf6f55759517fd0db26aafcf"
)
EXPECTED_SHARED_TEST_SHA256 = (
    "79e64906656871976d29b3ced66fc396f93c1271f25149e587065600ec5f47f2"
)


class RecoveredMaterializationError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecoveredMaterializationError(f"cannot read launcher evidence: {path}") from error
    if not isinstance(value, dict):
        raise RecoveredMaterializationError(f"launcher evidence is not an object: {path}")
    return value


def repo_path(repo_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RecoveredMaterializationError(f"launcher path escaped repository: {value}")
    return repo_root / relative


def validate_artifact(repo_root: Path, artifact: Any, label: str) -> Path:
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise RecoveredMaterializationError(f"{label} artifact contract changed")
    path = repo_path(repo_root, str(artifact["path"]))
    if path.is_symlink() or not path.is_file():
        raise RecoveredMaterializationError(f"missing {label}: {path}")
    actual = materializer.sha256_file(path)
    if actual != artifact["sha256"]:
        raise RecoveredMaterializationError(
            f"{label} sha256 changed: expected {artifact['sha256']}, got {actual}"
        )
    return path


def validate_repository_contract(repo_root: Path) -> dict[str, Any]:
    contract_path = repo_root / CONTRACT_PATH
    if materializer.sha256_file(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise RecoveredMaterializationError("launcher v2 contract sha256 changed")
    contract = read_json(contract_path)
    derived = contract.get("derived_plan", {})
    implementation = contract.get("implementation", {})
    if (
        contract.get("schema_version") != 2
        or contract.get("terminal") != "PREREQUISITE_NEEDED"
        or derived.get("path") != str(DERIVED_PLAN_PATH)
        or derived.get("expected_sha256") != EXPECTED_DERIVED_PLAN_SHA256
        or implementation.get("allowed_files")
        != [
            "scripts/analysis/run_external_text_layout_recovered_materialization.py",
            "tests/test_run_external_text_layout_recovered_materialization.py",
        ]
        or implementation.get("launcher_result") != str(RESULT_PATH)
        or implementation.get("new_dependency") is not False
        or implementation.get("shared_materializer_mutation") is not False
        or implementation.get("site_packages_write") is not False
        or contract.get("supersedes", {}).get("status")
        != "rejected_before_integration"
    ):
        raise RecoveredMaterializationError("launcher v2 contract changed")
    evidence = contract.get("evidence", {})
    artifacts = {
        label: validate_artifact(repo_root, artifact, f"launcher {label}")
        for label, artifact in evidence.items()
    }
    if (
        materializer.sha256_file(artifacts["shared_materializer"])
        != EXPECTED_SHARED_MATERIALIZER_SHA256
        or materializer.sha256_file(artifacts["shared_materializer_test"])
        != EXPECTED_SHARED_TEST_SHA256
    ):
        raise RecoveredMaterializationError("probe-bound shared source changed")
    archive_result = read_json(artifacts["archive_publication_result"])
    archive_verification = read_json(artifacts["archive_publication_verification"])
    if (
        archive_result.get("terminal") != "PASS"
        or archive_result.get("publication", {}).get("status") != "published"
        or archive_verification.get("terminal") != "PASS"
    ):
        raise RecoveredMaterializationError("archive publication evidence is not PASS")
    return contract


def validate_execution_authority(repo_root: Path) -> None:
    ledger = read_json(repo_root / LEDGER_PATH)
    active = ledger.get("active_iteration", {})
    statuses = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    required = {
        "external_text_layout_recovered_archive_publication": "passed",
        "external_text_layout_recovered_materializer_input_preregistration": "passed",
        "external_text_layout_recovered_materializer_launch_v2_preregistration": "passed",
        "external_text_layout_recovered_materializer_launch_v2_integration": "passed",
        "external_text_layout_support_train_only_diagnostic": "pending",
    }
    if active.get("terminal") != "PREREQUISITE_NEEDED" or any(
        statuses.get(name) != status for name, status in required.items()
    ):
        raise RecoveredMaterializationError("recovered launcher execution authority is closed")


def _validate_relative_link(repo_root: Path, registered: Any, label: str) -> Path:
    if not isinstance(registered, dict) or set(registered) != {"path", "target"}:
        raise RecoveredMaterializationError(f"{label} link contract changed")
    link = repo_path(repo_root, str(registered["path"]))
    if not link.is_symlink():
        raise RecoveredMaterializationError(f"missing {label} archive link")
    target = os.readlink(link)
    if Path(target).is_absolute() or target != registered["target"]:
        raise RecoveredMaterializationError(f"{label} archive target changed")
    try:
        resolved = link.resolve()
    except (OSError, RuntimeError) as error:
        raise RecoveredMaterializationError(f"{label} archive link is broken") from error
    if not resolved.is_dir() or resolved.is_symlink():
        raise RecoveredMaterializationError(f"{label} archive source changed")
    return link


def _prediction_identity(prediction_dir: Path) -> dict[str, Any]:
    if prediction_dir.is_symlink() or not prediction_dir.is_dir():
        raise RecoveredMaterializationError("archive prediction directory changed")
    predictions = list(prediction_dir.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in predictions):
        raise RecoveredMaterializationError("archive prediction surface changed")
    names = sorted(path.name for path in predictions)
    rows = [f"{name} {materializer.sha256_file(prediction_dir / name)}" for name in names]
    return {
        "content_sha256": materializer.sha256_rows(rows),
        "count": len(names),
        "filename_sha256": materializer.sha256_rows(names),
    }


def validate_archive_inputs(
    repo_root: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    evidence = contract["evidence"]
    overlay = read_json(repo_path(repo_root, evidence["recovered_input_v1"]["path"]))
    archive_result = read_json(
        repo_path(repo_root, evidence["archive_publication_result"]["path"])
    )
    required_links = overlay.get("input_overlay", {}).get(
        "required_archive_links", {}
    )
    actual: dict[str, Any] = {}
    expected = archive_result.get("publication", {}).get("linked_caches", {})
    for stage in ("primary", "second_stage"):
        link = _validate_relative_link(
            repo_root, required_links.get(stage), stage.replace("_", "-")
        )
        metrics = link / "metrics.csv"
        if metrics.is_symlink() or not metrics.is_file():
            raise RecoveredMaterializationError(f"{stage} archive metrics changed")
        actual[stage] = {
            "metrics_sha256": materializer.sha256_file(metrics),
            "prediction_set": _prediction_identity(link / "pred"),
        }
        if actual[stage] != expected.get(stage):
            raise RecoveredMaterializationError(
                f"{stage} archive cache identity changed"
            )
    return actual


def build_derived_plan(
    contract: dict[str, Any], original_plan: dict[str, Any]
) -> tuple[dict[str, Any], bytes]:
    semantic = contract["derived_plan"]["semantic_changes_from_original"]
    metrics_change = semantic["evidence.second_stage_metrics.sha256"]
    original_metrics = original_plan.get("evidence", {}).get("second_stage_metrics")
    if not isinstance(original_metrics, dict) or original_metrics.get("sha256") != metrics_change["before"]:
        raise RecoveredMaterializationError("original second-stage metrics identity changed")
    derived_plan = json.loads(json.dumps(original_plan))
    derived_plan["evidence"]["second_stage_metrics"]["sha256"] = metrics_change[
        "after"
    ]
    derived_plan["recovered_input_overlay"] = semantic["recovered_input_overlay"]
    restored = json.loads(json.dumps(derived_plan))
    del restored["recovered_input_overlay"]
    restored["evidence"]["second_stage_metrics"]["sha256"] = metrics_change[
        "before"
    ]
    if restored != original_plan:
        raise RecoveredMaterializationError("derived plan changed outside v2 contract")
    try:
        materializer.validate_plan(derived_plan)
    except RuntimeError as error:
        raise RecoveredMaterializationError(str(error)) from error
    payload = (
        json.dumps(derived_plan, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != contract["derived_plan"]["expected_sha256"]:
        raise RecoveredMaterializationError(
            f"derived plan sha256 changed: {actual_sha256}"
        )
    return derived_plan, payload


def write_or_validate_derived_plan(
    path: Path, payload: bytes, expected_sha256: str
) -> str:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RecoveredMaterializationError("existing derived plan changed")
        if materializer.sha256_file(path) != expected_sha256:
            raise RecoveredMaterializationError("existing derived plan sha256 changed")
        return "existing"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    if temporary.exists() or temporary.is_symlink():
        raise RecoveredMaterializationError("stale derived plan candidate exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    if materializer.sha256_file(path) != expected_sha256:
        raise RecoveredMaterializationError("derived plan write validation failed")
    return "written"


def validate_materialization_output(
    repo_root: Path, derived_plan_path: Path, expected_plan_sha256: str
) -> dict[str, Any]:
    output_root = repo_root / materializer.OUTPUT_ROOT
    manifest_path = output_root / "manifest.json"
    page_dir = output_root / "pages"
    if (
        output_root.is_symlink()
        or not output_root.is_dir()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or page_dir.is_symlink()
        or not page_dir.is_dir()
    ):
        raise RecoveredMaterializationError("materialization output surface changed")
    manifest = read_json(manifest_path)
    expected_plan = {
        "path": str(derived_plan_path.relative_to(repo_root)),
        "sha256": expected_plan_sha256,
    }
    rows = manifest.get("pages")
    if (
        manifest.get("terminal") != "PASS"
        or manifest.get("plan") != expected_plan
        or not isinstance(rows, list)
        or len(rows) != 275
    ):
        raise RecoveredMaterializationError("materialization manifest changed")
    expected_pages: set[str] = set()
    content_rows: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("file"), str):
            raise RecoveredMaterializationError("materialization page row changed")
        page = page_dir / f"{Path(row['file']).stem}.npz"
        expected_pages.add(page.name)
        if page.is_symlink() or not page.is_file():
            raise RecoveredMaterializationError("materialization page payload changed")
        actual_sha256 = materializer.sha256_file(page)
        if actual_sha256 != row.get("npz_sha256"):
            raise RecoveredMaterializationError("materialization page sha256 changed")
        content_rows.append(f"{row['file']} {actual_sha256}")
    if len(expected_pages) != len(rows) or {
        path.name for path in page_dir.iterdir()
    } != expected_pages:
        raise RecoveredMaterializationError("materialization page population changed")
    return {
        "content_sha256": materializer.sha256_rows(content_rows),
        "manifest": str(manifest_path.relative_to(repo_root)),
        "manifest_sha256": materializer.sha256_file(manifest_path),
        "output_root": str(output_root.relative_to(repo_root)),
        "terminal": "PASS",
        "train_count": len(rows),
    }


def build_launcher_result(
    contract: dict[str, Any],
    archive_inputs: dict[str, Any],
    materialization: dict[str, Any],
    plan_state_before: str,
) -> dict[str, Any]:
    metrics_change = contract["derived_plan"]["semantic_changes_from_original"][
        "evidence.second_stage_metrics.sha256"
    ]
    return {
        "archive_cache_identities": archive_inputs,
        "authority": {
            "candidate_inference": False,
            "quality_evaluation": False,
            "result_authority": "recovered_materializer_launcher_v2",
        },
        "derived_plan": {
            "path": contract["derived_plan"]["path"],
            "sha256": contract["derived_plan"]["expected_sha256"],
            "state_before": plan_state_before,
        },
        "historical_second_stage_metrics_sha256": metrics_change["before"],
        "materialization": materialization,
        "recovered_second_stage_metrics_sha256": metrics_change["after"],
        "schema_version": 2,
        "terminal": "PASS",
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(path)
    try:
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def validate_existing_result(
    repo_root: Path,
    contract: dict[str, Any],
    archive_inputs: dict[str, Any],
) -> dict[str, Any]:
    derived_plan_path = repo_path(repo_root, contract["derived_plan"]["path"])
    if (
        derived_plan_path.is_symlink()
        or not derived_plan_path.is_file()
        or materializer.sha256_file(derived_plan_path)
        != contract["derived_plan"]["expected_sha256"]
    ):
        raise RecoveredMaterializationError("terminal derived plan changed")
    materialization = validate_materialization_output(
        repo_root,
        derived_plan_path,
        contract["derived_plan"]["expected_sha256"],
    )
    result_path = repo_root / RESULT_PATH
    if result_path.is_symlink() or not result_path.is_file():
        raise RecoveredMaterializationError("launcher terminal result surface changed")
    result = read_json(result_path)
    if (
        result.get("terminal") != "PASS"
        or result.get("schema_version") != 2
        or result.get("archive_cache_identities") != archive_inputs
        or result.get("materialization") != materialization
        or result.get("derived_plan", {}).get("path")
        != contract["derived_plan"]["path"]
        or result.get("derived_plan", {}).get("sha256")
        != contract["derived_plan"]["expected_sha256"]
        or result.get("derived_plan", {}).get("state_before")
        not in {"absent", "existing"}
    ):
        raise RecoveredMaterializationError("launcher terminal result changed")
    return result


def run_launcher(repo_root: Path) -> dict[str, Any]:
    contract = validate_repository_contract(repo_root)
    original_plan_path = validate_artifact(
        repo_root, contract["evidence"]["original_plan"], "launcher original plan"
    )
    original_plan = read_json(original_plan_path)
    archive_inputs = validate_archive_inputs(repo_root, contract)
    _derived_plan, payload = build_derived_plan(contract, original_plan)
    result_path = repo_root / RESULT_PATH
    if result_path.exists() or result_path.is_symlink():
        return validate_existing_result(repo_root, contract, archive_inputs)
    validate_execution_authority(repo_root)
    derived_plan_path = repo_path(repo_root, contract["derived_plan"]["path"])
    plan_state_before = (
        "existing" if derived_plan_path.exists() or derived_plan_path.is_symlink() else "absent"
    )
    write_or_validate_derived_plan(
        derived_plan_path, payload, contract["derived_plan"]["expected_sha256"]
    )
    output_root = repo_root / materializer.OUTPUT_ROOT
    temporary_root = output_root.with_name(f".{output_root.name}.materializing")
    marker_path, _cleanup_root = materializer.published_transaction_paths(output_root)
    if output_root.exists() and not temporary_root.exists() and not marker_path.exists():
        materialization = validate_materialization_output(
            repo_root,
            derived_plan_path,
            contract["derived_plan"]["expected_sha256"],
        )
    else:
        materializer_result = materializer.materialize(
            repo_root=repo_root,
            plan_path=derived_plan_path.relative_to(repo_root),
            ledger_path=LEDGER_PATH,
            worker_count=1,
        )
        if materializer_result.get("terminal") != "PASS":
            raise RecoveredMaterializationError("shared materializer did not PASS")
        materialization = validate_materialization_output(
            repo_root,
            derived_plan_path,
            contract["derived_plan"]["expected_sha256"],
        )
    result = build_launcher_result(
        contract, archive_inputs, materialization, plan_state_before
    )
    try:
        write_result(result_path, result)
        if read_json(result_path) != result:
            raise RecoveredMaterializationError(
                "launcher result failed read-back validation"
            )
    except BaseException:
        result_path.unlink(missing_ok=True)
        result_path.with_name(f".{result_path.name}.writing").unlink(
            missing_ok=True
        )
        raise
    return result


def main() -> int:
    try:
        result = run_launcher(ROOT)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "reason": str(error),
                    "schema_version": 2,
                    "terminal": "PREREQUISITE_NEEDED",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
