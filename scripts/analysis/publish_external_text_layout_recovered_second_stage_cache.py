#!/usr/bin/env python3
"""Publish the audited recovered second-stage cache without model execution."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import external_text_layout_cache_metrics as cache_metrics  # noqa: E402
from scripts.analysis import reconstruct_external_text_layout_frozen_caches as reconstruction  # noqa: E402


CONTRACT_PATH = Path(
    "docs/external-text-layout-second-stage-recovered-cache-publication-v1.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "75a988d48fce658ca22e6e7deff9eab3759a4a13557161a99b4f4b3f62925163"
)
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
RESULT_PATH = Path(
    "outputs/external-text-layout-second-stage-recovered-publication-20260815/"
    "result.json"
)
EXPECTED_PATHS = {
    "archive_primary": (
        "outputs/archive/sign-separated-residual-repair-20260810/train275-primary"
    ),
    "archive_second_stage": (
        "outputs/archive/sign-separated-residual-repair-20260810/"
        "train275-frozen-pipeline"
    ),
    "final": "outputs/sign-separated-residual-repair-train275-frozen-pipeline-v1",
    "primary": "outputs/sign-separated-residual-repair-train275-primary-v1",
    "temporary": (
        "outputs/.sign-separated-residual-repair-train275-frozen-pipeline-v1."
        "materializing"
    ),
}
EXPECTED_PREDICTION_SET = {
    "content_sha256": "2ffa40fc0c9b2a7e721d560f6f12edfe2ccdc1c1988582fa7a8104665cdc088a",
    "count": 275,
    "filename_sha256": "8c75e1dbebc162f316c24137540add99e51877e07aedc6abb419de872c58b5de",
}


class PublicationError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicationError(f"cannot read publication evidence: {path}") from error
    if not isinstance(value, dict):
        raise PublicationError(f"publication evidence is not an object: {path}")
    return value


def repo_path(repo_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise PublicationError(f"publication path escaped repository: {value}")
    return repo_root / relative


def validate_artifact(repo_root: Path, artifact: Any, label: str) -> Path:
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise PublicationError(f"{label} artifact contract changed")
    path = repo_path(repo_root, str(artifact["path"]))
    if not path.is_file() or path.is_symlink():
        raise PublicationError(f"missing {label}: {path}")
    actual = reconstruction.sha256_file(path)
    if actual != artifact["sha256"]:
        raise PublicationError(
            f"{label} sha256 changed: expected {artifact['sha256']}, got {actual}"
        )
    return path


def validate_repository_contract(repo_root: Path) -> dict[str, Any]:
    path = repo_root / CONTRACT_PATH
    if reconstruction.sha256_file(path) != EXPECTED_CONTRACT_SHA256:
        raise PublicationError("recovered publication contract sha256 changed")
    contract = read_json(path)
    recovered = contract.get("recovered_identity", {})
    canonicalization = recovered.get("canonicalization", {})
    if (
        contract.get("schema_version") != 1
        or contract.get("state")
        != "preregistered_recovered_second_stage_cache_publication"
        or contract.get("terminal") != "PREREQUISITE_NEEDED"
        or contract.get("paths") != EXPECTED_PATHS
        or contract.get("implementation", {}).get("allowed_files")
        != [
            "scripts/analysis/"
            "publish_external_text_layout_recovered_second_stage_cache.py",
            "tests/test_publish_external_text_layout_recovered_second_stage_cache.py",
        ]
        or contract.get("implementation", {}).get("new_dependency") is not False
        or contract.get("implementation", {}).get("site_packages_write") is not False
        or contract.get("implementation", {}).get("result") != str(RESULT_PATH)
        or recovered.get("metrics_sha256")
        != "79fd61278e689a0003e37a5bdf20f856184b49c8fdb3af8ad9af03a3a13c451b"
        or recovered.get("prediction_set") != EXPECTED_PREDICTION_SET
        or canonicalization.get("current_repository_root")
        != str(repo_root.resolve())
        or canonicalization.get("expected_data_rows") != 275
        or canonicalization.get("expected_replacement_count") != 275
        or contract.get("historical_identity", {}).get("status")
        != "not_reproduced"
    ):
        raise PublicationError("recovered publication contract changed")
    for label, artifact in contract.get("evidence", {}).items():
        validate_artifact(repo_root, artifact, f"publication {label}")
    validate_artifact(
        repo_root,
        contract.get("implementation", {}).get("canonicalizer"),
        "publication canonicalizer",
    )
    return contract


def validate_execution_authority(repo_root: Path) -> None:
    ledger = read_json(repo_root / LEDGER_PATH)
    active = ledger.get("active_iteration", {})
    prerequisites = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    if (
        active.get("terminal") != "PREREQUISITE_NEEDED"
        or prerequisites.get(
            "external_text_layout_second_stage_recovered_cache_"
            "publication_preregistration"
        )
        != "passed"
        or prerequisites.get(
            "external_text_layout_second_stage_recovered_cache_"
            "publication_integration"
        )
        != "passed"
        or prerequisites.get("external_text_layout_second_stage_cache_salvage_audit")
        != "passed"
        or prerequisites.get(
            "external_text_layout_second_stage_cache_reconstruction"
        )
        != "pending"
    ):
        raise PublicationError("recovered publication execution authority is closed")


def _prediction_names(cache_dir: Path) -> list[str]:
    prediction_dir = cache_dir / "pred"
    if not prediction_dir.is_dir() or prediction_dir.is_symlink():
        raise PublicationError(f"prediction directory changed: {prediction_dir}")
    paths = list(prediction_dir.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise PublicationError(f"prediction surface changed: {prediction_dir}")
    return sorted(path.name for path in paths)


def validate_prediction_identity(
    cache_dir: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    names = _prediction_names(cache_dir)
    try:
        return reconstruction.validate_prediction_set(
            cache_dir / "pred", names, identity
        )
    except RuntimeError as error:
        raise PublicationError(str(error)) from error


def validate_cache_identity(
    cache_dir: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    names = _prediction_names(cache_dir)
    expected = {
        "metrics_sha256": identity["metrics_sha256"],
        "prediction_set": identity["prediction_set"],
    }
    try:
        return reconstruction.validate_reconstructed_cache(
            cache_dir, expected_names=names, expected=expected
        )
    except RuntimeError as error:
        raise PublicationError(str(error)) from error


def _assert_archive_paths_absent(repo_root: Path, paths: dict[str, str]) -> None:
    for name in ("archive_primary", "archive_second_stage"):
        path = repo_path(repo_root, paths[name])
        if path.exists() or path.is_symlink():
            raise PublicationError(f"archive path must remain absent: {path}")


def publish_registered_cache(
    repo_root: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    paths = contract["paths"]
    temporary = repo_path(repo_root, paths["temporary"])
    final = repo_path(repo_root, paths["final"])
    _assert_archive_paths_absent(repo_root, paths)
    if final.exists() or final.is_symlink():
        raise PublicationError(f"recovered final path already exists: {final}")
    if temporary.is_symlink() or not temporary.is_dir():
        raise PublicationError(f"recovered temporary cache is missing: {temporary}")
    identity = contract["recovered_identity"]
    prediction_set = validate_prediction_identity(
        temporary, identity["prediction_set"]
    )
    metrics_path = temporary / "metrics.csv"
    actual_metrics_sha256 = reconstruction.sha256_file(metrics_path)
    canonicalization = identity["canonicalization"]
    if actual_metrics_sha256 == canonicalization["source_metrics_sha256"]:
        try:
            metrics = cache_metrics.canonicalize_repository_root(
                metrics_path,
                current_repository_root=canonicalization[
                    "current_repository_root"
                ],
                frozen_historical_repository_root=canonicalization[
                    "frozen_historical_repository_root"
                ],
                expected_data_rows=canonicalization["expected_data_rows"],
                expected_replacement_count=canonicalization[
                    "expected_replacement_count"
                ],
                expected_metrics_sha256_before=canonicalization[
                    "source_metrics_sha256"
                ],
                expected_metrics_sha256_after=identity["metrics_sha256"],
            )
        except RuntimeError as error:
            raise PublicationError(str(error)) from error
        metrics_state_before = "source"
    elif actual_metrics_sha256 == identity["metrics_sha256"]:
        metrics = {
            "data_rows": canonicalization["expected_data_rows"],
            "metrics_sha256_after": identity["metrics_sha256"],
            "metrics_sha256_before": identity["metrics_sha256"],
            "replacement_count": 0,
        }
        metrics_state_before = "already_canonical"
    else:
        raise PublicationError(
            f"recovered temporary metrics identity is unknown: {actual_metrics_sha256}"
        )
    temporary_cache = validate_cache_identity(temporary, identity)
    temporary.replace(final)
    try:
        final_cache = validate_cache_identity(final, identity)
    except BaseException:
        final.replace(temporary)
        raise
    _assert_archive_paths_absent(repo_root, paths)
    return {
        "final_cache": final_cache,
        "metrics_canonicalization": metrics,
        "metrics_state_before": metrics_state_before,
        "prediction_set_before": prediction_set,
        "status": "published",
        "temporary_cache_before": temporary_cache,
    }


def build_publication_result(
    contract: dict[str, Any], publication: dict[str, Any]
) -> dict[str, Any]:
    return {
        "authority": {
            "archive_publication": False,
            "model_execution": False,
            "result_authority": "recovered_second_stage_cache_publication",
        },
        "historical_identity": contract["historical_identity"],
        "publication": publication,
        "recovered_identity": contract["recovered_identity"],
        "schema_version": 1,
        "terminal": "PASS",
    }


def validate_existing_publication(
    repo_root: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    paths = contract["paths"]
    _assert_archive_paths_absent(repo_root, paths)
    temporary = repo_path(repo_root, paths["temporary"])
    if temporary.exists() or temporary.is_symlink():
        raise PublicationError(
            "recovered temporary cache must be absent after publication"
        )
    final = repo_path(repo_root, paths["final"])
    if final.is_symlink() or not final.is_dir():
        raise PublicationError(f"recovered final cache is invalid: {final}")
    result_path = repo_root / RESULT_PATH
    if result_path.is_symlink() or not result_path.is_file():
        raise PublicationError("recovered final cache has no terminal PASS result")
    result = read_json(result_path)
    expected_authority = {
        "archive_publication": False,
        "model_execution": False,
        "result_authority": "recovered_second_stage_cache_publication",
    }
    publication = result.get("publication")
    if (
        result.get("authority") != expected_authority
        or result.get("historical_identity") != contract["historical_identity"]
        or result.get("recovered_identity") != contract["recovered_identity"]
        or result.get("schema_version") != 1
        or result.get("terminal") != "PASS"
        or not isinstance(publication, dict)
        or set(publication)
        != {
            "final_cache",
            "metrics_canonicalization",
            "metrics_state_before",
            "prediction_set_before",
            "status",
            "temporary_cache_before",
        }
        or publication.get("status") != "published"
    ):
        raise PublicationError("recovered publication result changed")
    final_cache = validate_cache_identity(final, contract["recovered_identity"])
    if publication["final_cache"] != final_cache:
        raise PublicationError("recovered publication result cache identity changed")
    return result


def run_publication(repo_root: Path) -> dict[str, Any]:
    contract = validate_repository_contract(repo_root)
    result_path = repo_root / RESULT_PATH
    final = repo_path(repo_root, contract["paths"]["final"])
    if final.exists() or final.is_symlink():
        return validate_existing_publication(repo_root, contract)
    validate_execution_authority(repo_root)
    if result_path.exists() or result_path.is_symlink():
        raise PublicationError(
            f"publication result exists without final cache: {result_path}"
        )
    primary_result = read_json(
        repo_root / contract["evidence"]["primary_recovery"]["path"]
    )
    primary = repo_path(repo_root, contract["paths"]["primary"])
    validate_cache_identity(
        primary,
        {
            "metrics_sha256": primary_result["cache"]["metrics_sha256"],
            "prediction_set": primary_result["cache"]["prediction_set"],
        },
    )
    with reconstruction.runtime.exclusive_run_lock(
        reconstruction.runtime.HOST_USER_RUN_LOCK_PATH
    ):
        reconstruction.materializer.assert_no_conflicting_model_processes()
        publication = publish_registered_cache(repo_root, contract)
        result = build_publication_result(contract, publication)
        try:
            write_result(result_path, result)
            if read_json(result_path) != result:
                raise PublicationError("publication result failed read-back validation")
        except BaseException:
            result_path.unlink(missing_ok=True)
            result_path.with_name(f".{result_path.name}.writing").unlink(
                missing_ok=True
            )
            temporary = repo_path(repo_root, contract["paths"]["temporary"])
            if final.is_dir() and not final.is_symlink() and not temporary.exists():
                final.replace(temporary)
            raise
    return result


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    if path.exists() or path.is_symlink() or temporary.exists():
        raise FileExistsError(path)
    try:
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    try:
        result = run_publication(ROOT)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {"reason": str(error), "schema_version": 1, "terminal": "PREREQUISITE_NEEDED"},
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
