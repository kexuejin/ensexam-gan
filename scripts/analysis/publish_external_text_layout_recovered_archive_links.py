#!/usr/bin/env python3
"""Publish validated recovered cache archive links without model execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import reconstruct_external_text_layout_frozen_caches as reconstruction  # noqa: E402


CONTRACT_PATH = Path("docs/external-text-layout-recovered-archive-publication-v1.json")
EXPECTED_CONTRACT_SHA256 = (
    "088ca14e0b9497d5c4c8150ded5a29b8c77c55b2edb451cc830cea9e15003fa2"
)
LEDGER_PATH = Path("docs/current-primary-quality-loop-ledger.json")
RESULT_PATH = Path(
    "outputs/external-text-layout-recovered-archive-publication-20260815/result.json"
)
EXPECTED_PATHS = {
    "archive_primary": (
        "outputs/archive/sign-separated-residual-repair-20260810/train275-primary"
    ),
    "archive_second_stage": (
        "outputs/archive/sign-separated-residual-repair-20260810/"
        "train275-frozen-pipeline"
    ),
    "primary": "outputs/sign-separated-residual-repair-train275-primary-v1",
    "second_stage": (
        "outputs/sign-separated-residual-repair-train275-frozen-pipeline-v1"
    ),
}
EXPECTED_CACHES = {
    "primary": {
        "metrics_sha256": (
            "efd58814583089e888482a7e1604efc1d19ee5f514085cbef0e0c6cabf479846"
        ),
        "prediction_set": {
            "content_sha256": (
                "6400c9413af963e3de280e348bd635cd962e5387c2e975e930036d320214274a"
            ),
            "count": 275,
            "filename_sha256": (
                "8c75e1dbebc162f316c24137540add99e51877e07aedc6abb419de872c58b5de"
            ),
        },
    },
    "second_stage": {
        "metrics_sha256": (
            "79fd61278e689a0003e37a5bdf20f856184b49c8fdb3af8ad9af03a3a13c451b"
        ),
        "prediction_set": {
            "content_sha256": (
                "2ffa40fc0c9b2a7e721d560f6f12edfe2ccdc1c1988582fa7a8104665cdc088a"
            ),
            "count": 275,
            "filename_sha256": (
                "8c75e1dbebc162f316c24137540add99e51877e07aedc6abb419de872c58b5de"
            ),
        },
        "provenance": (
            "semantic_equivalence_audit_pass_not_historical_payload_reproduction"
        ),
    },
}


class ArchivePublicationError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArchivePublicationError(f"cannot read archive evidence: {path}") from error
    if not isinstance(value, dict):
        raise ArchivePublicationError(f"archive evidence is not an object: {path}")
    return value


def repo_path(repo_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ArchivePublicationError(f"archive path escaped repository: {value}")
    return repo_root / relative


def validate_artifact(repo_root: Path, artifact: Any, label: str) -> Path:
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
        raise ArchivePublicationError(f"{label} artifact contract changed")
    path = repo_path(repo_root, str(artifact["path"]))
    if not path.is_file() or path.is_symlink():
        raise ArchivePublicationError(f"missing {label}: {path}")
    actual = reconstruction.sha256_file(path)
    if actual != artifact["sha256"]:
        raise ArchivePublicationError(
            f"{label} sha256 changed: expected {artifact['sha256']}, got {actual}"
        )
    return path


def validate_repository_contract(repo_root: Path) -> dict[str, Any]:
    path = repo_root / CONTRACT_PATH
    if reconstruction.sha256_file(path) != EXPECTED_CONTRACT_SHA256:
        raise ArchivePublicationError("recovered archive contract sha256 changed")
    contract = read_json(path)
    implementation = contract.get("implementation", {})
    if (
        contract.get("schema_version") != 1
        or contract.get("state") != "preregistered_recovered_archive_publication"
        or contract.get("terminal") != "PREREQUISITE_NEEDED"
        or contract.get("paths") != EXPECTED_PATHS
        or contract.get("expected_caches") != EXPECTED_CACHES
        or contract.get("historical_identity", {}).get("status")
        != "not_reproduced"
        or implementation.get("allowed_files")
        != [
            "scripts/analysis/"
            "publish_external_text_layout_recovered_archive_links.py",
            "tests/test_publish_external_text_layout_recovered_archive_links.py",
        ]
        or implementation.get("new_dependency") is not False
        or implementation.get("site_packages_write") is not False
        or implementation.get("result") != str(RESULT_PATH)
    ):
        raise ArchivePublicationError("recovered archive contract changed")
    evidence = contract.get("evidence", {})
    manifest = evidence.get("manifest")
    if not isinstance(manifest, dict) or set(manifest) != {"count", "path", "sha256"}:
        raise ArchivePublicationError("archive manifest contract changed")
    manifest_path = repo_path(repo_root, str(manifest["path"]))
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ArchivePublicationError(f"missing archive manifest: {manifest_path}")
    if reconstruction.sha256_file(manifest_path) != manifest["sha256"]:
        raise ArchivePublicationError("archive manifest sha256 changed")
    lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != manifest["count"]:
        raise ArchivePublicationError("archive manifest count changed")
    for label, artifact in evidence.items():
        if label != "manifest":
            validate_artifact(repo_root, artifact, f"archive {label}")
    validate_artifact(
        repo_root,
        implementation.get("validation_helper"),
        "archive validation helper",
    )
    return contract


def manifest_prediction_names(repo_root: Path, contract: dict[str, Any]) -> list[str]:
    manifest = contract["evidence"]["manifest"]
    path = repo_path(repo_root, manifest["path"])
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    try:
        names = reconstruction.expected_prediction_names(lines)
    except RuntimeError as error:
        raise ArchivePublicationError(str(error)) from error
    if len(names) != manifest["count"]:
        raise ArchivePublicationError("archive prediction-name count changed")
    return names


def validate_execution_authority(repo_root: Path) -> None:
    ledger = read_json(repo_root / LEDGER_PATH)
    active = ledger.get("active_iteration", {})
    prerequisites = {
        item.get("id"): item.get("status")
        for item in active.get("prerequisites", [])
        if isinstance(item, dict)
    }
    required = {
        "external_text_layout_recovered_archive_publication_preregistration": "passed",
        "external_text_layout_recovered_archive_publication_integration": "passed",
        "external_text_layout_second_stage_cache_reconstruction": "passed",
        "external_text_layout_second_stage_recovered_cache_publication": "passed",
        "external_text_layout_support_train_only_diagnostic": "pending",
    }
    if active.get("terminal") != "PREREQUISITE_NEEDED" or any(
        prerequisites.get(name) != status for name, status in required.items()
    ):
        raise ArchivePublicationError("recovered archive execution authority is closed")


def validate_source_caches(
    repo_root: Path, contract: dict[str, Any], names: list[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in ("primary", "second_stage"):
        path = repo_path(repo_root, contract["paths"][stage])
        expected = contract["expected_caches"][stage]
        try:
            result[stage] = reconstruction.validate_reconstructed_cache(
                path,
                expected_names=names,
                expected={
                    "metrics_sha256": expected["metrics_sha256"],
                    "prediction_set": expected["prediction_set"],
                },
            )
        except RuntimeError as error:
            raise ArchivePublicationError(str(error)) from error
    return result


def _temporary_link_path(final: Path) -> Path:
    return final.with_name(f".{final.name}.publishing")


def _relative_target(link: Path, source: Path) -> str:
    return os.path.relpath(source, start=link.parent)


def _validate_relative_link(link: Path, source: Path, label: str) -> str:
    if not link.is_symlink():
        raise ArchivePublicationError(f"{label} is not a relative symlink: {link}")
    target = os.readlink(link)
    expected = _relative_target(link, source)
    try:
        resolves_to_source = link.resolve() == source.resolve()
    except (OSError, RuntimeError):
        resolves_to_source = False
    if Path(target).is_absolute() or target != expected or not resolves_to_source:
        raise ArchivePublicationError(
            f"{label} is not the registered relative symlink: {link}"
        )
    return target


def _link_pairs(repo_root: Path, contract: dict[str, Any]):
    paths = contract["paths"]
    return {
        "primary": (
            repo_path(repo_root, paths["archive_primary"]),
            repo_path(repo_root, paths["primary"]),
        ),
        "second_stage": (
            repo_path(repo_root, paths["archive_second_stage"]),
            repo_path(repo_root, paths["second_stage"]),
        ),
    }


def preflight_link_states(
    repo_root: Path, contract: dict[str, Any]
) -> dict[str, str]:
    states: dict[str, str] = {}
    for stage, (final, source) in _link_pairs(repo_root, contract).items():
        temporary = _temporary_link_path(final)
        final_present = final.exists() or final.is_symlink()
        temporary_present = temporary.exists() or temporary.is_symlink()
        if final_present:
            _validate_relative_link(final, source, f"archive {stage}")
            if temporary_present:
                raise ArchivePublicationError(
                    f"archive {stage} has both final and temporary links"
                )
            states[stage] = "existing"
        elif temporary_present:
            _validate_relative_link(temporary, source, f"temporary archive {stage}")
            states[stage] = "temporary"
        else:
            states[stage] = "absent"
    return states


def _create_temporary_link(temporary: Path, source: Path) -> None:
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.symlink_to(_relative_target(temporary, source), target_is_directory=True)


def _promote_link(temporary: Path, final: Path) -> None:
    temporary.replace(final)


def rollback_links(
    repo_root: Path, contract: dict[str, Any], states_before: dict[str, str]
) -> None:
    for stage, (final, _source) in reversed(
        list(_link_pairs(repo_root, contract).items())
    ):
        temporary = _temporary_link_path(final)
        state = states_before[stage]
        if state == "existing":
            continue
        if state == "temporary" and final.is_symlink() and not temporary.is_symlink():
            final.replace(temporary)
            continue
        if final.is_symlink():
            final.unlink()
        if state == "absent" and temporary.is_symlink():
            temporary.unlink()


def validate_linked_caches(
    repo_root: Path, contract: dict[str, Any], names: list[str]
) -> tuple[dict[str, str], dict[str, Any]]:
    links: dict[str, str] = {}
    caches: dict[str, Any] = {}
    for stage, (final, source) in _link_pairs(repo_root, contract).items():
        links[stage] = _validate_relative_link(final, source, f"archive {stage}")
        expected = contract["expected_caches"][stage]
        try:
            caches[stage] = reconstruction.validate_cache(
                final,
                expected_names=names,
                expected={
                    "metrics_sha256": expected["metrics_sha256"],
                    "prediction_set": expected["prediction_set"],
                },
            )
        except RuntimeError as error:
            raise ArchivePublicationError(str(error)) from error
    return links, caches


def publish_registered_links(
    repo_root: Path, contract: dict[str, Any], names: list[str]
) -> dict[str, Any]:
    cache_identities = validate_source_caches(repo_root, contract, names)
    states_before = preflight_link_states(repo_root, contract)
    try:
        for stage, (final, source) in _link_pairs(repo_root, contract).items():
            if states_before[stage] == "absent":
                temporary = _temporary_link_path(final)
                _create_temporary_link(temporary, source)
                _validate_relative_link(
                    temporary, source, f"temporary archive {stage}"
                )
        for stage, (final, source) in _link_pairs(repo_root, contract).items():
            if states_before[stage] != "existing":
                temporary = _temporary_link_path(final)
                _validate_relative_link(
                    temporary, source, f"temporary archive {stage}"
                )
                _promote_link(temporary, final)
        links, linked_caches = validate_linked_caches(repo_root, contract, names)
    except BaseException:
        rollback_links(repo_root, contract, states_before)
        raise
    return {
        "cache_identities_before": cache_identities,
        "linked_caches": linked_caches,
        "links": links,
        "state_before": states_before,
        "status": "published",
    }


def build_publication_result(
    contract: dict[str, Any], publication: dict[str, Any]
) -> dict[str, Any]:
    return {
        "authority": {
            "cache_mutation": False,
            "model_execution": False,
            "result_authority": "recovered_archive_publication",
        },
        "historical_identity": contract["historical_identity"],
        "publication": publication,
        "recovered_second_stage_identity": contract["expected_caches"][
            "second_stage"
        ],
        "schema_version": 1,
        "terminal": "PASS",
    }


def validate_existing_publication(
    repo_root: Path,
    contract: dict[str, Any],
    names: list[str],
    source_caches: dict[str, Any],
) -> dict[str, Any]:
    states = preflight_link_states(repo_root, contract)
    if states != {"primary": "existing", "second_stage": "existing"}:
        raise ArchivePublicationError("archive links are not terminal")
    result_path = repo_root / RESULT_PATH
    if result_path.is_symlink() or not result_path.is_file():
        raise ArchivePublicationError("archive links have no terminal PASS result")
    result = read_json(result_path)
    publication = result.get("publication")
    if (
        result.get("authority")
        != {
            "cache_mutation": False,
            "model_execution": False,
            "result_authority": "recovered_archive_publication",
        }
        or result.get("historical_identity") != contract["historical_identity"]
        or result.get("recovered_second_stage_identity")
        != contract["expected_caches"]["second_stage"]
        or result.get("schema_version") != 1
        or result.get("terminal") != "PASS"
        or not isinstance(publication, dict)
        or set(publication)
        != {
            "cache_identities_before",
            "linked_caches",
            "links",
            "state_before",
            "status",
        }
        or publication.get("status") != "published"
        or publication.get("cache_identities_before") != source_caches
    ):
        raise ArchivePublicationError("archive publication result changed")
    links, linked_caches = validate_linked_caches(repo_root, contract, names)
    if publication["links"] != links or publication["linked_caches"] != linked_caches:
        raise ArchivePublicationError("archive publication terminal identity changed")
    return result


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


def run_publication(repo_root: Path) -> dict[str, Any]:
    contract = validate_repository_contract(repo_root)
    names = manifest_prediction_names(repo_root, contract)
    source_caches = validate_source_caches(repo_root, contract, names)
    states = preflight_link_states(repo_root, contract)
    result_path = repo_root / RESULT_PATH
    if result_path.exists() or result_path.is_symlink():
        return validate_existing_publication(
            repo_root, contract, names, source_caches
        )
    validate_execution_authority(repo_root)
    with reconstruction.runtime.exclusive_run_lock(
        reconstruction.runtime.HOST_USER_RUN_LOCK_PATH
    ):
        reconstruction.materializer.assert_no_conflicting_model_processes()
        source_caches = validate_source_caches(repo_root, contract, names)
        states = preflight_link_states(repo_root, contract)
        publication = publish_registered_links(repo_root, contract, names)
        result = build_publication_result(contract, publication)
        try:
            write_result(result_path, result)
            if read_json(result_path) != result:
                raise ArchivePublicationError(
                    "archive publication result failed read-back validation"
                )
        except BaseException:
            result_path.unlink(missing_ok=True)
            result_path.with_name(f".{result_path.name}.writing").unlink(
                missing_ok=True
            )
            rollback_links(repo_root, contract, states)
            raise
    return result


def main() -> int:
    try:
        result = run_publication(ROOT)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "reason": str(error),
                    "schema_version": 1,
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
