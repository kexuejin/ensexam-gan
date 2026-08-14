from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


from scripts.analysis import publish_external_text_layout_recovered_archive_links as publication
from scripts.analysis import reconstruct_external_text_layout_frozen_caches as reconstruction


MANIFEST_LINES = ["sources/a.jpg", "sources/b.jpeg"]
NAMES = ["a.png", "b.png"]


def write_cache(root: Path, relative: str, prefix: str) -> dict[str, object]:
    cache = root / relative
    predictions = cache / "pred"
    predictions.mkdir(parents=True)
    for name in NAMES:
        (predictions / name).write_bytes(f"{prefix}:{name}".encode())
    metrics = cache / "metrics.csv"
    metrics.write_text(f"cache,{prefix}\n", encoding="utf-8")
    rows = [
        f"{name} {reconstruction.sha256_file(predictions / name)}" for name in NAMES
    ]
    return {
        "metrics_sha256": reconstruction.sha256_file(metrics),
        "prediction_set": {
            "content_sha256": reconstruction.sha256_rows(rows),
            "count": len(NAMES),
            "filename_sha256": reconstruction.sha256_rows(NAMES),
        },
    }


def synthetic_contract(root: Path) -> dict[str, object]:
    del root
    return {
        "expected_caches": {},
        "historical_identity": {
            "metrics_payload_present": False,
            "metrics_sha256": "b" * 64,
            "status": "not_reproduced",
        },
        "paths": {
            "archive_primary": "archive/primary",
            "archive_second_stage": "archive/second-stage",
            "primary": "build/primary",
            "second_stage": "build/second-stage",
        },
    }


def write_caches(root: Path, contract: dict[str, object]) -> None:
    contract["expected_caches"] = {
        "primary": write_cache(root, contract["paths"]["primary"], "primary"),
        "second_stage": {
            **write_cache(
                root, contract["paths"]["second_stage"], "second-stage"
            ),
            "provenance": (
                "semantic_equivalence_audit_pass_not_historical_payload_reproduction"
            ),
        },
    }


def make_relative_link(link: Path, source: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(source, start=link.parent), target_is_directory=True)


class PublishExternalTextLayoutRecoveredArchiveLinksTest(unittest.TestCase):
    def test_repository_contract_is_exact(self) -> None:
        contract = publication.validate_repository_contract(publication.ROOT)
        self.assertEqual(contract["schema_version"], 1)
        self.assertEqual(
            contract["expected_caches"]["second_stage"]["metrics_sha256"],
            "79fd61278e689a0003e37a5bdf20f856184b49c8fdb3af8ad9af03a3a13c451b",
        )

    def test_original_publication_model_and_helper_routes_are_absent(self) -> None:
        source = Path(publication.__file__).read_text(encoding="utf-8")
        for route in (
            "publish_caches",
            "_load_historical_helper",
            "build_stage_command",
            "reconstruct_stage",
            "run_monitored_atomic_directory_command",
            "run_stage",
        ):
            self.assertNotIn(f"reconstruction.{route}(", source)
        self.assertNotIn("subprocess", source)

    def test_exact_caches_publish_relative_links(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = synthetic_contract(root)
            write_caches(root, contract)

            result = publication.publish_registered_links(root, contract, NAMES)

            self.assertEqual(result["status"], "published")
            for stage, archive_name in (
                ("primary", "archive_primary"),
                ("second_stage", "archive_second_stage"),
            ):
                link = root / contract["paths"][archive_name]
                source = root / contract["paths"][stage]
                self.assertTrue(link.is_symlink())
                self.assertFalse(Path(os.readlink(link)).is_absolute())
                self.assertEqual(link.resolve(), source.resolve())

    def test_wrong_cache_rejects_before_link_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = synthetic_contract(root)
            write_caches(root, contract)
            metrics = root / contract["paths"]["second_stage"] / "metrics.csv"
            metrics.write_text("changed\n", encoding="utf-8")

            with self.assertRaises(publication.ArchivePublicationError):
                publication.publish_registered_links(root, contract, NAMES)

            self.assertFalse((root / contract["paths"]["archive_primary"]).exists())
            self.assertFalse(
                (root / contract["paths"]["archive_second_stage"]).exists()
            )

    def test_second_destination_conflict_preflights_before_first_link(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = synthetic_contract(root)
            write_caches(root, contract)
            conflict = root / contract["paths"]["archive_second_stage"]
            conflict.mkdir(parents=True)

            with self.assertRaises(publication.ArchivePublicationError):
                publication.publish_registered_links(root, contract, NAMES)

            self.assertFalse((root / contract["paths"]["archive_primary"]).exists())

    def test_absolute_and_wrong_target_links_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = synthetic_contract(root)
            write_caches(root, contract)
            primary = root / contract["paths"]["primary"]
            archive = root / contract["paths"]["archive_primary"]
            archive.parent.mkdir(parents=True)
            archive.symlink_to(primary.resolve(), target_is_directory=True)
            with self.assertRaises(publication.ArchivePublicationError):
                publication.preflight_link_states(root, contract)
            archive.unlink()
            make_relative_link(archive, root / contract["paths"]["second_stage"])
            with self.assertRaises(publication.ArchivePublicationError):
                publication.preflight_link_states(root, contract)

    def test_partial_exact_final_link_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = synthetic_contract(root)
            write_caches(root, contract)
            primary = root / contract["paths"]["primary"]
            archive_primary = root / contract["paths"]["archive_primary"]
            make_relative_link(archive_primary, primary)

            result = publication.publish_registered_links(root, contract, NAMES)

            self.assertEqual(result["state_before"]["primary"], "existing")
            self.assertTrue(
                (root / contract["paths"]["archive_second_stage"]).is_symlink()
            )

    def test_exact_temporary_link_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = synthetic_contract(root)
            write_caches(root, contract)
            final = root / contract["paths"]["archive_primary"]
            source = root / contract["paths"]["primary"]
            temporary = publication._temporary_link_path(final)
            make_relative_link(temporary, source)

            result = publication.publish_registered_links(root, contract, NAMES)

            self.assertEqual(result["state_before"]["primary"], "temporary")
            self.assertTrue(final.is_symlink())
            self.assertFalse(temporary.exists() or temporary.is_symlink())

    def test_second_promotion_failure_rolls_back_current_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = synthetic_contract(root)
            write_caches(root, contract)
            original = publication._promote_link
            calls = 0

            def fail_second(temporary: Path, final: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("synthetic promotion failure")
                original(temporary, final)

            with mock.patch.object(
                publication, "_promote_link", side_effect=fail_second
            ):
                with self.assertRaises(OSError):
                    publication.publish_registered_links(root, contract, NAMES)

            for name in ("archive_primary", "archive_second_stage"):
                final = root / contract["paths"][name]
                self.assertFalse(final.exists() or final.is_symlink())
                temporary = publication._temporary_link_path(final)
                self.assertFalse(temporary.exists() or temporary.is_symlink())

    def test_final_validation_failure_rolls_back_current_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = synthetic_contract(root)
            write_caches(root, contract)
            with mock.patch.object(
                publication,
                "validate_linked_caches",
                side_effect=publication.ArchivePublicationError("invalid final"),
            ):
                with self.assertRaises(publication.ArchivePublicationError):
                    publication.publish_registered_links(root, contract, NAMES)
            self.assertFalse((root / contract["paths"]["archive_primary"]).exists())
            self.assertFalse(
                (root / contract["paths"]["archive_second_stage"]).exists()
            )

    def test_execution_authority_requires_integration_pass(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = root / publication.LEDGER_PATH
            ledger_path.parent.mkdir(parents=True)
            prerequisites = [
                {
                    "id": "external_text_layout_recovered_archive_publication_preregistration",
                    "status": "passed",
                },
                {
                    "id": "external_text_layout_second_stage_cache_reconstruction",
                    "status": "passed",
                },
                {
                    "id": "external_text_layout_second_stage_recovered_cache_publication",
                    "status": "passed",
                },
                {
                    "id": "external_text_layout_support_train_only_diagnostic",
                    "status": "pending",
                },
            ]
            ledger = {
                "active_iteration": {
                    "prerequisites": prerequisites,
                    "terminal": "PREREQUISITE_NEEDED",
                }
            }
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            with self.assertRaises(publication.ArchivePublicationError):
                publication.validate_execution_authority(root)
            prerequisites.append(
                {
                    "id": "external_text_layout_recovered_archive_publication_integration",
                    "status": "passed",
                }
            )
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            publication.validate_execution_authority(root)

    def test_existing_terminal_result_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = synthetic_contract(root)
            write_caches(root, contract)
            source_caches = publication.validate_source_caches(root, contract, NAMES)
            published = publication.publish_registered_links(root, contract, NAMES)
            result = publication.build_publication_result(contract, published)
            publication.write_result(root / publication.RESULT_PATH, result)

            self.assertEqual(
                publication.validate_existing_publication(
                    root, contract, NAMES, source_caches
                ),
                result,
            )
            with (
                mock.patch.object(
                    publication,
                    "validate_repository_contract",
                    return_value=contract,
                ),
                mock.patch.object(
                    publication, "manifest_prediction_names", return_value=NAMES
                ),
                mock.patch.object(
                    publication, "validate_execution_authority"
                ) as authority,
            ):
                self.assertEqual(publication.run_publication(root), result)
                authority.assert_not_called()

    def test_changed_terminal_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = synthetic_contract(root)
            write_caches(root, contract)
            source_caches = publication.validate_source_caches(root, contract, NAMES)
            published = publication.publish_registered_links(root, contract, NAMES)
            result = publication.build_publication_result(contract, published)
            result = json.loads(json.dumps(result))
            result["terminal"] = "PREREQUISITE_NEEDED"
            publication.write_result(root / publication.RESULT_PATH, result)
            with self.assertRaises(publication.ArchivePublicationError):
                publication.validate_existing_publication(
                    root, contract, NAMES, source_caches
                )

    def test_result_write_failure_rolls_back_current_attempt_links(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            contract = synthetic_contract(root)
            write_caches(root, contract)
            with (
                mock.patch.object(
                    publication,
                    "validate_repository_contract",
                    return_value=contract,
                ),
                mock.patch.object(
                    publication, "manifest_prediction_names", return_value=NAMES
                ),
                mock.patch.object(publication, "validate_execution_authority"),
                mock.patch.object(
                    publication.reconstruction.materializer,
                    "assert_no_conflicting_model_processes",
                ),
                mock.patch.object(
                    publication, "write_result", side_effect=OSError("disk full")
                ),
            ):
                with self.assertRaises(OSError):
                    publication.run_publication(root)

            for name in ("archive_primary", "archive_second_stage"):
                link = root / contract["paths"][name]
                self.assertFalse(link.exists() or link.is_symlink())
            self.assertFalse((root / publication.RESULT_PATH).exists())


if __name__ == "__main__":
    unittest.main()
