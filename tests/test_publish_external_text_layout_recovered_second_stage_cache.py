from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


from scripts.analysis import publish_external_text_layout_recovered_second_stage_cache as publication
from scripts.analysis import reconstruct_external_text_layout_frozen_caches as reconstruction


def write_cache(root: Path, contract: dict[str, object]) -> tuple[Path, bytes, bytes]:
    temporary = root / contract["paths"]["temporary"]
    predictions = temporary / "pred"
    predictions.mkdir(parents=True)
    names = ["a.png", "b.png"]
    for name in names:
        (predictions / name).write_bytes(f"prediction:{name}".encode())
    current = contract["recovered_identity"]["canonicalization"][
        "current_repository_root"
    ]
    historical = contract["recovered_identity"]["canonicalization"][
        "frozen_historical_repository_root"
    ]
    before = (
        "file,pred_path\r\n"
        f"a.jpg,{current}/outputs/cache/pred/a.png\r\n"
        f"b.jpg,{current}/outputs/cache/pred/b.png\r\n"
    ).encode()
    after = before.replace(current.encode(), historical.encode())
    (temporary / "metrics.csv").write_bytes(before)
    rows = [
        f"{name} {reconstruction.sha256_file(predictions / name)}" for name in names
    ]
    contract["recovered_identity"]["metrics_sha256"] = hashlib.sha256(
        after
    ).hexdigest()
    contract["recovered_identity"]["prediction_set"] = {
        "content_sha256": reconstruction.sha256_rows(rows),
        "count": len(names),
        "filename_sha256": reconstruction.sha256_rows(names),
    }
    canonicalization = contract["recovered_identity"]["canonicalization"]
    canonicalization["source_metrics_sha256"] = hashlib.sha256(before).hexdigest()
    canonicalization["expected_data_rows"] = 2
    canonicalization["expected_replacement_count"] = 2
    return temporary, before, after


def synthetic_contract(root: Path) -> dict[str, object]:
    return {
        "historical_identity": {
            "metrics_payload_present": False,
            "metrics_sha256": "f" * 64,
            "status": "not_reproduced",
        },
        "paths": {
            "archive_primary": "archive/primary",
            "archive_second_stage": "archive/second",
            "final": "outputs/cache",
            "primary": "outputs/primary",
            "temporary": "outputs/.cache.materializing",
        },
        "recovered_identity": {
            "canonicalization": {
                "current_repository_root": str(root),
                "expected_data_rows": 2,
                "expected_replacement_count": 2,
                "frozen_historical_repository_root": "/historical/root",
                "source_metrics_sha256": "pending",
            },
            "metrics_sha256": "pending",
            "prediction_set": {},
        },
    }


class PublishExternalTextLayoutRecoveredSecondStageCacheTest(unittest.TestCase):
    def test_repository_contract_is_exact(self) -> None:
        contract = publication.validate_repository_contract(publication.ROOT)
        self.assertEqual(contract["schema_version"], 1)
        self.assertEqual(
            contract["recovered_identity"]["metrics_sha256"],
            "79fd61278e689a0003e37a5bdf20f856184b49c8fdb3af8ad9af03a3a13c451b",
        )

    def test_model_and_historical_execution_routes_are_absent(self) -> None:
        source = Path(publication.__file__).read_text(encoding="utf-8")
        self.assertNotIn("subprocess", source)
        for helper in (
            "_load_historical_helper",
            "build_stage_command",
            "recover_existing_primary_cache",
            "reconstruct_stage",
            "run_monitored_atomic_directory_command",
            "run_stage",
        ):
            self.assertNotIn(f"reconstruction.{helper}(", source)

    def test_source_cache_is_validated_canonicalized_and_published(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            contract = synthetic_contract(root)
            temporary, _before, after = write_cache(root, contract)
            result = publication.publish_registered_cache(root, contract)
            final = root / contract["paths"]["final"]
            self.assertEqual(result["status"], "published")
            self.assertFalse(temporary.exists())
            self.assertEqual((final / "metrics.csv").read_bytes(), after)

    def test_wrong_candidate_hash_never_publishes_final(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            contract = synthetic_contract(root)
            temporary, before, _after = write_cache(root, contract)
            contract["recovered_identity"]["metrics_sha256"] = "0" * 64
            with self.assertRaises(publication.PublicationError):
                publication.publish_registered_cache(root, contract)
            self.assertTrue(temporary.is_dir())
            self.assertEqual((temporary / "metrics.csv").read_bytes(), before)
            self.assertFalse((root / contract["paths"]["final"]).exists())

    def test_wrong_prediction_never_mutates_metrics_or_publishes_final(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            contract = synthetic_contract(root)
            temporary, before, _after = write_cache(root, contract)
            (temporary / "pred" / "a.png").write_bytes(b"changed")
            with self.assertRaises(publication.PublicationError):
                publication.publish_registered_cache(root, contract)
            self.assertEqual((temporary / "metrics.csv").read_bytes(), before)
            self.assertFalse((root / contract["paths"]["final"]).exists())

    def test_wrong_replacement_count_never_mutates_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            contract = synthetic_contract(root)
            temporary, before, _after = write_cache(root, contract)
            contract["recovered_identity"]["canonicalization"][
                "expected_replacement_count"
            ] = 3
            with self.assertRaises(publication.PublicationError):
                publication.publish_registered_cache(root, contract)
            self.assertEqual((temporary / "metrics.csv").read_bytes(), before)

    def test_archive_conflict_rejects_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            contract = synthetic_contract(root)
            temporary, before, _after = write_cache(root, contract)
            archive = root / contract["paths"]["archive_second_stage"]
            archive.mkdir(parents=True)
            with self.assertRaises(publication.PublicationError):
                publication.publish_registered_cache(root, contract)
            self.assertEqual((temporary / "metrics.csv").read_bytes(), before)
            self.assertFalse((root / contract["paths"]["final"]).exists())

    def test_complete_temporary_validation_precedes_final_rename(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            contract = synthetic_contract(root)
            temporary, _before, after = write_cache(root, contract)
            with mock.patch.object(
                publication,
                "validate_cache_identity",
                side_effect=publication.PublicationError("invalid temporary"),
            ):
                with self.assertRaises(publication.PublicationError):
                    publication.publish_registered_cache(root, contract)
            self.assertTrue(temporary.is_dir())
            self.assertEqual((temporary / "metrics.csv").read_bytes(), after)
            self.assertFalse((root / contract["paths"]["final"]).exists())

    def test_failed_final_validation_restores_temporary_cache(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            contract = synthetic_contract(root)
            temporary, _before, after = write_cache(root, contract)
            with mock.patch.object(
                publication,
                "validate_cache_identity",
                side_effect=[
                    {"state": "temporary"},
                    publication.PublicationError("invalid final"),
                ],
            ):
                with self.assertRaises(publication.PublicationError):
                    publication.publish_registered_cache(root, contract)
            self.assertTrue(temporary.is_dir())
            self.assertEqual((temporary / "metrics.csv").read_bytes(), after)
            self.assertFalse((root / contract["paths"]["final"]).exists())

    def test_already_canonical_temporary_cache_resumes_before_rename(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            contract = synthetic_contract(root)
            temporary, _before, after = write_cache(root, contract)
            (temporary / "metrics.csv").write_bytes(after)
            result = publication.publish_registered_cache(root, contract)
            self.assertEqual(result["metrics_state_before"], "already_canonical")
            self.assertTrue((root / contract["paths"]["final"]).is_dir())

    def test_execution_authority_requires_integration_pass(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ledger_path = root / publication.LEDGER_PATH
            ledger_path.parent.mkdir(parents=True)
            prerequisites = [
                {
                    "id": (
                        "external_text_layout_second_stage_recovered_cache_"
                        "publication_preregistration"
                    ),
                    "status": "passed",
                },
                {
                    "id": "external_text_layout_second_stage_cache_salvage_audit",
                    "status": "passed",
                },
                {
                    "id": "external_text_layout_second_stage_cache_reconstruction",
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
            with self.assertRaises(publication.PublicationError):
                publication.validate_execution_authority(root)
            prerequisites.append(
                {
                    "id": (
                        "external_text_layout_second_stage_recovered_cache_"
                        "publication_integration"
                    ),
                    "status": "passed",
                }
            )
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            publication.validate_execution_authority(root)

    def test_existing_publication_requires_exact_pass_result(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            contract = synthetic_contract(root)
            temporary, _before, _after = write_cache(root, contract)
            published = publication.publish_registered_cache(root, contract)
            result = publication.build_publication_result(contract, published)
            result_path = root / publication.RESULT_PATH
            publication.write_result(result_path, result)

            self.assertEqual(
                publication.validate_existing_publication(root, contract), result
            )
            self.assertFalse(temporary.exists())
            with (
                mock.patch.object(
                    publication,
                    "validate_repository_contract",
                    return_value=contract,
                ),
                mock.patch.object(
                    publication, "validate_execution_authority"
                ) as authority,
            ):
                self.assertEqual(publication.run_publication(root), result)
                authority.assert_not_called()

            result_path.unlink()
            with self.assertRaises(publication.PublicationError):
                publication.validate_existing_publication(root, contract)

    def test_existing_result_with_changed_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            contract = synthetic_contract(root)
            write_cache(root, contract)
            published = publication.publish_registered_cache(root, contract)
            publication_result = publication.build_publication_result(
                contract, published
            )
            publication_result = json.loads(json.dumps(publication_result))
            publication_result["recovered_identity"]["metrics_sha256"] = "0" * 64
            result_path = root / publication.RESULT_PATH
            publication.write_result(result_path, publication_result)
            with self.assertRaises(publication.PublicationError):
                publication.validate_existing_publication(root, contract)

    def test_result_write_failure_restores_canonical_temporary_cache(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            contract = synthetic_contract(root)
            contract["evidence"] = {"primary_recovery": {"path": "primary.json"}}
            temporary, _before, after = write_cache(root, contract)
            primary = root / contract["paths"]["primary"]
            original_validate_cache = publication.validate_cache_identity

            def validate_cache(cache_dir: Path, identity: dict[str, object]):
                if cache_dir == primary:
                    return {"status": "primary-valid"}
                return original_validate_cache(cache_dir, identity)

            primary_result = {
                "cache": {"metrics_sha256": "unused", "prediction_set": {}}
            }
            with (
                mock.patch.object(
                    publication,
                    "validate_repository_contract",
                    return_value=contract,
                ),
                mock.patch.object(publication, "validate_execution_authority"),
                mock.patch.object(
                    publication, "read_json", return_value=primary_result
                ),
                mock.patch.object(
                    publication, "validate_cache_identity", side_effect=validate_cache
                ),
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

            self.assertTrue(temporary.is_dir())
            self.assertEqual((temporary / "metrics.csv").read_bytes(), after)
            self.assertFalse((root / contract["paths"]["final"]).exists())
            self.assertFalse((root / publication.RESULT_PATH).exists())


if __name__ == "__main__":
    unittest.main()
