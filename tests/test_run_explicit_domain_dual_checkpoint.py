import csv
import hashlib
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/infer/run_explicit_domain_dual_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("explicit_domain_dual_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(path: Path, rows: list[dict[str, str]], *, fields: list[str] | None = None) -> None:
    fieldnames = fields or ["image_path", "domain"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ExplicitDomainDualCheckpointTest(unittest.TestCase):
    def make_image(self, path: Path, payload: bytes = b"source-image") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def make_artifact_pair(self, directory: Path, name: str) -> tuple[Path, Path, str]:
        config = directory / f"{name}-config.yaml"
        weights = directory / f"{name}-weights.pth"
        config.write_text(f"model: {name}\n", encoding="utf-8")
        weights.write_bytes(f"weights-{name}".encode("utf-8"))
        return config, weights, sha256_file(weights)

    def make_main_args(
        self,
        manifest: Path,
        output_dir: Path,
        default_pair: tuple[Path, Path, str],
        hw5k_pair: tuple[Path, Path, str],
        *,
        acknowledge: bool = True,
    ) -> list[str]:
        default_config, default_weights, default_sha = default_pair
        hw5k_config, hw5k_weights, hw5k_sha = hw5k_pair
        args = [
            str(SCRIPT),
            "--manifest-csv",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--default-config",
            str(default_config),
            "--default-weights",
            str(default_weights),
            "--default-weights-sha256",
            default_sha,
            "--hw5k-config",
            str(hw5k_config),
            "--hw5k-weights",
            str(hw5k_weights),
            "--hw5k-weights-sha256",
            hw5k_sha,
        ]
        if acknowledge:
            args.append("--ack-research-specialist")
        return args

    def write_fake_primary_outputs(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        sample_list = Path(command[command.index("--samples-file") + 1])
        output_dir = Path(command[command.index("--output-dir") + 1])
        config_path = Path(command[command.index("--primary-config") + 1])
        weights_path = Path(command[command.index("--primary-weights") + 1])
        pred_dir = output_dir / "pred"
        pred_dir.mkdir(parents=True)
        metrics_rows: list[dict[str, str]] = []
        for source_value in sample_list.read_text(encoding="utf-8").splitlines():
            source = Path(source_value)
            prediction = pred_dir / f"{source.stem}.png"
            prediction.write_bytes(b"prediction-" + source.name.encode("utf-8"))
            metrics_rows.append(
                {
                    "image_path": str(source),
                    "image_sha256": sha256_file(source),
                    "pred_path": str(prediction),
                    "pred_sha256": sha256_file(prediction),
                    "metrics_skipped": "1",
                    "primary_config_sha256": sha256_file(config_path),
                    "primary_weights_sha256": sha256_file(weights_path),
                }
            )
        with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0]))
            writer.writeheader()
            writer.writerows(metrics_rows)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def write_branch_fixture(
        self,
        branch_output: Path,
        row: HARNESS.ManifestRow,
        artifact: HARNESS.ArtifactPair,
        *,
        overrides: dict[str, str] | None = None,
    ) -> Path:
        pred_dir = branch_output / "pred"
        pred_dir.mkdir(parents=True)
        prediction = pred_dir / row.prediction_name
        prediction.write_bytes(b"branch-prediction")
        metrics_row = {
            "image_path": str(row.resolved_image_path),
            "image_sha256": row.image_sha256,
            "pred_path": str(prediction),
            "pred_sha256": sha256_file(prediction),
            "metrics_skipped": "1",
            "primary_config_sha256": artifact.config_sha256,
            "primary_weights_sha256": artifact.weights_sha256,
        }
        if overrides:
            metrics_row.update(overrides)
        with (branch_output / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(HARNESS.REQUIRED_METRICS_FIELDS))
            writer.writeheader()
            writer.writerow(metrics_row)
        return prediction

    def test_domain_mapping_preserves_manifest_order_and_allows_empty_partition(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            first = self.make_image(tmp_path / "images" / "first.jpg")
            second = self.make_image(tmp_path / "images" / "second.jpg")
            third = self.make_image(tmp_path / "images" / "third.jpg")
            manifest = tmp_path / "manifest.csv"
            write_manifest(
                manifest,
                [
                    {"image_path": str(first), "domain": "unknown"},
                    {"image_path": str(second), "domain": "default"},
                    {"image_path": str(third), "domain": "unknown"},
                ],
            )

            rows = HARNESS.parse_manifest_rows(manifest)

            self.assertEqual([row.row_index for row in rows], [1, 2, 3])
            self.assertEqual([row.selected_branch for row in rows], ["default", "default", "default"])
            self.assertEqual(HARNESS.sorted_branch_counts(rows), {"default": 3, "hw5k": 0})
            self.assertEqual(HARNESS.DOMAIN_TO_BRANCH, {
                "default": "default",
                "unknown": "default",
                "hw5k": "hw5k",
            })

    def test_main_allows_empty_default_partition_when_only_hw5k_is_present(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            default_pair = self.make_artifact_pair(tmp_path, "default")
            hw5k_pair = self.make_artifact_pair(tmp_path, "hw5k")
            image = self.make_image(tmp_path / "images" / "specialist.jpg", b"hw5k")
            manifest = tmp_path / "manifest.csv"
            write_manifest(manifest, [{"image_path": str(image), "domain": "hw5k"}])
            output_dir = tmp_path / "output"
            calls: list[str] = []

            def fake_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
                output_dir_arg = Path(command[command.index("--output-dir") + 1])
                calls.append(output_dir_arg.parent.name)
                return self.write_fake_primary_outputs(command)

            args = self.make_main_args(manifest, output_dir, default_pair, hw5k_pair)
            with patch.object(HARNESS, "run_branch", side_effect=fake_runner), patch.object(
                HARNESS, "LOCK_PATH", tmp_path / "lock"
            ), patch.object(sys, "argv", args):
                HARNESS.main()

            self.assertEqual(calls, ["hw5k"])
            run_manifest = json_load(output_dir / "run_manifest.json")
            self.assertEqual(run_manifest["status"], "complete")
            self.assertEqual(run_manifest["branch_counts"], {"default": 0, "hw5k": 1})
            self.assertNotIn("default", run_manifest["branch_commands"])
            self.assertFalse((output_dir / "branches" / "default").exists())

    def test_missing_artifacts_and_expected_checkpoint_sha_mismatch_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config, weights, weights_sha = self.make_artifact_pair(tmp_path, "default")

            with self.subTest(artifact="config"):
                with self.assertRaises(FileNotFoundError):
                    HARNESS.resolve_artifact_pair(
                        branch="default",
                        config_value=str(tmp_path / "missing-config.yaml"),
                        weights_value=str(weights),
                        expected_weights_sha256=weights_sha,
                        research_only=False,
                    )

            with self.subTest(artifact="checkpoint"):
                with self.assertRaises(FileNotFoundError):
                    HARNESS.resolve_artifact_pair(
                        branch="default",
                        config_value=str(config),
                        weights_value=str(tmp_path / "missing-weights.pth"),
                        expected_weights_sha256=weights_sha,
                        research_only=False,
                    )

            with self.subTest(artifact="checkpoint SHA"):
                with self.assertRaisesRegex(ValueError, "checkpoint SHA mismatch"):
                    HARNESS.resolve_artifact_pair(
                        branch="default",
                        config_value=str(config),
                        weights_value=str(weights),
                        expected_weights_sha256="0" * 64,
                        research_only=False,
                    )

    def test_manifest_schema_domain_and_source_validation_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            image = self.make_image(tmp_path / "images" / "page.jpg")
            cases = [
                ("image_path\n" f"{image}\n", "exactly"),
                ("image_path,domain,extra\n" f"{image},default,value\n", "exactly"),
                ("image_path,domain\n" f"{image},\n", "invalid domain"),
                ("image_path,domain\n" f"{image},scut\n", "invalid domain"),
            ]
            for content, expected_error in cases:
                with self.subTest(expected_error=expected_error):
                    manifest = tmp_path / f"manifest-{len(list(tmp_path.glob('manifest-*')))}.csv"
                    manifest.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, expected_error):
                        HARNESS.parse_manifest_rows(manifest)

    def test_forbidden_direct_and_symlinked_label_paths_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            safe_image = self.make_image(tmp_path / "images" / "safe.jpg")
            direct_label = self.make_image(tmp_path / "label" / "direct.jpg")
            target_image = self.make_image(tmp_path / "targets" / "resolved.jpg")
            alias_dir = tmp_path / "aliases"
            alias_dir.mkdir()
            alias_to_label = alias_dir / "label-alias.jpg"
            alias_to_label.symlink_to(direct_label)
            alias_to_target = alias_dir / "target-alias.jpg"
            alias_to_target.symlink_to(target_image)

            for image_path in (direct_label, alias_to_label, alias_to_target):
                manifest = tmp_path / f"manifest-{image_path.name}.csv"
                write_manifest(manifest, [{"image_path": str(image_path), "domain": "default"}])
                with self.subTest(image_path=image_path):
                    with self.assertRaisesRegex(ValueError, "target/label path"):
                        HARNESS.parse_manifest_rows(manifest)

            safe_manifest = tmp_path / "safe-manifest.csv"
            write_manifest(safe_manifest, [{"image_path": str(safe_image), "domain": "default"}])
            self.assertEqual(len(HARNESS.parse_manifest_rows(safe_manifest)), 1)

    def test_duplicate_sources_and_prediction_name_collisions_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            first = self.make_image(tmp_path / "one" / "page.jpg")
            second = self.make_image(tmp_path / "two" / "page.jpg")
            duplicate_manifest = tmp_path / "duplicate.csv"
            write_manifest(
                duplicate_manifest,
                [
                    {"image_path": str(first), "domain": "default"},
                    {"image_path": str(first), "domain": "unknown"},
                ],
            )
            with self.assertRaisesRegex(ValueError, "duplicate source image"):
                HARNESS.parse_manifest_rows(duplicate_manifest)

            collision_manifest = tmp_path / "collision.csv"
            write_manifest(
                collision_manifest,
                [
                    {"image_path": str(first), "domain": "default"},
                    {"image_path": str(second), "domain": "hw5k"},
                ],
            )
            with self.assertRaisesRegex(ValueError, "colliding prediction filename"):
                HARNESS.parse_manifest_rows(collision_manifest)

    def test_hw5k_requires_acknowledgement_but_default_does_not(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            image = self.make_image(tmp_path / "images" / "page.jpg")
            default_manifest = tmp_path / "default.csv"
            hw5k_manifest = tmp_path / "hw5k.csv"
            write_manifest(default_manifest, [{"image_path": str(image), "domain": "default"}])
            write_manifest(hw5k_manifest, [{"image_path": str(image), "domain": "hw5k"}])

            default_rows = HARNESS.parse_manifest_rows(default_manifest)
            hw5k_rows = HARNESS.parse_manifest_rows(hw5k_manifest)
            HARNESS.require_specialist_ack(default_rows, False)
            with self.assertRaisesRegex(ValueError, "ack-research-specialist"):
                HARNESS.require_specialist_ack(hw5k_rows, False)
            HARNESS.require_specialist_ack(hw5k_rows, True)

    def test_branch_command_has_frozen_label_free_inference_arguments(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = tmp_path / "config.yaml"
            weights = tmp_path / "weights.pth"
            artifact = HARNESS.ArtifactPair(
                branch="hw5k",
                config_path=config,
                config_sha256="config-sha",
                weights_path=weights,
                weights_sha256="weights-sha",
                expected_weights_sha256="weights-sha",
                research_only=True,
            )
            command = HARNESS.build_branch_command(
                sample_list_path=tmp_path / "samples.txt",
                branch_output_dir=tmp_path / "run",
                artifact_pair=artifact,
                device="mps",
            )

            self.assertIn(str(HARNESS.PRIMARY_SCRIPT), command)
            self.assertIn("--skip-label-metrics", command)
            for flag, value in (
                ("--page-overlap", "32"),
                ("--batch-size", "8"),
                ("--copy-input-outside-mask", "mb"),
                ("--copy-mask-threshold-auto", "mb_cov8_step"),
                ("--copy-mask-threshold", "70"),
                ("--copy-mask-dilate", "0"),
            ):
                with self.subTest(flag=flag):
                    self.assertEqual(command[command.index(flag) + 1], value)
            self.assertEqual(command[command.index("--device") + 1], "mps")

    def test_serial_inference_lock_contention_fails_fast_and_releases(self) -> None:
        with TemporaryDirectory() as directory:
            lock_path = Path(directory) / "serial-inference.lock"
            with HARNESS.SerialInferenceLock(lock_path):
                with self.assertRaisesRegex(RuntimeError, "serial inference lock is already held"):
                    with HARNESS.SerialInferenceLock(lock_path):
                        self.fail("contending lock unexpectedly acquired")

            with HARNESS.SerialInferenceLock(lock_path):
                pass

    def test_main_runs_non_empty_branches_serially_and_merges_in_manifest_order(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            default_pair = self.make_artifact_pair(tmp_path, "default")
            hw5k_pair = self.make_artifact_pair(tmp_path, "hw5k")
            images = [
                self.make_image(tmp_path / "images" / "default.jpg", b"default"),
                self.make_image(tmp_path / "images" / "unknown.jpg", b"unknown"),
                self.make_image(tmp_path / "images" / "specialist.jpg", b"hw5k"),
            ]
            manifest = tmp_path / "manifest.csv"
            write_manifest(
                manifest,
                [
                    {"image_path": str(images[0]), "domain": "default"},
                    {"image_path": str(images[1]), "domain": "unknown"},
                    {"image_path": str(images[2]), "domain": "hw5k"},
                ],
            )
            output_dir = tmp_path / "output"
            calls: list[str] = []

            def fake_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
                output_dir_arg = Path(command[command.index("--output-dir") + 1])
                calls.append(output_dir_arg.parent.name)
                return self.write_fake_primary_outputs(command)

            args = self.make_main_args(manifest, output_dir, default_pair, hw5k_pair)
            with patch.object(HARNESS, "run_branch", side_effect=fake_runner), patch.object(
                HARNESS, "LOCK_PATH", tmp_path / "lock"
            ), patch.object(sys, "argv", args):
                HARNESS.main()

            self.assertEqual(calls, ["default", "hw5k"])
            run_manifest = json_load(output_dir / "run_manifest.json")
            self.assertEqual(run_manifest["status"], "complete")
            self.assertEqual(run_manifest["row_count"], 3)
            self.assertEqual(run_manifest["branch_counts"], {"default": 2, "hw5k": 1})
            self.assertFalse(run_manifest["inference_reads_labels"])
            route_rows = read_csv(output_dir / "route_decisions.csv")
            self.assertEqual([row["row_index"] for row in route_rows], ["1", "2", "3"])
            self.assertEqual([row["selected_branch"] for row in route_rows], ["default", "default", "hw5k"])
            self.assertEqual(len(list((output_dir / "pred").glob("*.png"))), 3)
            for row in route_rows:
                branch_prediction = Path(row["branch_prediction_path"])
                merged_prediction = Path(row["merged_prediction_path"])
                self.assertEqual(branch_prediction.read_bytes(), merged_prediction.read_bytes())
                self.assertEqual(row["prediction_sha256"], sha256_file(merged_prediction))

    def test_validate_prediction_set_and_merge_prove_prediction_sha(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = self.make_image(tmp_path / "images" / "page.jpg")
            manifest = tmp_path / "manifest.csv"
            write_manifest(manifest, [{"image_path": str(source), "domain": "default"}])
            row = HARNESS.parse_manifest_rows(manifest)[0]
            config, weights, weights_sha = self.make_artifact_pair(tmp_path, "default")
            artifact = HARNESS.resolve_artifact_pair(
                branch="default",
                config_value=str(config),
                weights_value=str(weights),
                expected_weights_sha256=weights_sha,
                research_only=False,
            )
            branch_output = tmp_path / "branch"
            prediction = self.write_branch_fixture(branch_output, row, artifact)

            branch_meta = HARNESS.validate_branch_outputs(
                branch="default",
                branch_rows=[row],
                branch_output_dir=branch_output,
                artifact_pair=artifact,
            )
            output_dir = tmp_path / "merged"
            output_dir.mkdir()
            route_rows = HARNESS.merge_branch_predictions(
                all_rows=[row],
                output_dir=output_dir,
                branch_output_dirs={"default": branch_output},
                branch_pred_meta={"default": branch_meta},
                artifacts_by_branch={"default": artifact},
            )
            merged_prediction = output_dir / "pred" / row.prediction_name
            self.assertEqual(merged_prediction.read_bytes(), prediction.read_bytes())
            self.assertEqual(route_rows[0]["prediction_sha256"], sha256_file(merged_prediction))

            for mutation in ("extra", "missing"):
                with self.subTest(prediction_set=mutation):
                    case_output = tmp_path / f"branch-{mutation}"
                    case_prediction = self.write_branch_fixture(case_output, row, artifact)
                    if mutation == "extra":
                        (case_output / "pred" / "extra.png").write_bytes(b"unexpected")
                    else:
                        case_prediction.unlink()
                    with self.assertRaisesRegex(ValueError, "prediction set mismatch"):
                        HARNESS.validate_branch_outputs(
                            branch="default",
                            branch_rows=[row],
                            branch_output_dir=case_output,
                            artifact_pair=artifact,
                        )

    def test_branch_metrics_source_prediction_and_artifact_shas_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = self.make_image(tmp_path / "images" / "page.jpg")
            manifest = tmp_path / "manifest.csv"
            write_manifest(manifest, [{"image_path": str(source), "domain": "default"}])
            row = HARNESS.parse_manifest_rows(manifest)[0]
            config, weights, weights_sha = self.make_artifact_pair(tmp_path, "default")
            artifact = HARNESS.resolve_artifact_pair(
                branch="default",
                config_value=str(config),
                weights_value=str(weights),
                expected_weights_sha256=weights_sha,
                research_only=False,
            )

            valid_output = tmp_path / "branch-valid"
            self.write_branch_fixture(valid_output, row, artifact)
            self.assertEqual(
                len(
                    HARNESS.validate_branch_outputs(
                        branch="default",
                        branch_rows=[row],
                        branch_output_dir=valid_output,
                        artifact_pair=artifact,
                    )
                ),
                1,
            )

            mismatch_cases = [
                ("image_sha256", "source image SHA mismatch"),
                ("pred_sha256", "prediction SHA mismatch"),
                ("primary_config_sha256", "config SHA mismatch"),
                ("primary_weights_sha256", "checkpoint SHA mismatch"),
            ]
            for field, expected_error in mismatch_cases:
                with self.subTest(field=field):
                    case_output = tmp_path / f"branch-{field}"
                    self.write_branch_fixture(case_output, row, artifact, overrides={field: "mismatch"})
                    with self.assertRaisesRegex(ValueError, expected_error):
                        HARNESS.validate_branch_outputs(
                            branch="default",
                            branch_rows=[row],
                            branch_output_dir=case_output,
                            artifact_pair=artifact,
                        )

    def test_branch_failure_writes_failed_manifest_without_fallback(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            default_pair = self.make_artifact_pair(tmp_path, "default")
            hw5k_pair = self.make_artifact_pair(tmp_path, "hw5k")
            image = self.make_image(tmp_path / "images" / "page.jpg")
            manifest = tmp_path / "manifest.csv"
            write_manifest(manifest, [{"image_path": str(image), "domain": "default"}])
            output_dir = tmp_path / "failed-output"

            def failing_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
                raise subprocess.CalledProcessError(9, command, output="stdout", stderr="branch failed")

            args = self.make_main_args(manifest, output_dir, default_pair, hw5k_pair, acknowledge=False)
            with patch.object(HARNESS, "run_branch", side_effect=failing_runner), patch.object(
                HARNESS, "LOCK_PATH", tmp_path / "lock"
            ), patch.object(sys, "argv", args):
                with self.assertRaises(subprocess.CalledProcessError):
                    HARNESS.main()

            failed_manifest = json_load(output_dir / "run_manifest.json")
            self.assertEqual(failed_manifest["status"], "failed")
            self.assertIn("CalledProcessError", failed_manifest["error"])
            self.assertNotIn("hw5k", failed_manifest["branch_commands"])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def json_load(path: Path) -> dict[str, object]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
