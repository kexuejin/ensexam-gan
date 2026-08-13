import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import torch

from scripts.analysis.audit_second_stage_alpha_support import (
    ABLATION_CHANNELS,
    CHANNELS,
    AuditError,
    OUTPUT_ROOT,
    build_page,
    evaluate_fold,
    load_raw_alpha_npz,
    run_audit,
    validate_authority as validate_audit_authority,
    validate_plan as validate_audit_plan,
)
from scripts.analysis.materialize_second_stage_alpha_train_only import (
    MaterializationError,
    RAW_ALPHA_KEY,
    infer_raw_alpha_full_page,
    load_registered_erasemap,
    validate_authority,
    validate_plan,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def write_rgb(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(values, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write test image: {path}")


def sha256_file_local(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_rows(rows: list[str]) -> str:
    payload = "\n".join(sorted(rows)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_metrics_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "image_sha256", "base_edit_threshold", "second_delta_threshold", "dark_threshold"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_synthetic_repo(root: Path, *, alpha_signal: bool) -> tuple[Path, Path]:
    names = [f"page-{index:03d}.png" for index in range(10)]
    docs = root / "docs"
    outputs = root / "outputs"
    data = root / "data-links" / "samples" / "SCUT-HW5K-mixed-20260729" / "train" / "all_labels"
    primary_dir = outputs / "archive" / "sign-separated-residual-repair-20260810" / "train275-primary" / "pred"
    second_dir = outputs / "archive" / "sign-separated-residual-repair-20260810" / "train275-frozen-pipeline" / "pred"
    alpha_root = outputs / "second-stage-alpha-support-materialization-20260812"
    alpha_dir = alpha_root / "pages"
    lists = root / "hardcase_lists"
    artifacts = root / "artifacts"
    scripts = root / "scripts" / "infer"
    for path in (docs, data, primary_dir, second_dir, alpha_dir, lists, artifacts / "current-primary", scripts):
        path.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    primary_metric_rows = []
    second_metric_rows = []
    primary_hash_rows = []
    second_hash_rows = []
    label_hash_rows = []
    alpha_page_rows = []

    for fold in range(5):
        for page_index in range(2):
            index = fold * 2 + page_index
            name = names[index]
            source_rel = f"data/train/{name}"
            manifest_rows.append(source_rel)
            primary = np.zeros((2, 2, 3), dtype=np.uint8)
            second = np.zeros((2, 2, 3), dtype=np.uint8)
            target = second.copy()
            target[0, 0] += 10
            target[1, 0] += 10
            if alpha_signal:
                raw_alpha = np.asarray([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
            else:
                raw_alpha = np.full((2, 2), 0.5, dtype=np.float32)
            write_rgb(primary_dir / name, primary)
            write_rgb(second_dir / name, second)
            write_rgb(data / name, target)
            np.savez_compressed(alpha_dir / f"{Path(name).stem}.npz", raw_alpha=raw_alpha)
            primary_hash = sha256_file_local(primary_dir / name)
            second_hash = sha256_file_local(second_dir / name)
            label_hash = sha256_file_local(data / name)
            npz_hash = sha256_file_local(alpha_dir / f"{Path(name).stem}.npz")
            primary_hash_rows.append(f"{name} {primary_hash}")
            second_hash_rows.append(f"{name} {second_hash}")
            label_hash_rows.append(f"{name} {label_hash}")
            primary_metric_rows.append(
                {
                    "file": name,
                    "image_sha256": primary_hash,
                    "base_edit_threshold": "12",
                    "second_delta_threshold": "32",
                    "dark_threshold": "0",
                }
            )
            second_metric_rows.append(
                {
                    "file": name,
                    "image_sha256": second_hash,
                    "base_edit_threshold": "12",
                    "second_delta_threshold": "32",
                    "dark_threshold": "0",
                }
            )
            alpha_page_rows.append(
                {
                    "file": name,
                    "height": 2,
                    "width": 2,
                    "source_prediction_sha256": primary_hash,
                    "npz_sha256": npz_hash,
                    "raw_alpha_min": float(raw_alpha.min()),
                    "raw_alpha_max": float(raw_alpha.max()),
                    "raw_alpha_mean": float(raw_alpha.mean()),
                }
            )

    manifest_path = lists / "manifest.txt"
    manifest_path.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    write_metrics_csv(outputs / "archive" / "sign-separated-residual-repair-20260810" / "train275-primary" / "metrics.csv", primary_metric_rows)
    write_metrics_csv(outputs / "archive" / "sign-separated-residual-repair-20260810" / "train275-frozen-pipeline" / "metrics.csv", second_metric_rows)

    config = artifacts / "current-primary" / "config.yaml"
    primary_checkpoint_artifact = artifacts / "current-primary" / "micro_region_probe_step0001.pth"
    checkpoint = root / "artifacts" / "current-second-stage-best.pt"
    role_plan = docs / "role-plan.json"
    base_role = docs / "base-role.json"
    source_output_kill = docs / "kill.md"
    inference_source = scripts / "patch_cleanup_erasemap.py"
    for path, content in [
        (config, "config"),
        (primary_checkpoint_artifact, "primary-checkpoint"),
        (checkpoint, "checkpoint"),
        (base_role, "{}\n"),
        (source_output_kill, "kill"),
        (inference_source, "infer"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    role_plan.write_text(json.dumps({"evidence": {"base_role_contract": {"path": str(base_role.relative_to(root)), "sha256": sha256_file_local(base_role)}}}), encoding="utf-8")

    plan = {
        "schema_version": 1,
        "iteration_id": "synthetic",
        "state": "synthetic",
        "next_boundary_on_pass": "next",
        "data": {
            "effective_train_count": len(names),
            "target_lighter_margin_gray": 2.0,
            "manifest": {"path": str(manifest_path.relative_to(root)), "sha256": sha256_file_local(manifest_path)},
        },
        "diagnostic": {"lambda": 1.0, "max_samples_per_class_per_page": 1},
        "acceptance": {
            "full_auc_ablation_margin_min": 0.03,
            "full_fold_auc_min": 0.55,
            "full_mean_fold_auc_min": 0.65,
            "macro_median_page_auc_min": 0.6,
            "positive_mean_above_preserve_min_folds": 4,
            "required_fold_count": 5,
            "required_terminal_on_pass": "PASS",
        },
        "evidence": {
            "current_primary_checkpoint": {"path": "artifacts/current-primary/micro_region_probe_step0001.pth", "sha256": sha256_file_local(primary_checkpoint_artifact)},
            "current_primary_config": {"path": str(config.relative_to(root)), "sha256": sha256_file_local(config)},
            "current_second_stage_checkpoint": {"path": str(checkpoint.relative_to(root)), "sha256": sha256_file_local(checkpoint)},
            "primary_metrics": {"path": "outputs/archive/sign-separated-residual-repair-20260810/train275-primary/metrics.csv", "sha256": sha256_file_local(outputs / "archive" / "sign-separated-residual-repair-20260810" / "train275-primary" / "metrics.csv")},
            "role_plan": {"path": str(role_plan.relative_to(root)), "sha256": sha256_file_local(role_plan)},
            "second_stage_inference_source": {"path": str(inference_source.relative_to(root)), "sha256": sha256_file_local(inference_source)},
            "second_stage_metrics": {"path": "outputs/archive/sign-separated-residual-repair-20260810/train275-frozen-pipeline/metrics.csv", "sha256": sha256_file_local(outputs / "archive" / "sign-separated-residual-repair-20260810" / "train275-frozen-pipeline" / "metrics.csv")},
            "source_output_kill": {"path": str(source_output_kill.relative_to(root)), "sha256": sha256_file_local(source_output_kill)},
            "primary_prediction_set": {
                "directory": str(primary_dir.relative_to(root)),
                "count": len(names),
                "filename_sha256": hash_rows(names),
                "content_sha256": hash_rows(primary_hash_rows),
            },
            "second_stage_prediction_set": {
                "directory": str(second_dir.relative_to(root)),
                "count": len(names),
                "filename_sha256": hash_rows(names),
                "content_sha256": hash_rows(second_hash_rows),
            },
            "train_label_set": {
                "directory": "data-links/samples/SCUT-HW5K-mixed-20260729/train/all_labels",
                "count": len(names),
                "content_sha256": hash_rows(label_hash_rows),
            },
        },
        "second_stage_alpha_materialization": {
            "output_root": str(alpha_root.relative_to(root)),
        },
    }
    plan_path = docs / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    ledger_path = docs / "ledger.json"
    ledger_path.write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    alpha_manifest = {
        "schema_version": 1,
        "terminal": "PASS",
        "provenance": "current_second_stage_erasemap_alpha_head_sigmoid",
        "target_access": False,
        "train_count": len(names),
        "tile_size": 160,
        "stride": 160,
        "batch_size": 32,
        "dtype": "float32",
        "encoding": "one_compressed_npz_per_page_with_exact_raw_alpha_key",
        "raw_alpha_key": "raw_alpha",
        "overlap_fusion": "arithmetic_mean_of_raw_patch_alpha_before_any_threshold",
        "output_root": str(alpha_root.relative_to(root)),
        "pages_directory": "pages",
        "plan": {"path": str(plan_path.relative_to(root)), "sha256": sha256_file_local(plan_path)},
        "source_manifest": {"path": str(manifest_path.relative_to(root)), "sha256": sha256_file_local(manifest_path)},
        "primary_config": {"path": str(config.relative_to(root)), "sha256": sha256_file_local(config)},
        "primary_checkpoint": {"path": str(primary_checkpoint_artifact.relative_to(root)), "sha256": sha256_file_local(primary_checkpoint_artifact)},
        "second_stage_checkpoint": {"path": str(checkpoint.relative_to(root)), "sha256": sha256_file_local(checkpoint)},
        "second_stage_inference_source": {"path": str(inference_source.relative_to(root)), "sha256": sha256_file_local(inference_source)},
        "pages": alpha_page_rows,
    }
    (alpha_root / "manifest.json").write_text(json.dumps(alpha_manifest), encoding="utf-8")
    return plan_path, ledger_path


class FakeEraseMap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, source: torch.Tensor):
        self.seen.append(source.detach().cpu().clone())
        alpha = source[:, :1].clamp(0.0, 1.0)
        clean = torch.zeros_like(source)
        pred = torch.clamp(source * (1.0 - alpha) + clean * alpha, 0.0, 1.0)
        return pred, alpha, clean


class SecondStageAlphaSupportPrerequisiteTest(unittest.TestCase):
    def registered(self) -> tuple[dict, dict]:
        plan = json.loads(
            (ROOT / "docs/second-stage-alpha-support-prerequisite-v1.json").read_text(
                encoding="utf-8"
            )
        )
        ledger = json.loads(
            (ROOT / "docs/current-primary-quality-loop-ledger.json").read_text(
                encoding="utf-8"
            )
        )
        return plan, ledger

    def test_registered_plan_is_frozen_and_terminal_authority_rejects_rerun(self) -> None:
        plan, ledger = self.registered()
        validate_plan(plan)
        validate_audit_plan(plan)
        with self.assertRaisesRegex(MaterializationError, "diagnostic is not pending"):
            validate_authority(ledger)
        with self.assertRaisesRegex(AuditError, "diagnostic is not pending"):
            validate_audit_authority(ledger)
        self.assertEqual(plan["representation"]["channels"], list(CHANNELS))
        self.assertEqual(plan["diagnostic"]["ablation_features"], list(ABLATION_CHANNELS))
        self.assertFalse(plan["authorization"]["model_training"])
        self.assertFalse(plan["authorization"]["candidate_inference"])
        statuses = {
            item["id"]: item["status"]
            for item in ledger["active_iteration"]["prerequisites"]
        }
        self.assertEqual(statuses["second_stage_alpha_support_train_only_diagnostic"], "passed")
        self.assertEqual(statuses["materially_new_support_successor_preregistration_v3"], "passed")
        self.assertEqual(
            statuses["independent_hw5k_expert_support_train_only_diagnostic"],
            "pending",
        )
        result = next(
            item
            for item in ledger["records"]
            if item["id"] == "second-stage-alpha-support-train275-diagnostic"
        )
        self.assertEqual(result["id"], "second-stage-alpha-support-train275-diagnostic")
        self.assertEqual(result["terminal"], "KILL")
        self.assertEqual(result["repeat_policy"], "do_not_repeat")

    def test_full_page_alpha_materialization_uses_bchw_rgb_and_is_aligned(self) -> None:
        image_bgr = np.zeros((5, 6, 3), dtype=np.uint8)
        image_bgr[..., 2] = np.array(
            [
                [0, 32, 64, 96, 128, 160],
                [16, 48, 80, 112, 144, 176],
                [32, 64, 96, 128, 160, 192],
                [48, 80, 112, 144, 176, 208],
                [64, 96, 128, 160, 192, 224],
            ],
            dtype=np.uint8,
        )
        model = FakeEraseMap()
        raw_alpha = infer_raw_alpha_full_page(
            model,
            image_bgr,
            torch.device("cpu"),
            tile_size=4,
            stride=3,
            batch_size=2,
        )
        self.assertEqual(raw_alpha.shape, (5, 6))
        self.assertEqual(raw_alpha.dtype, np.float32)
        self.assertTrue(np.isfinite(raw_alpha).all())
        expected = image_bgr[..., 2].astype(np.float32) / 255.0
        np.testing.assert_allclose(raw_alpha, expected, atol=1e-6)
        self.assertTrue(model.seen)
        first = model.seen[0]
        self.assertEqual(first.shape[1], 3)
        self.assertEqual(first.ndim, 4)
        rgb_expected = image_bgr[0, 0, ::-1].astype(np.float32) / 255.0
        np.testing.assert_allclose(first[0, :, 0, 0].numpy(), rgb_expected, atol=1e-6)

    def test_safe_loader_requires_weights_only_erasemap_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            checkpoint = Path(raw) / "good.pt"
            torch.save(
                {
                    "args": {"model_type": "erasemap"},
                    "model": {
                        "enc1.block.0.weight": torch.zeros((32, 3, 3, 3)),
                        "enc1.block.0.bias": torch.zeros(32),
                        "enc1.block.2.weight": torch.zeros((32, 32, 3, 3)),
                        "enc1.block.2.bias": torch.zeros(32),
                        "enc2.block.0.weight": torch.zeros((64, 32, 3, 3)),
                        "enc2.block.0.bias": torch.zeros(64),
                        "enc2.block.2.weight": torch.zeros((64, 64, 3, 3)),
                        "enc2.block.2.bias": torch.zeros(64),
                        "bottleneck.block.0.weight": torch.zeros((96, 64, 3, 3)),
                        "bottleneck.block.0.bias": torch.zeros(96),
                        "bottleneck.block.2.weight": torch.zeros((96, 96, 3, 3)),
                        "bottleneck.block.2.bias": torch.zeros(96),
                        "up2.weight": torch.zeros((96, 64, 2, 2)),
                        "up2.bias": torch.zeros(64),
                        "dec2.block.0.weight": torch.zeros((64, 128, 3, 3)),
                        "dec2.block.0.bias": torch.zeros(64),
                        "dec2.block.2.weight": torch.zeros((64, 64, 3, 3)),
                        "dec2.block.2.bias": torch.zeros(64),
                        "up1.weight": torch.zeros((64, 32, 2, 2)),
                        "up1.bias": torch.zeros(32),
                        "dec1.block.0.weight": torch.zeros((32, 64, 3, 3)),
                        "dec1.block.0.bias": torch.zeros(32),
                        "dec1.block.2.weight": torch.zeros((32, 32, 3, 3)),
                        "dec1.block.2.bias": torch.zeros(32),
                        "alpha_head.0.weight": torch.zeros((16, 32, 3, 3)),
                        "alpha_head.0.bias": torch.zeros(16),
                        "alpha_head.2.weight": torch.zeros((1, 16, 1, 1)),
                        "alpha_head.2.bias": torch.zeros(1),
                        "clean_head.0.weight": torch.zeros((16, 32, 3, 3)),
                        "clean_head.0.bias": torch.zeros(16),
                        "clean_head.2.weight": torch.zeros((3, 16, 1, 1)),
                        "clean_head.2.bias": torch.zeros(3),
                    },
                },
                checkpoint,
            )
            with mock.patch(
                "scripts.analysis.materialize_second_stage_alpha_train_only.torch.load",
                wraps=torch.load,
            ) as load_mock:
                model = load_registered_erasemap(checkpoint, torch.device("cpu"))
                self.assertEqual(type(model).__name__, "EraseMapCleanupNet")
                self.assertEqual(load_mock.call_args.kwargs["weights_only"], True)

            wrong = Path(raw) / "wrong.pt"
            torch.save({"args": {"model_type": "residual_delta"}, "model": {}}, wrong)
            with self.assertRaisesRegex(MaterializationError, "not erasemap"):
                load_registered_erasemap(wrong, torch.device("cpu"))

    def test_safe_loader_accepts_legacy_missing_model_type_when_state_matches(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            checkpoint = Path(raw) / "legacy.pt"
            from scripts.infer.patch_cleanup_erasemap import build_model

            torch.save({"args": {}, "model": build_model("erasemap").state_dict()}, checkpoint)
            loaded = load_registered_erasemap(checkpoint, torch.device("cpu"))
            self.assertEqual(type(loaded).__name__, "EraseMapCleanupNet")
            self.assertFalse(loaded.training)

    def test_safe_loader_rejects_legacy_missing_model_type_with_key_or_shape_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            from scripts.infer.patch_cleanup_erasemap import build_model

            bad_key = Path(raw) / "bad-key.pt"
            state_dict = build_model("erasemap").state_dict()
            state_dict.pop(next(iter(state_dict)))
            torch.save({"args": {}, "model": state_dict}, bad_key)
            with self.assertRaisesRegex(MaterializationError, "does not match erasemap keys"):
                load_registered_erasemap(bad_key, torch.device("cpu"))

            bad_shape = Path(raw) / "bad-shape.pt"
            state_dict = build_model("erasemap").state_dict()
            first_key = next(iter(state_dict))
            state_dict[first_key] = torch.zeros((1,), dtype=state_dict[first_key].dtype)
            torch.save({"args": {}, "model": state_dict}, bad_shape)
            with self.assertRaisesRegex(MaterializationError, "tensor shape changed"):
                load_registered_erasemap(bad_shape, torch.device("cpu"))
    def test_manifest_freezes_materialization_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output_root = root / "outputs" / "second-stage-alpha-support-materialization-20260812"
            output_root.mkdir(parents=True)
            plan_path = root / "docs" / "second-stage-alpha-support-prerequisite-v1.json"
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text("{}", encoding="utf-8")
            config = root / "artifacts" / "config.yaml"
            checkpoint = root / "artifacts" / "checkpoint.pt"
            primary_checkpoint = root / "artifacts" / "primary.pt"
            source = root / "scripts" / "infer.py"
            manifest_src = root / "lists" / "manifest.txt"
            for path in (config, checkpoint, primary_checkpoint, source, manifest_src):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")
            manifest_path = write_manifest(
                output_root,
                repo_root=root,
                plan_path=plan_path,
                config=config,
                primary_checkpoint=primary_checkpoint,
                checkpoint=checkpoint,
                inference_source=source,
                source_manifest=manifest_src,
                rows=[{
                    "file": "page.png",
                    "height": 2,
                    "width": 3,
                    "source_prediction_sha256": "a",
                    "npz_sha256": "b",
                    "raw_alpha_min": 0.1,
                    "raw_alpha_max": 0.9,
                    "raw_alpha_mean": 0.5,
                }],
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["provenance"], "current_second_stage_erasemap_alpha_head_sigmoid")
        self.assertEqual(payload["primary_checkpoint"]["path"], "artifacts/primary.pt")
        self.assertEqual(payload["tile_size"], 160)
        self.assertEqual(payload["stride"], 160)
        self.assertEqual(payload["batch_size"], 32)
        self.assertEqual(payload["dtype"], "float32")
        self.assertEqual(payload["encoding"], "one_compressed_npz_per_page_with_exact_raw_alpha_key")
        self.assertEqual(payload["raw_alpha_key"], RAW_ALPHA_KEY)
        self.assertEqual(
            payload["overlap_fusion"],
            "arithmetic_mean_of_raw_patch_alpha_before_any_threshold",
        )
        self.assertFalse(payload["target_access"])
        self.assertEqual(payload["pages_directory"], "pages")

    def test_manifest_requires_primary_checkpoint_entry_for_audit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path, ledger_path = build_synthetic_repo(root, alpha_signal=True)
            manifest_path = root / OUTPUT_ROOT / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("primary_checkpoint")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.validate_plan",
                autospec=True,
            ), mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.validate_authority",
                autospec=True,
            ), mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.effective_train_filenames",
                return_value=sorted([f"page-{index:03d}.png" for index in range(10)]),
            ):
                with self.assertRaisesRegex(AuditError, "primary_checkpoint contract changed"):
                    run_audit(repo_root=root, plan_path=plan_path.relative_to(root), ledger_path=ledger_path.relative_to(root))


    def test_page_features_are_output_rgb_plus_raw_alpha_only(self) -> None:
        second = np.asarray(
            [
                [[20, 30, 40], [50, 60, 70]],
                [[80, 90, 100], [110, 120, 130]],
            ],
            dtype=np.uint8,
        )
        target = second.copy()
        target[0, 0] += 10
        target[1, 0] += 10
        raw_alpha = np.asarray([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            second_dir = root / "second"
            label_dir = root / "labels"
            alpha_dir = root / "alpha"
            write_rgb(second_dir / "page.png", second)
            write_rgb(label_dir / "page.png", target)
            alpha_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(alpha_dir / "page.npz", raw_alpha=raw_alpha)
            alpha_row = {
                "file": "page.png",
                "height": 2,
                "width": 2,
                "source_prediction_sha256": "unused-for-page-test",
                "npz_sha256": "unused-for-page-test",
                "raw_alpha_min": float(raw_alpha.min()),
                "raw_alpha_max": float(raw_alpha.max()),
                "raw_alpha_mean": float(raw_alpha.mean()),
            }
            page = build_page(
                file_name="page.png",
                second_stage_dir=second_dir,
                label_dir=label_dir,
                alpha_dir=alpha_dir,
                alpha_manifest_row=alpha_row,
                second_stage_row={
                    "base_edit_threshold": "12",
                    "second_delta_threshold": "32",
                    "dark_threshold": "0",
                },
                margin_gray=2.0,
                sample_cap=1,
            )
        self.assertEqual(page["features"].shape, (2, 4))
        np.testing.assert_array_equal(page["ablation_features"], page["features"][:, :3])
        selected_rgb = np.rint(page["features"][:, :3] * 255.0).astype(np.uint8)
        positive_rows = {tuple(second.reshape(-1, 3)[index]) for index in (0, 2)}
        preserve_rows = {tuple(second.reshape(-1, 3)[index]) for index in (1, 3)}
        self.assertIn(tuple(selected_rgb[0]), positive_rows)
        self.assertIn(tuple(selected_rgb[1]), preserve_rows)
        self.assertGreater(page["features"][0, 3], page["features"][1, 3])

    def test_raw_alpha_loader_and_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "page.npz"
            values = np.zeros((3, 4), dtype=np.float32)
            np.savez_compressed(path, raw_alpha=values)
            loaded = load_raw_alpha_npz(path, expected_shape=(3, 4))
            self.assertEqual(loaded.shape, (3, 4))
            np.savez_compressed(path, wrong=values)
            with self.assertRaisesRegex(AuditError, "keys changed"):
                load_raw_alpha_npz(path)
        plan, ledger = self.registered()
        plan["diagnostic"]["lambda"] = 0.1
        with self.assertRaisesRegex(AuditError, "diagnostic field changed"):
            validate_audit_plan(plan)
        plan, ledger = self.registered()
        plan["second_stage_alpha_materialization"]["stride"] = 80
        with self.assertRaisesRegex(MaterializationError, "materialization field changed"):
            validate_plan(plan)
        for item in ledger["active_iteration"]["prerequisites"]:
            if item["id"] == "second_stage_alpha_support_train_only_diagnostic":
                item["status"] = "passed"
        with self.assertRaisesRegex(AuditError, "diagnostic is not pending"):
            validate_audit_authority(ledger)
        with self.assertRaisesRegex(MaterializationError, "diagnostic is not pending"):
            validate_authority(ledger)

    def test_registered_fold_evaluation_covers_full_and_ablation(self) -> None:
        pages = []
        labels = np.asarray([-1, -1, 1, 1], dtype=np.int8)
        signal = np.asarray([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
        for fold in range(5):
            for page_index in range(2):
                features = np.column_stack(
                    [
                        np.zeros(4, dtype=np.float32),
                        np.zeros(4, dtype=np.float32),
                        np.zeros(4, dtype=np.float32),
                        signal,
                    ]
                )
                pages.append(
                    {
                        "file": f"fold-{fold}-page-{page_index}.png",
                        "fold": fold,
                        "features": features,
                        "ablation_features": features[:, :3],
                        "labels": labels,
                        "samples_per_class": 2,
                    }
                )
        result = evaluate_fold(pages, 0, 1.0)
        self.assertEqual(result["test_page_count"], 2)
        self.assertGreater(result["full_auc"], result["ablation_auc"])


    def test_run_audit_synthetic_pass_and_kill(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path, ledger_path = build_synthetic_repo(root, alpha_signal=True)
            with mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.validate_plan",
                autospec=True,
            ), mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.validate_authority",
                autospec=True,
            ), mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.effective_train_filenames",
                return_value=sorted([f"page-{index:03d}.png" for index in range(10)]),
            ):
                result = run_audit(repo_root=root, plan_path=plan_path.relative_to(root), ledger_path=ledger_path.relative_to(root))
                self.assertEqual(result["terminal"], "PASS")
                self.assertTrue(result["acceptance"]["passed"])
                self.assertGreater(result["aggregates"]["full_auc_ablation_margin"], 0.03)

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path, ledger_path = build_synthetic_repo(root, alpha_signal=False)
            with mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.validate_plan",
                autospec=True,
            ), mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.validate_authority",
                autospec=True,
            ), mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.effective_train_filenames",
                return_value=sorted([f"page-{index:03d}.png" for index in range(10)]),
            ):
                result = run_audit(repo_root=root, plan_path=plan_path.relative_to(root), ledger_path=ledger_path.relative_to(root))
                self.assertEqual(result["terminal"], "KILL")
                self.assertFalse(result["acceptance"]["passed"])

    def test_run_audit_fail_closed_on_extra_npz_and_bad_summary(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path, ledger_path = build_synthetic_repo(root, alpha_signal=True)
            extra = root / OUTPUT_ROOT / "pages" / "unexpected.npz"
            np.savez_compressed(extra, raw_alpha=np.zeros((2, 2), dtype=np.float32))
            with mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.validate_plan",
                autospec=True,
            ), mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.validate_authority",
                autospec=True,
            ), mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.effective_train_filenames",
                return_value=sorted([f"page-{index:03d}.png" for index in range(10)]),
            ):
                with self.assertRaisesRegex(AuditError, "page filenames changed"):
                    run_audit(repo_root=root, plan_path=plan_path.relative_to(root), ledger_path=ledger_path.relative_to(root))

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path, ledger_path = build_synthetic_repo(root, alpha_signal=True)
            manifest_path = root / OUTPUT_ROOT / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["pages"][0]["height"] = 0
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.validate_plan",
                autospec=True,
            ), mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.validate_authority",
                autospec=True,
            ), mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.effective_train_filenames",
                return_value=sorted([f"page-{index:03d}.png" for index in range(10)]),
            ):
                with self.assertRaisesRegex(AuditError, "invalid alpha summary dimensions"):
                    run_audit(repo_root=root, plan_path=plan_path.relative_to(root), ledger_path=ledger_path.relative_to(root))

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path, ledger_path = build_synthetic_repo(root, alpha_signal=True)
            manifest_path = root / OUTPUT_ROOT / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["pages"][0]["file"] = "wrong.png"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.validate_plan",
                autospec=True,
            ), mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.validate_authority",
                autospec=True,
            ), mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.effective_train_filenames",
                return_value=sorted([f"page-{index:03d}.png" for index in range(10)]),
            ):
                with self.assertRaisesRegex(AuditError, "identities changed"):
                    run_audit(repo_root=root, plan_path=plan_path.relative_to(root), ledger_path=ledger_path.relative_to(root))

    def test_run_audit_fail_closed_on_manifest_and_npz_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path, ledger_path = build_synthetic_repo(root, alpha_signal=True)
            alpha_root = root / OUTPUT_ROOT
            manifest = json.loads((alpha_root / "manifest.json").read_text(encoding="utf-8"))
            manifest["pages_directory"] = "wrong"
            (alpha_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.validate_plan",
                autospec=True,
            ), mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.validate_authority",
                autospec=True,
            ), mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.effective_train_filenames",
                return_value=sorted([f"page-{index:03d}.png" for index in range(10)]),
            ):
                with self.assertRaisesRegex(AuditError, "manifest changed: pages_directory"):
                    run_audit(repo_root=root, plan_path=plan_path.relative_to(root), ledger_path=ledger_path.relative_to(root))

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            plan_path, ledger_path = build_synthetic_repo(root, alpha_signal=True)
            npz_path = root / OUTPUT_ROOT / "pages" / "page-000.npz"
            np.savez_compressed(npz_path, raw_alpha=np.zeros((2, 2), dtype=np.float64))
            manifest = json.loads((root / OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
            for row in manifest["pages"]:
                if row["file"] == "page-000.png":
                    row["npz_sha256"] = sha256_file_local(npz_path)
            (root / OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.validate_plan",
                autospec=True,
            ), mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.validate_authority",
                autospec=True,
            ), mock.patch(
                "scripts.analysis.audit_second_stage_alpha_support.effective_train_filenames",
                return_value=sorted([f"page-{index:03d}.png" for index in range(10)]),
            ):
                with self.assertRaisesRegex(AuditError, "dtype changed"):
                    run_audit(repo_root=root, plan_path=plan_path.relative_to(root), ledger_path=ledger_path.relative_to(root))


if __name__ == "__main__":
    unittest.main()
