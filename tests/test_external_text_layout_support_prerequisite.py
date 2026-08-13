from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from scripts.analysis.audit_external_text_layout_support import (
    ABLATION_CHANNELS,
    CHANNELS,
    AuditError,
    build_page,
    load_layout_npz,
    validate_materialization,
)
from scripts.analysis.materialize_external_text_layout_support_train_only import (
    MaterializationError,
    NPZ_KEYS,
    normalize_detections,
    rasterize_layout,
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def write_rgb(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(values, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write image: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExternalTextLayoutSupportTest(unittest.TestCase):
    def test_registered_plan_is_strictly_valid(self) -> None:
        plan = json.loads(
            (ROOT / "docs/external-text-layout-support-prerequisite-v1.json").read_text(
                encoding="utf-8"
            )
        )
        validate_plan(plan)

    def test_normalize_detections_clips_and_sorts_by_frozen_key(self) -> None:
        polygons = np.asarray(
            [
                [[9.4, 9.4], [12.4, 9.4], [12.4, 12.4], [9.4, 12.4]],
                [[2.4, 1.4], [4.4, 1.4], [4.4, 3.4], [2.4, 3.4]],
            ],
            dtype=np.float32,
        )
        normalized, scores = normalize_detections(
            polygons, np.asarray([0.8, 0.9], dtype=np.float32), height=10, width=10
        )
        np.testing.assert_array_equal(
            normalized[0], np.asarray([[2, 1], [4, 1], [4, 3], [2, 3]], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            normalized[1], np.asarray([[9, 9], [9, 9], [9, 9], [9, 9]], dtype=np.int32)
        )
        np.testing.assert_array_equal(scores, np.asarray([0.9, 0.8], dtype=np.float32))

    def test_rasterize_layout_uses_binary_occupancy_and_max_confidence(self) -> None:
        polygons = np.asarray(
            [
                [[1, 1], [3, 1], [3, 3], [1, 3]],
                [[2, 2], [4, 2], [4, 4], [2, 4]],
            ],
            dtype=np.int32,
        )
        occupancy, confidence = rasterize_layout(
            polygons,
            np.asarray([0.4, 0.8], dtype=np.float32),
            height=6,
            width=6,
        )
        self.assertEqual(occupancy.dtype, np.uint8)
        self.assertEqual(confidence.dtype, np.float32)
        self.assertEqual(int(occupancy[2, 2]), 1)
        self.assertAlmostEqual(float(confidence[2, 2]), 0.8, places=6)
        self.assertTrue(np.isin(occupancy, [0, 1]).all())

    def test_malformed_detector_output_fails_closed(self) -> None:
        with self.assertRaises(MaterializationError):
            normalize_detections(
                np.zeros((1, 4, 3), dtype=np.float32),
                np.ones(1, dtype=np.float32),
                height=4,
                width=4,
            )
        with self.assertRaises(MaterializationError):
            normalize_detections(
                np.zeros((1, 4, 2), dtype=np.float32),
                np.asarray([1.1], dtype=np.float32),
                height=4,
                width=4,
            )

    def test_npz_contract_rejects_extra_key(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "page.npz"
            np.savez_compressed(
                path,
                polygons=np.empty((0, 4, 2), dtype=np.int32),
                scores=np.empty((0,), dtype=np.float32),
                text_confidence=np.zeros((2, 2), dtype=np.float32),
                text_occupancy=np.zeros((2, 2), dtype=np.uint8),
                forbidden=np.zeros(1),
            )
            with self.assertRaises(AuditError):
                load_layout_npz(path, expected_shape=(2, 2))

    def test_npz_contract_reconstructs_raster(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "page.npz"
            polygons = np.asarray(
                [[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.int32
            )
            scores = np.asarray([0.75], dtype=np.float32)
            occupancy, confidence = rasterize_layout(
                polygons, scores, height=2, width=2
            )
            np.savez_compressed(
                path,
                polygons=polygons,
                scores=scores,
                text_confidence=confidence,
                text_occupancy=occupancy,
            )
            arrays = load_layout_npz(path, expected_shape=(2, 2))
            self.assertEqual(set(arrays), NPZ_KEYS)

    def test_five_channel_page_features_share_ablation_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            second_dir = root / "second"
            label_dir = root / "labels"
            layout_dir = root / "layout"
            second = np.asarray(
                [
                    [[10, 20, 30], [40, 50, 60]],
                    [[70, 80, 90], [100, 110, 120]],
                ],
                dtype=np.uint8,
            )
            target = second.copy()
            target[0, 0] += 10
            target[1, 0] += 10
            write_rgb(second_dir / "page.png", second)
            write_rgb(label_dir / "page.png", target)
            polygons = np.asarray(
                [[[0, 0], [0, 0], [0, 1], [0, 1]]], dtype=np.int32
            )
            scores = np.asarray([0.75], dtype=np.float32)
            occupancy, confidence = rasterize_layout(
                polygons, scores, height=2, width=2
            )
            layout_dir.mkdir(parents=True)
            np.savez_compressed(
                layout_dir / "page.npz",
                polygons=polygons,
                scores=scores,
                text_confidence=confidence,
                text_occupancy=occupancy,
            )
            page = build_page(
                file_name="page.png",
                second_stage_dir=second_dir,
                label_dir=label_dir,
                layout_dir=layout_dir,
                layout_row={"height": 2, "width": 2},
                second_stage_row={
                    "base_edit_threshold": "12",
                    "second_delta_threshold": "32",
                    "dark_threshold": "0",
                },
                margin_gray=2.0,
                sample_cap=1,
            )
            self.assertEqual(page["features"].shape[1], len(CHANNELS))
            self.assertEqual(page["ablation_features"].shape[1], len(ABLATION_CHANNELS))
            np.testing.assert_array_equal(
                page["ablation_features"], page["features"][:, :3]
            )

    def test_target_path_rejected_before_any_detector_factory(self) -> None:
        import scripts.analysis.materialize_external_text_layout_support_train_only as materializer

        with mock.patch.object(materializer, "validate_registered_inputs", side_effect=MaterializationError("forbidden source path")):
            detector_factory = mock.Mock(side_effect=AssertionError("detector called"))
            with self.assertRaises(MaterializationError):
                materializer.materialize(
                    repo_root=ROOT,
                    detector_factory=detector_factory,
                )
            detector_factory.assert_not_called()

    def test_materializer_failure_removes_temporary_output(self) -> None:
        import scripts.analysis.materialize_external_text_layout_support_train_only as materializer

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.png"
            write_rgb(source, np.zeros((2, 2, 3), dtype=np.uint8))
            output_root = root / "materialized"
            registered = {
                "file_names": ["source.png"],
                "output_root": output_root,
                "temporary_root": output_root.with_name(".materialized.materializing"),
                "sources": [("source.png", "source.png", source)],
            }

            class FailingDetector:
                def predict(self, **_kwargs):
                    raise RuntimeError("synthetic detector failure")

                def close(self):
                    pass

            with mock.patch.object(materializer, "validate_registered_inputs", return_value=registered):
                with self.assertRaises(MaterializationError):
                    materializer.materialize(
                        repo_root=ROOT,
                        detector_factory=lambda _spec: FailingDetector(),
                    )
            self.assertFalse(registered["temporary_root"].exists())
            self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
