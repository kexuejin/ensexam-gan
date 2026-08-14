from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest


from scripts.analysis import audit_external_text_layout_second_stage_cache_salvage as salvage


class ExternalTextLayoutSecondStageCacheSalvageTest(unittest.TestCase):
    def test_repository_contract_is_exact(self) -> None:
        contract = salvage.validate_repository_contract(salvage.ROOT)
        self.assertEqual(contract["schema_version"], 1)
        self.assertEqual(
            contract["canonical_candidate"]["metrics_sha256"],
            "79fd61278e689a0003e37a5bdf20f856184b49c8fdb3af8ad9af03a3a13c451b",
        )

    def test_all_metric_fields_have_exact_sources(self) -> None:
        recomputed = {
            "changed_px": 10,
            "mean_over_delta": 2.5,
            "mean_residual_delta": 3.5,
            "outside_px": 20,
            "over_px": 4,
            "overerase_ratio": 0.2,
            "residual_px": 5,
            "residual_ratio": 0.5,
        }
        row = {
            "base_edit_threshold": "12.0",
            "changed_px": "10",
            "dark_threshold": "0",
            "file": "page.jpg",
            "gate_ratio": "0.125",
            "image_path": "samples/all_images/page.jpg",
            "max_brighten_delta": "1000000000.0",
            "mean_over_delta": "2.5",
            "mean_residual_delta": "3.5",
            "outside_px": "20",
            "over_px": "4",
            "overerase_ratio": "0.2",
            "pred_path": "/repo/outputs/cache/pred/page.png",
            "residual_px": "5",
            "residual_ratio": "0.5",
            "second_delta_threshold": "32.0",
        }
        salvage.validate_metric_row(
            row,
            expected_file="page.jpg",
            expected_image_path="samples/all_images/page.jpg",
            expected_pred_path="/repo/outputs/cache/pred/page.png",
            expected_gate_ratio=0.125,
            constants={
                "base_edit_threshold": 12.0,
                "dark_threshold": 0,
                "max_brighten_delta": 1000000000.0,
                "second_delta_threshold": 32.0,
            },
            recomputed=recomputed,
        )

        for field in row:
            changed = dict(row)
            changed[field] = "different"
            with self.subTest(field=field), self.assertRaises(
                salvage.SemanticMismatch
            ):
                salvage.validate_metric_row(
                    changed,
                    expected_file="page.jpg",
                    expected_image_path="samples/all_images/page.jpg",
                    expected_pred_path="/repo/outputs/cache/pred/page.png",
                    expected_gate_ratio=0.125,
                    constants={
                        "base_edit_threshold": 12.0,
                        "dark_threshold": 0,
                        "max_brighten_delta": 1000000000.0,
                        "second_delta_threshold": 32.0,
                    },
                    recomputed=recomputed,
                )

    def test_canonical_candidate_is_read_only_and_hash_bound(self) -> None:
        current = "/current/root"
        historical = "/historical/root"
        before = (
            "file,pred_path\r\n"
            f"a.jpg,{current}/outputs/cache/pred/a.png\r\n"
            f"b.jpg,{current}/outputs/cache/pred/b.png\r\n"
        ).encode()
        after = before.replace(current.encode(), historical.encode())
        with tempfile.TemporaryDirectory() as raw:
            metrics = Path(raw) / "metrics.csv"
            metrics.write_bytes(before)
            result = salvage.validate_canonical_candidate(
                metrics,
                current_repository_root=current,
                frozen_historical_repository_root=historical,
                expected_replacement_count=2,
                expected_sha256=hashlib.sha256(after).hexdigest(),
            )
            self.assertEqual(metrics.read_bytes(), before)
            self.assertEqual(result["replacement_count"], 2)
            self.assertEqual(
                result["candidate_sha256"], hashlib.sha256(after).hexdigest()
            )

            with self.assertRaises(salvage.SemanticMismatch):
                salvage.validate_canonical_candidate(
                    metrics,
                    current_repository_root=current,
                    frozen_historical_repository_root=historical,
                    expected_replacement_count=3,
                    expected_sha256=hashlib.sha256(after).hexdigest(),
                )
            self.assertEqual(metrics.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
