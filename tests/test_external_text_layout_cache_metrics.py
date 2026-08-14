from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest


from scripts.analysis import external_text_layout_cache_metrics as cache_metrics


CURRENT_ROOT = "/Volumes/Tool/source/ensexam-gan-h0-monotonic-safe"
HISTORICAL_ROOT = "/private/tmp/ensexam-gan-h0-P0vNwp"


def metrics_payload(
    *,
    row_count: int = 2,
    current_root: str = CURRENT_ROOT,
    root_outside_pred_path: bool = False,
) -> bytes:
    rows = ["file,pred_path,score"]
    for index in range(row_count):
        source = f"page-{index}.jpg"
        if root_outside_pred_path and index == 0:
            source = f"{current_root}/sources/{source}"
        rows.append(
            f"{source},{current_root}/outputs/cache/pred/page-{index}.png,{index}"
        )
    return ("\n".join(rows) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class ExternalTextLayoutCacheMetricsTest(unittest.TestCase):
    def test_exact_pred_path_only_canonicalization_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            metrics = Path(raw) / "metrics.csv"
            before = metrics_payload()
            after = before.replace(
                CURRENT_ROOT.encode("utf-8"), HISTORICAL_ROOT.encode("utf-8")
            )
            metrics.write_bytes(before)

            result = cache_metrics.canonicalize_repository_root(
                metrics,
                current_repository_root=CURRENT_ROOT,
                frozen_historical_repository_root=HISTORICAL_ROOT,
                expected_data_rows=2,
                expected_replacement_count=2,
                expected_metrics_sha256_after=sha256_bytes(after),
                expected_metrics_sha256_before=sha256_bytes(before),
            )

            self.assertEqual(metrics.read_bytes(), after)
            self.assertEqual(result["replacement_count"], 2)
            self.assertEqual(result["metrics_sha256_before"], sha256_bytes(before))
            self.assertEqual(result["metrics_sha256_after"], sha256_bytes(after))
            self.assertFalse(metrics.with_name(".metrics.csv.canonicalizing").exists())

    def test_invalid_inputs_are_rejected_before_mutation(self) -> None:
        cases = (
            ("wrong_source_hash", metrics_payload(), 2, 2, "0" * 64, None),
            ("wrong_row_count", metrics_payload(), 3, 2, None, None),
            ("wrong_replacement_count", metrics_payload(), 2, 3, None, None),
            (
                "missing_csv_field",
                (
                    "file,pred_path,score\n"
                    f"page.jpg,{CURRENT_ROOT}/outputs/cache/pred/page.png\n"
                ).encode(),
                1,
                1,
                None,
                None,
            ),
            (
                "root_outside_pred_path",
                metrics_payload(root_outside_pred_path=True),
                2,
                2,
                None,
                None,
            ),
            ("wrong_candidate_hash", metrics_payload(), 2, 2, None, "0" * 64),
        )
        for name, before, rows, replacements, before_sha, after_sha in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as raw:
                metrics = Path(raw) / "metrics.csv"
                metrics.write_bytes(before)
                expected_after = after_sha or sha256_bytes(
                    before.replace(
                        CURRENT_ROOT.encode("utf-8"),
                        HISTORICAL_ROOT.encode("utf-8"),
                    )
                )
                with self.assertRaises(cache_metrics.MetricsCanonicalizationError):
                    cache_metrics.canonicalize_repository_root(
                        metrics,
                        current_repository_root=CURRENT_ROOT,
                        frozen_historical_repository_root=HISTORICAL_ROOT,
                        expected_data_rows=rows,
                        expected_replacement_count=replacements,
                        expected_metrics_sha256_after=expected_after,
                        expected_metrics_sha256_before=(
                            before_sha
                            if before_sha is not None
                            else sha256_bytes(before)
                        ),
                    )
                self.assertEqual(metrics.read_bytes(), before)
                self.assertFalse(
                    metrics.with_name(".metrics.csv.canonicalizing").exists()
                )


if __name__ == "__main__":
    unittest.main()
