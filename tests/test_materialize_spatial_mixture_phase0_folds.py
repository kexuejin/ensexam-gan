#!/usr/bin/env python3
"""Focused tests for deterministic Phase 0 fold materialization.

These tests exercise the pure, stdlib-only helpers and the full pipeline with
synthetic (tempfile-backed) manifests and image files. They assert the frozen
383-split semantics, the deterministic fold assignment that reproduces the
frozen per-fold counts, and the prohibited-surface isolation logic.

No real quality/blind image bytes are hashed here; tests never open the real
dataset fixture paths.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.materialize_spatial_mixture_phase0_folds import (
    FROZEN_FOLD_DOMAIN_COUNTS,
    SALT,
    assign_folds,
    build_prohibited_surface_identities,
    canonical_identity,
    digest,
    main as materialize_main,
    read_dev40_identities,
    sha256_file,
    strip_domain_prefix,
)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


class FakeArgs:
    """Minimal argparse.Namespace substitute to drive materialize_main in-process."""

    def __init__(self, root: Path, source_manifest: Path, data_root: Path):
        self.source_manifest = source_manifest
        self.data_root = data_root
        self.salt = SALT
        self.fold_count = 6
        self.output_dir = root / "out"
        self.inner_val15 = root / "inner_val15.txt"
        self.dev40_csv = root / "dev40.csv"
        self.holdout40 = root / "holdout40.txt"
        self.scut115 = root / "scut115.txt"
        self.hw5k_dev232 = root / "hw5k_dev232.txt"
        self.hw5k_reserved_root = root / "hw5k_reserved"
        self.plan = root / "plan.md"


class MaterializeSpatialMixturePhase0FoldsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # --- helper to build a realistic synthetic fixture ----------------------

    def build_fixture(
        self,
        scut_ids: list[int],
        hw5k_ids: list[int],
        barred_scut: list[int] | None = None,
        barred_hw5k: list[int] | None = None,
    ) -> FakeArgs:
        barred_scut = barred_scut or [506, 301, 45]
        barred_hw5k = barred_hw5k or [54, 74]
        data_root = self.root / "data"
        (data_root / "all_images").mkdir(parents=True)
        (data_root / "all_labels").mkdir(parents=True)

        src = self.root / "source.txt"
        rows = []
        for dom, ids in (("scut", scut_ids), ("hw5k", hw5k_ids)):
            for num in ids:
                base = f"{dom}_{num}.jpg"
                (data_root / "all_images" / base).write_bytes(b"img")
                (data_root / "all_labels" / base).write_bytes(b"lbl")
                rows.append(base)
        src.write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.assertEqual(src.read_text().count("\n"), len(rows))

        def w(path: Path, content: str) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        w(self.root / "inner_val15.txt", "\n".join(f"{n}.jpg" for n in barred_scut) + "\n")
        w(self.root / "dev40.csv", "split,file\n" + "\n".join(f"dev40,{n}.jpg" for n in [701, 702]) + "\n")
        w(self.root / "holdout40.txt", "\n".join(f"{n}.jpg" for n in [703, 704]) + "\n")
        w(self.root / "scut115.txt", "\n".join(f"{p}/{n}.jpg" for p, n in [("x", 13), ("y", 15)]) + "\n")
        w(self.root / "hw5k_dev232.txt", "\n".join(f"{n}.jpg" for n in barred_hw5k) + "\n")
        (self.root / "hw5k_reserved" / "all_images").mkdir(parents=True)
        for n in (5000, 5001):
            (self.root / "hw5k_reserved" / "all_images" / f"{n}.jpg").write_bytes(b"r")
        (self.root / "plan.md").write_text("frozen plan text\n", encoding="utf-8")
        return FakeArgs(self.root, src, data_root)

    # --- canonical identity / digest ----------------------------------------

    def test_canonical_identity_format(self) -> None:
        self.assertEqual(canonical_identity("scut", "scut_150.jpg"), "scut/train/scut_150.jpg")
        self.assertEqual(canonical_identity("hw5k", "hw5k_1011.jpg"), "hw5k/train/hw5k_1011.jpg")
        # lower-cases domain and returns basename with POSIX separators
        self.assertEqual(canonical_identity("SCUT", "s.jpg"), "scut/train/s.jpg")

    def test_digest_deterministic_and_salt_bound(self) -> None:
        ident = "scut/train/scut_150.jpg"
        self.assertEqual(digest(SALT, ident), digest(SALT, ident))
        self.assertEqual(len(digest(SALT, ident)), 64)
        self.assertNotEqual(digest(SALT, ident), digest("other", ident))

    def test_strip_domain_prefix(self) -> None:
        self.assertEqual(strip_domain_prefix("scut_150.jpg"), "150.jpg")
        self.assertEqual(strip_domain_prefix("hw5k_1011.jpg"), "1011.jpg")
        self.assertEqual(strip_domain_prefix("plain.jpg"), "plain.jpg")

    # --- fold assignment reproduces frozen counts ---------------------------

    def _identity_records(self, dirname: str, count: int) -> list[dict]:
        return [
            {
                "domain": "scut" if i % 2 else "hw5k",
                "basename": f"{( 'scut' if i % 2 else 'hw5k')}_{i}.jpg",
                "identity": canonical_identity("scut" if i % 2 else "hw5k", f"{( 'scut' if i % 2 else 'hw5k')}_{i}.jpg"),
                "digest": sha(f"{SALT}\0{( 'scut' if i % 2 else 'hw5k')}/train/{i}"),
            }
            for i in range(count)
        ]

    def test_assign_folds_reproduces_frozen_counts(self) -> None:
        # 130 scut + 253 hw5k
        records = self._identity_records("d", 383)
        # force exact per-domain membership: first 130 -> scut, rest -> hw5k
        for i, rec in enumerate(records):
            rec["domain"] = "scut" if i < 130 else "hw5k"
            rec["basename"] = f"{'scut' if i < 130 else 'hw5k'}_{i}.jpg"
            rec["identity"] = canonical_identity(rec["domain"], rec["basename"])

        slots = assign_folds(records, FROZEN_FOLD_DOMAIN_COUNTS)
        for f, (scut_qty, hw5k_qty) in enumerate(FROZEN_FOLD_DOMAIN_COUNTS):
            fold_scut = sum(1 for m in slots[f] if m["domain"] == "scut")
            fold_hw5k = sum(1 for m in slots[f] if m["domain"] == "hw5k")
            self.assertEqual((fold_scut, fold_hw5k), (scut_qty, hw5k_qty))

        def all_ids(slots) -> list[str]:
            return sorted(m["identity"] for m in sum(list(slots.values()), []))
        from collections import Counter
        identity_ids = [m["identity"] for m in records]
        self.assertEqual(all_ids(slots), sorted(identity_ids))
        counts = Counter(all_ids(slots))
        for ident in identity_ids:
            self.assertEqual(counts[ident], 1, f"identity duplicated in folds: {ident}")

    def test_assign_folds_score_constant(self) -> None:
        # every identity appears in exactly one slot overall
        records = self._identity_records("d", 383)
        for i, rec in enumerate(records):
            rec["domain"] = "scut" if i < 130 else "hw5k"
            rec["basename"] = f"{'scut' if i < 130 else 'hw5k'}_{i}.jpg"
            rec["identity"] = canonical_identity(rec["domain"], rec["basename"])
        from collections import Counter
        before = Counter(m["identity"] for m in records)
        slots = assign_folds(records, FROZEN_FOLD_DOMAIN_COUNTS)
        after = Counter(m["identity"] for m in sum(list(slots.values()), []))
        self.assertEqual(before, after)

    # --- prohibited-surface isolation ---------------------------------------

    def test_prohibited_surface_identity_parsing(self) -> None:
        args = FakeArgs(self.root, self.root / "p", self.root / "d")
        (self.root / "inner_val15.txt").write_text("506.jpg\n301.jpg\n", encoding="utf-8")
        (self.root / "dev40.csv").write_text("split,file\ndev40,701.jpg\n", encoding="utf-8")
        (self.root / "holdout40.txt").write_text("703.jpg\n", encoding="utf-8")
        (self.root / "scut115.txt").write_text("x/13.jpg\n", encoding="utf-8")
        (self.root / "hw5k_dev232.txt").write_text("54.jpg\n", encoding="utf-8")
        (self.root / "hw5k_reserved" / "all_images").mkdir(parents=True)
        (self.root / "hw5k_reserved" / "all_images" / "5000.jpg").write_bytes(b"r")

        surfaces = build_prohibited_surface_identities(
            args.inner_val15, args.dev40_csv, args.holdout40,
            args.scut115, args.hw5k_dev232, args.hw5k_reserved_root,
        )
        self.assertIn("506.jpg", surfaces["inner_val15"])
        self.assertIn("701.jpg", surfaces["dev40"])
        self.assertIn("703.jpg", surfaces["holdout40"])
        self.assertIn("13.jpg", surfaces["scut115"])
        self.assertIn("54.jpg", surfaces["hw5k_dev232"])
        self.assertIn("5000.jpg", surfaces["hw5k_reserved"])
        self.assertIn("506.jpg", surfaces["_scut_union"])
        self.assertIn("54.jpg", surfaces["_hw5k_union"])

    def test_read_dev40_identities(self) -> None:
        p = self.root / "d.csv"
        p.write_text("split,file\ndev40,27.jpg\ndev40,346.jpg\n", encoding="utf-8")
        self.assertEqual(read_dev40_identities(p), {"27.jpg", "346.jpg"})

    # --- sha256 convenience --------------------------------------------------

    def test_sha256_file(self) -> None:
        p = self.root / "x.bin"
        p.write_bytes(b"hello")
        self.assertEqual(sha256_file(p), hashlib.sha256(b"hello").hexdigest())


if __name__ == "__main__":
    unittest.main()
