#!/usr/bin/env python3
"""Materialize deterministic Phase 0 page-grouped folds for the spatial mixture.

This is the "train-role only, page-grouped" data-custody step of the frozen
implementation plan `docs/plans/2026-08-16-spatial-continuous-reconstruction-mixture-implementation.md`.

Scope discipline (matches the plan's non-goals and the success-selector policy):

- Only eligible train-role source/target file bytes are hashed. No quality or
  blind surface image is ever hashed or decoded.
- Prohibited manifests are parsed only for page identities (filenames/stems),
  never for pixel content.
- The tool never loads torch, cv2, or numpy, and never decodes an image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SALT = "spatial-mixture-phase0-v1"
DOMAINS = ("scut", "hw5k")
FOLD_COUNT = 6
# Frozen per-domain fold counts (SCUT, HW5K) from the implementation plan.
# Total per fold: f0..f3=64, f4=64, f5=63.
FROZEN_FOLD_DOMAIN_COUNTS: list[tuple[int, int]] = [
    (22, 42),
    (22, 42),
    (22, 42),
    (22, 42),
    (21, 43),
    (21, 42),
]


class FoldMaterializationError(RuntimeError):
    pass


# --- deterministic identity helpers -------------------------------------------


def canonical_identity(domain: str, basename: str) -> str:
    """Return `<domain>/train/<basename>` with lower-case domain and POSIX separators."""
    dom = domain.strip().lower()
    name = Path(basename).name.replace("\\", "/")
    return f"{dom}/train/{name}"


def digest(salt: str, identity: str) -> str:
    payload = (salt + "\0" + identity).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_sorted_lines(paths: list[Path]) -> str:
    return hashlib.sha256(("\n".join(sorted(str(p) for p in paths)) + "\n").encode()).hexdigest()


def rel_path_or_abs(path: Path) -> str:
    """Return `path` relative to cwd when inside it, else the absolute path.

    This keeps emitted manifest paths stable and deterministic for the normal
    repo use (cwd == repo root) while never crashing when the output directory
    lives outside the cwd (e.g. a temporary determinism re-check).
    """
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve() or Path.cwd() / "."))
    except ValueError:
        return str(path.resolve())


# --- manifest readers (identity only, never image bytes) -----------------------


def read_identity_lines(path: Path) -> set[str]:
    """Read a bare-filename manifest and return normalized lower-case filenames."""
    if not path.exists():
        return set()
    out: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        out.add(Path(value).name.lower())
    return out


def read_dev40_identities(path: Path) -> set[str]:
    """Read Dev40 identities from a CSV whose second column is the `file` identity."""
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("file", "")
            if value:
                out.add(Path(value).name.lower())
    return out


def strip_domain_prefix(name: str) -> str:
    """Return the numeric stem when a name is prefixed with a domain (e.g. `scut_123.jpg` -> `123.jpg`)."""
    lowered = name.lower()
    for dom in DOMAINS:
        prefix = f"{dom}_"
        if lowered.startswith(prefix):
            return lowered[len(prefix):]
    return lowered


# --- prohibited-surface identity sets (SCUT/HW5K families) ---------------------


def build_prohibited_surface_identities(
    inner_val15: Path,
    dev40_csv: Path,
    holdout40: Path,
    scut115: Path,
    hw5k_dev232: Path,
    hw5k_reserved_root: Path,
) -> dict[str, set[str]]:
    """Return `{surface: set_of_normalized_numeric_stems}` never touching image bytes.

    For merged-prefix comparisons we strip any `scut_`/`hw5k_` prefix so a bare
    prohibited stem (e.g. `150.jpg`) is comparable to an eligible `scut_150.jpg`
    (`strip_domain_prefix` -> `150.jpg`). Naming here expresses the SCUT/HW5K
    numbering; a prohibited SCUT stem is a SCUT page id and a prohibited HW5K
    stem is an HW5K page id. We keep the surface labels distinct.
    """

    def stems(lines: set[str]) -> set[str]:
        return {strip_domain_prefix(x) for x in lines if x}

    scut_barred: set[str] = set()
    hw5k_barred: set[str] = set()

    inner = stems(read_identity_lines(inner_val15))
    hold = stems(read_identity_lines(holdout40))
    dev = stems(read_dev40_identities(dev40_csv))
    scut115_s = stems(read_identity_lines(scut115))
    scut_barred |= inner | hold | dev | scut115_s

    dev232 = stems(read_identity_lines(hw5k_dev232))
    reserved = set()
    if hw5k_reserved_root.exists():
        reserved = stems(
            {p.name for p in hw5k_reserved_root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        )
    hw5k_barred |= dev232 | reserved

    return {
        "inner_val15": inner,
        "dev40": dev,
        "holdout40": hold,
        "scut115": scut115_s,
        "hw5k_dev232": dev232,
        "hw5k_reserved": reserved,
        "_scut_union": scut_barred,
        "_hw5k_union": hw5k_barred,
    }


# --- fold assignment -----------------------------------------------------------


def assign_folds(
    identities: list[dict[str, Any]],
    frozen_counts: list[tuple[int, int]],
) -> dict[int, list[dict[str, Any]]]:
    """Sort by (domain, digest, identity), then fill folds 0..5 to their frozen per-domain quotas.

    Deterministic sequential fill in fold order reproduces the frozen per-domain
    counts exactly: each fold is filled to its (scut, hw5k) quota from the sorted
    per-domain lists. The leftover of a partial final fold is an error.
    """
    by_domain: dict[str, list[dict[str, Any]]] = {d: [] for d in DOMAINS}
    for item in identities:
        by_domain[item["domain"]].append(item)

    slots: dict[int, list[dict[str, Any]]] = {f: [] for f in range(FOLD_COUNT)}
    for f in range(FOLD_COUNT):
        scut_qty, hw5k_qty = frozen_counts[f]
        for (domain, qty) in ((DOMAINS[0], scut_qty), (DOMAINS[1], hw5k_qty)):
            domain_sorted = sorted(by_domain[domain], key=lambda i: (i["digest"], i["identity"]))
            taken = domain_sorted[:qty]
            slots[f].extend(taken)
            by_domain[domain] = domain_sorted[qty:]

    for domain in DOMAINS:
        if by_domain[domain]:
            raise FoldMaterializationError(
                f"unassigned {domain} identities after fold fill: {len(by_domain[domain])}"
            )
    return slots


# --- master/plan helpers -------------------------------------------------------


def load_plan(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FoldMaterializationError(f"implementation plan not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FoldMaterializationError(f"plan is not a JSON object: {path}")
    return value


def plan_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --- main ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--salt", default=SALT)
    parser.add_argument("--fold-count", type=int, default=FOLD_COUNT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inner-val15", type=Path, default=Path("hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt"))
    parser.add_argument("--dev40-csv", type=Path, default=Path("outputs/source-edge-primary-edit-dev40-gate-20260816-v1/dev40_review_rows.csv"))
    parser.add_argument("--holdout40", type=Path, default=Path("hardcase_lists/scut_val_holdout_40.txt"))
    parser.add_argument("--scut115", type=Path, default=Path("docs/scut-test115-relative.txt"))
    parser.add_argument("--hw5k-dev232", type=Path, default=Path("hardcase_lists/hw5k_dev_232_20260729.txt"))
    parser.add_argument("--hw5k-reserved-root", type=Path, default=Path("data/hw5k/reserved"))
    parser.add_argument("--plan", type=Path, default=Path("docs/plans/2026-08-16-spatial-continuous-reconstruction-mixture-implementation.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fold_count != FOLD_COUNT:
        raise FoldMaterializationError(f"fold-count must be {FOLD_COUNT}, got {args.fold_count}")

    if len(FROZEN_FOLD_DOMAIN_COUNTS) != FOLD_COUNT:
        raise FoldMaterializationError("internal frozen-count table length mismatch")

    source_manifest = args.source_manifest.resolve()
    if not source_manifest.exists():
        raise FoldMaterializationError(f"source manifest not found: {source_manifest}")
    manifest_sha = sha256_file(source_manifest)

    data_root = args.data_root.resolve()
    images_dir = data_root / "all_images"
    labels_dir = data_root / "all_labels"
    for d in (images_dir, labels_dir):
        if not d.is_dir():
            raise FoldMaterializationError(f"missing data dir: {d}")

    # 1. read the source manifest entries (full prefixed basenames)
    entries: list[str] = []
    for raw in source_manifest.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        entries.append(value)
    if len(entries) != 383:
        # Do not hard-fail on count here; identity will be validated by the pool.
        raise FoldMaterializationError(
            f"source manifest must contain 383 identities, found {len(entries)}"
        )

    identities: list[dict[str, Any]] = []
    for entry in entries:
        lowered = entry.lower()
        domain: str | None = None
        for d in DOMAINS:
            if lowered.startswith(f"{d}_"):
                domain = d
                break
        if domain is None:
            raise FoldMaterializationError(f"entry lacks a domain prefix: {entry}")

        src = images_dir / lowered
        tgt = labels_dir / lowered
        if not src.is_file() or not tgt.is_file():
            raise FoldMaterializationError(
                f"missing source/target pair for {entry}: image={src.exists()} target={tgt.exists()}"
            )

        identity = canonical_identity(domain, lowered)
        identities.append(
            {
                "domain": domain,
                "basename": lowered,
                "identity": identity,
                "digest": digest(args.salt, identity),
                "source_path": rel_path_or_abs(src),
                "target_path": rel_path_or_abs(tgt),
                "source_sha256": sha256_file(src),
                "target_sha256": sha256_file(tgt),
            }
        )

    # pool validation against frozen totals
    pool = {d: 0 for d in DOMAINS}
    for item in identities:
        pool[item["domain"]] += 1
    if pool != {"scut": 130, "hw5k": 253}:
        raise FoldMaterializationError(
            f"eligible pool count mismatch, expected scut=130,hw5k=253 got {pool}"
        )
    if pool["scut"] != 130 or pool["hw5k"] != 253 or sum(pool.values()) != 383:
        raise FoldMaterializationError("eligible pool does not match frozen 383 split")

    # 2. verification against prohibited surfaces (identity only)
    prohibited = build_prohibited_surface_identities(
        args.inner_val15,
        args.dev40_csv,
        args.holdout40,
        args.scut115,
        args.hw5k_dev232,
        args.hw5k_reserved_root,
    )
    scut_barred = prohibited["_scut_union"]
    hw5k_barred = prohibited["_hw5k_union"]

    overlaps: list[str] = []
    for item in identities:
        stem = strip_domain_prefix(item["basename"])
        if item["domain"] == "scut" and stem in scut_barred:
            overlaps.append(f"{item['identity']} clashes SCUT surface {sorted(k for k,v in prohibited.items() if stem in v)}")
        if item["domain"] == "hw5k" and stem in hw5k_barred:
            overlaps.append(f"{item['identity']} clashes HW5K surface {sorted(k for k,v in prohibited.items() if stem in v)}")
    if overlaps:
        raise FoldMaterializationError("Phase 0 pool overlaps a prohibited surface:\n" + "\n".join(overlaps))

    # 3/4/5/6. split by domain, sort by digest then identity, fill folds to quota
    slots = assign_folds(identities, FROZEN_FOLD_DOMAIN_COUNTS)

    # 7. emit immutable manifests + master JSON
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_manifests: dict[str, Any] = {}
    for f in range(FOLD_COUNT):
        members = sorted(slots[f], key=lambda i: (i["domain"], i["digest"], i["identity"]))
        # .txt: one canonical identity per line
        lines_txt = "\n".join(m["identity"] for m in members) + "\n"
        txt_path = out_dir / f"fold{f}.txt"
        txt_path.write_text(lines_txt, encoding="utf-8")

        # .json: full metadata
        detail = {
            "fold": f,
            "scut_count": sum(1 for m in members if m["domain"] == "scut"),
            "hw5k_count": sum(1 for m in members if m["domain"] == "hw5k"),
            "total": len(members),
            "identities": [m["identity"] for m in members],
            "members": members,
        }
        json_path = out_dir / f"fold{f}.json"
        json_path.write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")

        fold_manifests[str(f)] = {
            "txt": rel_path_or_abs(txt_path),
            "json": rel_path_or_abs(json_path),
            "txt_sha256": sha256_file(txt_path),
            "json_sha256": sha256_file(json_path),
            "scut_count": detail["scut_count"],
            "hw5k_count": detail["hw5k_count"],
            "total": detail["total"],
        }

    all_identities = [i["identity"] for i in identities]
    overall = {
        "schema_version": 1,
        "program": "spatial-mixture-phase0-v1",
        "salt": args.salt,
        "fold_count": FOLD_COUNT,
        "frozen_fold_domain_counts": FROZEN_FOLD_DOMAIN_COUNTS,
        "source_manifest": {
            "path": rel_path_or_abs(source_manifest),
            "sha256": manifest_sha,
        },
        "pool": {
            "scut": pool["scut"],
            "hw5k": pool["hw5k"],
            "total": 383,
        },
        "identity_sha256": hashlib.sha256("\n".join(sorted(all_identities)).encode()).hexdigest(),
        "master_sha256": None,  # assigned after writing; placeholder replaced below
        "folds": fold_manifests,
        "prohibited_surfaces_checked": sorted(k for k in prohibited if not k.startswith("_")),
        "plan_sha256": hashlib.sha256(plan_text(args.plan.resolve()).encode()).hexdigest(),
    }
    master_path = out_dir / "master.json"
    master_path.write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")

    # Re-read and store file-level master sha (self-referential after write).
    master_sha = sha256_file(master_path)
    overall["master_sha256"] = master_sha
    master_path.write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")

    print(f"source_manifest={source_manifest}")
    print(f"pool=scut:{pool['scut']} hw5k:{pool['hw5k']} total:{sum(pool.values())}")
    for f in range(FOLD_COUNT):
        print(
            f"fold{f}: scut={fold_manifests[str(f)]['scut_count']} "
            f"hw5k={fold_manifests[str(f)]['hw5k_count']} "
            f"total={fold_manifests[str(f)]['total']}"
        )
    print(f"output_dir={out_dir}")
    print(f"master_sha256={master_sha}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FoldMaterializationError as exc:
        print(f"materialization error: {exc}", file=sys.stderr)
        raise SystemExit(2)
