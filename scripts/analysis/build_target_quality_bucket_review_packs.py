#!/usr/bin/env python3
"""Build review packs for selected target-quality triage buckets.

This wraps the existing product-quality page and crop pack builders so the
borderline review loop is reproducible from a triage CSV instead of relying on
inline CSV filters.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


DEFAULT_BUCKETS = ("auto_win_candidate", "ratio_noise_review")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage-csv", required=True, help="Output CSV from triage_target_quality_borderline.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--bucket",
        action="append",
        default=[],
        help="Triage bucket to include. May be repeated. Defaults to auto_win_candidate and ratio_noise_review.",
    )
    parser.add_argument("--name", default="high_priority")
    parser.add_argument("--thumb-width", type=int, default=520)
    parser.add_argument("--thumb-height", type=int, default=380)
    parser.add_argument("--max-contact-rows", type=int, default=120)
    parser.add_argument("--max-crops-per-row", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=340)
    parser.add_argument("--thumb-size", type=int, default=300)
    parser.add_argument("--max-contact-crops", type=int, default=120)
    parser.add_argument("--skip-combined", action="store_true")
    parser.add_argument("--skip-per-bucket", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_builder(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def build_pack(
    review_csv: Path,
    output_dir: Path,
    page_script: Path,
    crop_script: Path,
    args: argparse.Namespace,
) -> None:
    run_builder([
        sys.executable,
        str(page_script),
        "--review-csv",
        str(review_csv),
        "--output-dir",
        str(output_dir / "page_pack"),
        "--thumb-width",
        str(args.thumb_width),
        "--thumb-height",
        str(args.thumb_height),
        "--max-contact-rows",
        str(args.max_contact_rows),
    ])
    run_builder([
        sys.executable,
        str(crop_script),
        "--review-csv",
        str(review_csv),
        "--output-dir",
        str(output_dir / "crop_pack"),
        "--max-crops-per-row",
        str(args.max_crops_per_row),
        "--max-contact-crops",
        str(args.max_contact_crops),
        "--include-target-residual",
        "--crop-size",
        str(args.crop_size),
        "--thumb-size",
        str(args.thumb_size),
    ])


def main() -> None:
    args = parse_args()
    buckets = tuple(args.bucket) if args.bucket else DEFAULT_BUCKETS
    rows = read_rows(Path(args.triage_csv))
    selected = [row for row in rows if row.get("triage_bucket") in buckets]
    if not selected:
        raise ValueError(f"No rows matched buckets: {', '.join(buckets)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    page_script = script_dir / "build_product_quality_review_pack.py"
    crop_script = script_dir / "build_product_quality_crop_review_pack.py"

    combined_csv = output_dir / f"{args.name}_{len(selected)}_review_rows.csv"
    write_csv(combined_csv, selected)
    print(f"selected_rows={len(selected)}", flush=True)
    print(f"combined_csv={combined_csv}", flush=True)

    if not args.skip_combined:
        build_pack(
            combined_csv,
            output_dir / f"{args.name}_{len(selected)}",
            page_script,
            crop_script,
            args,
        )

    if args.skip_per_bucket:
        return

    bucket_csv_dir = output_dir / "bucket_csvs"
    for bucket in buckets:
        bucket_rows = [row for row in selected if row.get("triage_bucket") == bucket]
        if not bucket_rows:
            continue
        bucket_csv = bucket_csv_dir / f"{bucket}.csv"
        write_csv(bucket_csv, bucket_rows)
        print(f"bucket={bucket} rows={len(bucket_rows)} csv={bucket_csv}", flush=True)
        build_pack(
            bucket_csv,
            output_dir / bucket,
            page_script,
            crop_script,
            args,
        )


if __name__ == "__main__":
    main()
