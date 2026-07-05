#!/usr/bin/env python3
"""Download/convert ExamInk-Seg into EnsExamRealDataset-compatible folders.

Output layout:
  output_root/
    train/all_images/*.jpg
    train/all_labels/*.jpg
    train/all_masks/*.png
    test/all_images/*.jpg
    test/all_labels/*.jpg
    test/all_masks/*.png
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = "https://huggingface.co/datasets/ynyg/ExamInk-Seg/resolve/main/data"


def read_metadata(base_url: str, split: str) -> list[dict[str, str]]:
    url = f"{base_url.rstrip('/')}/{split}/metadata.jsonl"
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def target_split_name(source_split: str) -> str:
    return "test" if source_split == "val" else source_split


def output_name(path_value: str) -> str:
    return Path(path_value).name


def download_url(url: str, destination: Path, overwrite: bool, timeout: int, retries: int) -> None:
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                tmp.write_bytes(response.read())
            tmp.replace(destination)
            return
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"failed to download {url} after {retries} attempts: {last_error!r}")


def convert_split(
    *,
    base_url: str,
    output_root: Path,
    source_split: str,
    limit: int | None,
    overwrite: bool,
    timeout: int,
    retries: int,
) -> int:
    rows = read_metadata(base_url, source_split)
    if limit is not None:
        rows = rows[:limit]

    out_split = target_split_name(source_split)
    copied = 0
    for row in rows:
        source_name = output_name(row["file_name"])
        target_name = output_name(row["target_file_name"])
        mask_name = output_name(row["mask_file_name"])

        # Keep source/target basenames aligned for EnsExamRealDataset.
        target_out_name = source_name
        source_url = f"{base_url.rstrip('/')}/{source_split}/{row['file_name']}"
        target_url = f"{base_url.rstrip('/')}/{source_split}/{row['target_file_name']}"
        mask_url = f"{base_url.rstrip('/')}/{source_split}/{row['mask_file_name']}"

        download_url(source_url, output_root / out_split / "all_images" / source_name, overwrite, timeout, retries)
        download_url(target_url, output_root / out_split / "all_labels" / target_out_name, overwrite, timeout, retries)
        download_url(mask_url, output_root / out_split / "all_masks" / mask_name, overwrite, timeout, retries)
        copied += 1
        if copied % 10 == 0 or copied == len(rows):
            print(f"{source_split}: downloaded {copied}/{len(rows)}", flush=True)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--download-timeout", type=int, default=30)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    train_count = convert_split(
        base_url=args.base_url,
        output_root=output_root,
        source_split="train",
        limit=args.limit_train,
        overwrite=args.overwrite,
        timeout=args.download_timeout,
        retries=args.download_retries,
    )
    val_count = convert_split(
        base_url=args.base_url,
        output_root=output_root,
        source_split="val",
        limit=args.limit_val,
        overwrite=args.overwrite,
        timeout=args.download_timeout,
        retries=args.download_retries,
    )
    readme = output_root / "README.examink-seg.txt"
    readme.write_text(
        "Converted from ynyg/ExamInk-Seg for EnsExamRealDataset.\n"
        "License tag reported by Hugging Face: apache-2.0.\n"
        "Original split mapping: train -> train, val -> test.\n"
        "Explicit masks are stored under all_masks and loaded preferentially.\n",
        encoding="utf-8",
    )
    print(f"output_root={output_root}")
    print(f"train={train_count}")
    print(f"test={val_count}")


if __name__ == "__main__":
    main()
