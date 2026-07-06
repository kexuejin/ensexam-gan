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
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


DEFAULT_REPO_ID = "ynyg/ExamInk-Seg"
DEFAULT_REVISION = "main"


def request_json_with_headers(url: str, timeout: int, retries: int) -> tuple[object, object]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8")), response.headers
        except Exception as exc:  # noqa: BLE001 - report final URL with original failure.
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"failed to read JSON {url} after {retries} attempts: {last_error!r}")


def next_link(headers: object) -> str | None:
    link_value = headers.get("Link", "")
    for item in link_value.split(","):
        if 'rel="next"' not in item:
            continue
        match = re.search(r"<([^>]+)>", item)
        if match:
            return match.group(1)
    return None


def list_hf_files(repo_id: str, revision: str, directory: str, timeout: int, retries: int) -> list[str]:
    url = (
        f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}/{directory}"
        "?limit=1000"
    )
    paths: list[str] = []
    while url:
        rows, headers = request_json_with_headers(url, timeout, retries)
        if not isinstance(rows, list):
            raise RuntimeError(f"unexpected tree response for {directory}: {type(rows).__name__}")
        paths.extend(
            str(row["path"])
            for row in rows
            if isinstance(row, dict) and row.get("type") == "file" and row.get("path")
        )
        url = next_link(headers)
    if not paths:
        raise RuntimeError(f"no files found under {directory}")
    return paths


def target_split_name(source_split: str) -> str:
    return "test" if source_split == "val" else source_split


def hf_resolve_url(repo_id: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"


def natural_key(path_value: str) -> tuple[int, str]:
    stem = Path(path_value).stem
    if stem.isdigit():
        return int(stem), Path(path_value).name
    return 10**12, Path(path_value).name


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
                with tmp.open("wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            tmp.replace(destination)
            return
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    curl = shutil.which("curl")
    if curl:
        try:
            subprocess.run(
                [
                    curl,
                    "-L",
                    "--fail",
                    "--connect-timeout",
                    str(timeout),
                    "--max-time",
                    str(timeout),
                    "-o",
                    str(tmp),
                    url,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            tmp.replace(destination)
            return
        except Exception as exc:  # noqa: BLE001 - preserve urllib error in final message too.
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"failed to download {url} after {retries} urllib attempts "
                f"and curl fallback: urllib={last_error!r} curl={exc!r}"
            ) from exc
    raise RuntimeError(f"failed to download {url} after {retries} attempts: {last_error!r}")


def convert_split(
    *,
    repo_id: str,
    revision: str,
    output_root: Path,
    source_split: str,
    limit: int | None,
    overwrite: bool,
    timeout: int,
    retries: int,
) -> int:
    if limit == 0:
        print(f"{source_split}: skipped (limit=0)", flush=True)
        return 0
    source_paths = sorted(
        list_hf_files(repo_id, revision, f"data/{source_split}/source", timeout, retries),
        key=natural_key,
    )
    target_by_stem = {
        Path(path).stem: path
        for path in list_hf_files(repo_id, revision, f"data/{source_split}/target", timeout, retries)
    }
    mask_by_stem = {
        Path(path).stem: path
        for path in list_hf_files(repo_id, revision, f"data/{source_split}/mask", timeout, retries)
    }
    rows: list[tuple[str, str, str]] = []
    for source_path in source_paths:
        stem = Path(source_path).stem
        target_path = target_by_stem.get(stem)
        mask_path = mask_by_stem.get(stem)
        if target_path is None or mask_path is None:
            continue
        rows.append((source_path, target_path, mask_path))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"no matched source/target/mask triplets found for split {source_split}")

    out_split = target_split_name(source_split)
    copied = 0
    for source_path, target_path, mask_path in rows:
        source_name = Path(source_path).name
        mask_name = Path(mask_path).name
        target_out_name = source_name
        source_url = hf_resolve_url(repo_id, revision, source_path)
        target_url = hf_resolve_url(repo_id, revision, target_path)
        mask_url = hf_resolve_url(repo_id, revision, mask_path)

        download_url(source_url, output_root / out_split / "all_images" / source_name, overwrite, timeout, retries)
        download_url(target_url, output_root / out_split / "all_labels" / target_out_name, overwrite, timeout, retries)
        download_url(mask_url, output_root / out_split / "all_masks" / mask_name, overwrite, timeout, retries)
        copied += 1
        if len(rows) <= 10 or copied % 10 == 0 or copied == len(rows):
            print(f"{source_split}: downloaded {copied}/{len(rows)}", flush=True)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--download-timeout", type=int, default=30)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    train_count = convert_split(
        repo_id=args.repo_id,
        revision=args.revision,
        output_root=output_root,
        source_split="train",
        limit=args.limit_train,
        overwrite=args.overwrite,
        timeout=args.download_timeout,
        retries=args.download_retries,
    )
    val_count = convert_split(
        repo_id=args.repo_id,
        revision=args.revision,
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
