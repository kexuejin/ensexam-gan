#!/usr/bin/env python3
"""Canonicalize frozen cache metrics without changing unrelated CSV bytes."""

from __future__ import annotations

import csv
import hashlib
import io
import os
from pathlib import Path


class MetricsCanonicalizationError(RuntimeError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_rows(payload: bytes, metrics_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MetricsCanonicalizationError(
            f"metrics CSV is not UTF-8: {metrics_path}"
        ) from error
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or fieldnames.count("pred_path") != 1:
        raise MetricsCanonicalizationError(
            f"metrics CSV must contain exactly one pred_path field: {metrics_path}"
        )
    if any(
        None in row
        or set(row) != set(fieldnames)
        or any(value is None for value in row.values())
        for row in rows
    ):
        raise MetricsCanonicalizationError(
            f"metrics CSV row shape changed: {metrics_path}"
        )
    return fieldnames, rows


def _validate_root_locations(
    *,
    payload: bytes,
    metrics_path: Path,
    current_repository_root: str,
    frozen_historical_repository_root: str,
    expected_data_rows: int,
    expected_replacement_count: int,
) -> tuple[str, list[str], list[dict[str, str]]]:
    fieldnames, rows = _read_rows(payload, metrics_path)
    if len(rows) != expected_data_rows:
        raise MetricsCanonicalizationError(
            f"metrics row count changed: expected {expected_data_rows}, got {len(rows)}"
        )
    text = payload.decode("utf-8")
    actual_replacements = text.count(current_repository_root)
    if actual_replacements != expected_replacement_count:
        raise MetricsCanonicalizationError(
            "metrics repository-root replacement count changed: "
            f"expected {expected_replacement_count}, got {actual_replacements}"
        )
    if frozen_historical_repository_root in text:
        raise MetricsCanonicalizationError(
            "metrics already contain the frozen historical repository root"
        )
    for index, row in enumerate(rows, start=1):
        pred_path = row["pred_path"]
        if pred_path.count(current_repository_root) != 1:
            raise MetricsCanonicalizationError(
                f"metrics row {index} pred_path root count changed"
            )
        for field, value in row.items():
            if field != "pred_path" and current_repository_root in value:
                raise MetricsCanonicalizationError(
                    f"metrics row {index} contains repository root outside pred_path"
                )
            if frozen_historical_repository_root in value:
                raise MetricsCanonicalizationError(
                    f"metrics row {index} already contains historical root"
                )
    return text, fieldnames, rows


def canonicalize_repository_root(
    metrics_path: Path,
    *,
    current_repository_root: str,
    frozen_historical_repository_root: str,
    expected_data_rows: int,
    expected_replacement_count: int,
    expected_metrics_sha256_after: str,
    expected_metrics_sha256_before: str | None = None,
) -> dict[str, int | str]:
    if (
        not metrics_path.is_file()
        or metrics_path.is_symlink()
        or not current_repository_root
        or not frozen_historical_repository_root
        or current_repository_root == frozen_historical_repository_root
    ):
        raise MetricsCanonicalizationError(
            f"metrics canonicalization precondition failed: {metrics_path}"
        )
    before = metrics_path.read_bytes()
    before_sha256 = sha256_bytes(before)
    if (
        expected_metrics_sha256_before is not None
        and before_sha256 != expected_metrics_sha256_before
    ):
        raise MetricsCanonicalizationError(
            "metrics source sha256 changed: "
            f"expected {expected_metrics_sha256_before}, got {before_sha256}"
        )
    text, fieldnames, rows = _validate_root_locations(
        payload=before,
        metrics_path=metrics_path,
        current_repository_root=current_repository_root,
        frozen_historical_repository_root=frozen_historical_repository_root,
        expected_data_rows=expected_data_rows,
        expected_replacement_count=expected_replacement_count,
    )
    candidate = text.replace(
        current_repository_root, frozen_historical_repository_root
    ).encode("utf-8")
    candidate_sha256 = sha256_bytes(candidate)
    if candidate_sha256 != expected_metrics_sha256_after:
        raise MetricsCanonicalizationError(
            "canonical metrics sha256 changed: "
            f"expected {expected_metrics_sha256_after}, got {candidate_sha256}"
        )
    candidate_text = candidate.decode("utf-8")
    if (
        current_repository_root in candidate_text
        or candidate_text.count(frozen_historical_repository_root)
        != expected_replacement_count
    ):
        raise MetricsCanonicalizationError(
            "canonical metrics repository-root postcondition failed"
        )
    candidate_fields, candidate_rows = _read_rows(candidate, metrics_path)
    if candidate_fields != fieldnames or len(candidate_rows) != len(rows):
        raise MetricsCanonicalizationError("canonical metrics CSV shape changed")
    for before_row, after_row in zip(rows, candidate_rows):
        for field in fieldnames:
            expected = before_row[field]
            if field == "pred_path":
                expected = expected.replace(
                    current_repository_root, frozen_historical_repository_root
                )
            if after_row[field] != expected:
                raise MetricsCanonicalizationError(
                    f"canonical metrics changed field outside contract: {field}"
                )

    candidate_path = metrics_path.with_name(f".{metrics_path.name}.canonicalizing")
    if candidate_path.exists() or candidate_path.is_symlink():
        raise MetricsCanonicalizationError(
            f"stale canonical metrics candidate exists: {candidate_path}"
        )
    try:
        with candidate_path.open("xb") as handle:
            handle.write(candidate)
            handle.flush()
            os.fsync(handle.fileno())
        candidate_path.replace(metrics_path)
    except OSError:
        candidate_path.unlink(missing_ok=True)
        raise
    return {
        "data_rows": len(rows),
        "metrics_sha256_after": candidate_sha256,
        "metrics_sha256_before": before_sha256,
        "replacement_count": expected_replacement_count,
    }
