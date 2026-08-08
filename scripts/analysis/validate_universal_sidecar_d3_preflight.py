#!/usr/bin/env python3
"""Fail-closed preflight for the preregistered universal-sidecar D3 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import EnsExamRealDataset  # noqa: E402
from losses.losses import EnsExamLoss  # noqa: E402
from train import (  # noqa: E402
    validate_cached_baseline_tail_config,
    validate_universal_sidecar_config,
)


D2_CONFIG = Path(
    "configs/local/"
    "config.local-universal-sidecar-d2-d1-mixed-scut130-hw5k260-step80-mps.yaml"
)
D3_CONFIG = Path(
    "configs/local/"
    "config.local-universal-sidecar-d3-d1-mixed-scut130-hw5k260-step80-"
    "baseline-tail-mps.yaml"
)
EXPECTED_MANIFEST_SHA256 = (
    "92c78488cbc59e5b380fa0496f395dcfd69624b8aff58186e1559bcc66bfa21b"
)
EXPECTED_ROWS_SHA256 = (
    "592f6383164af92ec10008881a8b160cee6828132831ac66c4d3316d2742545a"
)
EXPECTED_TRAIN_MANIFEST_SHA256 = (
    "0385fb96aa7aee1812b95b90acd4198e2af39e96c895a7cd8cfb2681258470ca"
)
EXPECTED_BASELINE_CONFIG_SHA256 = (
    "8b47e383eb46c75171eec3b475e04a037f7afd9dc4bf51316120b197b5a8b42e"
)
EXPECTED_BASELINE_WEIGHTS_SHA256 = (
    "e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae"
)
EXPECTED_COUNT = 383
EXPECTED_PROTOCOL = "train_only_cached_baseline_tail_support"
EXPECTED_DIFFERENCES = {
    "loss.lambda_cached_baseline_tail_nonregress": 0.20,
    "loss.cached_baseline_tail_residual_alpha": 0.25,
    "loss.cached_baseline_tail_overerase_alpha": 1.0,
    "loss.cached_baseline_tail_fraction": 0.10,
    "loss.cached_baseline_tail_residual_threshold_px": 12.0,
    "loss.cached_baseline_tail_edit_threshold_px": 12.0,
    "loss.cached_baseline_tail_event_temperature_px": 0.25,
    "data.cached_baseline_tail_dir": (
        "./artifacts/caches/"
        "baseline-tail-universal-sidecar-d3-mixed-scut130-hw5k260-20260807"
    ),
    "train.save_dir": (
        "./artifacts/trials/"
        "universal-sidecar-d3-d1-mixed-scut130-hw5k260-step80-"
        "baseline-tail-20260808"
    ),
}
RUNTIME_LOSS_KEYS = {
    key.split(".", 1)[1]
    for key in EXPECTED_DIFFERENCES
    if key.startswith("loss.")
}


class PreflightError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PreflightError(f"missing config: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreflightError(f"config must be a mapping: {path}")
    return value


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    output: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        output.update(flatten(child, path))
    return output


def assert_exact_config_delta(d2: dict[str, Any], d3: dict[str, Any]) -> None:
    left = flatten(d2)
    right = flatten(d3)
    changed = {
        key: right.get(key)
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }
    if changed != EXPECTED_DIFFERENCES:
        raise PreflightError(
            "D3/D2 semantic differences do not match preregistration: "
            f"{changed}"
        )


def assert_runtime_support(d3: dict[str, Any]) -> None:
    missing_defaults = sorted(RUNTIME_LOSS_KEYS - set(EnsExamLoss._DEFAULTS))
    if missing_defaults:
        raise PreflightError(f"runtime loss defaults missing: {missing_defaults}")
    dataset_parameters = inspect.signature(EnsExamRealDataset.__init__).parameters
    if "cached_baseline_tail_dir" not in dataset_parameters:
        raise PreflightError("dataset runtime lacks cached_baseline_tail_dir")
    forward_parameters = inspect.signature(EnsExamLoss.forward).parameters
    if "cached_baseline_tail_gt" not in forward_parameters:
        raise PreflightError("loss forward lacks cached_baseline_tail_gt")
    if not callable(getattr(EnsExamLoss, "cached_baseline_tail_nonregress_loss", None)):
        raise PreflightError("cached baseline-tail loss implementation is missing")
    try:
        validate_cached_baseline_tail_config(d3)
        validate_universal_sidecar_config(d3)
    except (TypeError, ValueError) as exc:
        raise PreflightError(f"runtime config validation failed: {exc}") from exc


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def read_name_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise PreflightError(f"missing sample manifest: {path}")
    names = [
        Path(line.strip()).name
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(names) != len(set(names)):
        raise PreflightError(f"duplicate names in sample manifest: {path}")
    return names


def assert_gate_isolation(d3: dict[str, Any]) -> None:
    evaluation = d3.get("evaluation", {})
    if evaluation.get("skip_validation") is not True:
        raise PreflightError("D3 must keep validation disabled during training")
    if evaluation.get("skip_final_test") is not True:
        raise PreflightError("D3 must keep final test disabled during training")
    if evaluation.get("standalone_test_mode") not in {None, "none"}:
        raise PreflightError("standalone test gate is enabled")
    if evaluation.get("final_test_mode") not in {None, "none"}:
        raise PreflightError("final test gate is enabled")
    flattened = flatten(d3)
    forbidden_tokens = ("reserved_blind", "scut115", "holdout40", "promotion")
    enabled_forbidden = [
        key
        for key, value in flattened.items()
        if any(token in key.lower() for token in forbidden_tokens)
        and value not in {None, False, "", "none", 0}
    ]
    if enabled_forbidden:
        raise PreflightError(f"later gates enabled in D3: {enabled_forbidden}")


def assert_output_dir_clean(repo_root: Path, d3: dict[str, Any]) -> None:
    output_dir = resolve_repo_path(repo_root, d3["train"]["save_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PreflightError(f"D3 save_dir is not new or empty: {output_dir}")


def assert_cache(
    repo_root: Path,
    d3: dict[str, Any],
    *,
    expected_manifest_sha256: str,
    expected_rows_sha256: str,
    expected_train_manifest_sha256: str,
    expected_baseline_config_sha256: str,
    expected_baseline_weights_sha256: str,
    expected_count: int,
) -> dict[str, Any]:
    cache_dir = resolve_repo_path(
        repo_root, d3["data"]["cached_baseline_tail_dir"]
    )
    manifest_path = cache_dir / "manifest.json"
    rows_path = cache_dir / "cache_rows.csv"
    if not manifest_path.is_file() or not rows_path.is_file():
        raise PreflightError(f"cache manifest or rows missing: {cache_dir}")
    manifest_sha = sha256_file(manifest_path)
    rows_sha = sha256_file(rows_path)
    if manifest_sha != expected_manifest_sha256:
        raise PreflightError(f"cache manifest hash mismatch: {manifest_sha}")
    if rows_sha != expected_rows_sha256:
        raise PreflightError(f"cache rows hash mismatch: {rows_sha}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "protocol": EXPECTED_PROTOCOL,
        "sample_count": expected_count,
        "rows_csv_sha256": expected_rows_sha256,
        "train_file_list_sha256": expected_train_manifest_sha256,
        "primary_config_sha256": expected_baseline_config_sha256,
        "primary_weights_sha256": expected_baseline_weights_sha256,
    }
    mismatches = {
        key: manifest.get(key)
        for key, expected in expected_fields.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise PreflightError(f"cache manifest field mismatch: {mismatches}")

    with rows_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row_names = [Path(row.get("file", "")).name for row in rows]
    if len(rows) != expected_count or len(row_names) != len(set(row_names)):
        raise PreflightError("cache row count or uniqueness mismatch")
    residual_names = {
        path.name for path in (cache_dir / "residual_safe").glob("*.png")
    }
    outside_names = {
        path.name for path in (cache_dir / "outside_safe").glob("*.png")
    }
    expected_mask_names = {f"{Path(name).stem}.png" for name in row_names}
    if residual_names != expected_mask_names or outside_names != expected_mask_names:
        raise PreflightError("cache mask names/counts do not match rows")
    required_hash_columns = (
        "source_path",
        "source_sha256",
        "label_path",
        "label_sha256",
        "residual_safe_sha256",
        "outside_safe_sha256",
    )
    for row in rows:
        missing_columns = [key for key in required_hash_columns if not row.get(key)]
        if missing_columns:
            raise PreflightError(
                f"cache row {row.get('file')} lacks hash evidence: {missing_columns}"
            )
        source_path = resolve_repo_path(repo_root, row["source_path"])
        label_path = resolve_repo_path(repo_root, row["label_path"])
        residual_path = cache_dir / "residual_safe" / f"{Path(row['file']).stem}.png"
        outside_path = cache_dir / "outside_safe" / f"{Path(row['file']).stem}.png"
        evidence = (
            ("source", source_path, row["source_sha256"]),
            ("label", label_path, row["label_sha256"]),
            ("residual_safe", residual_path, row["residual_safe_sha256"]),
            ("outside_safe", outside_path, row["outside_safe_sha256"]),
        )
        for label, path, expected_hash in evidence:
            if not path.is_file():
                raise PreflightError(
                    f"cache row {row['file']} missing {label}: {path}"
                )
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise PreflightError(
                    f"cache row {row['file']} {label} hash mismatch: {actual_hash}"
                )

    train_manifest = resolve_repo_path(repo_root, d3["data"]["train_file_list"])
    if sha256_file(train_manifest) != expected_train_manifest_sha256:
        raise PreflightError("registered train manifest hash mismatch")
    train_names = read_name_manifest(train_manifest)
    if row_names != train_names:
        raise PreflightError("cache row order does not match train manifest")
    inner_val = repo_root / "hardcase_lists/scut_train_hard_proxy_inner_val_15_20260726.txt"
    inner_names = set(read_name_manifest(inner_val))
    overlap = sorted(set(row_names) & inner_names)
    if overlap:
        raise PreflightError(f"cache overlaps inner-val15: {overlap[:5]}")

    baseline_config = repo_root / "artifacts/current-primary/config.yaml"
    baseline_weights = (
        repo_root / "artifacts/current-primary/micro_region_probe_step0001.pth"
    )
    if not baseline_config.is_file() or not baseline_weights.is_file():
        raise PreflightError("frozen current-primary files are missing")
    if sha256_file(baseline_config) != expected_baseline_config_sha256:
        raise PreflightError("current-primary config hash mismatch")
    if sha256_file(baseline_weights) != expected_baseline_weights_sha256:
        raise PreflightError("current-primary weights hash mismatch")
    return {
        "cache_dir": str(cache_dir),
        "manifest_sha256": manifest_sha,
        "rows_sha256": rows_sha,
        "sample_count": len(rows),
        "inner_val_overlap": 0,
    }


def run_preflight(
    *,
    repo_root: Path = ROOT,
    d2_config: Path | None = None,
    d3_config: Path | None = None,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
    expected_rows_sha256: str = EXPECTED_ROWS_SHA256,
    expected_train_manifest_sha256: str = EXPECTED_TRAIN_MANIFEST_SHA256,
    expected_baseline_config_sha256: str = EXPECTED_BASELINE_CONFIG_SHA256,
    expected_baseline_weights_sha256: str = EXPECTED_BASELINE_WEIGHTS_SHA256,
    expected_count: int = EXPECTED_COUNT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    d2_path = d2_config or (repo_root / D2_CONFIG)
    d3_path = d3_config or (repo_root / D3_CONFIG)
    try:
        d2 = read_yaml(d2_path)
        d3 = read_yaml(d3_path)
        assert_exact_config_delta(d2, d3)
        assert_runtime_support(d3)
        assert_gate_isolation(d3)
        assert_output_dir_clean(repo_root, d3)
        cache = assert_cache(
            repo_root,
            d3,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_rows_sha256=expected_rows_sha256,
            expected_train_manifest_sha256=expected_train_manifest_sha256,
            expected_baseline_config_sha256=expected_baseline_config_sha256,
            expected_baseline_weights_sha256=expected_baseline_weights_sha256,
            expected_count=expected_count,
        )
    except (KeyError, OSError, PreflightError, TypeError, ValueError) as exc:
        return {
            "terminal": "PREREQUISITE_NEEDED",
            "runnable": False,
            "first_gate": "scut_inner_val15",
            "reason": str(exc),
        }
    return {
        "terminal": "PASS",
        "runnable": True,
        "first_gate": "scut_inner_val15",
        "config": str(d3_path),
        "cache": cache,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_preflight(repo_root=args.repo_root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if result["terminal"] == "PASS" else 2)


if __name__ == "__main__":
    main()
