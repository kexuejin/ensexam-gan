#!/usr/bin/env python3
"""Run explicit-domain dual-checkpoint primary inference as a research harness."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.path_utils import normalize_path  # noqa: E402


SCRIPT_PATH = Path(__file__).resolve()
PRIMARY_SCRIPT = ROOT / "scripts" / "infer" / "run_primary_full_page.py"
LOCK_PATH = ROOT / ".omx" / "state" / "explicit_domain_dual_checkpoint.lock"
FORBIDDEN_SAMPLE_COMPONENTS = {"label", "labels", "target", "targets", "all_labels"}
ALLOWED_DOMAINS = {"default", "unknown", "hw5k"}
DOMAIN_TO_BRANCH = {
    "default": "default",
    "unknown": "default",
    "hw5k": "hw5k",
}
BRANCH_ORDER = ("default", "hw5k")
PROTOCOL_VERSION = "2026-08-03-explicit-domain-dual-checkpoint-v1"
FIXED_INFERENCE_ARGS = {
    "page_overlap": 32,
    "batch_size": 8,
    "copy_input_outside_mask": "mb",
    "copy_mask_threshold_auto": "mb_cov8_step",
    "copy_mask_threshold": 70,
    "copy_mask_dilate": 0,
}
REQUIRED_METRICS_FIELDS = {
    "image_path",
    "image_sha256",
    "pred_path",
    "pred_sha256",
    "metrics_skipped",
    "primary_config_sha256",
    "primary_weights_sha256",
}
DEFAULT_CONFIG = ROOT / "artifacts" / "current-primary" / "config.yaml"
DEFAULT_WEIGHTS = ROOT / "artifacts" / "current-primary" / "micro_region_probe_step0001.pth"
DEFAULT_WEIGHTS_SHA256 = "e6acf784bf6737eccbd68438acdc566f62cab699a52e2e57a995e7ef08958bae"
HW5K_CONFIG = (
    ROOT
    / "artifacts"
    / "trials"
    / "hw5k-mixed-scut130-hw5k130-50pct-guard-jointtail-lite-step6400-respress-bs4-20260730"
    / "ensexam"
    / "20260801_183409"
    / "config.yaml"
)
HW5K_WEIGHTS = (
    ROOT
    / "artifacts"
    / "trials"
    / "hw5k-mixed-scut130-hw5k130-50pct-guard-jointtail-lite-step6400-respress-bs4-20260730"
    / "ensexam"
    / "20260801_183409"
    / "epoch_1.pth"
)
HW5K_WEIGHTS_SHA256 = "8da25117dd883f95059b6d7067e3dc3580da11339de365ef904f711db4a1f490"


@dataclass(frozen=True)
class ArtifactPair:
    branch: str
    config_path: Path
    config_sha256: str
    weights_path: Path
    weights_sha256: str
    expected_weights_sha256: str
    research_only: bool


@dataclass(frozen=True)
class ManifestRow:
    row_index: int
    input_image_path: str
    resolved_image_path: Path
    image_sha256: str
    caller_domain: str
    selected_branch: str
    prediction_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True, help="CSV with exact columns: image_path,domain")
    parser.add_argument("--output-dir", required=True, help="New directory for audit artifacts and merged predictions")
    parser.add_argument("--device", default="auto", help="Device forwarded to run_primary_full_page.py")
    parser.add_argument(
        "--default-config",
        default=str(DEFAULT_CONFIG),
        help="Current-primary config path",
    )
    parser.add_argument(
        "--default-weights",
        default=str(DEFAULT_WEIGHTS),
        help="Current-primary checkpoint path",
    )
    parser.add_argument(
        "--default-weights-sha256",
        default=DEFAULT_WEIGHTS_SHA256,
        help="Expected SHA-256 for the current-primary checkpoint",
    )
    parser.add_argument(
        "--hw5k-config",
        default=str(HW5K_CONFIG),
        help="Research-only HW5K config path",
    )
    parser.add_argument(
        "--hw5k-weights",
        default=str(HW5K_WEIGHTS),
        help="Research-only HW5K checkpoint path",
    )
    parser.add_argument(
        "--hw5k-weights-sha256",
        default=HW5K_WEIGHTS_SHA256,
        help="Expected SHA-256 for the research-only HW5K checkpoint",
    )
    parser.add_argument(
        "--ack-research-specialist",
        action="store_true",
        help="Required whenever any hw5k row is present",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prediction_name(image_path: Path) -> str:
    return f"{image_path.stem}.png"


def assert_source_image_path(path: Path) -> None:
    for candidate in (path, path.resolve()):
        components = {part.lower() for part in candidate.parts}
        if components & FORBIDDEN_SAMPLE_COMPONENTS:
            raise ValueError(f"manifest includes a target/label path: {path}")


def resolve_existing_file(path_text: str, *, base_dir: Path | None = None) -> Path:
    path = Path(path_text)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    path = Path(normalize_path(str(path)))
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def parse_manifest_rows(manifest_csv: Path) -> list[ManifestRow]:
    if not manifest_csv.is_file():
        raise FileNotFoundError(manifest_csv)
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest CSV is missing a header row")
        expected_fields = ["image_path", "domain"]
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f"manifest CSV must contain exactly {expected_fields}, got {reader.fieldnames}"
            )
        rows: list[ManifestRow] = []
        seen_paths: set[Path] = set()
        seen_prediction_names: set[str] = set()
        for row_index, raw_row in enumerate(reader, start=1):
            extra_values = raw_row.get(None)
            if extra_values:
                raise ValueError(f"manifest row {row_index} has extra columns: {extra_values}")
            image_value = (raw_row.get("image_path") or "").strip()
            domain_value = (raw_row.get("domain") or "").strip()
            if not image_value:
                raise ValueError(f"manifest row {row_index} is missing image_path")
            if domain_value not in ALLOWED_DOMAINS:
                raise ValueError(
                    f"manifest row {row_index} has invalid domain {domain_value!r}; "
                    f"expected one of {sorted(ALLOWED_DOMAINS)}"
                )
            candidate_image_path = Path(image_value)
            if not candidate_image_path.is_absolute():
                candidate_image_path = manifest_csv.parent / candidate_image_path
            assert_source_image_path(candidate_image_path)
            resolved_image = resolve_existing_file(image_value, base_dir=manifest_csv.parent)
            assert_source_image_path(resolved_image)
            if resolved_image in seen_paths:
                raise ValueError(f"manifest contains duplicate source image: {resolved_image}")
            seen_paths.add(resolved_image)
            pred_name = prediction_name(resolved_image)
            if pred_name in seen_prediction_names:
                raise ValueError(f"manifest has colliding prediction filename: {pred_name}")
            seen_prediction_names.add(pred_name)
            rows.append(
                ManifestRow(
                    row_index=row_index,
                    input_image_path=image_value,
                    resolved_image_path=resolved_image,
                    image_sha256=sha256_file(resolved_image),
                    caller_domain=domain_value,
                    selected_branch=DOMAIN_TO_BRANCH[domain_value],
                    prediction_name=pred_name,
                )
            )
    if not rows:
        raise ValueError("manifest CSV contains no input rows")
    return rows


def resolve_artifact_pair(
    *,
    branch: str,
    config_value: str,
    weights_value: str,
    expected_weights_sha256: str,
    research_only: bool,
) -> ArtifactPair:
    config_path = resolve_existing_file(config_value)
    weights_path = resolve_existing_file(weights_value)
    config_sha256 = sha256_file(config_path)
    weights_sha256 = sha256_file(weights_path)
    if weights_sha256 != expected_weights_sha256:
        raise ValueError(
            f"{branch} checkpoint SHA mismatch: expected {expected_weights_sha256}, got {weights_sha256}"
        )
    return ArtifactPair(
        branch=branch,
        config_path=config_path,
        config_sha256=config_sha256,
        weights_path=weights_path,
        weights_sha256=weights_sha256,
        expected_weights_sha256=expected_weights_sha256,
        research_only=research_only,
    )


def require_specialist_ack(rows: list[ManifestRow], ack_research_specialist: bool) -> None:
    needs_ack = any(row.selected_branch == "hw5k" for row in rows)
    if needs_ack and not ack_research_specialist:
        raise ValueError("--ack-research-specialist is required when manifest contains hw5k rows")


def sorted_branch_counts(rows: list[ManifestRow]) -> dict[str, int]:
    return {branch: sum(1 for row in rows if row.selected_branch == branch) for branch in BRANCH_ORDER}


class SerialInferenceLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any | None = None

    def __enter__(self) -> dict[str, str]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"serial inference lock is already held: {self.path}") from exc
        self.handle.seek(0)
        self.handle.truncate()
        payload = {"pid": os.getpid(), "script": str(SCRIPT_PATH), "cwd": str(Path.cwd())}
        self.handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self.handle.flush()
        return {"lock_path": str(self.path), "lock_mode": "LOCK_EX|LOCK_NB", "lock_holder_pid": str(os.getpid())}

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.handle is None:
            return
        try:
            self.handle.seek(0)
            self.handle.truncate()
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def materialize_branch_sample_list(branch_dir: Path, rows: list[ManifestRow]) -> Path:
    sample_list_path = branch_dir / "samples.txt"
    content = "".join(f"{row.resolved_image_path}\n" for row in rows)
    write_text_atomic(sample_list_path, content)
    return sample_list_path


def build_branch_command(
    *,
    sample_list_path: Path,
    branch_output_dir: Path,
    artifact_pair: ArtifactPair,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        str(PRIMARY_SCRIPT),
        "--samples-file",
        str(sample_list_path),
        "--output-dir",
        str(branch_output_dir),
        "--primary-config",
        str(artifact_pair.config_path),
        "--primary-weights",
        str(artifact_pair.weights_path),
        "--device",
        device,
        "--page-overlap",
        str(FIXED_INFERENCE_ARGS["page_overlap"]),
        "--batch-size",
        str(FIXED_INFERENCE_ARGS["batch_size"]),
        "--copy-input-outside-mask",
        str(FIXED_INFERENCE_ARGS["copy_input_outside_mask"]),
        "--copy-mask-threshold-auto",
        str(FIXED_INFERENCE_ARGS["copy_mask_threshold_auto"]),
        "--copy-mask-threshold",
        str(FIXED_INFERENCE_ARGS["copy_mask_threshold"]),
        "--copy-mask-dilate",
        str(FIXED_INFERENCE_ARGS["copy_mask_dilate"]),
        "--skip-label-metrics",
    ]


def run_branch(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=str(cwd), check=True, text=True, capture_output=True)


def read_metrics_csv(metrics_path: Path) -> list[dict[str, str]]:
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"metrics CSV is empty: {metrics_path}")
    return rows


def validate_branch_outputs(
    *,
    branch: str,
    branch_rows: list[ManifestRow],
    branch_output_dir: Path,
    artifact_pair: ArtifactPair,
) -> dict[Path, dict[str, str]]:
    pred_dir = branch_output_dir / "pred"
    if not pred_dir.is_dir():
        raise FileNotFoundError(pred_dir)
    expected_names = [row.prediction_name for row in branch_rows]
    actual_entries = list(pred_dir.iterdir())
    actual_names = sorted(path.name for path in actual_entries)
    if sorted(expected_names) != actual_names:
        raise ValueError(
            f"{branch} prediction set mismatch: expected {sorted(expected_names)}, got {actual_names}"
        )
    metrics_rows = read_metrics_csv(branch_output_dir / "metrics.csv")
    if len(metrics_rows) != len(branch_rows):
        raise ValueError(
            f"{branch} metrics row count mismatch: expected {len(branch_rows)}, got {len(metrics_rows)}"
        )
    by_image_path = {str(row.resolved_image_path): row for row in branch_rows}
    pred_sha_by_path: dict[Path, dict[str, str]] = {}
    seen_metric_paths: set[Path] = set()
    seen_metric_images: set[Path] = set()
    for metrics_row in metrics_rows:
        missing_fields = REQUIRED_METRICS_FIELDS - set(metrics_row)
        if missing_fields:
            raise ValueError(
                f"{branch} metrics is missing required fields: {sorted(missing_fields)}"
            )
        image_path = metrics_row.get("image_path", "")
        pred_path_value = metrics_row.get("pred_path", "")
        pred_sha = metrics_row.get("pred_sha256", "")
        if image_path not in by_image_path:
            raise ValueError(f"{branch} metrics references unexpected image path: {image_path}")
        manifest_row = by_image_path[image_path]
        pred_path = Path(pred_path_value).resolve()
        expected_pred_path = (pred_dir / manifest_row.prediction_name).resolve()
        if pred_path != expected_pred_path:
            raise ValueError(
                f"{branch} metrics pred_path mismatch for {image_path}: "
                f"expected {expected_pred_path}, got {pred_path}"
            )
        if pred_path in seen_metric_paths:
            raise ValueError(f"{branch} metrics contains duplicate prediction path: {pred_path}")
        seen_metric_paths.add(pred_path)
        resolved_image_path = manifest_row.resolved_image_path
        if resolved_image_path in seen_metric_images:
            raise ValueError(f"{branch} metrics contains duplicate image path: {image_path}")
        seen_metric_images.add(resolved_image_path)
        if metrics_row.get("image_sha256") != manifest_row.image_sha256:
            raise ValueError(f"{branch} metrics source image SHA mismatch for {image_path}")
        if metrics_row.get("primary_config_sha256") != artifact_pair.config_sha256:
            raise ValueError(f"{branch} metrics config SHA mismatch for {image_path}")
        if metrics_row.get("primary_weights_sha256") != artifact_pair.weights_sha256:
            raise ValueError(f"{branch} metrics checkpoint SHA mismatch for {image_path}")
        if metrics_row.get("metrics_skipped") != "1":
            raise ValueError(f"{branch} metrics must report metrics_skipped=1 for {image_path}")
        if not pred_path.is_file():
            raise FileNotFoundError(pred_path)
        actual_pred_sha = sha256_file(pred_path)
        if pred_sha != actual_pred_sha:
            raise ValueError(
                f"{branch} prediction SHA mismatch for {pred_path}: expected {pred_sha}, got {actual_pred_sha}"
            )
        pred_sha_by_path[pred_path] = {
            "prediction_sha256": actual_pred_sha,
            "image_sha256": manifest_row.image_sha256,
        }
    expected_metric_paths = {(pred_dir / row.prediction_name).resolve() for row in branch_rows}
    if seen_metric_paths != expected_metric_paths:
        raise ValueError(f"{branch} metrics prediction coverage mismatch")
    expected_metric_images = {row.resolved_image_path for row in branch_rows}
    if seen_metric_images != expected_metric_images:
        raise ValueError(f"{branch} metrics source image coverage mismatch")
    return pred_sha_by_path


def merge_branch_predictions(
    *,
    all_rows: list[ManifestRow],
    output_dir: Path,
    branch_output_dirs: dict[str, Path],
    branch_pred_meta: dict[str, dict[Path, dict[str, str]]],
    artifacts_by_branch: dict[str, ArtifactPair],
) -> list[dict[str, str]]:
    pred_dir = output_dir / "pred"
    pred_dir.mkdir()
    route_rows: list[dict[str, str]] = []
    for row in all_rows:
        branch_output_dir = branch_output_dirs[row.selected_branch]
        source_pred = (branch_output_dir / "pred" / row.prediction_name).resolve()
        if source_pred not in branch_pred_meta[row.selected_branch]:
            raise ValueError(f"missing validated branch prediction for {source_pred}")
        merged_pred = (pred_dir / row.prediction_name).resolve()
        if merged_pred.exists():
            raise ValueError(f"merged prediction collision: {merged_pred}")
        shutil.copy2(source_pred, merged_pred)
        source_sha = branch_pred_meta[row.selected_branch][source_pred]["prediction_sha256"]
        merged_sha = sha256_file(merged_pred)
        if source_sha != merged_sha:
            raise ValueError(f"merged prediction SHA mismatch for {merged_pred}")
        artifact_pair = artifacts_by_branch[row.selected_branch]
        route_rows.append(
            {
                "row_index": str(row.row_index),
                "image_path": str(row.resolved_image_path),
                "image_sha256": row.image_sha256,
                "caller_domain": row.caller_domain,
                "selected_branch": row.selected_branch,
                "primary_config_path": str(artifact_pair.config_path),
                "primary_config_sha256": artifact_pair.config_sha256,
                "primary_weights_path": str(artifact_pair.weights_path),
                "primary_weights_sha256": artifact_pair.weights_sha256,
                "branch_prediction_path": str(source_pred),
                "merged_prediction_path": str(merged_pred),
                "prediction_sha256": merged_sha,
            }
        )
    return route_rows


def write_route_csv(path: Path, rows: list[dict[str, str]]) -> str:
    fieldnames = [
        "row_index",
        "image_path",
        "image_sha256",
        "caller_domain",
        "selected_branch",
        "primary_config_path",
        "primary_config_sha256",
        "primary_weights_path",
        "primary_weights_sha256",
        "branch_prediction_path",
        "merged_prediction_path",
        "prediction_sha256",
    ]
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)
    return sha256_file(path)


def build_manifest(
    *,
    status: str,
    args: argparse.Namespace,
    manifest_csv: Path,
    manifest_sha256: str,
    rows: list[ManifestRow],
    output_dir: Path,
    lock_info: dict[str, str],
    artifacts_by_branch: dict[str, ArtifactPair],
    branch_commands: dict[str, list[str]],
    branch_output_dirs: dict[str, Path],
    route_csv_path: Path | None,
    route_csv_sha256: str | None,
    error: str | None = None,
) -> dict[str, Any]:
    branch_counts = sorted_branch_counts(rows)
    artifacts_payload = {}
    for branch in BRANCH_ORDER:
        artifact = artifacts_by_branch[branch]
        artifacts_payload[branch] = {
            "config_path": str(artifact.config_path),
            "config_sha256": artifact.config_sha256,
            "weights_path": str(artifact.weights_path),
            "weights_sha256": artifact.weights_sha256,
            "expected_weights_sha256": artifact.expected_weights_sha256,
            "research_only": artifact.research_only,
        }
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "script_path": str(SCRIPT_PATH),
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "input_manifest_path": str(manifest_csv),
        "input_manifest_sha256": manifest_sha256,
        "row_count": len(rows),
        "branch_counts": branch_counts,
        "allowed_domains": sorted(ALLOWED_DOMAINS),
        "domain_mapping": DOMAIN_TO_BRANCH,
        "fixed_inference_args": FIXED_INFERENCE_ARGS,
        "device": args.device,
        "inference_reads_labels": False,
        "label_use_policy": "source-only primary inference; labels/targets are forbidden",
        "research_specialist_acknowledged": bool(args.ack_research_specialist),
        "research_specialist_status": "research_only/gate_qualified_nonpromotion",
        "artifacts": artifacts_payload,
        "branch_commands": branch_commands,
        "branch_output_dirs": {branch: str(path) for branch, path in branch_output_dirs.items()},
        "output_dir": str(output_dir),
        "merged_pred_dir": str(output_dir / "pred"),
        "lock": lock_info,
    }
    if route_csv_path is not None:
        payload["route_decisions_csv_path"] = str(route_csv_path)
    if route_csv_sha256 is not None:
        payload["route_decisions_csv_sha256"] = route_csv_sha256
    if error is not None:
        payload["error"] = error
    return payload


def main() -> None:
    args = parse_args()
    manifest_csv = resolve_existing_file(args.manifest_csv)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not PRIMARY_SCRIPT.is_file():
        raise FileNotFoundError(PRIMARY_SCRIPT)

    rows = parse_manifest_rows(manifest_csv)
    require_specialist_ack(rows, args.ack_research_specialist)
    artifacts_by_branch = {
        "default": resolve_artifact_pair(
            branch="default",
            config_value=args.default_config,
            weights_value=args.default_weights,
            expected_weights_sha256=args.default_weights_sha256,
            research_only=False,
        ),
        "hw5k": resolve_artifact_pair(
            branch="hw5k",
            config_value=args.hw5k_config,
            weights_value=args.hw5k_weights,
            expected_weights_sha256=args.hw5k_weights_sha256,
            research_only=True,
        ),
    }
    manifest_sha256 = sha256_file(manifest_csv)

    route_csv_path: Path | None = None
    route_csv_sha256: str | None = None
    branch_commands: dict[str, list[str]] = {}
    branch_output_dirs: dict[str, Path] = {}
    lock_info: dict[str, str] = {}
    manifest_path: Path | None = None
    with SerialInferenceLock(LOCK_PATH) as acquired_lock:
        lock_info = acquired_lock
        output_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = output_dir / "run_manifest.json"
        running_manifest = build_manifest(
            status="running",
            args=args,
            manifest_csv=manifest_csv,
            manifest_sha256=manifest_sha256,
            rows=rows,
            output_dir=output_dir,
            lock_info=lock_info,
            artifacts_by_branch=artifacts_by_branch,
            branch_commands=branch_commands,
            branch_output_dirs=branch_output_dirs,
            route_csv_path=route_csv_path,
            route_csv_sha256=route_csv_sha256,
        )
        write_json_atomic(manifest_path, running_manifest)
        try:
            branch_pred_meta: dict[str, dict[Path, dict[str, str]]] = {}
            for branch in BRANCH_ORDER:
                branch_rows = [row for row in rows if row.selected_branch == branch]
                if not branch_rows:
                    continue
                branch_dir = output_dir / "branches" / branch
                branch_dir.mkdir(parents=True, exist_ok=False)
                sample_list_path = materialize_branch_sample_list(branch_dir, branch_rows)
                branch_output_dir = branch_dir / "run_primary_full_page"
                command = build_branch_command(
                    sample_list_path=sample_list_path,
                    branch_output_dir=branch_output_dir,
                    artifact_pair=artifacts_by_branch[branch],
                    device=args.device,
                )
                branch_commands[branch] = command
                branch_output_dirs[branch] = branch_output_dir
                write_json_atomic(
                    manifest_path,
                    build_manifest(
                        status="running",
                        args=args,
                        manifest_csv=manifest_csv,
                        manifest_sha256=manifest_sha256,
                        rows=rows,
                        output_dir=output_dir,
                        lock_info=lock_info,
                        artifacts_by_branch=artifacts_by_branch,
                        branch_commands=branch_commands,
                        branch_output_dirs=branch_output_dirs,
                        route_csv_path=route_csv_path,
                        route_csv_sha256=route_csv_sha256,
                    ),
                )
                run_branch(command, cwd=ROOT)
                branch_pred_meta[branch] = validate_branch_outputs(
                    branch=branch,
                    branch_rows=branch_rows,
                    branch_output_dir=branch_output_dir,
                    artifact_pair=artifacts_by_branch[branch],
                )
            route_rows = merge_branch_predictions(
                all_rows=rows,
                output_dir=output_dir,
                branch_output_dirs=branch_output_dirs,
                branch_pred_meta=branch_pred_meta,
                artifacts_by_branch=artifacts_by_branch,
            )
            route_csv_path = output_dir / "route_decisions.csv"
            route_csv_sha256 = write_route_csv(route_csv_path, route_rows)
            complete_manifest = build_manifest(
                status="complete",
                args=args,
                manifest_csv=manifest_csv,
                manifest_sha256=manifest_sha256,
                rows=rows,
                output_dir=output_dir,
                lock_info=lock_info,
                artifacts_by_branch=artifacts_by_branch,
                branch_commands=branch_commands,
                branch_output_dirs=branch_output_dirs,
                route_csv_path=route_csv_path,
                route_csv_sha256=route_csv_sha256,
            )
            write_json_atomic(manifest_path, complete_manifest)
        except Exception as exc:
            if manifest_path is not None:
                failed_manifest = build_manifest(
                    status="failed",
                    args=args,
                    manifest_csv=manifest_csv,
                    manifest_sha256=manifest_sha256,
                    rows=rows,
                    output_dir=output_dir,
                    lock_info=lock_info,
                    artifacts_by_branch=artifacts_by_branch,
                    branch_commands=branch_commands,
                    branch_output_dirs=branch_output_dirs,
                    route_csv_path=route_csv_path,
                    route_csv_sha256=route_csv_sha256,
                    error=f"{type(exc).__name__}: {exc}",
                )
                write_json_atomic(manifest_path, failed_manifest)
            raise


if __name__ == "__main__":
    main()
