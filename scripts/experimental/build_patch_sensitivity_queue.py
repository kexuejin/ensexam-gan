#!/usr/bin/env python3
"""Build a reproducible one-patch sensitivity sweep queue.

The nearworst-safe probe is highly sensitive to the exact training patch. This
script turns a ranked patch-index CSV into per-patch index files plus runnable
train/eval commands, so experiments can be replayed without relying on random
DataLoader order or visual inspection.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


DEFAULT_SCUT_BASELINE = "outputs/scut_test115_second_stage_baseline_20260705/pred"
DEFAULT_HOLDOUT_BASELINE = "outputs/holdout40_second_stage_readiness_20260705/pred"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-index-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--experiment-prefix", default="patch_sensitivity")
    parser.add_argument("--date-tag", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--train-config", default="configs/local/config.local-current-primary-continuation-mps.yaml")
    parser.add_argument("--candidate-config", default="configs/local/config.local-current-primary-continuation-mps.yaml")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-pages", type=int, default=16)
    parser.add_argument("--lambda-input-preserve", type=float, default=24.0)
    parser.add_argument("--lambda-mb-leak", type=float, default=2.0)
    parser.add_argument("--scut-samples-file", default="docs/scut-test115-relative.txt")
    parser.add_argument("--holdout-samples-file", default="docs/holdout40-relative.txt")
    parser.add_argument("--scut-baseline-pred-dir", default=DEFAULT_SCUT_BASELINE)
    parser.add_argument("--holdout-baseline-pred-dir", default=DEFAULT_HOLDOUT_BASELINE)
    parser.add_argument("--cleanup-checkpoint", default="artifacts/current-second-stage-best.pt")
    parser.add_argument("--min-copy-mask-cov8", type=float, default=0.806133)
    parser.add_argument("--max-primary-edit-px", type=int, default=98868)
    parser.add_argument("--candidate-copy-threshold", type=int, default=98)
    parser.add_argument("--cleanup-alpha-threshold", type=float, default=0.3)
    parser.add_argument("--cleanup-tile-size", type=int, default=160)
    parser.add_argument("--cleanup-stride", type=int, default=160)
    parser.add_argument("--base-edit-threshold", type=float, default=12.0)
    parser.add_argument("--second-delta-threshold", type=float, default=32.0)
    parser.add_argument("--dark-threshold", type=int, default=0)
    return parser.parse_args()


def read_patch_rows(path: Path, offset: int, limit: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty patch index: {path}")
    required = {"file", "x1", "y1", "x2", "y2"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {sorted(missing)}")
    if offset < 0 or limit <= 0:
        raise ValueError("--offset must be >= 0 and --limit must be > 0")
    return rows[offset: offset + limit]


def shell_quote(value: str | Path) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def safe_stem(row: dict[str, str], ordinal: int) -> str:
    file_stem = Path(row["file"]).stem
    return f"{ordinal:03d}_{file_stem}_x{int(row['x1'])}_y{int(row['y1'])}"


def train_command(args: argparse.Namespace, patch_csv: Path, train_dir: Path) -> str:
    return " ".join([
        "$ENSEXAM_PYTHON",
        "scripts/train/micro_train_region_probe.py",
        "--config", shell_quote(args.train_config),
        "--output-dir", shell_quote(train_dir),
        "--max-steps", str(args.max_steps),
        "--batch-size", str(args.batch_size),
        "--train-pages", str(args.train_pages),
        "--patch-index-file", shell_quote(patch_csv),
        "--loss-override", shell_quote(f"lambda_input_preserve={args.lambda_input_preserve}"),
        "--loss-override", shell_quote(f"lambda_mb_leak={args.lambda_mb_leak}"),
        "--trace-batches-file", shell_quote(train_dir / "trace_batches.csv"),
        "--log-every", "1",
        "--save-every", str(args.max_steps),
    ])


def gate_command(
    args: argparse.Namespace,
    samples_file: str,
    baseline_pred_dir: str,
    train_dir: Path,
    eval_dir: Path,
) -> str:
    checkpoint = train_dir / f"micro_region_probe_step{args.max_steps:04d}.pth"
    return " ".join([
        "$ENSEXAM_PYTHON",
        "scripts/infer/run_hybrid_second_stage_gate.py",
        "--samples-file", shell_quote(samples_file),
        "--output-dir", shell_quote(eval_dir),
        "--baseline-pred-dir", shell_quote(baseline_pred_dir),
        "--candidate-config", shell_quote(args.candidate_config),
        "--candidate-weights", shell_quote(checkpoint),
        "--cleanup-checkpoint", shell_quote(args.cleanup_checkpoint),
        "--device", "mps",
        "--candidate-copy-mask", "mb",
        "--candidate-copy-threshold", str(args.candidate_copy_threshold),
        "--min-copy-mask-cov8", str(args.min_copy_mask_cov8),
        "--max-primary-edit-px", str(args.max_primary_edit_px),
        "--cleanup-alpha-threshold", str(args.cleanup_alpha_threshold),
        "--cleanup-tile-size", str(args.cleanup_tile_size),
        "--cleanup-stride", str(args.cleanup_stride),
        "--base-edit-threshold", str(args.base_edit_threshold),
        "--second-delta-threshold", str(args.second_delta_threshold),
        "--dark-threshold", str(args.dark_threshold),
        "--save-candidate",
    ])


def write_patch_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerow({key: row[key] for key in writer.fieldnames})


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    patch_dir = output_dir / "patch_indices"
    command_dir = output_dir / "commands"
    train_root = output_dir / "train"
    eval_root = output_dir / "eval"
    for directory in (patch_dir, command_dir, train_root, eval_root):
        directory.mkdir(parents=True, exist_ok=True)

    rows = read_patch_rows(Path(args.patch_index_csv), args.offset, args.limit)
    manifest_path = output_dir / "manifest.csv"
    train_commands: list[str] = []
    eval_commands: list[str] = []

    manifest_fields = [
        "ordinal",
        "name",
        "file",
        "x1",
        "y1",
        "x2",
        "y2",
        "patch_index_csv",
        "train_output_dir",
        "scut_output_dir",
        "holdout_output_dir",
    ]
    extra_fields = [key for key in rows[0].keys() if key not in set(manifest_fields)]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields + extra_fields)
        writer.writeheader()
        for ordinal, row in enumerate(rows, start=args.offset + 1):
            name = safe_stem(row, ordinal)
            patch_csv = patch_dir / f"{name}.csv"
            train_dir = train_root / f"{args.experiment_prefix}_{name}_{args.date_tag}"
            scut_dir = eval_root / f"scut115_{args.experiment_prefix}_{name}_{args.date_tag}"
            holdout_dir = eval_root / f"holdout40_{args.experiment_prefix}_{name}_{args.date_tag}"
            write_patch_csv(patch_csv, row)

            train_cmd = train_command(args, patch_csv, train_dir)
            scut_cmd = gate_command(args, args.scut_samples_file, args.scut_baseline_pred_dir, train_dir, scut_dir)
            holdout_cmd = gate_command(
                args,
                args.holdout_samples_file,
                args.holdout_baseline_pred_dir,
                train_dir,
                holdout_dir,
            )
            train_commands.append(train_cmd)
            eval_commands.extend([scut_cmd, holdout_cmd])

            writer.writerow({
                **{key: row.get(key, "") for key in extra_fields},
                "ordinal": ordinal,
                "name": name,
                "file": row["file"],
                "x1": row["x1"],
                "y1": row["y1"],
                "x2": row["x2"],
                "y2": row["y2"],
                "patch_index_csv": patch_csv,
                "train_output_dir": train_dir,
                "scut_output_dir": scut_dir,
                "holdout_output_dir": holdout_dir,
            })

    train_script = command_dir / "01_train.sh"
    eval_script = command_dir / "02_eval.sh"
    all_script = command_dir / "run_all.sh"
    train_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\nsource .env\n\n" + "\n".join(train_commands) + "\n")
    eval_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\nsource .env\n\n" + "\n".join(eval_commands) + "\n")
    all_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        f"bash {shell_quote(train_script)}\n"
        f"bash {shell_quote(eval_script)}\n"
    )
    for script in (train_script, eval_script, all_script):
        script.chmod(0o755)

    print(f"patches={len(rows)}")
    print(f"manifest={manifest_path}")
    print(f"train_commands={train_script}")
    print(f"eval_commands={eval_script}")
    print(f"run_all={all_script}")


if __name__ == "__main__":
    main()
