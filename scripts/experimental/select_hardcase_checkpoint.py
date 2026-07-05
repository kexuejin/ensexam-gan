#!/usr/bin/env python3
"""Evaluate candidate checkpoints on hard SCUT pages and select the best one."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--baseline-pred-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--page-overlap", type=int, default=32)
    parser.add_argument("--max-overerase", type=float, default=0.0015)
    parser.add_argument("--overerase-penalty", type=float, default=8.0)
    parser.add_argument(
        "--copy-input-outside-mask",
        choices=("none", "ms", "mb"),
        default="none",
        help="Pass-through inference postprocess for eval_hardcase_worst_pages.py.",
    )
    parser.add_argument("--copy-mask-threshold", type=int, default=32)
    parser.add_argument(
        "--copy-mask-threshold-map",
        default="",
        help="Pass-through per-file threshold overrides for eval_hardcase_worst_pages.py.",
    )
    parser.add_argument(
        "--copy-mask-threshold-auto",
        choices=("none", "mb_cov8_step"),
        default="none",
        help="Pass-through inference-time threshold heuristic for eval_hardcase_worst_pages.py.",
    )
    parser.add_argument("--copy-mask-dilate", type=int, default=0)
    parser.add_argument(
        "--include-final-checkpoint",
        action="store_true",
        help=(
            "Also evaluate the final aggregate checkpoint when step checkpoints "
            "exist. By default it is skipped because micro_train_region_probe.py "
            "saves both micro_region_probe_stepNNNN.pth and a duplicate final "
            "micro_region_probe.pth."
        ),
    )
    return parser.parse_args()


def read_metrics(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(path.open()))

    def avg(key: str) -> float:
        return sum(float(row[key]) for row in rows) / max(len(rows), 1)

    return {
        "baseline_residual": avg("baseline_residual_ratio"),
        "baseline_overerase": avg("baseline_overerase_ratio"),
        "residual": avg("residual_ratio"),
        "overerase": avg("overerase_ratio"),
    }


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    name = path.stem
    if "step" in name:
        try:
            return (int(name.rsplit("step", 1)[1]), name)
        except ValueError:
            pass
    return (10**9, name)


def run_eval(args: argparse.Namespace, checkpoint: Path, eval_dir: Path) -> None:
    if (eval_dir / "hardcase_worst_metrics.csv").exists():
        return
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_hardcase_worst_pages.py"),
        "--config",
        args.config,
        "--weights",
        str(checkpoint),
        "--samples-file",
        args.samples_file,
        "--baseline-pred-dir",
        args.baseline_pred_dir,
        "--output-dir",
        str(eval_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--page-overlap",
        str(args.page_overlap),
    ]
    if args.copy_input_outside_mask != "none":
        cmd.extend([
            "--copy-input-outside-mask",
            args.copy_input_outside_mask,
            "--copy-mask-threshold",
            str(args.copy_mask_threshold),
            "--copy-mask-threshold-map",
            args.copy_mask_threshold_map,
            "--copy-mask-threshold-auto",
            args.copy_mask_threshold_auto,
            "--copy-mask-dilate",
            str(args.copy_mask_dilate),
        ])
    with (eval_dir.with_suffix(".eval.log")).open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(checkpoint_dir.glob("*.pth"), key=checkpoint_sort_key)
    if not args.include_final_checkpoint:
        has_step_checkpoints = any("step" in checkpoint.stem for checkpoint in checkpoints)
        if has_step_checkpoints:
            checkpoints = [
                checkpoint
                for checkpoint in checkpoints
                if checkpoint.stem != "micro_region_probe"
            ]
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}")

    rows: list[dict[str, str | float]] = []
    for checkpoint in checkpoints:
        eval_dir = output_dir / f"eval_{checkpoint.stem}"
        run_eval(args, checkpoint, eval_dir)
        metrics = read_metrics(eval_dir / "hardcase_worst_metrics.csv")
        residual_gain = metrics["baseline_residual"] - metrics["residual"]
        overerase_delta = metrics["overerase"] - metrics["baseline_overerase"]
        score = residual_gain - args.overerase_penalty * max(overerase_delta, 0.0)
        rows.append({
            "checkpoint": checkpoint.name,
            "checkpoint_path": str(checkpoint),
            "eval_dir": str(eval_dir),
            "baseline_residual": metrics["baseline_residual"],
            "residual": metrics["residual"],
            "residual_gain": residual_gain,
            "baseline_overerase": metrics["baseline_overerase"],
            "overerase": metrics["overerase"],
            "overerase_delta": overerase_delta,
            "score": score,
            "passes_overerase_gate": metrics["overerase"] <= args.max_overerase,
        })

    gated = [row for row in rows if row["passes_overerase_gate"]]
    candidates = gated if gated else rows
    best = sorted(candidates, key=lambda row: (float(row["score"]), -float(row["overerase"])), reverse=True)[0]

    summary_csv = output_dir / "selector_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_txt = output_dir / "best_checkpoint.txt"
    best_txt.write_text(
        "\n".join([
            f"best_checkpoint={best['checkpoint_path']}",
            f"best_eval_dir={best['eval_dir']}",
            f"residual={float(best['residual']):.6f}",
            f"residual_gain={float(best['residual_gain']):+.6f}",
            f"overerase={float(best['overerase']):.6f}",
            f"overerase_delta={float(best['overerase_delta']):+.6f}",
            f"score={float(best['score']):+.6f}",
            f"passes_overerase_gate={best['passes_overerase_gate']}",
            f"summary_csv={summary_csv}",
        ]) + "\n",
        encoding="utf-8",
    )

    print(best_txt.read_text(), end="")


if __name__ == "__main__":
    main()
