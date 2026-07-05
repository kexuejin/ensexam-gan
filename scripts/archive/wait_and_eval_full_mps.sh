#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  . "$ROOT/.env"
fi
PYTHON="${ENSEXAM_PYTHON:-python}"
TRAIN_SESSION="ensexam_full_mps"
RUNS_ROOT="${1:-checkpoints-full-mps-fast/ensexam}"
CONFIG_PATH="${2:-configs/local/config.local-full-mps-fast.yaml}"

while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  sleep 60
done

cd "$ROOT"

run_dir="$(find "$RUNS_ROOT" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)"
weights="$run_dir/best.pth"
if [[ ! -f "$weights" ]]; then
  weights="$run_dir/latest.pth"
fi

"$PYTHON" -u test.py \
  --config "$CONFIG_PATH" \
  --weights "$weights" \
  --eval-mode both \
  --output-dir "$run_dir/final-test" \
  2>&1 | tee "$run_dir/final-test.log"
