#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  . "$ROOT/.env"
fi
PYTHON="${ENSEXAM_PYTHON:-python}"
TRAIN_PID="${TRAIN_PID:-}"
TRAIN_SESSION="${TRAIN_SESSION:-${1:-ensexam-hardcase-finetune}}"
RUN_DIR="${RUN_DIR:-${2:-}}"
RUNS_ROOT="${RUNS_ROOT:-${3:-checkpoints-hardcase-mps/ensexam}}"
CONFIG_PATH="${CONFIG_PATH:-${4:-configs/local/config.local-hardcase-mps.yaml}}"
SAMPLES_FILE="${SAMPLES_FILE:-${5:-docs/smoke-holdout3-absolute.txt}}"
BASELINE_PRED_DIR="${BASELINE_PRED_DIR:-${6:-artifacts/current-holdout40-primary-pred}}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-1800}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

while true; do
  if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    log "training pid $TRAIN_PID still running; sleeping ${CHECK_INTERVAL_SECONDS}s"
    sleep "$CHECK_INTERVAL_SECONDS"
    continue
  fi

  if [[ -z "$TRAIN_PID" ]] && tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
    log "tmux session $TRAIN_SESSION still exists; sleeping ${CHECK_INTERVAL_SECONDS}s"
    sleep "$CHECK_INTERVAL_SECONDS"
    continue
  fi

  if [[ -z "$TRAIN_PID" ]] && pgrep -f "train.py --config ${CONFIG_PATH}" >/dev/null 2>&1; then
    log "matching training command still running; sleeping ${CHECK_INTERVAL_SECONDS}s"
    sleep "$CHECK_INTERVAL_SECONDS"
    continue
  fi

  break
done

cd "$ROOT"

if [[ -n "$RUN_DIR" ]]; then
  run_dir="$RUN_DIR"
else
  run_dir="$(find "$RUNS_ROOT" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)"
fi
weights="$run_dir/best.pth"
if [[ ! -f "$weights" ]]; then
  weights="$run_dir/latest.pth"
fi
if [[ ! -f "$weights" ]]; then
  echo "No checkpoint found under $run_dir" >&2
  exit 1
fi

output_dir="$ROOT/outputs/hardcase_finetune_worst_eval/$(basename "$run_dir")_final_best"
mkdir -p "$output_dir"

log "evaluating $weights"
PYTORCH_ENABLE_MPS_FALLBACK=1 "$PYTHON" -u scripts/eval_hardcase_worst_pages.py \
  --config "$CONFIG_PATH" \
  --weights "$weights" \
  --samples-file "$SAMPLES_FILE" \
  --baseline-pred-dir "$BASELINE_PRED_DIR" \
  --output-dir "$output_dir" \
  --device auto \
  2>&1 | tee "$output_dir/eval.log"
