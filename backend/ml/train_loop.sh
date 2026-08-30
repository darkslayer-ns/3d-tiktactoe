#!/usr/bin/env bash
# Train distillation in short resumable invocations so each completes well
# within the WSL background-process lifetime. Each invocation runs a few
# epochs, saves the checkpoint (with `epoch` in config), and the next call
# resumes from it.
#
# Usage: ./train_loop.sh [data] [target_epochs] [epochs_per_run]
set -euo pipefail

DATA="${1:-/tmp/d3_final.bin}"
TARGET="${2:-60}"
PER_RUN="${3:-3}"
DMODEL="${4:-64}"
LAYERS="${5:-2}"

LOG=/tmp/opencode/train_loop.log
PROG=/tmp/opencode/train_distill.log

for ((e = 0; e < TARGET; e += PER_RUN)); do
  echo "[run $e] starting epochs $((e+1))..$((e+PER_RUN))"
  if [[ -f model_universal.pt ]]; then
    python3 -m training.train_distill "$DATA" \
      --d-model "$DMODEL" --layers "$LAYERS" --epochs "$PER_RUN" \
      --device cuda --save model_universal.pt --resume \
      --max-epochs "$TARGET" --progress "$PROG" \
      >> "$LOG" 2>&1
  else
    python3 -m training.train_distill "$DATA" \
      --d-model "$DMODEL" --layers "$LAYERS" --epochs "$PER_RUN" \
      --device cuda --save model_universal.pt \
      --max-epochs "$TARGET" --progress "$PROG" \
      >> "$LOG" 2>&1
  fi
done
echo "TRAINING LOOP COMPLETE" >> "$LOG"