#!/usr/bin/env bash
# Generate distillation data in small resumable chunks so each chunk finishes
# well within the WSL background-process lifetime, then append to one file.
# Usage: ./gen_chunks.sh <out> <n> <total_games> <games_per_chunk> [workers]
set -euo pipefail

OUT="${1:-/tmp/distill_data.bin}"
N="${2:-3}"
TOTAL="${3:-200000}"
CHUNK="${4:-4000}"
WORKERS="${5:-32}"
EXPLORE="${6:-0.05}"

rm -f "$OUT"
for ((g = 0; g < TOTAL; g += CHUNK)); do
  games=$((TOTAL - g))
  if ((games > CHUNK)); then games=$CHUNK; fi
  echo "[$g/$TOTAL] chunk: $games games, explore=$EXPLORE"
  if ((g == 0)); then
    ./distill -n "$N" -games "$games" -workers "$WORKERS" -explore "$EXPLORE" -out "$OUT"
  else
    ./distill -n "$N" -games "$games" -workers "$WORKERS" -explore "$EXPLORE" -out "$OUT" -append
  fi
done
echo "ALL CHUNKS DONE"