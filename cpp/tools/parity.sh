#!/usr/bin/env bash
# One-shot parity gate: export weights, build fixtures, compile, and run both
# the golden fixture test and the randomized cross-check against PyTorch.
#
# Usage: tools/parity.sh [path/to/model_universal.pt]
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:-$(python3 -c 'from backend.ml.model_agent import base_checkpoint_path; print(base_checkpoint_path(3))')}"
if [[ ! -f "$CKPT" ]]; then
  echo "checkpoint not found: $CKPT" >&2
  exit 1
fi
CKPT="$(realpath "$CKPT")"

echo "== weights =="
python3 tools/export_weights.py "$CKPT" model.bin
echo "== fixtures =="
python3 tools/make_fixtures.py "$CKPT" fixtures
echo "== build =="
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" >/dev/null
echo "== golden fixture parity =="
./build/parity_test model.bin fixtures
echo "== randomized cross-check (ctypes vs torch) =="
python3 tools/check_parity.py "$CKPT" model.bin build/libmodel.so