#!/usr/bin/env bash
# Start the FastAPI backend (and optionally the frontend dev server).
# Usage:
#   ./scripts/dev.sh backend        # backend only (default)
#   ./scripts/dev.sh frontend       # frontend only
#   ./scripts/dev.sh both           # backend + frontend
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${BACKEND_PORT:-8100}"
FRONT_PORT="${FRONT_PORT:-5173}"
LOG="$ROOT/.logs"

mkdir -p "$LOG"

kill_port() {
  local p="$1"
  local pids
  pids="$(lsof -ti tcp:"$p" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "  freeing port $p (killing $pids)"
    kill $pids 2>/dev/null || true
    sleep 1
  fi
}

start_backend() {
  kill_port "$PORT"
  echo "  starting backend on :$PORT ..."
  cd "$ROOT"
  nohup python3 -m uvicorn backend.server.app:app --host 127.0.0.1 --port "$PORT" \
    > "$LOG/backend.log" 2>&1 &
  echo $! > "$LOG/backend.pid"
  # wait until healthy
  for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
      echo "  backend ready ✓  (log: $LOG/backend.log)"
      return 0
    fi
    sleep 0.5
  done
  echo "  backend FAILED to start; see $LOG/backend.log"
  return 1
}

start_frontend() {
  kill_port "$FRONT_PORT"
  echo "  starting frontend on :$FRONT_PORT ..."
  cd "$ROOT/frontend"
  nohup npm run dev > "$LOG/frontend.log" 2>&1 &
  echo $! > "$LOG/frontend.pid"
  for i in $(seq 1 40); do
    if curl -sf "http://127.0.0.1:$FRONT_PORT/" >/dev/null 2>&1; then
      echo "  frontend ready ✓  http://localhost:$FRONT_PORT  (log: $LOG/frontend.log)"
      return 0
    fi
    sleep 0.5
  done
  echo "  frontend FAILED to start; see $LOG/frontend.log"
  return 1
}

cmd="${1:-backend}"
case "$cmd" in
  backend)  start_backend ;;
  frontend) start_frontend ;;
  both)     start_backend && start_frontend ;;
  *) echo "usage: $0 [backend|frontend|both]"; exit 1 ;;
esac