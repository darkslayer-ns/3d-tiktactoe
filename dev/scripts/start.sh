#!/usr/bin/env bash
# Start backend + frontend dev servers and expose the frontend via ngrok.
# Usage:
#   ./scripts/start.sh            # start everything
#   ./scripts/start.sh stop       # stop everything
#   ./scripts/start.sh status     # show what's running
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${BACKEND_PORT:-8100}"
FRONT_PORT="${FRONT_PORT:-5173}"
NGROK_URL="${NGROK_URL:-https://krissy-pushed-nonscandalously.ngrok-free.dev}"
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

stop_all() {
  echo "stopping backend, frontend, ngrok..."
  for f in backend.pid frontend.pid ngrok.pid; do
    if [ -f "$LOG/$f" ]; then
      pid="$(cat "$LOG/$f" 2>/dev/null || true)"
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "  killing $f ($pid)"
        kill "$pid" 2>/dev/null || true
      fi
      rm -f "$LOG/$f"
    fi
  done
  kill_port "$FRONT_PORT" 2>/dev/null || true
  kill_port "$PORT" 2>/dev/null || true
  kill_port 4040 2>/dev/null || true
  echo "stopped."
}

status_all() {
  for spec in "$PORT:backend" "$FRONT_PORT:frontend" "4040:ngrok-admin"; do
    p="${spec%%:*}"
    name="${spec##*:}"
    if lsof -ti tcp:"$p" >/dev/null 2>&1; then
      echo "  $name running on :$p ✓"
    else
      echo "  $name not running"
    fi
  done
}

start_backend() {
  kill_port "$PORT"
  echo "  starting backend on :$PORT ..."
  cd "$ROOT"
  nohup python3 -m uvicorn backend.server.app:app --host 127.0.0.1 --port "$PORT" \
    > "$LOG/backend.log" 2>&1 &
  echo $! > "$LOG/backend.pid"
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

start_ngrok() {
  kill_port 4040
  echo "  starting ngrok tunnel for :$FRONT_PORT ..."
  cd "$ROOT"
  nohup ngrok http "$FRONT_PORT" --url "$NGROK_URL" > "$LOG/ngrok.log" 2>&1 &
  echo $! > "$LOG/ngrok.pid"
  for i in $(seq 1 20); do
    if curl -sf "http://127.0.0.1:4040/api/tunnels" >/dev/null 2>&1; then
      echo "  ngrok ready ✓  -> $NGROK_URL  (log: $LOG/ngrok.log)"
      return 0
    fi
    sleep 0.5
  done
  echo "  ngrok FAILED to start; see $LOG/ngrok.log"
  return 1
}

case "${1:-start}" in
  start)
    start_backend && start_frontend && start_ngrok
    echo ""
    echo "all up — open $NGROK_URL on your phone"
    ;;
  stop)
    stop_all
    ;;
  status)
    status_all
    ;;
  restart)
    stop_all
    "$0" start
    ;;
  *)
    echo "usage: $0 [start|stop|status|restart]"
    exit 1
    ;;
esac