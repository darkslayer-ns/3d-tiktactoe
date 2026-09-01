#!/usr/bin/env bash
# Re-sync the repo from WSL to a native Windows folder so Android Studio on
# Windows can build it without the \\wsl.localhost mount permission issues.
#
# Usage:
#   ./scripts/sync-to-windows.sh            # default target C:\Users\nikol\3d_tiktactoe
#   DEST=/mnt/c/Users/other ./scripts/sync-to-windows.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-/mnt/c/Users/nikol/3d_tiktactoe}"

mkdir -p "$DEST"
rsync -a --delete \
  --exclude 'node_modules' \
  --exclude '.git' \
  --exclude 'cpp/build' \
  --exclude 'cpp/fixtures' \
  --exclude '.logs' \
  --exclude 'dist' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '*.log' \
  --exclude 'android/.gradle' \
  --exclude 'android/app/build' \
  "$ROOT/" "$DEST/"

echo "synced -> $DEST"
echo "then on Windows:"
echo "  cd C:\\Users\\nikol\\3d_tiktactoe\\mobile-rn && npm install"
echo "  open C:\\Users\\nikol\\3d_tiktactoe\\mobile-rn\\android in Android Studio"