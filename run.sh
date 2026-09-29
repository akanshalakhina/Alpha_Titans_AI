#!/usr/bin/env bash
# Alpha Titans — Content Engine
# Starts the API and the UI together on one port. Run from anywhere:
#     ./run.sh                 # http://127.0.0.1:8000
#     AT_PORT=9000 ./run.sh
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_ROOT"

PY="${PYTHON:-python3}"

# --- preflight --------------------------------------------------------------
command -v "$PY" >/dev/null 2>&1 || { echo "✗ $PY not found"; exit 1; }

for bin in ffmpeg ffprobe; do
  command -v "$bin" >/dev/null 2>&1 || {
    echo "✗ $bin is not on PATH — the pipeline cannot run without it."
    echo "  Debian/Ubuntu: sudo apt install ffmpeg"
    echo "  macOS:         brew install ffmpeg"
    echo "  Windows:       winget install Gyan.FFmpeg"
    exit 1
  }
done

"$PY" - <<'EOF' || { echo "  → pip install -r requirements.txt"; exit 1; }
import importlib.util, sys
missing = [m for m in ("fastapi", "uvicorn", "multipart")
           if importlib.util.find_spec(m) is None]
if missing:
    print("✗ Python dependencies missing: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
EOF

mkdir -p uploads outputs data/music

HOST="${AT_HOST:-127.0.0.1}"
PORT="${AT_PORT:-8000}"

echo "▸ Alpha Titans — Content Engine"
echo "  open http://${HOST}:${PORT}"
echo

# --app-dir puts server/ on sys.path without needing to cd into it, so every
# path in the app stays relative to this folder.
exec "$PY" -m uvicorn main:app --app-dir "$APP_ROOT/server" \
     --host "$HOST" --port "$PORT" "$@"
