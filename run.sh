#!/usr/bin/env bash
set -euo pipefail

# ReactorX local launcher — fully automatic & local
# - Creates .venv if missing
# - Installs requirements (idempotent)
# - Picks free port, launches Gradio on 127.0.0.1 by default
# - All compute stays on-device; no external service needed after install
# Usage: ./run.sh [--host 127.0.0.1] [--port 7860] [--share]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Python discovery (3.10+) ---
find_python() {
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
      if "$c" -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
        echo "$c"; return 0
      fi
    fi
  done
  return 1
}

if [[ ! -d "$ROOT/.venv" ]]; then
  PY=""
  if ! PY=$(find_python); then
    echo "ReactorX: Python 3.10+ not found. Install from https://www.python.org/downloads/" >&2
    echo "  Ubuntu: sudo apt install python3.11 python3.11-venv" >&2
    echo "  macOS:  brew install python@3.11" >&2
    exit 1
  fi
  echo "[ReactorX] Creating virtual environment with $PY ..."
  "$PY" -m venv "$ROOT/.venv"
fi
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "ReactorX could not create its virtual environment" >&2
  exit 1
fi

DEFAULT_PORT="${GRADIO_SERVER_PORT:-7860}"
HOST="127.0.0.1"
REQUESTED_PORT=""
APP_ARGS=()
SHOW_HELP=false

pick_port() {
  "$ROOT/.venv/bin/python" - "$1" "$2" <<'PY'
import socket, sys
host = sys.argv[1]
start = int(sys.argv[2])
for port in range(start, start + 50):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try: sock.bind((host, port))
        except OSError: continue
        print(port); raise SystemExit(0)
raise SystemExit(f"ReactorX could not find a free port near {start}")
PY
}

while (($#)); do
  case "$1" in
    --host) HOST="${2:?Missing value for --host}"; shift 2 ;;
    --host=*) HOST="${1#*=}"; shift ;;
    --port) REQUESTED_PORT="${2:?Missing value for --port}"; shift 2 ;;
    --port=*) REQUESTED_PORT="${1#*=}"; shift ;;
    --share) APP_ARGS+=("--share"); shift ;;
    --help|-h) SHOW_HELP=true; APP_ARGS+=("$1"); shift ;;
    *) APP_ARGS+=("$1"); shift ;;
  esac
done

if [[ "$SHOW_HELP" == true ]]; then
  echo "ReactorX — 100% local face-swap engine"
  echo "Usage: ./run.sh [--host IP] [--port PORT] [--share]"
  echo "  --host  Bind address (default 127.0.0.1, use 0.0.0.0 for LAN)"
  echo "  --port  Preferred port (default 7860, auto-increments if busy)"
  echo "  --share Create a temporary public Gradio link (optional)"
  echo ""
fi

START_PORT="${REQUESTED_PORT:-$DEFAULT_PORT}"
PORT="$START_PORT"
if ! "$ROOT/.venv/bin/python" - "$HOST" "$START_PORT" <<'PY' >/dev/null 2>&1
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
PY
then
  PORT="$(pick_port "$HOST" "$((START_PORT + 1))")"
  printf 'ReactorX: port %s is busy, using %s instead\n' "$START_PORT" "$PORT" >&2
fi

# Device-aware deps (GPU vs CPU)
has_nvidia_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then return 0; fi
  if command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -qi nvidia; then return 0; fi
  if [[ -f /proc/driver/nvidia/version ]]; then return 0; fi
  return 1
}
if has_nvidia_gpu; then
  "$ROOT/.venv/bin/python" -m pip uninstall -y onnxruntime opencv-python >/dev/null 2>&1 || true
  if ! "$ROOT/.venv/bin/python" -m pip install -q -r "$ROOT/requirements.txt" 2>&1; then
    echo "[ReactorX] GPU install failed, falling back to CPU" >&2
    sed 's/onnxruntime-gpu/onnxruntime/' "$ROOT/requirements.txt" > /tmp/req_cpu.txt
    "$ROOT/.venv/bin/python" -m pip install -q -r /tmp/req_cpu.txt
  fi
else
  "$ROOT/.venv/bin/python" -m pip uninstall -y onnxruntime onnxruntime-gpu opencv-python >/dev/null 2>&1 || true
  sed 's/onnxruntime-gpu/onnxruntime/' "$ROOT/requirements.txt" > /tmp/req_cpu.txt
  "$ROOT/.venv/bin/python" -m pip install -q -r /tmp/req_cpu.txt
fi

# Friendly banner
echo ""
echo "  ReactorX running 100% locally"
echo "  URL: http://${HOST}:${PORT}  (no data leaves your device)"
echo "  Models: ${REACTORX_MODELS:-$ROOT/models}"
echo "  Outputs: $ROOT/outputs"
echo ""

exec "$ROOT/.venv/bin/python" "$ROOT/app.py" --host "$HOST" --port "$PORT" "${APP_ARGS[@]}"
