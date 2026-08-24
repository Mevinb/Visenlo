#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -d "$ROOT/.venv" ]]; then
  python3 -m venv "$ROOT/.venv"
fi
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "ReactorX could not create its virtual environment" >&2
  exit 1
fi
DEFAULT_PORT="${GRADIO_SERVER_PORT:-7860}"
HOST="127.0.0.1"
REQUESTED_PORT=""
APP_ARGS=()

pick_port() {
  "$ROOT/.venv/bin/python" - "$1" "$2" <<'PY'
import socket
import sys

host = sys.argv[1]
start = int(sys.argv[2])
for port in range(start, start + 50):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            continue
        print(port)
        raise SystemExit(0)
raise SystemExit(f"ReactorX could not find a free port near {start}")
PY
}

while (($#)); do
  case "$1" in
    --host)
      HOST="${2:?Missing value for --host}"
      shift 2
      ;;
    --host=*)
      HOST="${1#*=}"
      shift
      ;;
    --port)
      REQUESTED_PORT="${2:?Missing value for --port}"
      shift 2
      ;;
    --port=*)
      REQUESTED_PORT="${1#*=}"
      shift
      ;;
    --share)
      APP_ARGS+=("--share")
      shift
      ;;
    *)
      APP_ARGS+=("$1")
      shift
      ;;
  esac
done

START_PORT="${REQUESTED_PORT:-$DEFAULT_PORT}"
PORT="$START_PORT"
if ! "$ROOT/.venv/bin/python" - "$HOST" "$START_PORT" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
PY
then
  PORT="$(pick_port "$HOST" "$((START_PORT + 1))")"
  printf 'ReactorX: port %s is busy, using %s instead\n' "$START_PORT" "$PORT" >&2
fi
# The CPU and GPU onnxruntime wheels and both opencv wheels install the same
# import packages and collide. Keep only the flavors requirements.txt wants.
"$ROOT/.venv/bin/python" -m pip uninstall -y onnxruntime opencv-python >/dev/null 2>&1 || true
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"
exec "$ROOT/.venv/bin/python" "$ROOT/app.py" --host "$HOST" --port "$PORT" "${APP_ARGS[@]}"
