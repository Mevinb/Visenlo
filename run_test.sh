#!/usr/bin/env bash
set -e
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Automatically find and export NVIDIA library path for RTX GPU acceleration
NV_PATHS=$(find "$ROOT/.venv/lib/python*/site-packages/nvidia" -type d -name "lib" 2>/dev/null | tr "\n" ":")
if [[ -n "$NV_PATHS" ]]; then
    export LD_LIBRARY_PATH="${NV_PATHS}:${LD_LIBRARY_PATH:-}"
fi

exec "$ROOT/.venv/bin/python" "$ROOT/scripts/run_model_tests.py" "$@"
