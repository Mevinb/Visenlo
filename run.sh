#!/usr/bin/env bash
set -euo pipefail

# Keep setup, offline reuse, port selection and argument handling in one launcher.
VISENLO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$VISENLO_ROOT/.venv/bin/python" ]]; then
  exec "$VISENLO_ROOT/.venv/bin/python" "$VISENLO_ROOT/launcher.py" "$@"
fi
for candidate in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3,10))' 2>/dev/null; then
    exec "$candidate" "$VISENLO_ROOT/launcher.py" "$@"
  fi
done
printf '%s\n' 'Visenlo requires Python 3.10 or newer. Install Python, then run this script again.' >&2
exit 1
