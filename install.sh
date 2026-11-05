#!/usr/bin/env bash
# Visenlo — One-shot local installer & launcher for Linux / macOS
# Usage:
#   bash install.sh [--port 7860] [--host 127.0.0.1]
#   OR hosted: curl -fsSL https://YOUR_WEBSITE/install.sh | bash
# What it does completely locally:
#   1. Checks Python 3.10+ exists (or installs via guidance)
#   2. Creates .venv, installs pip deps from requirements.txt
#   3. Checks local models and displays explicit setup instructions
#   4. Runs health check
#   5. Launches Gradio on 127.0.0.1:7860 (fully local, no data leaves device)

set -euo pipefail

# Where to get the full project if user just ran: curl .../install.sh | bash (no files yet)
REPO_URL="${VISENLO_REPO:-${REACTORX_REPO:-https://github.com/Mevinb/Visenlo.git}}"
REPO_ZIP="${VISENLO_ZIP:-${REACTORX_ZIP:-https://github.com/Mevinb/Visenlo/archive/refs/heads/main.zip}}"

# Fix: BASH_SOURCE is unbound when piped (curl | bash) under set -u — use :- fallback
_SRC="${BASH_SOURCE[0]:-${0:-}}"
if [[ "$_SRC" == "bash" ]] || [[ "$_SRC" == "-bash" ]] || [[ -z "$_SRC" ]]; then
  ROOT="$PWD"
else
  ROOT="$(cd "$(dirname "$_SRC")" 2>/dev/null && pwd || echo "$PWD")"
fi
# If piped via curl or ROOT doesn't contain project, fetch it
if [[ ! -f "$ROOT/app.py" ]]; then
  if [[ -f "$PWD/app.py" ]]; then
    ROOT="$PWD"
  elif [[ -f "./Visenlo/app.py" ]]; then
    ROOT="$PWD/Visenlo"
  else
    echo "[Visenlo] Project files not found — fetching from $REPO_URL ..."
    if command -v git >/dev/null 2>&1; then
      if [[ -d "./Visenlo/.git" ]]; then
        echo "[Visenlo] Updating existing Visenlo/ ..."
        git -C "./Visenlo" pull --ff-only || true
        ROOT="$PWD/Visenlo"
      else
        git clone "$REPO_URL" Visenlo
        ROOT="$PWD/Visenlo"
      fi
    elif command -v curl >/dev/null 2>&1; then
      echo "[Visenlo] git not found, downloading zip via curl ..."
      curl -L -o /tmp/visenlo.zip "$REPO_ZIP"
      if command -v unzip >/dev/null 2>&1; then
        unzip -q -o /tmp/visenlo.zip -d /tmp
        # zip extracts to Visenlo-main
        FOUND="$(find /tmp -maxdepth 2 -name "app.py" -type f | head -1)"
        if [[ -n "$FOUND" ]]; then
          DIR="$(dirname "$FOUND")"
          mkdir -p ./Visenlo
          cp -r "$DIR"/* ./Visenlo/
          ROOT="$PWD/Visenlo"
        fi
      else
        echo "[Visenlo] Please install git or unzip, or manually: git clone $REPO_URL" >&2
        exit 1
      fi
    else
      echo "[Visenlo] No git/curl found. Install git and run: git clone $REPO_URL && cd Visenlo && ./install.sh" >&2
      exit 1
    fi
  fi
fi
echo "[Visenlo] Using project at: $ROOT"

# Use the same launcher for installation and subsequent offline starts.
if [[ ! -f "$ROOT/launcher.py" ]]; then
  printf '%s\n' 'Visenlo project files are incomplete: launcher.py is missing.' >&2
  exit 1
fi
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  exec "$ROOT/.venv/bin/python" "$ROOT/launcher.py" "$@"
fi
for candidate in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3,10))' 2>/dev/null; then
    exec "$candidate" "$ROOT/launcher.py" "$@"
  fi
done
printf '%s\n' 'Visenlo requires Python 3.10 or newer. Install Python, then run this script again.' >&2
exit 1
