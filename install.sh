#!/usr/bin/env bash
# ReactorX — One-shot local installer & launcher for Linux / macOS
# Usage:
#   bash install.sh [--port 7860] [--host 127.0.0.1]
#   OR hosted: curl -fsSL https://YOUR_WEBSITE/install.sh | bash
# What it does completely locally:
#   1. Checks Python 3.10+ exists (or installs via guidance)
#   2. Creates .venv, installs pip deps from requirements.txt
#   3. Verifies / downloads models if needed (buffalo_l auto, others via helper)
#   4. Runs health check
#   5. Launches Gradio on 127.0.0.1:7860 (fully local, no data leaves device)

set -euo pipefail

# Where to get the full project if user just ran: curl .../install.sh | bash (no files yet)
REPO_URL="${REACTORX_REPO:-https://github.com/Mevinb/Reactor-X.git}"
REPO_ZIP="${REACTORX_ZIP:-https://github.com/Mevinb/Reactor-X/archive/refs/heads/main.zip}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")"
# If piped via curl or ROOT doesn't contain project, fetch it
if [[ ! -f "$ROOT/app.py" ]]; then
  if [[ -f "$PWD/app.py" ]]; then
    ROOT="$PWD"
  elif [[ -f "./ReactorX/app.py" ]]; then
    ROOT="$PWD/ReactorX"
  else
    echo "[ReactorX] Project files not found — fetching from $REPO_URL ..."
    if command -v git >/dev/null 2>&1; then
      if [[ -d "./ReactorX/.git" ]]; then
        echo "[ReactorX] Updating existing ReactorX/ ..."
        git -C "./ReactorX" pull --ff-only || true
        ROOT="$PWD/ReactorX"
      else
        git clone "$REPO_URL" ReactorX
        ROOT="$PWD/ReactorX"
      fi
    elif command -v curl >/dev/null 2>&1; then
      echo "[ReactorX] git not found, downloading zip via curl ..."
      curl -L -o /tmp/reactorx.zip "$REPO_ZIP"
      if command -v unzip >/dev/null 2>&1; then
        unzip -q -o /tmp/reactorx.zip -d /tmp
        # zip extracts to ReactorX-main
        FOUND="$(find /tmp -maxdepth 2 -name "app.py" -type f | head -1)"
        if [[ -n "$FOUND" ]]; then
          DIR="$(dirname "$FOUND")"
          mkdir -p ./ReactorX
          cp -r "$DIR"/* ./ReactorX/
          ROOT="$PWD/ReactorX"
        fi
      else
        echo "[ReactorX] Please install git or unzip, or manually: git clone $REPO_URL" >&2
        exit 1
      fi
    else
      echo "[ReactorX] No git/curl found. Install git and run: git clone $REPO_URL && cd ReactorX && ./install.sh" >&2
      exit 1
    fi
  fi
fi
echo "[ReactorX] Using project at: $ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${CYAN}[ReactorX]${NC} $*"; }
ok()    { echo -e "${GREEN}[ok]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
fail()  { echo -e "${RED}[fail]${NC} $*"; }

# --- 1. Python check ---
find_python() {
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
      ver=$("$c" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
      major=${ver%%.*}; minor=${ver##*.}
      if [[ "$major" -eq 3 && "$minor" -ge 10 ]] || [[ "$major" -gt 3 ]]; then
        echo "$c"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON=""
if ! PYTHON=$(find_python); then
  fail "Python 3.10+ is required but not found."
  echo "  Install Python 3.10+ from https://www.python.org/downloads/"
  echo "  macOS: brew install python@3.11"
  echo "  Ubuntu/Debian: sudo apt update && sudo apt install python3.11 python3.11-venv python3-pip"
  exit 1
fi
PY_VER=$($PYTHON --version 2>&1)
ok "Found $PY_VER ($PYTHON)"

# Check venv module
if ! $PYTHON -m venv --help >/dev/null 2>&1; then
  fail "python3-venv not installed."
  echo "  Ubuntu/Debian: sudo apt install python3.11-venv"
  echo "  Fedora: sudo dnf install python3-venv"
  exit 1
fi

# --- 2. Create .venv ---
VENV="$ROOT/.venv"
if [[ ! -d "$VENV" ]]; then
  info "Creating virtual environment at $VENV ..."
  $PYTHON -m venv "$VENV"
  ok "Virtual environment created"
else
  ok "Using existing venv at $VENV"
fi

PIP="$VENV/bin/python -m pip"
PYBIN="$VENV/bin/python"

# Upgrade pip quietly
info "Installing dependencies (this may take 2-5 minutes first time)..."
$PYBIN -m pip install --upgrade pip -q
# Remove conflicting wheels if present
$PYBIN -m pip uninstall -y onnxruntime opencv-python >/dev/null 2>&1 || true
$PYBIN -m pip install -r "$ROOT/requirements.txt"

ok "Dependencies installed"

# --- 3. System info ---
info "System check:"
$PYBIN -c "
import platform, sys
print(f'  OS: {platform.system()} {platform.release()}')
print(f'  Python: {sys.version.split()[0]}')
try:
    import onnxruntime as ort
    print(f'  ONNXRuntime: {ort.__version__} providers={ort.get_available_providers()}')
except: print('  ONNXRuntime: not available')
try:
    import cv2; print(f'  OpenCV: {cv2.__version__}')
except: pass
try:
    import torch; print(f'  PyTorch: {torch.__version__} cuda={torch.cuda.is_available()}')
except: pass
"

# --- 4. Models ---
info "Checking models in $ROOT/models ..."
if [[ -f "$ROOT/scripts/download_models.py" ]]; then
  $PYBIN "$ROOT/scripts/download_models.py" --check || true
else
  # Minimal check
  for f in "insightface/models/buffalo_l/det_10g.onnx" "inswapper_128.onnx"; do
    if [[ -f "$ROOT/models/$f" ]]; then ok "found models/$f"
    else warn "missing models/$f (will auto-download or see README)"
    fi
  done
fi

# --- 5. Self-check ---
if [[ -f "$ROOT/scripts/selfcheck.py" ]]; then
  info "Running self-check..."
  if $PYBIN "$ROOT/scripts/selfcheck.py"; then
    ok "Self-check passed"
  else
    warn "Self-check reported failures (optional models may be missing, core still works)"
  fi
fi

# --- 6. Launch ---
info "Starting ReactorX locally..."
echo ""
echo -e "${GREEN}  ReactorX will open at http://127.0.0.1:7860${NC}"
echo -e "  All processing stays 100% on your device. No images leave your machine."
echo -e "  Press Ctrl+C to stop."
echo ""

# Pass through args to run.sh / app.py
EXTRA_ARGS=("$@")
if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
  EXTRA_ARGS=()
fi

# Prefer run.sh if present for port logic
if [[ -x "$ROOT/run.sh" ]]; then
  exec "$ROOT/run.sh" "${EXTRA_ARGS[@]}"
else
  exec "$PYBIN" "$ROOT/app.py" --host 127.0.0.1 --port 7860 "${EXTRA_ARGS[@]}"
fi
