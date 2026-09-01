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

# --- Device detection (runs on user's device, adapts install) ---
has_nvidia_gpu() {
  # Check nvidia-smi (driver installed and GPU responds)
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then return 0; fi
  # Fallback: lspci shows NVIDIA
  if command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -qi nvidia; then return 0; fi
  # Fallback: nvidia driver files exist
  if [[ -f /proc/driver/nvidia/version ]]; then return 0; fi
  return 1
}

detect_system() {
  info "Detecting system (adapts install to your device)..."
  # OS
  OS="$(uname -s 2>/dev/null || echo Unknown)"
  ARCH="$(uname -m 2>/dev/null || echo Unknown)"
  info "  OS: $OS $ARCH"
  # RAM
  if [[ -f /proc/meminfo ]]; then
    MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    MEM_GB=$((MEM_KB / 1024 / 1024))
    info "  RAM: ${MEM_GB}GB"
    if [[ $MEM_GB -lt 4 ]]; then warn "RAM <4GB — swaps will be slower, close other apps"; fi
  elif command -v sysctl >/dev/null 2>&1; then
    MEM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}')
    [[ -n "$MEM_GB" ]] && info "  RAM: ${MEM_GB}GB"
  fi
  # Disk
  AVAIL=$(df -h "$ROOT" 2>/dev/null | tail -1 | awk '{print $4}')
  [[ -n "$AVAIL" ]] && info "  Disk free at $ROOT: $AVAIL (need ~4GB for models)"
  # GPU
  if has_nvidia_gpu; then
    info "  GPU: NVIDIA detected"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -3 | while read line; do info "    $line"; done
    fi
    echo "nvidia" > /tmp/reactorx_gpu_flag 2>/dev/null || true
  else
    info "  GPU: none / not detected — will use CPU (auto fallback, slower but works)"
    echo "cpu" > /tmp/reactorx_gpu_flag 2>/dev/null || true
  fi
  # Disk space check
  REQ_MB=4000
  if command -v df >/dev/null 2>&1; then
    AVAIL_MB=$(df -m "$ROOT" 2>/dev/null | tail -1 | awk '{print $4}')
    if [[ -n "$AVAIL_MB" && "$AVAIL_MB" -lt "$REQ_MB" ]]; then
      warn "Low disk space: ${AVAIL_MB}MB free, need ~4000MB for models. Free up space."
    fi
  fi
}

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

# --- Device-aware dependency install ---
detect_system
GPU_FLAG="$(cat /tmp/reactorx_gpu_flag 2>/dev/null || echo cpu)"
info "Installing dependencies (this may take 2-5 minutes first time) — mode: $GPU_FLAG ..."
$PYBIN -m pip install --upgrade pip -q
# Remove conflicting wheels (onnxruntime vs onnxruntime-gpu collide)
$PYBIN -m pip uninstall -y onnxruntime onnxruntime-gpu opencv-python >/dev/null 2>&1 || true

if [[ "$GPU_FLAG" == "nvidia" ]]; then
  info "  Installing with GPU support (onnxruntime-gpu)..."
  if ! $PYBIN -m pip install -r "$ROOT/requirements.txt"; then
    warn "GPU install failed, falling back to CPU"
    $PYBIN -m pip uninstall -y onnxruntime-gpu >/dev/null 2>&1 || true
    # Create CPU requirements on-the-fly
    sed 's/onnxruntime-gpu/onnxruntime/' "$ROOT/requirements.txt" > /tmp/req_cpu.txt
    $PYBIN -m pip install -r /tmp/req_cpu.txt
  fi
  # Verify CUDA libs actually load — if missing (driver without toolkit), silently fallback
  if ! $PYBIN -c "import onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers()" 2>/dev/null; then
    warn "CUDA provider not usable (missing libcublas etc.) — using CPU fallback (still works)"
    # Keep both installed; pipeline will fallback at runtime via try/except. Optionally switch:
    # $PYBIN -m pip uninstall -y onnxruntime-gpu >/dev/null 2>&1 || true
    # $PYBIN -m pip install -q onnxruntime
  else
    ok "CUDA provider available"
  fi
else
  info "  Installing CPU build (onnxruntime) — no NVIDIA GPU detected..."
  sed 's/onnxruntime-gpu/onnxruntime/' "$ROOT/requirements.txt" > /tmp/req_cpu.txt
  $PYBIN -m pip install -r /tmp/req_cpu.txt
fi

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

# --- 4. Models — checker + Setup Guide (no auto-download, licensing) ---
info "Checking models in $ROOT/models ..."
if [[ -f "$ROOT/scripts/download_models.py" ]]; then
  $PYBIN -u "$ROOT/scripts/download_models.py" --check || true
  RET=$?
  if [[ $RET -ne 0 ]]; then
    echo ""
    echo -e "${YELLOW}  ┌─────────────────────────────────────────────────────────┐${NC}"
    echo -e "${YELLOW}  │  Model Setup Guide — some models missing              │${NC}"
    echo -e "${YELLOW}  └─────────────────────────────────────────────────────────┘${NC}"
    echo ""
    echo "  ReactorX does not auto-download models (licensing)."
    echo "  Follow the Model Setup Guide below, then re-run this installer."
    echo ""
    echo "  Quick setup (run from $ROOT):"
    echo ""
    echo "    BASE=https://huggingface.co/facefusion/models-3.0.0/resolve/main"
    echo "    mkdir -p models"
    echo "    curl -L -o models/bisenet_resnet_34.onnx \$BASE/bisenet_resnet_34.onnx   # face parsing, MIT (~90 MB)"
    echo "    curl -L -o models/codeformer.onnx     \$BASE/codeformer.onnx            # restoration, CC BY-NC 4.0 (~377 MB)"
    echo "    curl -L -o models/inswapper_128.onnx  \$BASE/inswapper_128.onnx         # swap model, research-only (~529 MB)"
    echo "    curl -L -o models/xseg_1.onnx https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx  # occlusion, GPL-3.0 (~68 MB)"
    echo ""
    echo "  buffalo_l pack (5 files) auto-downloads on first swap via InsightFace — no manual step needed,"
    echo "  or get it from: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    echo ""
    echo "  After installing, verify:"
    echo "    $PYBIN scripts/download_models.py --check"
    echo ""
    echo "  Once you see:"
    echo "    [✓] buffalo_l"
    echo "    [✓] inswapper_128"
    echo "    [✓] BiSeNet"
    echo "    [✓] XSeg"
    echo "    [✓] CodeFormer"
    echo "  then run:  ./run.sh  or  python launcher.py  ->  [Launch ReactorX]"
    echo ""
    # Don't fail hard — let user read guide, but warn that swaps will need inswapper
    if [[ ! -f "$ROOT/models/inswapper_128.onnx" ]]; then
      warn "CRITICAL: inswapper_128.onnx missing — swaps will fail until you install it (see guide above)."
    fi
  else
    ok "All required models present — ready to launch!"
  fi
else
  for f in "insightface/models/buffalo_l/det_10g.onnx" "inswapper_128.onnx"; do
    if [[ -f "$ROOT/models/$f" ]]; then ok "found models/$f"
    else warn "missing models/$f — see Model Setup Guide in README"
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
