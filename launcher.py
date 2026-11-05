#!/usr/bin/env python3
"""
Visenlo cross-platform bootstrap launcher.

- Works on Windows / macOS / Linux with a single command:
    python launcher.py
  or
    python launcher.py --port 7861 --host 127.0.0.1

- Fully automatic:
  * Checks Python 3.10+
  * Creates .venv if missing
  * Installs requirements.txt (idempotent)
  * Verifies models, runs self-check
  * Finds free port and launches Gradio 100% locally

All heavy compute stays on-device; no data leaves the machine.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
IS_WIN = sys.platform.startswith("win")
VENV_PY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")
REQ = ROOT / "requirements.txt"          # CPU runtime
REQ_GPU = ROOT / "requirements-gpu.txt"  # NVIDIA runtime (same packages + onnxruntime-gpu)
REQ_BASE = ROOT / "requirements-base.txt"
APP = ROOT / "app.py"

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
NC = "\033[0m"


def info(msg): print(f"{CYAN}[Visenlo]{NC} {msg}")
def ok(msg): print(f"{GREEN}[ok]{NC} {msg}")
def warn(msg): print(f"{YELLOW}[warn]{NC} {msg}")
def fail(msg): print(f"{RED}[fail]{NC} {msg}")


def check_python():
    if sys.version_info < (3, 10):
        fail(f"Python 3.10+ required, found {sys.version.split()[0]}")
        print("  Install from https://www.python.org/downloads/")
        sys.exit(1)
    ok(f"Python {sys.version.split()[0]}")


def ensure_venv():
    if not VENV_PY.exists():
        info(f"Creating virtual environment at {VENV} ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
        ok("Virtual environment created")
    else:
        ok(f"Using existing venv: {VENV}")
    if not VENV_PY.exists():
        fail("Could not create virtual environment")
        sys.exit(1)


def has_nvidia_gpu() -> bool:
    import shutil, pathlib
    # nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            subprocess.check_call(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except: pass
    # Windows nvidia-smi location
    if IS_WIN and pathlib.Path("C:/Windows/System32/nvidia-smi.exe").exists():
        return True
    # lspci
    if shutil.which("lspci"):
        try:
            out = subprocess.check_output(["lspci"], text=True, stderr=subprocess.DEVNULL)
            if "nvidia" in out.lower(): return True
        except: pass
    if pathlib.Path("/proc/driver/nvidia/version").exists():
        return True
    return False

def dependencies_ready() -> bool:
    """Keep an already-working install usable without network access.

    Only the shared package set is version-checked; either onnxruntime build
    (CPU or GPU) counts as satisfying the runtime requirement.
    """
    code = '''
import importlib
from importlib.metadata import version
from packaging.requirements import Requirement
from pathlib import Path
for module in ("numpy", "cv2", "skimage", "PIL", "gradio", "insightface", "onnxruntime", "onnx"):
    importlib.import_module(module)
for line in Path(__import__("sys").argv[1]).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("-r"):
        continue
    req = Requirement(line)
    if req.marker and not req.marker.evaluate():
        continue
    installed = version(req.name)
    if installed not in req.specifier:
        raise RuntimeError(f"{req.name} {installed} does not satisfy {req.specifier}")
'''
    base = REQ_BASE if REQ_BASE.exists() else REQ
    return subprocess.call([str(VENV_PY), "-c", code, str(base)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def pip_install():
    info("Installing dependencies (2-5 min first time) ...")
    subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "-q"])
    gpu = has_nvidia_gpu() and REQ_GPU.exists()
    if gpu:
        info("  NVIDIA GPU detected — installing requirements-gpu.txt (falls back to CPU if CUDA libs are missing)...")
        try:
            subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "-q", "-r", str(REQ_GPU)])
        except subprocess.CalledProcessError:
            warn("GPU install failed, falling back to the CPU runtime")
            subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "-q", "-r", str(REQ)])
        # verify CUDA actually usable
        ret = subprocess.call([str(VENV_PY), "-c", "import onnxruntime as ort; exit(0 if 'CUDAExecutionProvider' in ort.get_available_providers() else 1)"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if ret != 0:
            warn("CUDA provider not usable (missing libcublas) — pipeline will use CPU (still works)")
        else:
            ok("CUDA provider available")
    else:
        info("  No NVIDIA GPU detected — installing the CPU runtime (requirements.txt)...")
        subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "-q", "-r", str(REQ)])
    ok("Dependencies installed")


def find_free_port(host: str, start: int) -> int:
    for port in range(start, min(start + 50, 65536)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    fail(f"Could not find free port near {start}")
    sys.exit(1)


def system_check():
    info("System check:")
    code = """
import platform, sys
print(f"  OS: {platform.system()} {platform.release()}")
print(f"  Python: {sys.version.split()[0]}")
try:
 import onnxruntime as ort
 print(f"  ONNXRuntime: {ort.__version__} providers={ort.get_available_providers()}")
except Exception as e: print(f"  ONNXRuntime: {e}")
try: import cv2; print(f"  OpenCV: {cv2.__version__}")
except: pass
"""
    subprocess.call([str(VENV_PY), "-c", code])


def model_check():
    helper = ROOT / "scripts" / "download_models.py"
    if not helper.exists():
        warn("Model checker is missing; use the application's Model setup panel")
        return
    ret = subprocess.call([str(VENV_PY), str(helper), "--check"])
    if ret:
        warn("Required local models are missing — open Model setup for instructions")
        print("  Models are never downloaded automatically during a swap.")
        print("  In the local app, open Model setup, review the model licenses,")
        print("  acknowledge the terms, and choose Install required models.")
    else:
        ok("Required local models ready")


def selfcheck():
    sc = ROOT / "scripts" / "selfcheck.py"
    if sc.exists():
        info("Running self-check ...")
        ret = subprocess.call([str(VENV_PY), str(sc)])
        if ret == 0: ok("Self-check passed")
        else: warn("Self-check had failures (optional models may be missing)")


def main():
    parser = argparse.ArgumentParser(description="Visenlo local launcher (100% on-device)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1, use 0.0.0.0 for LAN)")
    parser.add_argument("--port", type=int, default=7860, help="Preferred port (default 7860)")
    parser.add_argument("--share", action="store_true", help="Create temporary public Gradio link")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install (use existing venv as-is)")
    parser.add_argument("--reinstall", action="store_true", help="Re-run dependency installation even when the current environment is ready")
    parser.add_argument("--skip-check", action="store_true", help="Skip startup system and model checks")
    parser.add_argument("--self-check", action="store_true", help="Run the extended diagnostic suite before launch")
    parser.add_argument("--no-browser", action="store_true", help="Start the server without opening a browser")
    parser.add_argument("--auth", help="user:password login for the studio (use with --host 0.0.0.0 or --share)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    print(f"\n{CYAN}Visenlo — 100% local face-swap engine{NC}")
    print(f"  All processing stays on your device. No images leave your machine.\n")

    check_python()
    ensure_venv()
    if not args.skip_install:
        if not args.reinstall and dependencies_ready():
            ok("Existing dependencies ready — no download or reinstall needed")
        else:
            pip_install()
    if not args.skip_check:
        system_check()
        model_check()
    if args.self_check:
        selfcheck()

    port = find_free_port(args.host, args.port)
    if port != args.port:
        warn(f"Port {args.port} busy, using {port} instead")

    info(f"Starting Visenlo at http://{args.host}:{port}")
    print(f"  Models: {os.environ.get('VISENLO_MODELS', os.environ.get('REACTORX_MODELS', str(ROOT / 'models')))}")
    print(f"  Outputs: {ROOT / 'outputs'}\n")

    cmd = [str(VENV_PY), str(APP), "--host", args.host, "--port", str(port)]
    if args.share:
        cmd.append("--share")
    if args.no_browser:
        cmd.append("--no-browser")
    if args.auth:
        cmd += ["--auth", args.auth]
    # Preserve startup instructions even when stdout is piped.
    sys.stdout.flush()
    sys.stderr.flush()
    # Replace current process
    os.execv(str(VENV_PY), cmd)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        fail(f"Setup did not complete (exit {exc.returncode}). See the installation error above.")
        print("  Use Python 3.11 or 3.12 and install the C++ build tools required by InsightFace.")
        print("  Keep this terminal open, fix the reported issue, then run the same run script again.")
        sys.exit(exc.returncode or 1)
    except KeyboardInterrupt:
        print("\nVisenlo stopped.")
        sys.exit(130)
