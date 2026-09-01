#!/usr/bin/env python3
"""
ReactorX cross-platform bootstrap launcher.

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
REQ = ROOT / "requirements.txt"
APP = ROOT / "app.py"

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
NC = "\033[0m"


def info(msg): print(f"{CYAN}[ReactorX]{NC} {msg}")
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

def pip_install():
    info("Installing dependencies (2-5 min first time) ...")
    # remove conflicting wheels
    subprocess.call([str(VENV_PY), "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu", "opencv-python"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "-q"])
    gpu = has_nvidia_gpu()
    if gpu:
        info("  GPU detected — installing onnxruntime-gpu (will fallback to CPU if CUDA libs missing)...")
        try:
            subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "-q", "-r", str(REQ)])
        except subprocess.CalledProcessError:
            warn("GPU install failed, falling back to CPU")
            # rewrite req to cpu
            txt = REQ.read_text().replace("onnxruntime-gpu", "onnxruntime")
            tmp = ROOT / ".tmp_req_cpu.txt"
            tmp.write_text(txt)
            subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "-q", "-r", str(tmp)])
            tmp.unlink(missing_ok=True)
        # verify CUDA actually usable
        ret = subprocess.call([str(VENV_PY), "-c", "import onnxruntime as ort; exit(0 if 'CUDAExecutionProvider' in ort.get_available_providers() else 1)"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if ret != 0:
            warn("CUDA provider not usable (missing libcublas) — pipeline will use CPU (still works)")
        else:
            ok("CUDA provider available")
    else:
        info("  No GPU detected — installing CPU build (onnxruntime)...")
        txt = REQ.read_text().replace("onnxruntime-gpu", "onnxruntime")
        tmp = ROOT / ".tmp_req_cpu.txt"
        tmp.write_text(txt)
        subprocess.check_call([str(VENV_PY), "-m", "pip", "install", "-q", "-r", str(tmp)])
        tmp.unlink(missing_ok=True)
    ok("Dependencies installed")


def find_free_port(host: str, start: int) -> int:
    for port in range(start, start + 50):
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
    if helper.exists():
        ret = subprocess.call([str(VENV_PY), str(helper), "--check"])
        if ret != 0:
            warn("Some models missing — follow Model Setup Guide (no auto-download, licensing)")
            print("  Guide:")
            print("    BASE=https://huggingface.co/facefusion/models-3.0.0/resolve/main")
            print("    curl -L -o models/inswapper_128.onnx  $BASE/inswapper_128.onnx")
            print("    curl -L -o models/bisenet_resnet_34.onnx $BASE/bisenet_resnet_34.onnx")
            print("    curl -L -o models/codeformer.onnx $BASE/codeformer.onnx")
            print("    curl -L -o models/xseg_1.onnx https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx")
            print("  Then re-run: python scripts/download_models.py --check")
            print("  Or auto (if you accept licenses): python scripts/download_models.py --download")
            print("  buffalo_l auto-downloads on first swap")
            # verify critical
            if not (ROOT / "models" / "inswapper_128.onnx").exists():
                warn("CRITICAL: inswapper_128.onnx still missing — swaps will fail until installed")
            print("\n  Once you see: [✓] buffalo_l [✓] inswapper_128 [✓] BiSeNet [✓] XSeg [✓] CodeFormer -> [Launch ReactorX]")
        else:
            ok("All models ready — [Launch ReactorX]")
    else:
        for rel in ["models/inswapper_128.onnx", "models/insightface/models/buffalo_l/det_10g.onnx"]:
            p = ROOT / rel
            if p.exists(): ok(f"found {rel}")
            else: warn(f"missing {rel} (see README Model Setup Guide)")


def selfcheck():
    sc = ROOT / "scripts" / "selfcheck.py"
    if sc.exists():
        info("Running self-check ...")
        ret = subprocess.call([str(VENV_PY), str(sc)])
        if ret == 0: ok("Self-check passed")
        else: warn("Self-check had failures (optional models may be missing)")


def main():
    parser = argparse.ArgumentParser(description="ReactorX local launcher (100% on-device)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1, use 0.0.0.0 for LAN)")
    parser.add_argument("--port", type=int, default=7860, help="Preferred port (default 7860)")
    parser.add_argument("--share", action="store_true", help="Create temporary public Gradio link")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install (use existing venv as-is)")
    parser.add_argument("--skip-check", action="store_true", help="Skip system/model/self checks")
    args = parser.parse_args()

    print(f"\n{CYAN}ReactorX — 100% local face-swap engine{NC}")
    print(f"  All processing stays on your device. No images leave your machine.\n")

    check_python()
    ensure_venv()
    if not args.skip_install:
        pip_install()
    if not args.skip_check:
        system_check()
        model_check()
        selfcheck()

    port = find_free_port(args.host, args.port)
    if port != args.port:
        warn(f"Port {args.port} busy, using {port} instead")

    info(f"Starting ReactorX at http://{args.host}:{port}")
    print(f"  Models: {os.environ.get('REACTORX_MODELS', str(ROOT / 'models'))}")
    print(f"  Outputs: {ROOT / 'outputs'}\n")

    cmd = [str(VENV_PY), str(APP), "--host", args.host, "--port", str(port)]
    if args.share:
        cmd.append("--share")
    # Replace current process
    os.execv(str(VENV_PY), cmd)


if __name__ == "__main__":
    main()
