#!/usr/bin/env python3
"""Create a distributable ZIP for Option B (no git needed).

Excludes: .venv, models/*.onnx, outputs, .git, caches, etc.
Usage: python scripts/make_dist.py  -> dist/ReactorX-v1.zip
"""
from pathlib import Path
import zipfile
import fnmatch

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
DIST.mkdir(exist_ok=True)

EXCLUDE_DIRS = {".venv", "venv", ".git", "__pycache__", ".ruff_cache", ".pytest_cache", ".mypy_cache", "dist", "outputs", "output", ".gradio", "models"}
EXCLUDE_FILES = {"*.onnx", "*.pth", "*.ckpt", "*.safetensors"}
KEEP_MODELS = set()  # we exclude all weights; downloader gets them via download_models.py

def should_exclude(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    for part in rel.parts:
        if part in EXCLUDE_DIRS:
            return True
    for pat in EXCLUDE_FILES:
        if fnmatch.fnmatch(path.name, pat):
            return True
    return False

out = DIST / "ReactorX-v1.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in ROOT.rglob("*"):
        if p.is_dir():
            continue
        if should_exclude(p):
            continue
        # skip large hidden files
        if p.stat().st_size > 10_000_000 and p.suffix in (".zip",):
            continue
        arc = Path("ReactorX") / p.relative_to(ROOT)
        z.write(p, arc)

print(f"[ok] Created {out} ({out.stat().st_size/1e6:.1f} MB)")
print("  Contains: app.py, reactorx/, requirements.txt, install.sh/.bat/.ps1, launcher.py, scripts/, etc.")
print("  Excludes: .venv, models/*.onnx, outputs — downloader runs install.sh to fetch deps & models locally")
