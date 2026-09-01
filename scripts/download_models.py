#!/usr/bin/env python3
"""Helper to verify and optionally download ReactorX models.

All downloads happen locally on the user's device — no cloud execution.
buffalo_l pack auto-downloads via insightface; other models are fetched
from Hugging Face if missing.

Usage:
  python scripts/download_models.py --check          # verify only
  python scripts/download_models.py                  # download missing
  python scripts/download_models.py --all            # force re-download check
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("REACTORX_MODELS", ROOT / "models"))

# URLs mirror README
BASE_300 = "https://huggingface.co/facefusion/models-3.0.0/resolve/main"
BASE_310 = "https://huggingface.co/facefusion/models-3.1.0/resolve/main"

FILES = [
    # (relative path under MODELS, url, description)
    ("bisenet_resnet_34.onnx", f"{BASE_300}/bisenet_resnet_34.onnx", "face parsing (BiSeNet) — MIT"),
    ("codeformer.onnx",       f"{BASE_300}/codeformer.onnx",       "CodeFormer restoration — CC BY-NC 4.0"),
    ("xseg_1.onnx",           f"{BASE_310}/xseg_1.onnx",           "occlusion mask (XSeg) — GPL-3.0"),
    ("inswapper_128.onnx",    f"{BASE_300}/inswapper_128.onnx",    "swap model 128px — research-only"),
]

OPTIONAL_LOCAL = [
    ("reswapper_256.onnx", "swap model 256px — research-only, optional (no public host)"),
]

BUFFALO_FILES = ["det_10g.onnx", "2d106det.onnx", "1k3d68.onnx", "genderage.onnx", "w600k_r50.onnx"]


def check():
    ok = True
    print(f"Models dir: {MODELS}")
    # buffalo_l
    bdir = MODELS / "insightface" / "models" / "buffalo_l"
    for f in BUFFALO_FILES:
        p = bdir / f
        if p.is_file():
            print(f"  [ok] buffalo_l/{f} ({p.stat().st_size/1e6:.1f} MB)")
        else:
            print(f"  [missing] buffalo_l/{f} — will auto-download on first pipeline init")
            ok = False
    for rel, _, desc in FILES:
        p = MODELS / rel
        alt = MODELS / "restoration" / "codeformer.onnx" if rel == "codeformer.onnx" else None
        exists = p.is_file() or (alt is not None and alt.is_file())
        if exists:
            sz = (p if p.is_file() else alt).stat().st_size/1e6
            print(f"  [ok] {rel} ({sz:.1f} MB) — {desc}")
        else:
            print(f"  [missing] {rel} — {desc}")
            ok = False
    for rel, desc in OPTIONAL_LOCAL:
        p = MODELS / rel
        if p.is_file():
            print(f"  [ok] {rel} ({p.stat().st_size/1e6:.1f} MB)")
        else:
            print(f"  [info] {rel} missing — {desc}")
    if ok:
        print("\nAll required models present.")
    else:
        print("\nSome models missing — run without --check to download, or see README for manual curl.")
    return 0 if ok else 1


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"  downloading {Path(dest).name} ({url}) -> {dest} ...", flush=True)
    print(f"    -> {url}", flush=True)
    last_pct = -1
    last_mb = -1
    def _report(block, bsize, total):
        nonlocal last_pct, last_mb
        if total > 0:
            pct = min(100, block*bsize*100//total)
            mb = block*bsize//1_000_000
            # Print every 5% or every 2MB to avoid spam, always flush
            if pct // 5 != last_pct // 5 or mb - last_mb >= 2 or block % 500 == 0:
                last_pct = pct
                last_mb = mb
                # Use \r when TTY, otherwise newline for pipe/tee visibility
                if sys.stdout.isatty():
                    print(f"\r    {pct}% ({mb} MB / {total//1_000_000} MB)", end="", flush=True)
                else:
                    print(f"    {pct}% ({mb} MB / {total//1_000_000} MB)", flush=True)
        else:
            # Unknown total — print dots
            if block % 100 == 0:
                print(".", end="", flush=True)
    # Follow redirects and handle large files with timeout
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "ReactorX-installer/1.0")]
    urllib.request.install_opener(opener)
    try:
        # Always use reporthook so piped (tee) also shows progress
        urllib.request.urlretrieve(url, tmp, reporthook=_report)
        if sys.stdout.isatty():
            print(flush=True)
        else:
            print(f"    100% done", flush=True)
        tmp.rename(dest)
        print(f"  [ok] saved {dest.name} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)
        return True
    except Exception as e:
        print(f"\n  [fail] {dest.name}: {e}", flush=True)
        if tmp.exists(): tmp.unlink(missing_ok=True)
        return False


def main():
    ap = argparse.ArgumentParser(description="ReactorX model helper (local) — licensing: manual setup by default")
    ap.add_argument("--check", action="store_true", help="only check, don't download (default)")
    ap.add_argument("--download", action="store_true", help="actually download missing models (ensure you comply with each model's license)")
    ap.add_argument("--all", action="store_true", help="check all including optional")
    args = ap.parse_args()

    # Default / --check: show checker + guide, no network
    if args.check or not args.download:
        # If user ran without flags, show check + guide (no auto-download)
        if not args.download:
            print("ReactorX Model Setup Guide (no auto-download — licensing)")
            print("="*60)
        ret = check()
        if ret != 0:
            print("\nModel Setup Guide — run these from your ReactorX folder:")
            print("  BASE=https://huggingface.co/facefusion/models-3.0.0/resolve/main")
            print("  mkdir -p models")
            for rel, url, desc in FILES:
                dest = MODELS / rel
                alt = MODELS / "restoration" / "codeformer.onnx" if rel == "codeformer.onnx" else None
                if not (dest.is_file() or (alt and alt.is_file())):
                    print(f"  curl -L -o {dest} {url}   # {desc}")
            print("  # optional XSeg: curl -L -o models/xseg_1.onnx https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx")
            print("  # buffalo_l auto-downloads on first swap; or: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")
            print("\nAfter installing, re-run: python scripts/download_models.py --check")
            print("Once you see:")
            print("  [✓] buffalo_l  [✓] inswapper_128  [✓] BiSeNet  [✓] XSeg  [✓] CodeFormer")
            print("  -> [Launch ReactorX]")
            print("\nTo auto-download despite licensing, run: python scripts/download_models.py --download")
        sys.exit(ret)

    # --download: actually fetch (user explicitly opted in)
    print("Downloading missing models (you confirmed licensing compliance)...")
    print(f"Models dir: {MODELS}")
    MODELS.mkdir(parents=True, exist_ok=True)
    any_missing = False
    for rel, url, desc in FILES:
        dest = MODELS / rel
        alt = MODELS / "restoration" / "codeformer.onnx" if rel == "codeformer.onnx" else None
        if dest.is_file() or (alt and alt.is_file()):
            print(f"  [skip] {rel} already present")
            continue
        any_missing = True
        ok = download(url, dest)
        if not ok:
            print(f"    manual: curl -L -o {dest} {url}")
    for rel, desc in OPTIONAL_LOCAL:
        p = MODELS / rel
        if not p.is_file():
            print(f"  [info] {rel} still missing — {desc}")

    bdir = MODELS / "insightface" / "models" / "buffalo_l"
    missing_buffalo = [f for f in BUFFALO_FILES if not (bdir/f).exists()]
    if missing_buffalo:
        print(f"  [info] buffalo_l missing {missing_buffalo} — will auto-download via insightface on first run")

    if not any_missing and not missing_buffalo:
        print("\nAll downloadable models present.")
    else:
        print("\nDone. Re-run with --check to verify.")
        check()

if __name__ == "__main__":
    main()
