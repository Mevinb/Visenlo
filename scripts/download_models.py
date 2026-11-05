#!/usr/bin/env python3
"""Check local models, or explicitly download the required face-swap models.

No flags / --check: local checks only, with no network access.
--download: fetch missing buffalo_l analysis and InSwapper weights.
--download --all: also fetch optional parsing, restoration and occlusion weights.
Review upstream model licenses before downloading or using them.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("VISENLO_MODELS", os.environ.get("REACTORX_MODELS", ROOT / "models")))
BASE_300 = "https://huggingface.co/facefusion/models-3.0.0/resolve/main"
BASE_310 = "https://huggingface.co/facefusion/models-3.1.0/resolve/main"
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
FILES = [
    ("inswapper_128.onnx", f"{BASE_300}/inswapper_128.onnx", "required swap model — research-only"),
    ("inswapper_128_fp16.onnx", f"{BASE_300}/inswapper_128_fp16.onnx", "optional FP16 twin of the swap model (2x faster tiles on GPU) — research-only"),
    ("bisenet_resnet_34.onnx", f"{BASE_300}/bisenet_resnet_34.onnx", "optional face parsing — see upstream license"),
    ("codeformer.onnx", f"{BASE_300}/codeformer.onnx", "optional restoration — CC BY-NC 4.0"),
    ("gfpgan_1.4.onnx", f"{BASE_300}/gfpgan_1.4.onnx", "optional restoration (identity-friendly) — Apache-2.0"),
    ("gpen_bfr_512.onnx", f"{BASE_300}/gpen_bfr_512.onnx", "optional restoration (most faithful) — Apache-2.0"),
    ("restoreformer_plus_plus.onnx", f"{BASE_300}/restoreformer_plus_plus.onnx", "optional restoration (heavy blur) — Apache-2.0"),
    ("xseg_1.onnx", f"{BASE_310}/xseg_1.onnx", "optional occlusion mask — see upstream license"),
]
OPTIONAL_LOCAL = [("reswapper_256.onnx", "optional alternative swap model (no canonical public host)")]
BUFFALO_FILES = ["det_10g.onnx", "2d106det.onnx", "1k3d68.onnx", "genderage.onnx", "w600k_r50.onnx"]


def present(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def model_path(name: str) -> Path:
    path = MODELS / name
    alternative = MODELS / "restoration" / name
    return alternative if name == "codeformer.onnx" and not present(path) and present(alternative) else path


def check(all_models=False):
    ready = True
    print(f"Models dir: {MODELS}")
    for name in BUFFALO_FILES:
        path = MODELS / "insightface/models/buffalo_l" / name
        found = present(path)
        print(f"  [{'ok' if found else 'missing'}] buffalo_l/{name} — required analysis pack")
        ready = ready and found
    for name, _, description in FILES:
        found = present(model_path(name))
        required = name == "inswapper_128.onnx" or all_models
        label = "ok" if found else "missing" if required else "optional"
        print(f"  [{label}] {name} — {description}")
        if required and not found:
            ready = False
    for name, description in OPTIONAL_LOCAL:
        print(f"  [{'ok' if present(MODELS / name) else 'optional'}] {name} — {description}")
    print("\n" + ("Required models are ready." if ready else "Required models are missing; follow the setup instructions below."))
    print("Optional finishing models improve masks or texture; their absence does not prevent a basic swap.")
    return 0 if ready else 1


def download(url: str, dest: Path):
    """Download to a temporary file; retain existing assets if transfer fails."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=dest.name + ".", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            request = urllib.request.Request(url, headers={"User-Agent": "Visenlo-installer/1.0"})
            print(f"  downloading {dest.name} from {url}", flush=True)
            with urllib.request.urlopen(request, timeout=60) as response:
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" in content_type or "application/json" in content_type:
                    raise ValueError("Server returned a page instead of model weights")
                expected = response.headers.get("Content-Length")
                shutil.copyfileobj(response, output, length=1024 * 1024)
            output.flush()
            size = output.tell()
        if not size or (expected and size != int(expected)):
            raise ValueError("Model download was empty or incomplete")
        temporary.replace(dest)
        print(f"  [ok] saved {dest.name} ({size / 1e6:.1f} MB)", flush=True)
        return True
    except Exception as exc:
        print(f"  [fail] {dest.name}: {exc}", flush=True)
        return False
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_buffalo():
    destination = MODELS / "insightface/models/buffalo_l"
    missing = [name for name in BUFFALO_FILES if not present(destination / name)]
    if not missing:
        print("  [skip] buffalo_l analysis pack already present")
        return True
    MODELS.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="buffalo-", dir=MODELS) as temp:
            archive_path = Path(temp) / "buffalo_l.zip"
            if not download(BUFFALO_URL, archive_path):
                return False
            with zipfile.ZipFile(archive_path) as archive:
                # Extract only named model files; never trust ZIP paths or extractall.
                members = {}
                for item in archive.infolist():
                    name = Path(item.filename).name
                    if name in BUFFALO_FILES and not item.is_dir():
                        if name in members:
                            raise ValueError(f"Duplicate analysis model in archive: {name}")
                        members[name] = item
                if any(name not in members or members[name].file_size == 0 for name in missing):
                    raise ValueError("Analysis archive does not contain the complete buffalo_l pack")
                for name in missing:
                    staged = Path(temp) / name
                    with archive.open(members[name]) as source, staged.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    destination.mkdir(parents=True, exist_ok=True)
                    staged.replace(destination / name)
                    print(f"  [ok] buffalo_l/{name}")
        return True
    except Exception as exc:
        print(f"  [fail] buffalo_l: {exc}")
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description="Visenlo local model setup; review upstream model licenses first")
    parser.add_argument("--check", action="store_true", help="check files only; never download (default)")
    parser.add_argument("--download", action="store_true", help="explicitly download missing required models")
    parser.add_argument("--all", action="store_true", help="also check/download optional finishing models")
    args = parser.parse_args(argv)
    if args.check and args.download:
        parser.error("--check and --download cannot be used together")
    if not args.download:
        result = check(args.all)
        print("\nReview the upstream licenses before downloading model weights:")
        print("  buffalo_l: https://github.com/deepinsight/insightface/tree/master/model_zoo")
        print("  swap/finishing weights: https://huggingface.co/facefusion/models-3.0.0")
        print("Then run: python scripts/download_models.py --download")
        print("Optional finishing models: python scripts/download_models.py --download --all")
        print("Re-check: python scripts/download_models.py --check")
        return result
    print("Downloading requested models. Their licenses govern permitted use.")
    success = download_buffalo()
    for name, url, _ in FILES:
        if name != "inswapper_128.onnx" and not args.all:
            continue
        destination = model_path(name)
        if present(destination):
            print(f"  [skip] {name} already present")
        elif not download(url, destination):
            success = False
    status = check(args.all)
    return status if success else 1


if __name__ == "__main__":
    sys.exit(main())
