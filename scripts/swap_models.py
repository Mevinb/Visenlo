"""Check/download registry models using pinned Hugging Face revisions and SHA256.

Run: .venv/bin/python scripts/swap_models.py --download --all
Existing files are never overwritten. No images leave this computer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visenlo.swappers import MODELS, missing_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", action="append", choices=list(MODELS))
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        cached_token = Path.home() / ".cache" / "huggingface" / "token"
        if cached_token.is_file():
            token = cached_token.read_text().strip()
    if "\n" in token or "\r" in token:
        raise ValueError("HF_TOKEN contains invalid characters")
    root = Path(os.environ.get("VISENLO_MODELS", os.environ.get("REACTORX_MODELS", ROOT / "models")))
    selected = list(MODELS) if args.all else (args.model or ["inswapper_128.onnx"])
    repositories, seen, failures = {}, set(), []
    for name in selected:
        spec = MODELS[name]
        print(f"{name}: {spec.license}; missing={missing_assets(root, name)}", flush=True)
        if spec.family == "mobilefaceswap" and missing_assets(root, name):
            print("  install with: python scripts/setup_mobilefaceswap.py", flush=True)
        if not args.download:
            continue
        for filename, release in spec.assets:
            if filename in seen:
                continue
            seen.add(filename)
            dest = root / filename
            if dest.is_file():
                print(f"  preserving existing {filename}", flush=True)
                continue
            try:
                repo = release if "/" in release else f"facefusion/models-{release}"
                if repo not in repositories:
                    headers = {"Authorization": "Bearer " + token} if token else {}
                    request = urllib.request.Request(f"https://huggingface.co/api/models/{repo}?blobs=true", headers=headers)
                    with urllib.request.urlopen(request, timeout=60) as response:
                        repositories[repo] = json.load(response)
                metadata = repositories[repo]
                entry = next(item for item in metadata["siblings"] if item["rfilename"] == filename)
                expected = entry.get("lfs", {}).get("sha256")
                if not expected:
                    raise RuntimeError("No upstream SHA256 available")
                url = f"https://huggingface.co/{repo}/resolve/{metadata['sha']}/{filename}"
                part = dest.with_suffix(dest.suffix + ".part")
                root.mkdir(parents=True, exist_ok=True)
                print(f"  downloading {filename} ({entry['lfs']['size'] / 1e6:.1f} MB)", flush=True)
                # Keep credentials off the process command line and error logs.
                curl_config = "header = " + json.dumps("Authorization: Bearer " + token) + "\n" if token else ""
                subprocess.run(["curl", "--config", "-", "--fail", "--location", "--silent", "--show-error", "--retry", "3",
                                "--retry-all-errors",
                                "--connect-timeout", "30", "--max-time", "1800", "--continue-at", "-",
                                "--output", str(part), url], check=True, input=curl_config, text=True)
                digest = hashlib.sha256()
                with part.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise RuntimeError(f"SHA256 mismatch: remove {part} before retrying")
                part.rename(dest)
                receipt = {"url": url, "sha256": expected, "revision": metadata["sha"], "license": spec.license}
                dest.with_suffix(".download.json").write_text(json.dumps(receipt, indent=2) + "\n")
                print(f"  verified {filename}", flush=True)
            except Exception as exc:
                failures.append(filename)
                print(f"  FAILED {filename}: {exc}", flush=True)
    return bool(failures or any(missing_assets(root, name) for name in selected))


if __name__ == "__main__":
    sys.exit(main())
