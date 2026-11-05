"""Install isolated MobileFaceSwap runtime and original author checkpoint archive."""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "29552474c2f621c27818ff7fc32447e1a60a96fb"


def main():
    env = ROOT / ".venvs/mobilefaceswap"
    source = ROOT / "models/runtime-source/MobileFaceSwap"
    if not (env / "bin/python").exists():
        subprocess.run(["uv", "venv", "--python", "3.10", str(env)], check=True)
    subprocess.run(["uv", "pip", "install", "--python", str(env / "bin/python"),
                    "paddlepaddle==2.6.2", "numpy==1.26.4", "opencv-python-headless==4.10.0.84",
                    "setuptools==69.5.1", "protobuf==4.25.9"], check=True)
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Seanseattle/MobileFaceSwap.git", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "fetch", "origin", REVISION], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", REVISION], check=True)
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise RuntimeError("Existing MobileFaceSwap source revision differs; preserved unchanged")
    models = ROOT / "models"
    required = ["MobileFaceSwap_224.pdparams", "arcface.pdparams"]
    if all((models / name).is_file() for name in required):
        return
    archive = models / "mobilefaceswap-checkpoints.tar"
    url = "https://drive.usercontent.google.com/download?id=1ZIzGLDB15GRAZAbkfNR0hNWdgQpxeA_r&export=download&confirm=t"
    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3",
                    "--retry-all-errors", "--connect-timeout", "30", "--max-time", "1800",
                    "--continue-at", "-", "--output", str(archive), url], check=True)
    # Extract only the two expected regular files, never archive paths or links.
    with tarfile.open(archive) as tar:
        for name in required:
            if (models / name).exists():
                continue
            member = next(m for m in tar.getmembers() if m.isfile() and Path(m.name).name == name)
            digest = hashlib.sha256()
            with tar.extractfile(member) as src, (models / (name + ".part")).open("wb") as dest:
                for block in iter(lambda: src.read(1024 * 1024), b""):
                    digest.update(block)
                    dest.write(block)
            (models / (name + ".part")).rename(models / name)
            (models / (name + ".download.json")).write_text(json.dumps({"url": url, "sha256": digest.hexdigest(),
                "integrity": "locally recorded; author does not publish checksum"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
