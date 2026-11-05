"""Install FaceDancer's isolated Python 3.10 CPU runtime and original layer code."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REVISION = "af640de56c4105b56ae4754aa07e3c51b20a91d6"


def main():
    env = ROOT / ".venvs/facedancer"
    source = ROOT / "models/runtime-source/FaceDancer"
    if not (env / "bin/python").exists():
        subprocess.run(["uv", "venv", "--python", "3.10", str(env)], check=True)
    subprocess.run(["uv", "pip", "install", "--python", str(env / "bin/python"),
                    "tensorflow-cpu==2.15.1", "tensorflow-addons==0.23.0",
                    "numpy==1.26.4", "scipy==1.14.1", "opencv-python-headless==4.10.0.84"], check=True)
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/felixrosberg/FaceDancer.git", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "fetch", "origin", REVISION], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", REVISION], check=True)
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise RuntimeError(f"Existing FaceDancer source is {actual}, expected {REVISION}; preserved unchanged")
    print("FaceDancer source revision:")
    subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True)


if __name__ == "__main__":
    main()
