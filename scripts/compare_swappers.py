"""Reproducible local comparison. Each model runs in an isolated CPU process.

Use --all for all installed models; --quality best_likeness exercises automatic
selection. Native/manual is the controlled model comparison default.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visenlo.swappers import MODELS, missing_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", action="append")
    parser.add_argument("--reference", default=str(ROOT / "testdata/reference_female.jpg"))
    parser.add_argument("--target", default=str(ROOT / "testdata/target_mixed.jpg"))
    parser.add_argument("--output", default=str(ROOT / "outputs/model-comparison"))
    parser.add_argument("--quality", choices=["manual", "best_likeness"], default="manual")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    if args.worker:
        import cv2
        import numpy as np
        from visenlo import PipelineConfig, VisenloPipeline
        config = PipelineConfig(save_swaps=False, quality_mode=args.quality)
        engine = VisenloPipeline(str(ROOT / "models"), config)
        engine._providers = ["CPUExecutionProvider"]
        reference, target = cv2.imread(args.reference), cv2.imread(args.target)
        if reference is None or target is None:
            raise ValueError("Cannot read comparison inputs")
        start = time.monotonic()
        output, status = engine.process([reference], target, swapper_name=args.model[0])
        if output.shape != target.shape or output.dtype != np.uint8:
            raise AssertionError("Output image contract changed")
        name = args.model[0].replace(".onnx", "")
        if not cv2.imwrite(str(directory / (name + ".png")), output):
            raise RuntimeError("Cannot write comparison image")
        result = {"model": args.model[0], "status": status, "seconds": time.monotonic() - start,
                  "config": asdict(config), "provider": "CPUExecutionProvider",
                  "reference_sha256": hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
                  "target_sha256": hashlib.sha256(Path(args.target).read_bytes()).hexdigest()}
        (directory / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
        return 0
    names = list(MODELS) if args.all else (args.model or ["inswapper_128.onnx"])
    summary = []
    for name in names:
        missing = missing_assets(ROOT / "models", name.partition("@")[0])
        if missing:
            summary.append({"model": name, "result": "missing weights", "missing": missing})
            continue
        print(f"Testing {name}", flush=True)
        command = [sys.executable, __file__, "--worker", "--model", name,
                   "--reference", args.reference, "--target", args.target,
                   "--output", str(directory), "--quality", args.quality]
        with (directory / (name.replace(".onnx", "") + ".log")).open("w") as log:
            try:
                run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                status = "passed" if run.returncode == 0 else f"failed ({run.returncode})"
            except subprocess.TimeoutExpired:
                status = f"CPU timeout ({args.timeout}s); GPU validation pending"
        summary.append({"model": name, "result": status})
        print(status, flush=True)
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return int(any(item["result"].startswith("failed") for item in summary))


if __name__ == "__main__":
    sys.exit(main())
