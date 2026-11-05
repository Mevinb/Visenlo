"""Timed real-image swap for before/after comparisons.

Usage: .venv/bin/python scripts/bench_swap.py [--mode best_likeness|manual] [--model inswapper_128.onnx]
Writes /tmp/visenlo-bench-<mode>.png and prints the pipeline status line and timing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from visenlo import PipelineConfig, VisenloPipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="best_likeness")
    parser.add_argument("--model", default="inswapper_128.onnx")
    parser.add_argument("--ref", default=str(ROOT / "testdata" / "reference_female.jpg"))
    parser.add_argument("--target", default=str(ROOT / "testdata" / "target_mixed.jpg"))
    parser.add_argument("--match", default="gender")
    parser.add_argument("--no-codeformer", action="store_true")
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    ref = cv2.imread(args.ref)
    tgt = cv2.imread(args.target)
    cfg = PipelineConfig(save_swaps=False, quality_mode=args.mode,
                         codeformer_enabled=not args.no_codeformer)
    pipe = VisenloPipeline(str(ROOT / "models"), cfg)
    for i in range(args.repeat):
        t0 = time.perf_counter()
        result, status = pipe.process([ref], tgt, swapper_name=args.model, match_mode=args.match)
        dt = time.perf_counter() - t0
        print(f"run {i}: {dt:.2f}s  {status}")
    out = f"/tmp/visenlo-bench-{args.mode}.png"
    cv2.imwrite(out, result)
    print("wrote", out)


if __name__ == "__main__":
    main()
