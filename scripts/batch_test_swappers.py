#!/usr/bin/env python3
"""Batch test all face swap models on test images and organize into per-model folders."""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

# Ensure CUDA libraries in .venv are visible
ROOT = Path(__file__).resolve().parents[1]
venv_nvidia = ROOT / ".venv/lib/python3.12/site-packages/nvidia"
if venv_nvidia.is_dir():
    nv_libs = [str(p) for p in venv_nvidia.glob("*/lib") if p.is_dir()]
    if nv_libs:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(nv_libs) + (":" + existing if existing else "")

sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from visenlo import PipelineConfig, VisenloPipeline
from visenlo.swappers import MODELS, missing_assets, runtime_ready


def generate_html_report(results_dir: Path, models: list, targets: list, metrics: dict):
    """Generate an interactive HTML gallery to compare all models across all images."""
    html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'/>",
        "<title>ReactorX Model Swap Comparison</title>",
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }",
        "  h1 { text-align: center; color: #38bdf8; margin-bottom: 8px; }",
        "  .subtitle { text-align: center; color: #94a3b8; margin-bottom: 24px; font-size: 14px; }",
        "  .nav-tabs { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-bottom: 24px; }",
        "  .tab-btn { background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.2s; }",
        "  .tab-btn:hover { background: #334155; color: #fff; }",
        "  .tab-btn.active { background: #0284c7; color: #fff; border-color: #38bdf8; }",
        "  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 20px; }",
        "  .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; }",
        "  .card img { width: 100%; height: auto; display: block; border-bottom: 1px solid #334155; background: #000; }",
        "  .card-info { padding: 12px; font-size: 12px; }",
        "  .card-info .title { font-weight: 600; color: #e2e8f0; margin-bottom: 4px; word-break: break-all; }",
        "  .badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 600; }",
        "  .badge-pass { background: #065f46; color: #34d399; }",
        "  .badge-fail { background: #991b1b; color: #f87171; }",
        "  .metric { color: #94a3b8; margin-top: 4px; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>ReactorX Model Swap Comparison Gallery</h1>",
        f"<div class='subtitle'>Evaluated {len(models)} models across {len(targets)} test images</div>",
        "<div class='nav-tabs'>",
    ]

    clean_names = [m.replace(".onnx", "").replace(".h5", "").replace(".pdparams", "") for m in models]
    for i, (m, c) in enumerate(zip(models, clean_names)):
        active = " active" if i == 0 else ""
        html.append(f"<button class='tab-btn{active}' onclick=\"showModel('{c}')\">{c}</button>")

    html.append("</div>")

    for i, (model_name, clean_name) in enumerate(zip(models, clean_names)):
        display = "grid" if i == 0 else "none"
        html.append(f"<div id='model-{clean_name}' class='grid' style='display: {display};'>")
        for tgt in targets:
            img_rel = f"{clean_name}/{tgt.name}"
            m_info = metrics.get(model_name, {}).get(tgt.name, {})
            score = m_info.get("similarity", "N/A")
            sec = m_info.get("seconds", "N/A")
            status = m_info.get("status", "unknown")
            is_pass = "verified" in str(status).lower() or "success" in str(status).lower()
            badge_cls = "badge-pass" if is_pass else "badge-fail"

            html.append("  <div class='card'>")
            html.append(f"    <a href='{img_rel}' target='_blank'><img src='{img_rel}' alt='{tgt.stem}' loading='lazy'/></a>")
            html.append("    <div class='card-info'>")
            html.append(f"      <div class='title'>{tgt.name}</div>")
            html.append(f"      <div><span class='badge {badge_cls}'>{status}</span></div>")
            html.append(f"      <div class='metric'>ArcFace Similarity: <b>{score}</b> | Time: <b>{sec}s</b></div>")
            html.append("    </div>")
            html.append("  </div>")
        html.append("</div>")

    html.append("""
<script>
function showModel(cleanName) {
  document.querySelectorAll('.grid').forEach(g => g.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  const target = document.getElementById('model-' + cleanName);
  if (target) target.style.display = 'grid';
  event.target.classList.add('active');
}
</script>
</body>
</html>
""")

    report_path = results_dir / "index.html"
    report_path.write_text("\n".join(html))
    print(f"Generated gallery report at: {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=list(MODELS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--reference", default=str(ROOT / "test_images/refrence imh/ChatGPT Image Sep 20, 2026, 08_53_29 PM.png"))
    parser.add_argument("--targets-dir", default=str(ROOT / "test_images/images _needed_to_be_swapped"))
    parser.add_argument("--output-dir", default=str(ROOT / "test_images/swapped_results"))
    parser.add_argument("--quality", choices=["manual", "best_likeness"], default="manual")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    targets_dir = Path(args.targets_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    ref_img = cv2.imread(str(ref_path))
    if ref_img is None:
        raise ValueError(f"Could not load reference image: {ref_path}")

    target_paths = sorted(targets_dir.glob("*.png")) + sorted(targets_dir.glob("*.jpg")) + sorted(targets_dir.glob("*.jpeg"))
    if not target_paths:
        raise FileNotFoundError(f"No target images found in: {targets_dir}")

    models_to_run = list(MODELS) if (args.all or not args.model) else args.model

    print(f"=== Starting Batch Face Swap Evaluation ===")
    print(f"Models to evaluate: {len(models_to_run)}")
    print(f"Target images: {len(target_paths)}")
    print(f"Reference: {ref_path.name}")
    print(f"Output directory: {out_dir}")
    print("===========================================\n", flush=True)

    config = PipelineConfig(save_swaps=False, quality_mode=args.quality)
    pipeline = VisenloPipeline(str(ROOT / "models"), config)

    metrics_file = out_dir / "evaluation_metrics.json"
    metrics = {}
    if metrics_file.is_file():
        try:
            metrics = json.loads(metrics_file.read_text())
        except Exception:
            metrics = {}

    total_swaps = len(models_to_run) * len(target_paths)
    current_swap = 0
    start_total = time.monotonic()

    for m_idx, model_name in enumerate(models_to_run, 1):
        clean_name = model_name.replace(".onnx", "").replace(".h5", "").replace(".pdparams", "")
        model_out_dir = out_dir / clean_name
        model_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{m_idx}/{len(models_to_run)}] Model: {model_name} -> {model_out_dir.name}/", flush=True)

        if model_name not in metrics:
            metrics[model_name] = {}

        for t_idx, tgt_path in enumerate(target_paths, 1):
            current_swap += 1
            out_file = model_out_dir / tgt_path.name

            if out_file.is_file() and not args.force and tgt_path.name in metrics[model_name]:
                print(f"  ({t_idx}/{len(target_paths)}) Skipping existing: {tgt_path.name}")
                continue

            tgt_img = cv2.imread(str(tgt_path))
            if tgt_img is None:
                print(f"  ({t_idx}/{len(target_paths)}) Error reading {tgt_path.name}")
                continue

            t0 = time.monotonic()
            try:
                out_img, status = pipeline.process([ref_img], tgt_img, swapper_name=model_name)
                elapsed = time.monotonic() - t0

                # Extract similarity score if present in status
                sim_str = "N/A"
                if "reference similarity:" in status:
                    parts = status.split("reference similarity:")
                    if len(parts) > 1:
                        sim_str = parts[1].split("|")[0].strip()

                cv2.imwrite(str(out_file), out_img)

                metrics[model_name][tgt_path.name] = {
                    "status": "Success",
                    "similarity": sim_str,
                    "seconds": round(elapsed, 2),
                    "full_status": status,
                }
                print(f"  ({t_idx}/{len(target_paths)}) {tgt_path.name}: {elapsed:.2f}s (sim: {sim_str})", flush=True)
            except Exception as e:
                elapsed = time.monotonic() - t0
                print(f"  ({t_idx}/{len(target_paths)}) FAILED on {tgt_path.name}: {e}", flush=True)
                metrics[model_name][tgt_path.name] = {
                    "status": f"Failed: {str(e)}",
                    "similarity": "N/A",
                    "seconds": round(elapsed, 2),
                }

            # Save metrics progress periodically
            metrics_file.write_text(json.dumps(metrics, indent=2))

        # Generate / update HTML report after each model completes
        generate_html_report(out_dir, models_to_run[:m_idx], target_paths, metrics)
        print(f"--- Completed model {model_name} ---\n", flush=True)

    elapsed_total = time.monotonic() - start_total
    print(f"\n=== All evaluations complete! Total elapsed: {elapsed_total:.2f}s ===")
    generate_html_report(out_dir, models_to_run, target_paths, metrics)


if __name__ == "__main__":
    main()
