#!/usr/bin/env python3
"""Run batch face-swap tests across models on test_images.

Defaults:
  Reference:   test_images/refrence imh/
  Targets:     test_images/images _needed_to_be_swapped/
  Output:      test_images/swapped_results/<model_name>/
  HTML Report: test_images/swapped_results/index.html
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Auto-configure CUDA library search path for RTX GPU acceleration
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
from visenlo.swappers import MODELS

# Curated list of high-performing models (strong visible face swaps)
TOP_MODELS = [
    "inswapper_128.onnx",
    "inswapper_128_fp16.onnx",
    "reswapper_256.onnx",
    "ghost_1_256.onnx",
    "ghost_2_256.onnx",
    "ghost_3_256.onnx",
    "simswap_256.onnx",
    "simswap_unofficial_512.onnx",
    "alphaface_256.onnx",
    "hififace_unofficial_256.onnx",
]


def generate_html_report(results_dir: Path, ref_path: Path, models: list, targets: list, metrics: dict):
    """Generate a responsive HTML gallery comparing Original Target vs Swapped Result."""
    ref_rel = os.path.relpath(ref_path, results_dir)
    clean_names = [m.replace(".onnx", "").replace(".h5", "").replace(".pdparams", "") for m in models]

    html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'/>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'/>",
        "  <title>ReactorX Face Swap Model Comparison</title>",
        "  <style>",
        "    :root { --bg: #0b0f19; --card-bg: #151d30; --border: #24304f; --accent: #38bdf8; --text: #f1f5f9; --text-muted: #94a3b8; }",
        "    * { box-sizing: border-box; }",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; }",
        "    h1 { text-align: center; color: var(--accent); margin: 0 0 8px; font-size: 26px; }",
        "    .meta { text-align: center; color: var(--text-muted); font-size: 14px; margin-bottom: 24px; }",
        "    .ref-container { max-width: 480px; margin: 0 auto 28px; background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 14px; text-align: center; }",
        "    .ref-img { max-height: 220px; border-radius: 8px; margin-top: 8px; border: 2px solid var(--accent); }",
        "    .nav-tabs { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-bottom: 24px; }",
        "    .tab-btn { background: var(--card-bg); color: #cbd5e1; border: 1px solid var(--border); padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.15s; }",
        "    .tab-btn:hover { background: #1e293b; color: #fff; }",
        "    .tab-btn.active { background: #0284c7; color: #fff; border-color: var(--accent); }",
        "    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(460px, 1fr)); gap: 20px; }",
        "    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }",
        "    .compare-row { display: flex; width: 100%; border-bottom: 1px solid var(--border); background: #000; }",
        "    .compare-pane { flex: 1; position: relative; }",
        "    .compare-pane img { width: 100%; height: 320px; object-fit: contain; display: block; background: #070b14; }",
        "    .pane-tag { position: absolute; top: 8px; left: 8px; background: rgba(0,0,0,0.75); color: #fff; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; text-transform: uppercase; }",
        "    .card-info { padding: 12px 16px; font-size: 13px; }",
        "    .card-info .name { font-weight: 600; color: #e2e8f0; margin-bottom: 6px; word-break: break-all; }",
        "    .stats { color: var(--text-muted); font-size: 12px; }",
        "    .badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; background: #065f46; color: #34d399; margin-left: 6px; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>ReactorX Face Swap Model Comparison</h1>",
        f"  <div class='meta'>Comparing <b>{len(models)}</b> models across <b>{len(targets)}</b> test images</div>",
        "  <div class='ref-container'>",
        "    <div style='font-weight: 600; color: var(--accent); font-size: 13px;'>SOURCE / REFERENCE FACE (Identity Source)</div>",
        f"    <img src='{ref_rel}' class='ref-img' alt='Reference face'/>",
        f"    <div style='font-size: 12px; color: var(--text-muted); margin-top: 6px;'>{ref_path.name}</div>",
        "  </div>",
        "  <div class='nav-tabs'>",
    ]

    for i, clean in enumerate(clean_names):
        active = " active" if i == 0 else ""
        html.append(f"    <button class='tab-btn{active}' onclick=\"showModel('{clean}')\">{clean}</button>")

    html.append("  </div>")

    for i, (model_name, clean_name) in enumerate(zip(models, clean_names)):
        display = "grid" if i == 0 else "none"
        html.append(f"  <div id='model-{clean_name}' class='grid' style='display: {display};'>")

        for tgt in targets:
            swapped_rel = f"{clean_name}/{tgt.name}"
            target_rel = os.path.relpath(tgt, results_dir)
            m_info = metrics.get(model_name, {}).get(tgt.name, {})
            score = m_info.get("similarity", "N/A")
            sec = m_info.get("seconds", "N/A")

            html.append("    <div class='card'>")
            html.append("      <div class='compare-row'>")
            html.append("        <div class='compare-pane'>")
            html.append(f"          <span class='pane-tag'>Original Target</span>")
            html.append(f"          <a href='{target_rel}' target='_blank'><img src='{target_rel}' loading='lazy' alt='Target'/></a>")
            html.append("        </div>")
            html.append("        <div class='compare-pane'>")
            html.append(f"          <span class='pane-tag' style='color: var(--accent);'>Swapped ({clean_name})</span>")
            html.append(f"          <a href='{swapped_rel}' target='_blank'><img src='{swapped_rel}' loading='lazy' alt='Swapped'/></a>")
            html.append("        </div>")
            html.append("      </div>")
            html.append("      <div class='card-info'>")
            html.append(f"        <div class='name'>{tgt.name} <span class='badge'>Success</span></div>")
            html.append(f"        <div class='stats'>ArcFace Similarity to Ref: <b style='color: var(--accent);'>{score}</b> | Inference Time: <b>{sec}s</b></div>")
            html.append("      </div>")
            html.append("    </div>")

        html.append("  </div>")

    html.append("""
  <script>
    function showModel(clean) {
      document.querySelectorAll('.grid').forEach(g => g.style.display = 'none');
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      const el = document.getElementById('model-' + clean);
      if (el) el.style.display = 'grid';
      event.target.classList.add('active');
    }
  </script>
</body>
</html>
""")

    report_file = results_dir / "index.html"
    report_file.write_text("\n".join(html))
    return report_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Run all 25 models")
    parser.add_argument("--top", action="store_true", help="Run only top 10 strong face swap models")
    parser.add_argument("--model", action="append", choices=list(MODELS), help="Run specific model(s)")
    default_ref = ROOT / "test_images/refrence imh"
    parser.add_argument("--reference", default=str(default_ref),
                        help="Path to source reference image or folder containing reference image(s)")
    parser.add_argument("--targets-dir", default=str(ROOT / "test_images/images _needed_to_be_swapped"),
                        help="Folder of target images to swap")
    parser.add_argument("--output-dir", default=str(ROOT / "test_images/swapped_results"),
                        help="Output directory for swapped model folders")
    parser.add_argument("--force", action="store_true", help="Re-swap and overwrite existing images")
    parser.add_argument("--quality", choices=["manual", "best_likeness"], default="manual")
    args = parser.parse_args()

    # Resolve reference image: accepts a specific file or auto-selects any image in folder
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    ref_input = Path(args.reference)

    if ref_input.is_file():
        ref_path = ref_input
    elif ref_input.is_dir():
        candidates = sorted([p for p in ref_input.iterdir() if p.suffix.lower() in image_exts])
        if not candidates:
            raise FileNotFoundError(f"No image files found in reference directory: {ref_input}")
        ref_path = candidates[0]
    elif ref_input.parent.is_dir():
        # User passed a nonexistent file path; look in its parent folder
        candidates = sorted([p for p in ref_input.parent.iterdir() if p.suffix.lower() in image_exts])
        if not candidates:
            raise FileNotFoundError(f"Reference file {ref_input} does not exist and no images found in {ref_input.parent}")
        ref_path = candidates[0]
    else:
        # Fallback to default ref directory
        candidates = sorted([p for p in default_ref.iterdir() if p.suffix.lower() in image_exts]) if default_ref.is_dir() else []
        if not candidates:
            raise FileNotFoundError(f"Could not find any reference image in {ref_input} or {default_ref}")
        ref_path = candidates[0]

    ref_img = cv2.imread(str(ref_path))
    if ref_img is None:
        raise ValueError(f"Could not load reference image: {ref_path}")

    targets_dir = Path(args.targets_dir)
    if not targets_dir.is_dir():
        raise FileNotFoundError(f"Target directory not found: {targets_dir}")

    target_paths = sorted([p for p in targets_dir.iterdir() if p.suffix.lower() in image_exts])
    if not target_paths:
        raise FileNotFoundError(f"No target images found in {targets_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine models
    if args.top:
        models = TOP_MODELS
    elif args.model:
        models = args.model
    else:
        models = list(MODELS)

    print("=" * 65)
    print("ReactorX Model Face Swap Evaluation Runner")
    print(f"  Reference Image: {ref_path.name} (from {ref_path.parent.name}/)")
    print(f"  Target Images:   {len(target_paths)} in {targets_dir.name}/")
    print(f"  Models to test:  {len(models)}")
    print(f"  Output folder:   {out_dir}/")
    print("=" * 65, flush=True)

    config = PipelineConfig(save_swaps=False, quality_mode=args.quality)
    pipeline = VisenloPipeline(str(ROOT / "models"), config)

    metrics_file = out_dir / "evaluation_metrics.json"
    metrics = {}
    if metrics_file.is_file():
        try:
            metrics = json.loads(metrics_file.read_text())
        except Exception:
            metrics = {}

    last_ref = metrics.get("_reference_file")
    force_run = args.force
    if last_ref and last_ref != ref_path.name:
        print(f"  [Notice] Reference image changed: {last_ref} -> {ref_path.name}.")
        print("  Generating fresh swaps for the new reference identity...\n")
        force_run = True
    metrics["_reference_file"] = ref_path.name

    start_time = time.monotonic()

    for m_idx, model_name in enumerate(models, 1):
        clean_name = model_name.replace(".onnx", "").replace(".h5", "").replace(".pdparams", "")
        model_out_dir = out_dir / clean_name
        model_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{m_idx}/{len(models)}] Running Model: {model_name} -> {clean_name}/", flush=True)
        if model_name not in metrics or not isinstance(metrics[model_name], dict):
            metrics[model_name] = {}

        for t_idx, tgt_path in enumerate(target_paths, 1):
            out_file = model_out_dir / tgt_path.name

            if out_file.is_file() and not force_run and tgt_path.name in metrics[model_name]:
                print(f"  ({t_idx}/{len(target_paths)}) Skipping already swapped: {tgt_path.name}")
                continue

            tgt_img = cv2.imread(str(tgt_path))
            if tgt_img is None:
                print(f"  ({t_idx}/{len(target_paths)}) Error reading {tgt_path.name}")
                continue

            t0 = time.monotonic()
            try:
                out_img, status = pipeline.process([ref_img], tgt_img, swapper_name=model_name)
                elapsed = time.monotonic() - t0

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
                print(f"  ({t_idx}/{len(target_paths)}) FAILED: {tgt_path.name}: {e}", flush=True)
                metrics[model_name][tgt_path.name] = {
                    "status": f"Failed: {str(e)}",
                    "similarity": "N/A",
                    "seconds": round(elapsed, 2),
                }

            metrics_file.write_text(json.dumps(metrics, indent=2))

        # Regenerate HTML report with completed models
        generate_html_report(out_dir, ref_path, models[:m_idx], target_paths, metrics)

    total_elapsed = time.monotonic() - start_time
    report_file = generate_html_report(out_dir, ref_path, models, target_paths, metrics)
    print("\n" + "=" * 60)
    print(f"All model tests completed in {total_elapsed:.2f}s!")
    print(f"Swapped folders: {out_dir}/<model_name>/")
    print(f"HTML Comparison Gallery: file://{report_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
