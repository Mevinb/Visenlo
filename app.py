"""Standalone Gradio launcher for ReactorX Swap Engine v1."""

from __future__ import annotations

import argparse
import logging
import os
import threading
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from reactorx import PipelineConfig, ReactorXPipeline

ROOT = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("REACTORX_MODELS", ROOT / "models"))
_pipeline = None
_pipeline_lock = threading.Lock()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reactorx.app")
APP_CSS = """
body { background: #10110f; }
.rx-title { letter-spacing: .08em; text-transform: uppercase; }
.rx-status { border-left: 3px solid #b9ff66; padding-left: 12px; }
/* Swapped images gallery — fixed height with internal scroll, viewport-fixed lightbox.
   Gradio 5 Gallery uses .grid-wrap (scroll container), .grid-container (grid),
   .thumbnail-lg (tiles) and .preview (lightbox which is absolute by default). */
.rx-gallery .grid-wrap { height: 520px !important; max-height: 520px !important; min-height: 280px !important; overflow-y: auto !important; overflow-x: hidden !important; border-radius: 8px; scrollbar-width: thin; scrollbar-color: #333 #1a1a1a; }
.rx-gallery .grid-wrap.fixed-height { height: 520px !important; max-height: 520px !important; min-height: 280px !important; overflow-y: auto !important; overflow-x: hidden !important; }
.rx-gallery .grid-container { height: auto !important; align-content: start; }
.rx-gallery .thumbnail-lg img, .rx-gallery .gallery-item img { object-fit: contain !important; background: #1a1a1a; }
.rx-gallery .preview { position: fixed !important; inset: 0 !important; width: 100vw !important; height: 100vh !important; max-width: 100vw !important; max-height: 100vh !important; display: flex !important; flex-direction: column !important; align-items: center !important; justify-content: center !important; background: rgba(16,17,15,0.92) !important; -webkit-backdrop-filter: blur(8px) !important; backdrop-filter: blur(8px) !important; z-index: 9999 !important; padding: 24px !important; box-sizing: border-box !important; border-radius: 0 !important; }
.rx-gallery .preview:before { background: transparent !important; opacity: 1 !important; }
.rx-gallery .preview .media-button { height: auto !important; flex: 1 !important; min-height: 0 !important; width: 100% !important; display: flex !important; align-items: center !important; justify-content: center !important; }
.rx-gallery .preview .media-button img, .preview .media-button img { object-fit: contain !important; max-width: 92vw !important; max-height: 82vh !important; width: auto !important; height: auto !important; cursor: zoom-out; }
.rx-gallery .thumbnails img { object-fit: cover !important; }
@media (max-width: 900px) {
  .rx-gallery .grid-wrap, .rx-gallery .grid-wrap.fixed-height { height: 420px !important; max-height: 60vh !important; }
}
/* Target images gallery — thumbnail grid with preview, auto-scales when many images */
.rx-targets .grid-wrap { height: auto !important; max-height: 520px !important; min-height: 240px !important; overflow-y: auto !important; overflow-x: hidden !important; border-radius: 8px; scrollbar-width: thin; }
.rx-targets .grid-wrap.fixed-height { max-height: 520px !important; overflow-y: auto !important; min-height: 260px !important; }
.rx-targets .grid-container { height: auto !important; }
.rx-targets .thumbnail-lg img, .rx-targets .gallery-item img { object-fit: contain !important; background: #1a1a1a; border-radius: 6px; }
.rx-targets .preview { position: fixed !important; inset: 0 !important; width: 100vw !important; height: 100vh !important; max-width: 100vw !important; max-height: 100vh !important; display: flex !important; flex-direction: column !important; align-items: center !important; justify-content: center !important; background: rgba(16,17,15,0.92) !important; -webkit-backdrop-filter: blur(8px) !important; backdrop-filter: blur(8px) !important; z-index: 9999 !important; padding: 24px !important; box-sizing: border-box !important; border-radius: 0 !important; }
.rx-targets .preview:before { background: transparent !important; opacity: 1 !important; }
.rx-targets .preview .media-button { height: auto !important; flex: 1 !important; min-height: 0 !important; width: 100% !important; display: flex !important; align-items: center !important; justify-content: center !important; }
.rx-targets .preview img, .rx-targets .preview .media-button img { object-fit: contain !important; max-width: 92vw !important; max-height: 86vh !important; width: auto !important; height: auto !important; cursor: zoom-out; }
.rx-targets .empty { min-height: 220px; display: flex; align-items: center; justify-content: center; opacity: 0.85; }
/* Make thumbnails shrink gracefully when many images are present:
   override Gradio's fixed column count so the grid becomes responsive.
   auto-fit collapses empty tracks — single image -> large full-width,
   2-3 -> medium, 6+ -> small tiles that wrap. */
.rx-targets .grid-container { grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)) !important; }
@media (max-width: 900px) {
  .rx-targets .grid-container { grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)) !important; }
}
"""

SWAPPER_CHOICES = [
    "inswapper_128.onnx",
    "inswapper_128.onnx@256",
    "inswapper_128.onnx@512",
    "inswapper_128.onnx@1024",
    "inswapper_128.onnx@2048",
    "reswapper_256.onnx",
]


def get_pipeline(min_face_size, verify_threshold, color_strength, codeformer_enabled,
                 codeformer_weight, sharpen_strength, occluder_enabled):
    global _pipeline
    config = PipelineConfig(min_face_size=int(min_face_size),
                            verification_threshold=float(verify_threshold),
                            color_strength=float(color_strength),
                            codeformer_enabled=bool(codeformer_enabled),
                            codeformer_weight=float(codeformer_weight),
                            sharpen_strength=float(sharpen_strength),
                            occluder_enabled=bool(occluder_enabled))
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = ReactorXPipeline(str(MODELS), config)
        else:
            _pipeline.update_config(config)
    return _pipeline


def _bgr(image):
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _upload_path(item):
    """Resolve a gr.File entry to a filesystem path (str, Path, dict, or file object)."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        # Gradio FileData dict (e.g. {"name": "/tmp/...", "orig_name": "photo.jpg"})
        return item.get("name") or item.get("path") or item.get("orig_name") or ""
    # pathlib.Path or file-like with .name
    if isinstance(item, Path):
        return str(item)
    return getattr(item, "name", str(item) if item is not None else "")


def _target_path(item):
    """Normalize a single File or Gallery entry to a filesystem path string.

    Gallery interactive values are (path, caption) tuples where path may itself
    be a FileData dict; File values are plain paths/dicts.
    """
    if isinstance(item, (list, tuple)):
        if len(item) == 0:
            return ""
        # Gallery tuple (media, caption) — caption is str/None
        if len(item) == 2 and (item[1] is None or isinstance(item[1], str)):
            return _upload_path(item[0])
        # Fallback: treat as generic sequence, take first element
        return _upload_path(item[0])
    return _upload_path(item)


def run_swap(targets, ref1, ref2, ref3, ref4, target_index, source_index, match_mode,
             swapper_model, min_face_size, verify_threshold, color_strength,
             codeformer_enabled, codeformer_weight, sharpen_strength,
             occluder_enabled):
    """Swap one or many selected targets against the same references.

    Streams results so the gallery and report update after each image; a
    failing image is reported and skipped without stopping the rest. The
    pipeline auto-saves every completed swap as <date>_<NN>.png.
    """
    if not targets:
        raise gr.Error("Select at least one target image.")
    if ref1 is None:
        raise gr.Error("Reference 1 is required.")
    mode = "index" if match_mode.startswith("Manual") else "gender"
    files = list(targets) if isinstance(targets, (list, tuple)) else [targets]
    logger.info("Swap requested: %d image(s), swapper=%s target_face_idx=%d "
                "source_face_idx=%d match_mode=%s codeformer=%s sharpen=%.2f occluder=%s",
                len(files), swapper_model, target_index, source_index, mode,
                codeformer_enabled, sharpen_strength, occluder_enabled)
    references = [_bgr(image) for image in (ref1, ref2, ref3, ref4) if image is not None]
    pipeline = get_pipeline(min_face_size, verify_threshold, color_strength,
                            codeformer_enabled, codeformer_weight, sharpen_strength,
                            occluder_enabled)
    gallery, report = [], []
    total = len(files)
    for i, item in enumerate(files):
        path = _target_path(item)
        name = Path(path).name if path else f"image {i + 1}"
        # With a single image selected the plain name keeps the report clean;
        # batches get numbered lines like "[2/5] photo.jpg: ...".
        prefix = f"[{i + 1}/{total}] " if total > 1 else ""
        try:
            with Image.open(path) as image:
                image.load()
                result, status = pipeline.process(references, _bgr(image),
                                                  int(source_index), int(target_index),
                                                  swapper_model, match_mode=mode)
            gallery.append(Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)))
            report.append(f"{prefix}{name}: {status}")
        except Exception as exc:
            logger.warning("Swap failed for %s (%s)", name, exc, exc_info=True)
            # Avoid leaking absolute temp paths in user-visible report.
            safe_msg = str(exc).replace(path, name) if path else str(exc)
            report.append(f"{prefix}{name}: FAILED - {safe_msg}")
        yield gallery.copy(), "\n".join(report)


def build_ui():
    with gr.Blocks(
        title="ReactorX Swap Engine v1",
        theme=gr.themes.Monochrome(),
        css=APP_CSS,
    ) as app:
        gr.Markdown("# ReactorX Swap Engine v1\nIdentity transfer with target geometry and scene preservation.", elem_classes="rx-title")
        with gr.Row():
            with gr.Column(scale=5):
                targets = gr.Gallery(
                    label="Target images (one or many) — drag & drop, click to add, click thumbnail to preview",
                    columns=3,
                    rows=2,
                    height="auto",
                    object_fit="contain",
                    allow_preview=True,
                    show_download_button=False,
                    show_fullscreen_button=True,
                    file_types=["image"],
                    type="filepath",
                    interactive=True,
                    elem_classes="rx-targets",
                )
                with gr.Row():
                    upload_btn = gr.UploadButton("Add images", file_count="multiple", file_types=["image"], variant="secondary", size="sm")
                    clear_btn = gr.Button("Clear", variant="stop", size="sm")
                gr.Markdown("Add one image or many — all are swapped with the references and controls below. "
                            "Thumbnails scale down automatically when many images are added; click any thumbnail for full-size preview.")
            with gr.Column(scale=4):
                with gr.Row():
                    ref1 = gr.Image(type="pil", label="Reference 1", height=200)
                    ref2 = gr.Image(type="pil", label="Reference 2", height=200)
                with gr.Row():
                    ref3 = gr.Image(type="pil", label="Reference 3", height=200)
                    ref4 = gr.Image(type="pil", label="Reference 4", height=200)
        with gr.Accordion("Controls", open=True):
            with gr.Row():
                target_index = gr.Slider(0, 10, value=0, step=1, label="Target face index")
                source_index = gr.Slider(0, 10, value=0, step=1, label="Reference face index")
                min_face_size = gr.Slider(24, 256, value=48, step=4, label="Minimum face size")
            with gr.Row():
                match_mode = gr.Radio(
                    choices=["Manual (target index)", "Gender match (auto)"],
                    value="Manual (target index)",
                    label="Face matching",
                    info="Manual uses the target face index. Gender match detects the "
                         "reference's gender and swaps the leftmost target face of that "
                         "gender (target face index is then ignored).",
                )
                swapper_model = gr.Dropdown(
                    choices=SWAPPER_CHOICES,
                    value="inswapper_128.onnx",
                    label="Face swap model",
                    info="128px standard, @256/@512/@1024/@2048 pixel-boost (sharper, slower), or the 256px model.",
                )
                verify_threshold = gr.Slider(0, 1, value=.30, step=.01, label="Identity threshold")
                color_strength = gr.Slider(0, 1, value=.25, step=.05, label="Color matching", info="How much to blend the swapped face toward target lighting. Lower keeps the reference identity stronger.")
            with gr.Row():
                codeformer_enabled = gr.Checkbox(value=False, label="Enable CodeFormer restoration", info="Restores facial texture on the aligned swapped face before blending.")
                codeformer_weight = gr.Slider(0, 1, value=.8, step=.05, label="CodeFormer fidelity weight", info="Lower = more restoration, higher = keeps the swapped face structure closer to the swap output.")
                sharpen_strength = gr.Slider(0, 1, value=.5, step=.05, label="Sharpen strength", info="Single size-aware unsharp pass on the swapped face interior. Increases clarity without changing the swap model.")
            occluder_enabled = gr.Checkbox(value=True, label="Occlusion mask (XSeg)", info="Keeps hair strands, glasses and other objects in front of the face. Requires models/xseg_1.onnx (ignored if absent). Face parsing uses models/bisenet_resnet_34.onnx when present for tighter masks.")
            swap = gr.Button("Run identity swap", variant="primary")
            output = gr.Gallery(
                label="Swapped images — click any image to view full size / zoom",
                columns=3,
                height=520,
                object_fit="contain",
                allow_preview=True,
                show_fullscreen_button=True,
                show_download_button=True,
                interactive=False,
                elem_classes="rx-gallery",
            )
            status = gr.Textbox(label="Pipeline report", lines=4, elem_classes="rx-status")

            def _toggle_target_index(mode):
                return gr.update(visible=mode.startswith("Manual"))

            def _append_uploads(new_files, current_gallery):
                """Merge newly uploaded files into the existing gallery without replacing."""
                if not new_files:
                    return current_gallery
                # UploadButton may return single path or list
                if isinstance(new_files, (str, dict)) or hasattr(new_files, "name"):
                    new_files = [new_files]
                # Also handle tuple from Gallery if needed
                if not isinstance(new_files, (list, tuple)):
                    new_files = [new_files]
                added = []
                for f in new_files:
                    if isinstance(f, (list, tuple)):
                        p = _target_path(f)
                    else:
                        p = _upload_path(f)
                    if p:
                        added.append((p, None))
                cur = current_gallery or []
                return cur + added

            match_mode.change(_toggle_target_index, match_mode, target_index)
            upload_btn.upload(_append_uploads, inputs=[upload_btn, targets], outputs=[targets])
            clear_btn.click(lambda: [], outputs=[targets])
            swap.click(run_swap,
                       [targets, ref1, ref2, ref3, ref4, target_index, source_index,
                        match_mode, swapper_model, min_face_size, verify_threshold,
                        color_strength, codeformer_enabled, codeformer_weight,
                        sharpen_strength, occluder_enabled],
                       [output, status],
                       concurrency_limit=1)
        gr.Markdown("Every completed swap is auto-saved to `outputs/` as "
                    "`<date>_<NN>.png`. Use only images you own or have permission "
                    "to edit. ReactorX runs locally.")
    return app


def main():
    parser = argparse.ArgumentParser(description="ReactorX standalone local app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    logger.info("ReactorX Swap Engine v1")
    logger.info("  models dir:  %s", MODELS)
    logger.info("  server:      %s:%d", args.host, args.port)
    logger.info("  share mode:  %s", args.share)
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
