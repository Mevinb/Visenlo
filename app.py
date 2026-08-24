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
"""

SWAPPER_CHOICES = [
    "inswapper_128.onnx",
    "inswapper_128@256",
    "inswapper_128@512",
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


def run_swap(target, ref1, ref2, ref3, ref4, target_index, source_index,
             swapper_model, min_face_size, verify_threshold, color_strength,
             codeformer_enabled, codeformer_weight, sharpen_strength,
             occluder_enabled):
    if target is None or ref1 is None:
        raise gr.Error("A target image and Reference 1 are required.")
    logger.info("Swap requested: swapper=%s target_face_idx=%d source_face_idx=%d "
                "codeformer=%s sharpen=%.2f occluder=%s",
                swapper_model, target_index, source_index,
                codeformer_enabled, sharpen_strength, occluder_enabled)
    references = [_bgr(image) for image in (ref1, ref2, ref3, ref4) if image is not None]
    try:
        result, status = get_pipeline(
            min_face_size, verify_threshold, color_strength,
            codeformer_enabled, codeformer_weight, sharpen_strength,
            occluder_enabled).process(
            references, _bgr(target), int(source_index), int(target_index), swapper_model
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), status


def build_ui():
    with gr.Blocks(
        title="ReactorX Swap Engine v1",
        theme=gr.themes.Monochrome(),
        css=APP_CSS,
    ) as app:
        gr.Markdown("# ReactorX Swap Engine v1\nIdentity transfer with target geometry and scene preservation.", elem_classes="rx-title")
        with gr.Row():
            with gr.Column(scale=5):
                target = gr.Image(type="pil", label="Target image", height=420)
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
                swapper_model = gr.Dropdown(
                    choices=SWAPPER_CHOICES,
                    value="inswapper_128.onnx",
                    label="Face swap model",
                    info="128px standard, @256/@512 pixel-boost (sharper, slower), or the 256px model.",
                )
                verify_threshold = gr.Slider(0, 1, value=.30, step=.01, label="Identity threshold")
                color_strength = gr.Slider(0, 1, value=.25, step=.05, label="Color matching", info="How much to blend the swapped face toward target lighting. Lower keeps the reference identity stronger.")
            with gr.Row():
                codeformer_enabled = gr.Checkbox(value=False, label="Enable CodeFormer restoration", info="Restores facial texture on the aligned swapped face before blending.")
                codeformer_weight = gr.Slider(0, 1, value=.8, step=.05, label="CodeFormer fidelity weight", info="Lower = more restoration, higher = keeps the swapped face structure closer to the swap output.")
                sharpen_strength = gr.Slider(0, 1, value=.5, step=.05, label="Sharpen strength", info="Single size-aware unsharp pass on the swapped face interior. Increases clarity without changing the swap model.")
            occluder_enabled = gr.Checkbox(value=True, label="Occlusion mask (XSeg)", info="Keeps hair strands, glasses and other objects in front of the face. Requires models/xseg_1.onnx (ignored if absent). Face parsing uses models/bisenet_resnet_34.onnx when present for tighter masks.")
            swap = gr.Button("Run identity swap", variant="primary")
            output = gr.Image(type="pil", label="Final image", format="png")
            status = gr.Textbox(label="Pipeline report", elem_classes="rx-status")
            swap.click(run_swap,
                       [target, ref1, ref2, ref3, ref4, target_index, source_index,
                        swapper_model, min_face_size, verify_threshold, color_strength,
                        codeformer_enabled, codeformer_weight, sharpen_strength,
                        occluder_enabled],
                       [output, status],
                       concurrency_limit=1)
        gr.Markdown("Use only images you own or have permission to edit. ReactorX runs locally.")
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
