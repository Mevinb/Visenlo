"""Standalone Gradio studio for Visenlo (images + video, multi-person mapping)."""

from __future__ import annotations

import argparse
import html
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from visenlo import FaceAssignment, PipelineConfig, SwapCancelled, VisenloPipeline
from visenlo.identity import default_bank
from visenlo.pipeline import gender_label
from visenlo.presets import delete_preset, list_presets, load_preset, save_preset
from visenlo.restoration import RESTORERS, restorer_path
from visenlo.swappers import MODELS as SWAP_MODELS, missing_assets, model_choices, runtime_ready

ROOT = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("VISENLO_MODELS", os.environ.get("REACTORX_MODELS", ROOT / "models")))
_pipeline = None
_pipeline_lock = threading.Lock()
_run_lock = threading.Lock()
_setup_lock = threading.Lock()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("visenlo").propagate = False
logger = logging.getLogger("visenlo.app")
APP_CSS = (ROOT / "app.css").read_text()

SWAPPER_CHOICES = model_choices()
PERSON_SLOTS = ("A", "B", "C", "D")
MAX_MAP_FACES = 8
RESTORER_LABELS = {"Off": None, **{spec.label: key for key, spec in RESTORERS.items()}}
MOUTH_MODES = {"Swap lips, keep teeth": "swap_lips_keep_teeth",
               "Keep entire target mouth": "keep_target_mouth",
               "Swap entire mouth": "swap_mouth"}
EYE_MODES = {"Swap eyes (default)": "swap", "Keep target eyes / gaze": "keep_target_eyes"}
MASK_MODES = {"Parsed face region (recommended)": "region", "Classic box": "box"}
BLEND_MODES = {"Multi-band (seamless)": "multiband", "Alpha feather": "alpha"}
PRECISION_MODES = {"Auto (FP16 on GPU)": "auto", "FP32 only": "fp32", "FP16": "fp16"}
ASSIGN_MODES = ("Single face (index)", "Gender match (auto)", "All faces → Person A",
                "Looks like anchor photo", "Map faces manually")

# --- Model checker (licensing: no auto-download) ---
MODEL_SPECS = [
    ("buffalo_l",        "models/insightface/models/buffalo_l", ["det_10g.onnx","2d106det.onnx","1k3d68.onnx","genderage.onnx","w600k_r50.onnx"], "Face analysis (detection+landmarks+identity)", "Install explicitly · research-only"),
    ("inswapper_128",    "models/inswapper_128.onnx",           None, "Default swap model 128px", "Research-only"),
    ("BiSeNet",          "models/bisenet_resnet_34.onnx",       None, "Face parsing (~90 MB)", "MIT"),
    ("CodeFormer",       "models/codeformer.onnx",              None, "Restoration (~377 MB, optional)", "CC BY-NC 4.0"),
    ("XSeg",             "models/xseg_1.onnx",                  None, "Occlusion mask (~68 MB, optional)", "GPL-3.0"),
    ("InSwapper FP16",   "models/inswapper_128_fp16.onnx",      None, "FP16 twin · 2x faster tiles on GPU", "Research-only"),
    ("GFPGAN 1.4",       "models/gfpgan_1.4.onnx",              None, "Restoration (~340 MB, optional)", "Apache-2.0"),
    ("GPEN BFR 512",     "models/gpen_bfr_512.onnx",            None, "Restoration (~280 MB, optional)", "Apache-2.0"),
    ("RestoreFormer++",  "models/restoreformer_plus_plus.onnx", None, "Restoration (optional)", "Apache-2.0"),
]


def _check_models(selected="inswapper_128.onnx"):
    """Check local assets without initializing models or making network requests."""
    filename = str(selected or "").partition("@")[0]
    if filename not in SWAP_MODELS:
        return '<p class="model-warning">Select a supported face-swap model.</p>', False
    spec = SWAP_MODELS[filename]
    analysis = MODELS / "insightface" / "models" / "buffalo_l"
    analysis_missing = [f for f in MODEL_SPECS[0][2]
                        if not (analysis / f).is_file() or (analysis / f).stat().st_size == 0]
    missing = missing_assets(MODELS, filename)
    # Treat empty/interrupted downloads as missing too.
    for name in [filename] + ([spec.converter + ".onnx"] if spec.converter else []):
        paths = [MODELS / name, MODELS / "insightface" / "models" / name]
        if name not in missing and not any(p.is_file() and p.stat().st_size > 0 for p in paths):
            missing.append(name)
    runtime = runtime_ready(filename)
    can_launch = not analysis_missing and not missing and runtime
    rows = []
    def row(name, ok, note):
        rows.append(f'<div class="model-row"><span><i class="model-dot {"ready" if ok else "missing"}"></i>'
                    f'{html.escape(name)}</span><span>{html.escape(note)}</span></div>')
    row("Face analysis · buffalo_l", not analysis_missing,
        "Installed" if not analysis_missing else f"{5-len(analysis_missing)}/5 files · required")
    row(spec.name, not missing and runtime,
        f"Ready · {spec.size}px native" if not missing and runtime else
        ("Missing: " + ", ".join(missing) if missing else "Additional runtime needed"))
    for name, rel, _, desc, lic in MODEL_SPECS[2:]:
        path = MODELS / rel.removeprefix("models/")
        alt = MODELS / "restoration" / path.name
        ok = any(p.is_file() and p.stat().st_size > 0 for p in (path, alt))
        row(name, ok, "Installed · optional" if ok else "Not installed · optional")
    notes = []
    if not can_launch:
        notes.append('<p>Complete the setup below to install the required models on this device. '
                     'Existing files are kept. Nothing downloads until you acknowledge the licenses and click Install.</p>')
        if analysis_missing or "inswapper_128.onnx" in missing:
            notes.append('<p>The core setup installs the buffalo_l analysis pack and InSwapper 128. '
                         'It needs an internet connection once; your photos stay on this device.</p>')
        if missing and filename != "inswapper_128.onnx":
            command = (f"python scripts/setup_{spec.family}.py" if spec.family in ("mobilefaceswap", "facedancer")
                       else f"python scripts/swap_models.py --download --model {filename}")
            notes.append(f'<p>Selected model setup: <code>{html.escape(command)}</code></p>')
        if not runtime:
            notes.append(f'<p>Runtime setup: <code>python scripts/setup_{spec.family}.py</code></p>')
    else:
        notes.append('<p class="model-success">Required assets are installed. You can start a swap.</p>')
    if spec.family == "facedancer":
        notes.append('<p>Experimental backend: real generator inference remains unverified. Official weights require approved access.</p>')
    notes.append('<p class="model-footnote">Optional models improve masking and restoration. Missing optional assets do not block a swap.</p>')
    return '<div class="model-check">' + ''.join(rows + notes) + '</div>', can_launch


def check_models(selected="inswapper_128.onnx"):
    """Refresh readiness without downloads; also used after local setup."""
    markup, can = _check_models(selected)
    if _setup_lock.locked():
        return markup, "Installing models · progress appears in the setup report", gr.update(interactive=False)
    return markup, ("Required models ready" if can else "Selected model needs setup · open Model setup"), gr.update(interactive=can)


def install_models(licenses_accepted, include_optional, selected="inswapper_128.onnx"):
    """Stream one explicitly requested, fixed local installer command."""
    if not licenses_accepted:
        raise gr.Error("Review the model licenses and check the acknowledgement before installing.")
    if not _setup_lock.acquire(blocking=False):
        raise gr.Error("Model installation is already running. Follow its setup report and wait for it to finish.")
    process = None
    reader = None
    stop_reading = threading.Event()
    lines = []
    markup = ""
    started = time.monotonic()
    outcome = "Installation could not start."

    def update(message):
        # Keep browser updates bounded even if a downloader prints noisy errors.
        return (message, markup, "Installing models · keep this app running",
                gr.update(interactive=False), gr.update(interactive=False))

    try:
        markup, _ = _check_models(selected)
        yield update("Preparing local model setup. Existing model files will be kept.")
        command = [sys.executable, "-u", str(ROOT / "scripts" / "download_models.py"), "--download"]
        if include_optional:
            command.append("--all")
        environment = dict(os.environ, VISENLO_MODELS=str(MODELS.resolve()), PYTHONUNBUFFERED="1")
        process = subprocess.Popen(command, cwd=str(ROOT), env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        messages = queue.Queue(maxsize=256)

        def enqueue(message):
            while not stop_reading.is_set():
                try:
                    messages.put(message, timeout=.1)
                    return
                except queue.Full:
                    pass

        def read_output():
            try:
                for line in process.stdout:
                    enqueue(line.rstrip()[:1000])
                    if stop_reading.is_set():
                        break
            finally:
                enqueue(None)

        reader = threading.Thread(target=read_output, name="visenlo-model-setup", daemon=True)
        reader.start()
        last_update = 0.0
        finished = False
        while not finished:
            try:
                message = messages.get(timeout=.5)
                if message is None:
                    finished = True
                else:
                    lines.append(message)
                    lines = lines[-120:]
            except queue.Empty:
                pass
            now = time.monotonic()
            if now - started > 7200:
                raise TimeoutError("Installation exceeded two hours. Check your connection and try again; existing models are kept.")
            if finished or now - last_update >= 1:
                elapsed = int(now - started)
                report = ("\n".join(lines) if lines else "Connecting to the model download server…")
                yield update(f"Installing models · {elapsed}s elapsed. Large downloads can take several minutes.\n\n" + report)
                last_update = now
        returncode = process.wait(timeout=10)
        if returncode == 0:
            outcome = "Model installation finished."
        else:
            outcome = ("Installation did not finish. Check the setup report below and your internet connection, "
                       "then click Install to retry. Completed downloads are kept.")
    except (OSError, subprocess.SubprocessError, TimeoutError) as exc:
        logger.warning("Local model installation failed: %s", exc)
        outcome = f"Installation could not finish: {exc}. Existing model files are kept. You can retry."
    finally:
        stop_reading.set()
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if reader is not None:
                # Normal completion drains the bounded queue before closing.
                reader.join(timeout=1)
            if process.stdout is not None:
                process.stdout.close()
        _setup_lock.release()
    markup, can = _check_models(selected)
    readiness = "Required models ready · you can start a swap" if can else "Selected model still needs setup"
    if not can and process is not None and process.returncode == 0:
        outcome += " Select inswapper_128.onnx to use the core setup, or follow the selected model’s additional setup instructions above."
    yield (outcome + ("\n\n" + "\n".join(lines[-120:]) if lines else ""), markup, readiness,
           gr.update(interactive=can), gr.update(interactive=True))


# --- Settings <-> PipelineConfig -------------------------------------------
SETTING_KEYS = ("quality_mode", "swapper_model", "min_face_size", "verify_threshold", "color_strength",
                "restorer", "codeformer_weight", "restoration_blend", "sharpen_strength", "occluder_enabled",
                "mouth_mode", "eye_mode", "mask_mode", "blend_mode", "precision", "dealias", "debug_masks")
SETTING_DEFAULTS = {
    "quality_mode": "Best likeness", "swapper_model": "inswapper_128.onnx", "min_face_size": 48,
    "verify_threshold": .30, "color_strength": .25, "restorer": "CodeFormer", "codeformer_weight": .9,
    "restoration_blend": .7, "sharpen_strength": .5, "occluder_enabled": True,
    "mouth_mode": "Swap lips, keep teeth", "eye_mode": "Swap eyes (default)",
    "mask_mode": "Parsed face region (recommended)", "blend_mode": "Multi-band (seamless)",
    "precision": "Auto (FP16 on GPU)", "dealias": False, "debug_masks": False,
}


def settings_to_config(settings: dict) -> PipelineConfig:
    s = {**SETTING_DEFAULTS, **{k: v for k, v in settings.items() if v is not None}}
    restorer = RESTORER_LABELS.get(str(s["restorer"]).replace(" (not installed)", ""), "codeformer")
    return PipelineConfig(
        min_face_size=int(s["min_face_size"]),
        verification_threshold=float(s["verify_threshold"]),
        color_strength=float(s["color_strength"]),
        codeformer_enabled=restorer is not None,
        restorer=restorer or "codeformer",
        codeformer_weight=float(s["codeformer_weight"]),
        restoration_blend=float(s["restoration_blend"]),
        sharpen_strength=float(s["sharpen_strength"]),
        occluder_enabled=bool(s["occluder_enabled"]),
        quality_mode="best_likeness" if str(s["quality_mode"]).startswith("Best") else "manual",
        mouth_mode=MOUTH_MODES.get(str(s["mouth_mode"]), "swap_lips_keep_teeth"),
        eye_mode=EYE_MODES.get(str(s["eye_mode"]), "swap"),
        mask_mode=MASK_MODES.get(str(s["mask_mode"]), "region"),
        blend_mode=BLEND_MODES.get(str(s["blend_mode"]), "multiband"),
        inswapper_precision=PRECISION_MODES.get(str(s["precision"]), "auto"),
        boost_dealias=bool(s["dealias"]),
        debug_masks=bool(s["debug_masks"]),
    )


def get_pipeline(config: PipelineConfig | None = None, **legacy):
    """Return the shared pipeline, updating its configuration."""
    global _pipeline
    if config is None:
        config = settings_to_config(legacy) if legacy else PipelineConfig()
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = VisenloPipeline(str(MODELS), config)
        else:
            _pipeline.update_config(config)
    return _pipeline


def _bgr(image):
    return cv2.cvtColor(np.asarray(ImageOps.exif_transpose(image).convert("RGB")), cv2.COLOR_RGB2BGR)


def _rgb(image_bgr):
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


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


def _paths(items):
    if not items:
        return []
    items = list(items) if isinstance(items, (list, tuple)) else [items]
    return [p for p in (_target_path(i) for i in items) if p]


def _load_bgr(path):
    with Image.open(path) as image:
        image.load()
        return _bgr(image)


# --- People / references -----------------------------------------------------
def _bank():
    return default_bank(str(MODELS))


def identity_choices():
    return ["(none)"] + _bank().names()


def _reference_sets(pipeline, ref_galleries, identity_names, source_index):
    """Build {slot: ReferenceSet} for every person slot that has input."""
    sets = {}
    for slot, gallery, ident in zip(PERSON_SLOTS, ref_galleries, identity_names):
        paths = _paths(gallery)
        if paths:
            sets[slot] = pipeline.analyze_references([_load_bgr(p) for p in paths[:4]], int(source_index),
                                                     label=f"Person {slot}")
        elif ident and ident != "(none)":
            refset = _bank().as_reference_set(ident)
            refset.label = f"Person {slot} ({ident})"
            sets[slot] = refset
    return sets


def save_identity(name, gallery, source_index):
    paths = _paths(gallery)
    if not paths:
        raise gr.Error("Add at least one reference photo for Person A before saving.")
    if not str(name or "").strip():
        raise gr.Error("Give the identity a name.")
    pipeline = get_pipeline()
    refset = pipeline.analyze_references([_load_bgr(p) for p in paths[:4]], int(source_index), label=str(name))
    saved = _bank().save(str(name).strip(), refset)
    choices = identity_choices()
    return (f"Saved identity '{saved.name}' ({saved.gender_label}, {saved.references} reference(s)).",
            *[gr.update(choices=choices) for _ in PERSON_SLOTS])


def delete_identity(name):
    if not name or name == "(none)":
        raise gr.Error("Pick a saved identity to delete.")
    _bank().delete(name)
    choices = identity_choices()
    return (f"Deleted identity '{name}'.", *[gr.update(choices=choices, value="(none)") for _ in PERSON_SLOTS])


# --- Face picker --------------------------------------------------------------
def detect_target_faces(targets, min_face_size):
    """Thumbnails of the faces in the first target image, for mapping."""
    paths = _paths(targets)
    empty = [gr.update(visible=False, value="Keep") for _ in range(MAX_MAP_FACES)]
    if not paths or not _check_models()[1]:
        return [], "Add a target image to see its faces here.", *empty
    try:
        pipeline = get_pipeline(PipelineConfig(min_face_size=int(min_face_size)))
        image = _load_bgr(paths[0])
        records = pipeline.detect_faces(image)
    except Exception as exc:
        logger.warning("face detection for picker failed: %s", exc)
        return [], f"Face detection unavailable: {exc}", *empty
    thumbs = []
    for i, record in enumerate(records):
        x1, y1, x2, y2 = record.bbox
        crop = image[y1:y2, x1:x2]
        if crop.size:
            thumbs.append((_rgb(crop), f"Face {i} · {gender_label(record.gender)} · ~{record.age}"))
    updates = []
    for i in range(MAX_MAP_FACES):
        if i < len(records):
            updates.append(gr.update(visible=True, value="Keep",
                                     label=f"Face {i} ({gender_label(records[i].gender)}, ~{records[i].age})"))
        else:
            updates.append(gr.update(visible=False, value="Keep"))
    note = (f"{len(records)} face(s) found in {Path(paths[0]).name}, numbered left to right. "
            "Use the index, gender match, or map each face to a person below.")
    return thumbs, note, *updates


# --- Debug overlay -------------------------------------------------------------
_MASK_COLORS = {"face_mask": (255, 160, 60), "interior": (60, 200, 255), "occlusion": (60, 60, 255),
                "mouth": (255, 60, 220), "eyes": (60, 255, 120), "blend_mask": (190, 255, 120)}


def mask_overlays(result_bgr, debug):
    """One tinted overlay image per debug mask."""
    out = []
    for key, mask in debug.items():
        if mask is None:
            continue
        base = key.split("_", 1)[1] if key.startswith("face") and "_" in key else key
        color = next((c for name, c in _MASK_COLORS.items() if base.startswith(name)), (200, 200, 200))
        tint = np.zeros_like(result_bgr)
        tint[:] = color
        alpha = (np.clip(mask, 0, 1) * .6)[:, :, None]
        blended = (result_bgr.astype(np.float32) * (1 - alpha) + tint.astype(np.float32) * alpha).astype(np.uint8)
        out.append((_rgb(blended), key))
    return out


# --- Runs ---------------------------------------------------------------------
def _assignments(sets, assign_mode, target_index, anchor_image, map_values, pipeline):
    if "A" not in sets:
        raise gr.Error("Person A needs at least one reference photo or a saved identity.")
    mode = str(assign_mode)
    if mode.startswith("Single"):
        return [FaceAssignment(sets["A"], "index", int(target_index))]
    if mode.startswith("Gender"):
        return [FaceAssignment(sets["A"], "gender")]
    if mode.startswith("All"):
        return [FaceAssignment(sets["A"], "all")]
    if mode.startswith("Looks"):
        if anchor_image is None:
            raise gr.Error("Upload an anchor photo of the person to replace.")
        anchors = pipeline.detect_faces(_bgr(anchor_image))
        if not anchors:
            raise gr.Error("No face found in the anchor photo.")
        return [FaceAssignment(sets["A"], "similar", anchor=np.asarray(anchors[0].embedding, np.float32))]
    assignments = []
    for index, value in enumerate(map_values):
        slot = str(value or "Keep").replace("Person ", "").strip()
        if slot in sets:
            assignments.append(FaceAssignment(sets[slot], "index", index))
        elif slot not in ("Keep", "") and slot in PERSON_SLOTS:
            raise gr.Error(f"Face {index} is mapped to Person {slot}, which has no reference.")
    if not assignments:
        raise gr.Error("Map at least one face to a person (or choose another assignment mode).")
    return assignments


def cancel_run():
    if _pipeline is not None:
        _pipeline.cancel_event.set()
    return "Cancelling after the current step…"


def run_images(targets, ref_a, ref_b, ref_c, ref_d, id_a, id_b, id_c, id_d, anchor_image, assign_mode,
               target_index, source_index, *rest):
    map_values = rest[:MAX_MAP_FACES]
    settings = dict(zip(SETTING_KEYS, rest[MAX_MAP_FACES:]))
    with _run_lock:
        yield from _run_images(targets, (ref_a, ref_b, ref_c, ref_d), (id_a, id_b, id_c, id_d), anchor_image,
                               assign_mode, target_index, source_index, map_values, settings)


def _run_images(targets, refs, idents, anchor_image, assign_mode, target_index, source_index, map_values, settings):
    paths = _paths(targets)
    if not paths:
        raise gr.Error("Select at least one target image.")
    if not _check_models(settings.get("swapper_model"))[1]:
        raise gr.Error("Required model files are missing. Open Model setup and re-check after installing.")
    yield [], "Preparing models. The first swap takes longer while the local models load.", [], gr.update(), []
    pipeline = get_pipeline(settings_to_config(settings))
    pipeline.cancel_event.clear()
    sets = _reference_sets(pipeline, refs, idents, source_index)
    assignments = _assignments(sets, assign_mode, target_index, anchor_image, map_values, pipeline)
    save_preset("last", settings)
    gallery, report, pairs, masks = [], [], [], []
    total, failures = len(paths), 0
    for i, path in enumerate(paths):
        name = Path(path).name
        prefix = f"[{i + 1}/{total}] " if total > 1 else ""
        yield gallery.copy(), f"Processing image {i + 1} of {total}…\n" + "\n".join(report), pairs, gr.update(), masks
        try:
            output = pipeline.swap(_load_bgr(path), assignments, settings.get("swapper_model", "inswapper_128.onnx"))
            gallery.append((_rgb(output.image), name))
            result_path = output.saved_path
            if not result_path:
                fd, result_path = tempfile.mkstemp(prefix="visenlo-", suffix=".png")
                os.close(fd)
                cv2.imwrite(result_path, output.image)
            pairs.append((path, result_path))
            report.append(f"{prefix}{name}: {output.status}")
            if output.debug:
                masks.extend((img, f"{name} · {key}") for img, key in mask_overlays(output.image, output.debug))
        except SwapCancelled:
            report.append(f"{prefix}{name}: cancelled")
            break
        except Exception as exc:
            failures += 1
            logger.warning("Swap failed for %s (%s)", name, exc, exc_info=True)
            report.append(f"{prefix}{name}: FAILED - {str(exc).replace(path, name)}")
        yield gallery.copy(), "\n".join(report), pairs, gr.update(), masks
    slider = gr.update(value=(pairs[0][0], pairs[0][1])) if pairs else gr.update()
    summary = f"Finished: {len(gallery)} of {total} images completed" + (f" · {failures} failed" if failures else "")
    yield gallery.copy(), summary + "\n\n" + "\n".join(report), pairs, slider, masks


def show_pair(evt: gr.SelectData, pairs):
    if not pairs or evt.index is None or evt.index >= len(pairs):
        return gr.update()
    before, after = pairs[evt.index]
    return gr.update(value=(before, after))


def run_video(video_path, ref_a, ref_b, ref_c, ref_d, id_a, id_b, id_c, id_d, anchor_image, assign_mode,
              target_index, source_index, smoothing, every, *settings_values):
    from visenlo.video import swap_video
    settings = dict(zip(SETTING_KEYS, settings_values))
    if not video_path:
        raise gr.Error("Upload a video first.")
    if not _check_models(settings.get("swapper_model"))[1]:
        raise gr.Error("Required model files are missing. Open Model setup and re-check after installing.")
    with _run_lock:
        yield None, None, "Preparing models…"
        pipeline = get_pipeline(settings_to_config(settings))
        sets = _reference_sets(pipeline, (ref_a, ref_b, ref_c, ref_d), (id_a, id_b, id_c, id_d), source_index)
        if str(assign_mode).startswith("Map"):
            raise gr.Error("Manual face mapping is for still images; use index, gender, all or anchor for video.")
        assignments = _assignments(sets, assign_mode, target_index, anchor_image, [], pipeline)
        save_preset("last", settings)
        out_dir = ROOT / "outputs"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{time.strftime('%Y-%m-%d')}_{Path(video_path).stem}_swapped.mp4"
        updates = queue.Queue()
        done = {}

        def progress(p):
            updates.put(p)

        def worker():
            try:
                done["result"] = swap_video(pipeline, video_path, str(out_path), assignments,
                                            settings.get("swapper_model", "inswapper_128.onnx"), progress=progress,
                                            smoothing=float(smoothing), every=int(every))
            except Exception as exc:  # surfaced below
                done["error"] = exc
            finally:
                updates.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = updates.get()
            if item is None:
                break
            preview = _rgb(item.preview) if item.preview is not None else None
            yield None, preview, (f"Frame {item.frame}/{item.total} · {item.swapped_faces} face swaps · "
                                  f"{item.elapsed:.0f}s elapsed · {item.fraction * 100:.0f}%")
        if "error" in done:
            if isinstance(done["error"], SwapCancelled):
                yield None, None, "Video swap cancelled."
                return
            raise gr.Error(f"Video swap failed: {done['error']}")
        result = done["result"]
        notes = ("\n" + "\n".join("• " + n for n in result.notes)) if result.notes else ""
        yield (str(result.output_path), None,
               f"Done: {result.frames_done} frames, {result.swapped_faces} face swaps in {result.seconds:.0f}s · "
               f"audio {'kept' if result.audio_copied else 'not available'} · saved {out_path.name}{notes}")


# --- Presets --------------------------------------------------------------------
def apply_preset(name):
    if not name:
        raise gr.Error("Pick a preset.")
    data = load_preset(name)
    return [gr.update(value=data.get(key, SETTING_DEFAULTS[key])) for key in SETTING_KEYS]


def store_preset(name, *values):
    if not str(name or "").strip():
        raise gr.Error("Give the preset a name.")
    saved = save_preset(str(name).strip(), dict(zip(SETTING_KEYS, values)))
    return gr.update(choices=list_presets(), value=saved), f"Saved preset '{saved}'."


def remove_preset(name):
    if not name:
        raise gr.Error("Pick a preset to delete.")
    delete_preset(name)
    return gr.update(choices=list_presets(), value=None), f"Deleted preset '{name}'."


# --- UI -------------------------------------------------------------------------
def build_ui():
    theme = gr.themes.Base(primary_hue="lime", neutral_hue="zinc",
                           font=["Arial", "sans-serif"], font_mono=["ui-monospace", "monospace"])
    palette = dict(body_background_fill="#10120f", body_text_color="#e9ede5",
                   body_text_color_subdued="#9ba595", background_fill_primary="#191c17",
                   background_fill_secondary="#20251d", block_background_fill="#191c17",
                   block_border_color="#343c2e", border_color_primary="#343c2e",
                   input_background_fill="#11150e", input_border_color="#434d3c",
                   button_primary_background_fill="#b9ec80", button_primary_text_color="#18200f",
                   button_primary_border_color="#b9ec80", button_primary_background_fill_hover="#cef3a7",
                   button_secondary_background_fill="#242a1f", button_secondary_text_color="#e9ede5",
                   button_secondary_border_color="#434d3c", block_label_background_fill="#191c17",
                   block_label_text_color="#d6dfcd", block_title_text_color="#e9ede5")
    theme.set(**{key+suffix: value for key, value in palette.items() for suffix in ("", "_dark")})
    restorer_choices = ["Off"] + [spec.label + ("" if restorer_path(str(MODELS), key) else " (not installed)")
                                  for key, spec in RESTORERS.items()]
    installed_default = "CodeFormer" if restorer_path(str(MODELS), "codeformer") else "Off"
    with gr.Blocks(title="Visenlo · local face studio", theme=theme, css=APP_CSS,
                   analytics_enabled=False) as app:
        gr.HTML('<header class="studio-header"><div class="studio-brand"><span class="brand-symbol">v.</span>'
                '<span>visenlo<span class="brand-divider">/</span><small>local face studio</small></span></div>'
                '<span class="local-badge"><i></i> Processing stays on your device</span></header>'
                '<div class="studio-intro"><span class="studio-eyebrow">YOUR IMAGES. YOUR MACHINE.</span>'
                '<h1>A different face.<br><em>The same moment.</em></h1>'
                '<p>Bring a reference. Choose your scene or clip. Swap one face, several people, or a whole '
                'video — locally, with control over the details.</p></div>')
        init_html, init_can = _check_models()
        with gr.Accordion("Model setup · local assets and readiness", open=not init_can):
            model_html = gr.HTML(init_html)
            gr.HTML('<div class="setup-intro"><h2>Set up once. Create locally.</h2>'
                    '<p>Install the required face-analysis and swap models here. '
                    'Downloads come from GitHub and Hugging Face; images are never uploaded. '
                    'Model licenses determine how you can use the results.</p>'
                    '<p><a href="https://github.com/deepinsight/insightface/tree/master/model_zoo" '
                    'target="_blank" rel="noopener noreferrer">Review InsightFace model terms ↗</a> '
                    '<span>·</span> <a href="https://huggingface.co/facefusion/models-3.0.0" '
                    'target="_blank" rel="noopener noreferrer">Swap and finishing model sources ↗</a> '
                    '<span>·</span> <a href="https://github.com/sczhou/CodeFormer/blob/master/LICENSE" '
                    'target="_blank" rel="noopener noreferrer">CodeFormer license ↗</a></p></div>')
            licenses_accepted = gr.Checkbox(value=False, label="I have reviewed the model licenses and agree to their permitted use.",
                                            elem_id="model-license-acknowledgement")
            include_optional = gr.Checkbox(value=False, label="Also install optional parsing, restoration (CodeFormer, GFPGAN, GPEN, RestoreFormer++), occlusion and FP16 models",
                info="Takes more time and disk space; their upstream licenses also apply.")
            install_btn = gr.Button("Install required models", variant="primary", interactive=False, elem_id="install-models")
            setup_report = gr.Textbox(label="Model setup report", value="No downloads started. Existing models are kept when you install or retry.",
                                      lines=4, max_lines=10, interactive=False, elem_id="model-setup-report")
            with gr.Row():
                recheck_btn = gr.Button("Re-check models", size="sm")
                launch_info = gr.Markdown("Required models ready" if init_can else "Setup needed before your first swap")

        with gr.Row(equal_height=False, elem_classes="workspace"):
            with gr.Column(scale=6, min_width=280):
                gr.HTML('<div class="step-heading"><span>01</span><h2>Choose your scene</h2><small>IMAGES OR VIDEO</small></div>')
                with gr.Tabs() as scene_tabs:
                    with gr.Tab("Images", id="images"):
                        targets = gr.Gallery(label="Target images", columns=3, rows=1, height=300,
                            object_fit="contain", allow_preview=True, show_download_button=False,
                            show_fullscreen_button=True, file_types=["image"], type="filepath",
                            interactive=True, elem_classes="vs-targets")
                        with gr.Row():
                            upload_btn = gr.UploadButton("+ Add more images", file_count="multiple", file_types=["image"], size="sm")
                            clear_btn = gr.Button("Clear targets", size="sm")
                        face_picker = gr.Gallery(label="Faces in the first target (left → right)", columns=6, height=130,
                                                 object_fit="contain", allow_preview=False, interactive=False,
                                                 show_download_button=False, elem_classes="vs-faces")
                        face_note = gr.Markdown("Add a target image to see its faces here.", elem_classes="quiet-note")
                    with gr.Tab("Video", id="video"):
                        video_in = gr.Video(label="Target video", sources=["upload"], height=300)
                        with gr.Row():
                            smoothing = gr.Slider(0, .9, value=.5, step=.05, label="Temporal smoothing",
                                                  info="Steadies the face alignment between frames.")
                            every = gr.Slider(1, 4, value=1, step=1, label="Swap every Nth frame",
                                              info="2–4 for quick previews; untouched frames are copied.")
                        gr.Markdown("Video uses Manual quality (your selected model once per face). Index and gender "
                                    "picks are resolved on the first frame and then tracked by identity.", elem_classes="quiet-note")
            with gr.Column(scale=4, min_width=280):
                gr.HTML('<div class="step-heading"><span>02</span><h2>Add people</h2><small>REFERENCES OR SAVED IDENTITIES</small></div>')
                ident_choices = identity_choices()
                ref_a = gr.Gallery(label="Person A · reference photos (required unless a saved identity is picked)", columns=4, height=170,
                                   object_fit="contain", file_types=["image"], type="filepath", interactive=True,
                                   show_download_button=False, elem_id="primary-reference")
                with gr.Row():
                    id_a = gr.Dropdown(ident_choices, value="(none)", label="…or saved identity A", scale=2)
                    source_index = gr.Slider(0, 10, value=0, step=1, label="Face index inside references", scale=1)
                with gr.Row():
                    save_name = gr.Textbox(label="Save Person A as", placeholder="e.g. Maya", scale=2)
                    save_btn = gr.Button("Save identity", size="sm", scale=1)
                    delete_id_btn = gr.Button("Delete selected", size="sm", scale=1)
                identity_note = gr.Markdown("Saved identities keep only the face embedding, never the photos.", elem_classes="quiet-note")
                with gr.Accordion("More people · B, C, D (for mapping several faces)", open=False):
                    ref_b = gr.Gallery(label="Person B", columns=4, height=120, object_fit="contain", file_types=["image"],
                                       type="filepath", interactive=True, show_download_button=False)
                    id_b = gr.Dropdown(ident_choices, value="(none)", label="…or saved identity B")
                    ref_c = gr.Gallery(label="Person C", columns=4, height=120, object_fit="contain", file_types=["image"],
                                       type="filepath", interactive=True, show_download_button=False)
                    id_c = gr.Dropdown(ident_choices, value="(none)", label="…or saved identity C")
                    ref_d = gr.Gallery(label="Person D", columns=4, height=120, object_fit="contain", file_types=["image"],
                                       type="filepath", interactive=True, show_download_button=False)
                    id_d = gr.Dropdown(ident_choices, value="(none)", label="…or saved identity D")

        gr.HTML('<div class="step-heading settings-heading"><span>03</span><h2>Make it yours</h2></div>')
        with gr.Row():
            quality_mode = gr.Radio(["Best likeness", "Manual"], value="Best likeness", label="Quality mode",
                info="Best likeness compares resolutions up to the face's real size and keeps the sharpest likeness. Manual runs your selected model once.")
            swapper_model = gr.Dropdown(choices=SWAPPER_CHOICES, value="inswapper_128.onnx", label="Face swap model",
                info="@256 / @512 set the aligned face resolution, not the image dimensions.", filterable=True)
        with gr.Group():
            assign_mode = gr.Radio(list(ASSIGN_MODES), value=ASSIGN_MODES[0], label="Which face gets replaced",
                                   info="Faces are numbered 0.. from left to right (see the face strip under the targets).")
            with gr.Row():
                target_index = gr.Slider(0, 10, value=0, step=1, label="Target face index")
                anchor_image = gr.Image(type="pil", label="Anchor photo · the person to replace", height=120,
                                        sources=["upload"], visible=False)
            with gr.Row(visible=False) as map_row:
                map_dropdowns = [gr.Dropdown(["Keep"] + [f"Person {s}" for s in PERSON_SLOTS], value="Keep",
                                             label=f"Face {i}", visible=False, min_width=110) for i in range(MAX_MAP_FACES)]
        with gr.Accordion("Fine-tune · blending, restoration, masks", open=False):
            with gr.Row():
                min_face_size = gr.Slider(24, 256, value=48, step=4, label="Minimum face size")
                verify_threshold = gr.Slider(0, 1, value=.30, step=.01, label="Identity threshold",
                    info="Flags low similarity in the report; results are still shown for review.")
                color_strength = gr.Slider(0, 1, value=.25, step=.05, label="Color matching")
                sharpen_strength = gr.Slider(0, 1, value=.5, step=.05, label="Sharpening")
            with gr.Row():
                restorer = gr.Dropdown(restorer_choices, value=installed_default, label="Face restorer",
                    info="Accepted only when reference similarity is preserved. GFPGAN/GPEN keep identity better; CodeFormer cleans more.")
                codeformer_weight = gr.Slider(0, 1, value=.9, step=.05, label="CodeFormer fidelity (w)")
                restoration_blend = gr.Slider(0, 1, value=.7, step=.05, label="Restoration blend",
                    info="How much of the restored face is mixed in (1 = full replacement).")
            with gr.Row():
                mask_mode = gr.Radio(list(MASK_MODES), value=list(MASK_MODES)[0], label="Blend mask")
                blend_mode = gr.Radio(list(BLEND_MODES), value=list(BLEND_MODES)[0], label="Blending")
                occluder_enabled = gr.Checkbox(value=True, label="Preserve objects in front of the face · XSeg")
            with gr.Row():
                mouth_mode = gr.Radio(list(MOUTH_MODES), value=list(MOUTH_MODES)[0], label="Mouth handling")
                eye_mode = gr.Radio(list(EYE_MODES), value=list(EYE_MODES)[0], label="Eyes")
            with gr.Row():
                precision = gr.Radio(list(PRECISION_MODES), value=list(PRECISION_MODES)[0], label="InSwapper precision",
                    info="FP16 twin of the same weights: ~2x faster pixel-boost tiles on NVIDIA GPUs.")
                dealias = gr.Checkbox(value=False, label="Soften pixel-boost tile grid (median · costs sharpness)")
                debug_masks = gr.Checkbox(value=False, label="Show masks (debug view)")
            with gr.Row():
                preset_dd = gr.Dropdown(list_presets(), label="Presets", scale=2)
                preset_name = gr.Textbox(label="Save current settings as", placeholder="e.g. portraits", scale=2)
                load_preset_btn = gr.Button("Load", size="sm")
                save_preset_btn = gr.Button("Save", size="sm")
                delete_preset_btn = gr.Button("Delete", size="sm")
            preset_note = gr.Markdown("Settings are also remembered as the `last` preset after each run.", elem_classes="quiet-note")
        with gr.Row():
            swap = gr.Button("Run image swap →", variant="primary", interactive=init_can, elem_id="run-swap", scale=3)
            swap_video_btn = gr.Button("Run video swap →", variant="secondary", interactive=init_can, scale=2)
            cancel_btn = gr.Button("Cancel", size="sm", scale=1)
        gr.Markdown("First run loads the models. Results appear as each image finishes.", elem_classes="quiet-note")

        gr.HTML('<div class="step-heading results-heading"><span>04</span><h2>Your results</h2><small>FULL-RESOLUTION PNG · MP4</small></div>')
        with gr.Tabs():
            with gr.Tab("Images"):
                output = gr.Gallery(label="Swapped images", format="png", columns=3, height=350, object_fit="contain",
                    allow_preview=True, show_fullscreen_button=True, show_download_button=True,
                    interactive=False, elem_classes="vs-gallery")
                compare = gr.ImageSlider(label="Before / after · click a result above to compare", type="filepath",
                                         interactive=False, height=420)
                mask_gallery = gr.Gallery(label="Masks (debug)", columns=4, height=160, object_fit="contain",
                                          visible=False, interactive=False, show_download_button=True)
            with gr.Tab("Video"):
                video_out = gr.Video(label="Swapped video", interactive=False, height=360)
                video_preview = gr.Image(label="Live preview", interactive=False, height=240)
        status = gr.Textbox(label="Processing report", value=("Your workspace is ready. Add a target and a reference to begin." if init_can else
                             "Open Model setup above and install the required models before your first swap."),
                            lines=3, interactive=False, elem_classes="vs-status")
        pairs_state = gr.State([])
        gr.HTML('<footer class="studio-footer"><span>visenlo <b>·</b> made to run locally</span>'
                '<span>Results save to outputs/ · Use images you own or have permission to edit.</span></footer>')

        settings_components = [quality_mode, swapper_model, min_face_size, verify_threshold, color_strength,
                               restorer, codeformer_weight, restoration_blend, sharpen_strength, occluder_enabled,
                               mouth_mode, eye_mode, mask_mode, blend_mode, precision, dealias, debug_masks]
        people_components = [ref_a, ref_b, ref_c, ref_d, id_a, id_b, id_c, id_d, anchor_image, assign_mode,
                             target_index, source_index]

        def _append_uploads(new_files, current_gallery):
            if not new_files:
                return current_gallery or []
            if not isinstance(new_files, (list, tuple)):
                new_files = [new_files]
            current = list(current_gallery or [])
            seen = {_target_path(item) for item in current}
            for item in new_files:
                path = _target_path(item)
                if path and path not in seen:
                    current.append((path, None))
                    seen.add(path)
            return current

        def _assign_visibility(mode):
            mode = str(mode)
            return (gr.update(visible=mode.startswith("Single")),
                    gr.update(visible=mode.startswith("Looks")),
                    gr.update(visible=mode.startswith("Map")))

        recheck_btn.click(check_models, [swapper_model], [model_html, launch_info, swap], api_name="check_models")
        swapper_model.change(check_models, [swapper_model], [model_html, launch_info, swap], api_name=False)
        licenses_accepted.change(lambda accepted: gr.update(interactive=bool(accepted) and not _setup_lock.locked()),
                                 licenses_accepted, install_btn, api_name=False)
        setup_event = install_btn.click(install_models, [licenses_accepted, include_optional, swapper_model],
                          [setup_report, model_html, launch_info, swap, install_btn],
                          concurrency_limit=1, concurrency_id="model_setup", api_name="install_models")
        # Re-read current controls if the user changed them while downloading.
        setup_complete = setup_event.then(check_models, [swapper_model],
                                           [model_html, launch_info, swap], api_name=False)
        setup_complete.then(lambda accepted: gr.update(interactive=bool(accepted) and not _setup_lock.locked()),
                            licenses_accepted, install_btn, api_name=False)
        assign_mode.change(_assign_visibility, assign_mode, [target_index, anchor_image, map_row], api_name=False)
        debug_masks.change(lambda on: gr.update(visible=bool(on)), debug_masks, mask_gallery, api_name=False)
        upload_btn.upload(_append_uploads, [upload_btn, targets], [targets])
        clear_btn.click(lambda: [], outputs=[targets])
        targets.change(detect_target_faces, [targets, min_face_size], [face_picker, face_note] + map_dropdowns,
                       api_name="detect_faces", concurrency_limit=1)
        save_btn.click(save_identity, [save_name, ref_a, source_index], [identity_note, id_a, id_b, id_c, id_d],
                       api_name="save_identity")
        delete_id_btn.click(delete_identity, [id_a], [identity_note, id_a, id_b, id_c, id_d], api_name=False)
        load_preset_btn.click(apply_preset, [preset_dd], settings_components, api_name=False)
        save_preset_btn.click(store_preset, [preset_name] + settings_components, [preset_dd, preset_note], api_name=False)
        delete_preset_btn.click(remove_preset, [preset_dd], [preset_dd, preset_note], api_name=False)
        run_event = swap.click(run_images, [targets] + people_components + map_dropdowns + settings_components,
                               [output, status, pairs_state, compare, mask_gallery],
                               concurrency_limit=1, api_name="run_swap")
        video_event = swap_video_btn.click(run_video, [video_in] + people_components + [smoothing, every] + settings_components,
                                           [video_out, video_preview, status], concurrency_limit=1, api_name="run_video")
        cancel_btn.click(cancel_run, None, status, cancels=[run_event, video_event], api_name=False)
        output.select(show_pair, [pairs_state], compare, api_name=False)
    return app


def main():
    parser = argparse.ArgumentParser(description="Visenlo standalone local app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--auth", help="user:password required to open the studio (recommended with --host 0.0.0.0 or --share)")
    parser.add_argument("--no-browser", action="store_true", help="Print the local address without opening a browser")
    args = parser.parse_args()
    auth = None
    if args.auth:
        user, _, password = args.auth.partition(":")
        if not user or not password:
            parser.error("--auth expects user:password")
        auth = (user, password)
    elif args.share or args.host not in ("127.0.0.1", "localhost"):
        logger.warning("The studio is reachable from other machines without --auth user:password")
    logger.info("Visenlo Swap Engine v2")
    logger.info("  models dir:  %s", MODELS)
    logger.info("  server:      %s:%d", args.host, args.port)
    logger.info("  share mode:  %s", args.share)
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share,
                      inbrowser=not args.no_browser, auth=auth)


if __name__ == "__main__":
    main()
