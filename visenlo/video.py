"""Video face swap: frame loop with identity tracking and temporal smoothing.

Frames are decoded with OpenCV, swapped through the same per-face stages as
still images (Manual quality mode — Best likeness would flicker), written to
a temporary MP4 and finally re-encoded with ffmpeg (libx264) while copying
the original audio track. Without ffmpeg the silent MP4 is kept as is.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .engine import cosine
from .pipeline import FaceAssignment, SwapCancelled, parse_swapper_spec

logger = logging.getLogger("visenlo.video")


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    seconds: float


@dataclass
class VideoProgress:
    frame: int
    total: int
    swapped_faces: int
    elapsed: float
    preview: np.ndarray | None = None

    @property
    def fraction(self):
        return 0.0 if self.total <= 0 else min(1.0, self.frame / self.total)


@dataclass
class _Track:
    assignment: FaceAssignment
    anchor: np.ndarray | None
    kps: np.ndarray | None = None
    missed: int = 0
    swaps: int = 0


@dataclass
class VideoResult:
    output_path: str
    info: VideoInfo
    frames_done: int
    swapped_faces: int
    seconds: float
    audio_copied: bool
    notes: list[str] = field(default_factory=list)


def probe(path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return VideoInfo(width, height, fps, frames, frames / fps if fps else 0.0)


def smooth_kps(previous, current, alpha=.5, max_jump=.12):
    """Exponential smoothing of the 5 alignment points. Large jumps (cuts,
    fast motion) bypass the filter so the swap never lags behind the face."""
    if previous is None:
        return np.asarray(current, np.float32)
    previous = np.asarray(previous, np.float32)
    current = np.asarray(current, np.float32)
    scale = max(float(np.linalg.norm(current[1] - current[0])), 1.0)
    if float(np.abs(current - previous).max()) > scale * max_jump * 4:
        return current
    return (alpha * previous + (1 - alpha) * current).astype(np.float32)


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _mux(temp_video, source, output, fps, crf=18):
    """Re-encode with libx264 and copy the source audio (if any)."""
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(temp_video), "-i", str(source),
               "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-r", f"{fps:.3f}", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[:400] or "ffmpeg failed")


def swap_video(pipeline, input_path, output_path, assignments, swapper_name="inswapper_128.onnx",
               progress: Callable[[VideoProgress], None] | None = None,
               cancel: threading.Event | None = None, smoothing=.5, start_frame=0, end_frame=None,
               every=1, preview_every=15, crf=18, max_missed=10) -> VideoResult:
    """Swap faces in a video file.

    `assignments` use the same FaceAssignment objects as still images. Index
    and gender modes are resolved on the first frame that contains faces and
    then tracked by identity similarity; "all" swaps every face in every frame.
    """
    info = probe(input_path)
    output_path = str(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    cancel = cancel or pipeline.cancel_event
    cancel.clear()
    started = time.perf_counter()
    notes = []
    with pipeline._lock:
        cfg = pipeline.config
        import copy
        video_cfg = copy.deepcopy(cfg)
        video_cfg.quality_mode = "manual"
        video_cfg.save_swaps = False
        video_cfg.debug_masks = False
        pipeline.config = video_cfg
        try:
            model_file, boost = parse_swapper_spec(swapper_name)
            swapper = pipeline._ensure_swapper(model_file, boost)
            tracks = [_Track(a, a.anchor if a.match_mode == "similar" else None) for a in assignments]
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {input_path}")
            total = info.frames if end_frame is None else min(info.frames, int(end_frame))
            temp_dir = tempfile.mkdtemp(prefix="visenlo-video-")
            temp_video = os.path.join(temp_dir, "swapped.mp4")
            writer = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height))
            if not writer.isOpened():
                raise RuntimeError("Could not open a video writer (mp4v)")
            frame_index = 0
            swapped_faces = 0
            last_preview = None
            try:
                if start_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
                    frame_index = int(start_frame)
                while True:
                    if cancel.is_set():
                        raise SwapCancelled("Video swap cancelled")
                    ok, frame = cap.read()
                    if not ok or (end_frame is not None and frame_index >= end_frame):
                        break
                    out = frame
                    if frame_index % max(1, int(every)) == 0:
                        out, count = _swap_frame(pipeline, frame, tracks, swapper, swapper_name, model_file,
                                                 boost, video_cfg, smoothing, max_missed)
                        swapped_faces += count
                    writer.write(out)
                    frame_index += 1
                    if progress is not None and (frame_index % max(1, preview_every) == 0 or frame_index == total):
                        last_preview = out
                        progress(VideoProgress(frame_index, total, swapped_faces, time.perf_counter() - started, out))
            finally:
                cap.release()
                writer.release()
            audio = False
            if has_ffmpeg():
                try:
                    _mux(temp_video, input_path, output_path, info.fps, crf)
                    audio = True
                except Exception as exc:
                    notes.append(f"ffmpeg re-encode failed ({exc}); kept the silent mp4v file")
                    shutil.move(temp_video, output_path)
            else:
                notes.append("ffmpeg not found; output has no audio and uses the mp4v codec")
                shutil.move(temp_video, output_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
            for track in tracks:
                if track.swaps == 0:
                    notes.append(f"{track.assignment.reference.label or 'reference'}: no matching face found in any frame")
            return VideoResult(output_path, info, frame_index, swapped_faces, time.perf_counter() - started, audio, notes)
        finally:
            pipeline.config = cfg


def _index_of(records, record):
    return next(i for i, item in enumerate(records) if item is record)


def _swap_frame(pipeline, frame, tracks, swapper, swapper_name, model_file, boost, cfg, smoothing, max_missed):
    targets = pipeline._detect_adaptive(frame)
    count = 0
    if not targets:
        for track in tracks:
            track.missed += 1
        return frame, 0
    taken = []
    out = frame
    for track in tracks:
        assignment = track.assignment
        record = None
        mode = assignment.match_mode
        if mode == "all":
            for candidate in targets:
                if any(candidate is t for t in taken):
                    continue
                out, _, _ = pipeline._swap_face(out, candidate, assignment.reference, swapper, swapper_name,
                                                model_file, boost, _index_of(targets, candidate), cfg)
                taken.append(candidate)
                count += 1
                track.swaps += 1
            continue
        if track.anchor is None:
            # First sighting: resolve index/gender once, then track by identity.
            try:
                plan = pipeline.resolve_assignments(targets, [FaceAssignment(assignment.reference, mode,
                                                                            assignment.target_index)], cfg)
            except ValueError:
                track.missed += 1
                continue
            record = plan[0][0]
            track.anchor = np.asarray(record.embedding, np.float32)
        else:
            pool = [t for t in targets if not any(t is k for k in taken) and t.embedding is not None]
            if pool:
                scores = [cosine(track.anchor, t.embedding) for t in pool]
                best = int(np.argmax(scores))
                if scores[best] >= cfg.similar_threshold:
                    record = pool[best]
                    # Drift the anchor slowly so pose/lighting changes stay tracked.
                    track.anchor = track.anchor * .9 + np.asarray(record.embedding, np.float32) * .1
        if record is None:
            track.missed += 1
            if track.missed > max_missed:
                track.kps = None
            continue
        track.missed = 0
        record.face.kps = smooth_kps(track.kps, record.face.kps, smoothing)
        track.kps = record.face.kps
        out, _, _ = pipeline._swap_face(out, record, assignment.reference, swapper, swapper_name,
                                        model_file, boost, _index_of(targets, record), cfg)
        taken.append(record)
        count += 1
        track.swaps += 1
    return out, count
