"""Visenlo pipeline orchestration.

Stages (one target face):
  analyse references -> detect target faces -> select face(s) -> build the
  per-face TargetContext (masks, ROI) once -> swap candidates (pixel boosts x
  finishes) -> score in crop space -> optional guarded restoration -> paste.

Everything after the swap works on the face ROI only; identity scoring embeds
the aligned crop at the known landmarks instead of re-detecting the frame.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, NamedTuple

import cv2
import numpy as np

from .engine import (
    FaceRecord,
    build_face_mask,
    clamp_bbox,
    color_match,
    composite_region_mask,
    cosine,
    crop_roi,
    dense_landmarks,
    parse_face,
    quality_score,
    recover_occlusions,
    refine_kps,
    region_blend_mask,
    sharpen_face_region,
    sharpness,
    union_roi,
    virtual_face,
    weighted_identity,
)
from .restoration import DEFAULT_RESTORER, RESTORERS, FaceRestorer
from .restoration.utils import paste_restored_back
from .swappers import (
    MODELS as SWAP_MODELS,
    FaceDancerSwapper,
    InswapperBackend,
    MobileFaceSwapper,
    OnnxSwapper,
    Reswapper256,
    missing_assets,
)
from .runtime import preferred_providers, provider_names, uses_cuda

logger = logging.getLogger("visenlo")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

_ARCFACE_BASE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], np.float32)

MATCH_MODES = ("index", "gender", "similar", "all")


class SwapCancelled(RuntimeError):
    """Raised when the caller's cancel event is set between stages."""


def allocate_output_path(directory, ext=".png", stamp=None):
    """Return a non-clashing '<date>_<NN><ext>' path inside `directory`.

    Numbering starts at 00 for each date and keeps incrementing, skipping
    names already on disk so restarts continue the sequence instead of
    overwriting previous swaps. Uses atomic O_EXCL to avoid races between
    concurrent processes/threads.
    """
    import errno

    os.makedirs(directory, exist_ok=True)
    prefix = stamp or time.strftime("%Y-%m-%d")
    index = 0
    while True:
        cand = os.path.join(directory, f"{prefix}_{index:02d}{ext}")
        try:
            # Atomic create — fails with EEXIST if another process already claimed it.
            # Keep the zero-byte reservation so concurrent allocators skip it; caller
            # will overwrite with the real PNG (Path.write_bytes / cv2.tofile both truncate).
            fd = os.open(cand, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            os.close(fd)
            return cand
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                index += 1
                if index > 9999:
                    raise RuntimeError(f"Output directory exhausted: {directory}")
                continue
            raise


def parse_swapper_spec(swapper_name):
    """Return the model filename and optional pixel-boost factor."""
    model_file, _, boost_raw = swapper_name.partition("@")
    if not os.path.splitext(model_file)[1]:
        model_file += ".onnx"
    boost = 1
    if boost_raw:
        if not boost_raw.isdigit():
            raise ValueError(f"Invalid pixel-boost suffix: @{boost_raw}")
        requested = int(boost_raw)
        # UI suffixes name the desired aligned resolution, not a scale factor.
        if requested in (256, 512, 1024, 2048):
            native = SWAP_MODELS[model_file].size if model_file in SWAP_MODELS else 128
            if requested < native or requested % native:
                raise ValueError(f"{model_file} requires a multiple of its native {native}px resolution")
            boost = requested // native
        elif requested in (2, 4, 8, 16):
            boost = requested
        else:
            raise ValueError(f"Unsupported pixel-boost size: @{requested}")
    return model_file, boost


def candidate_boosts_for(swapper_name, model_file, boost, quality_mode):
    """Resolve the exact boost factors to run for this request.

    A suffix is an explicit user choice and always wins. Registered backends
    compare supported native/boost resolutions up to 512px in Best likeness.
    """
    explicit = bool(str(swapper_name).partition("@")[2])
    is_inswapper = os.path.basename(model_file) == "inswapper_128.onnx"
    if str(quality_mode).strip().lower() == "best_likeness" and is_inswapper and not explicit:
        return (1, 2, 4)
    spec = SWAP_MODELS.get(model_file)
    if str(quality_mode).strip().lower() == "best_likeness" and not explicit and spec and spec.family != "reswapper":
        return (1,) + tuple(size // spec.size for size in (256, 512) if size > spec.size and size % spec.size == 0)
    return (int(boost),)


class Candidate(NamedTuple):
    boost: int
    profile: str
    image: np.ndarray
    score: float | None
    raw_score: float | None
    sharpness: float = 0.0
    debug: dict | None = None


def select_likeness_candidate(candidates, tolerance=.01):
    """Pick the best candidate from ``(boost, profile, image, final, raw[, sharpness])``.

    Scores within ``tolerance`` are indistinguishable for ArcFace, so among
    those we keep the normal visual finish first, then the *sharpest* result
    (higher pixel boost when sharpness is unknown). The previous rule picked
    the lowest boost, which systematically returned the blurriest swap.
    """
    items = [item if isinstance(item, Candidate) else Candidate(*item) for item in candidates]
    verified = [item for item in items if item.score is not None]
    if not verified:
        return candidates[0]
    best_score = max(item.score for item in verified)
    eligible = [item for item in verified if item.score >= best_score - tolerance]
    chosen = min(eligible, key=lambda item: (item.profile != "default", -item.sharpness, -item.boost))
    return candidates[items.index(chosen)]


def gender_label(gender):
    """Human label for an insightface gender id (0 female, 1 male)."""
    if gender is None:
        return "?"
    return "M" if int(gender) == 1 else "F"


def source_gender_vote(records):
    """Majority-vote the gender (0 female, 1 male) across reference records.

    Ties fall back to the first known gender; returns None when no record
    carries gender information (e.g. an analysis pack without genderage).
    """
    known = [record.gender for record in records if record.gender is not None]
    if not known:
        return None
    females = sum(1 for gender in known if int(gender) == 0)
    males = len(known) - females
    if females == males:
        return int(known[0])
    return 0 if females > males else 1


def select_target_record(records, match_mode="index", target_index=0, source_gender=None,
                         anchor=None, threshold=.25, exclude=()):
    """Choose the single target face to swap.

    match_mode="index" (default): the face at `target_index`, bounds-checked.
    match_mode="gender": the leftmost face matching `source_gender` (records are
    already sorted left-to-right), or None when nothing matches.
    match_mode="similar": the face whose embedding is closest to `anchor`
    (cosine >= threshold), or None.
    """
    pool = [record for record in records if id(record) not in {id(r) for r in exclude}]
    if match_mode not in ("gender", "similar"):
        if not 0 <= int(target_index) < len(records):
            raise ValueError(
                f"Target face index {target_index} is out of range; "
                f"the target has {len(records)} face(s)")
        return records[int(target_index)]
    if match_mode == "gender":
        if source_gender is None:
            return None
        for record in pool:
            if record.gender == source_gender:
                return record
        return None
    if match_mode == "similar":
        if anchor is None:
            return None
        scored = [(cosine(anchor, record.embedding), record) for record in pool
                  if record.embedding is not None]
        if not scored:
            return None
        best_score, best = max(scored, key=lambda item: item[0])
        return best if best_score >= threshold else None
    return None


def arcface_kps(size):
    """Five-point template of an arcface-aligned crop of `size` px.

    Mirrors insightface's face_align.estimate_norm so recognition embeddings
    computed directly on such crops use the correct landmark positions.
    """
    if size % 112 == 0:
        ratio, diff_x = size / 112.0, 0.0
    else:
        ratio = size / 128.0
        diff_x = 8.0 * ratio
    dst = _ARCFACE_BASE * ratio
    dst[:, 0] += diff_x
    return dst.astype(np.float32)


@dataclass
class PipelineConfig:
    min_face_size: int = 48
    reference_quality: float = .20
    verification_threshold: float = .30
    color_strength: float = .25
    codeformer_enabled: bool = True
    codeformer_weight: float = .9
    codeformer_verify_identity: bool = True
    # Restorer key from visenlo.restoration.RESTORERS and how much of the
    # restored crop is mixed into the swap (1.0 = full replacement).
    restorer: str = DEFAULT_RESTORER
    restoration_blend: float = .7
    sharpen_strength: float = .5
    det_size: int = 640
    det_size_max: int = 1280
    occluder_enabled: bool = True
    # Best likeness compares native/256/512 pixel-boost candidates before the
    # optional restorer. Manual preserves the user-selected swapper setting.
    quality_mode: str = "best_likeness"
    likeness_tolerance: float = .01
    # Swap the reference lips by default, but retain only target teeth/mouth
    # interior where the 128px generator is most likely to create artifacts.
    mouth_mode: str = "swap_lips_keep_teeth"
    # "swap" (default) or "keep_target_eyes" to preserve the target's gaze.
    eye_mode: str = "swap"
    # Deprecated compatibility setting for callers from v1.  New callers use
    # mouth_mode; this remains so existing integrations do not crash.
    keep_target_mouth: bool = True
    # Paste-back: "box" is the classic erode+feather square; "region" also
    # restricts the blend to the parsed face interior. "multiband" blends a
    # Laplacian pyramid instead of a single alpha feather.
    mask_mode: str = "region"
    blend_mode: str = "multiband"
    # Replace SCRFD's 5 points with centroids of the 106-point contours.
    refine_landmarks: bool = True
    # "auto" uses inswapper_128_fp16.onnx on CUDA when present; "fp32"/"fp16".
    inswapper_precision: str = "auto"
    # Optional median over the tile footprint. Off by default: it removes the
    # faint polyphase grid but also most of the sharpness pixel boost adds.
    boost_dealias: bool = False
    # Best likeness never tries boosts that would upsample the source face
    # pixels by more than this factor (explicit @256/@512 still run as asked).
    boost_slack: float = 2.0
    similar_threshold: float = .25
    debug_masks: bool = False
    save_swaps: bool = True


@dataclass
class ReferenceSet:
    """One person: aggregated identity plus the accepted reference records."""
    identity: np.ndarray
    records: list[FaceRecord]
    images: list[np.ndarray]
    label: str = ""

    @property
    def gender(self):
        return source_gender_vote(self.records)

    @property
    def best_index(self) -> int:
        return max(range(len(self.records)), key=lambda i: self.records[i].quality)


@dataclass
class FaceAssignment:
    """Which target face a ReferenceSet should replace."""
    reference: ReferenceSet
    match_mode: str = "index"
    target_index: int = 0
    anchor: np.ndarray | None = None


@dataclass
class FaceReport:
    face_index: int
    bbox: tuple
    boost: int
    profile: str
    similarity: float | None
    raw_similarity: float | None
    verified: bool
    restoration: str
    seconds: float
    label: str = ""
    extras: list[str] = field(default_factory=list)

    def line(self):
        sim = "n/a" if self.similarity is None else f"{self.similarity:.3f}"
        tag = f"face {self.face_index}" + (f" ← {self.label}" if self.label else "")
        return (f"{tag}: x{self.boost} {self.profile} | similarity {sim} | "
                f"{'verified' if self.verified else 'LOW CONFIDENCE'} | restoration {self.restoration} | "
                f"{self.seconds:.2f}s")


@dataclass
class SwapOutput:
    image: np.ndarray
    status: str
    faces: list[FaceReport]
    debug: dict = field(default_factory=dict)
    saved_path: str | None = None
    target_faces: list[FaceRecord] = field(default_factory=list)


@dataclass
class TargetContext:
    """Everything about one target face that does not depend on the swap."""
    record: FaceRecord
    kps: np.ndarray
    roi: tuple
    face_mask: np.ndarray
    interior: np.ndarray
    occlusion: np.ndarray | None
    mouth: np.ndarray | None
    eyes: np.ndarray | None
    face_px: int
    ffhq_matrix: np.ndarray | None
    parsing_available: bool
    region_crop_labels: np.ndarray | None = None


def _shift_record(record, dx, dy):
    """A light view of `record` translated into ROI coordinates."""
    x1, y1, x2, y2 = record.bbox
    landmarks = getattr(record, "landmarks", None)
    landmarks = None if landmarks is None or not len(landmarks) else np.asarray(landmarks, np.float32) - (dx, dy)
    kps = getattr(getattr(record, "face", None), "kps", None)
    face = SimpleNamespace(kps=None if kps is None else np.asarray(kps, np.float32) - (dx, dy))
    return SimpleNamespace(bbox=(x1 - dx, y1 - dy, x2 - dx, y2 - dy), landmarks=landmarks,
                           masks=getattr(record, "masks", None), face=face,
                           parsing_available=getattr(record, "parsing_available", False))


class VisenloPipeline:
    def __init__(self, models_path: str, config: PipelineConfig | None = None):
        import copy
        self.models_path = models_path
        self.config = copy.deepcopy(config) if config is not None else PipelineConfig()
        # Swapped results land in outputs/ next to the project root (the parent
        # of the models directory), named <date>_<NN>.png.
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(models_path)), "outputs")
        self._lock = threading.RLock()
        self._analysis = None
        self._swapper = None
        self._swapper_path = None
        self._swapper_precision = None
        self._providers = None
        self._parser = None
        self._occluder = None
        self._restorers: dict[str, FaceRestorer] = {}
        self.cancel_event = threading.Event()
        self.on_progress: Callable[[str], None] | None = None
        os.makedirs(models_path, exist_ok=True)

    # ------------------------------------------------------------------ setup
    def _check_cancel(self):
        if self.cancel_event.is_set():
            raise SwapCancelled("Swap cancelled")

    def _progress(self, message):
        if self.on_progress is not None:
            try:
                self.on_progress(message)
            except Exception:  # pragma: no cover - UI callback must not break the swap
                pass

    def _fallback_providers_to_cpu(self, exc: Exception) -> bool:
        """If CUDA provider failed (missing libcublas/cudnn), fall back to CPU once."""
        msg = str(exc).lower()
        if any(k in msg for k in ("cuda", "cublas", "cudnn", "provider")) and self._providers and uses_cuda(self._providers):
            logger.warning("CUDA provider failed (%s) — falling back to CPUExecutionProvider", exc)
            self._providers = ["CPUExecutionProvider"]
            return True
        return False

    def update_config(self, config: PipelineConfig):
        """Swap configuration atomically; safe against a running process()."""
        import copy
        with self._lock:
            self.config = copy.deepcopy(config)

    def _ensure_parser(self):
        if self._parser is not None:
            return
        path = os.path.join(self.models_path, "bisenet_resnet_34.onnx")
        if not os.path.isfile(path):
            logger.info("face parsing model not found (%s) - geometric masks will be used", path)
            return
        try:
            from .parsing import BisenetParser
            self._parser = BisenetParser(path, self._providers)
            logger.info("Face parsing ready: bisenet_resnet_34")
        except Exception as exc:
            logger.warning("Face parsing unavailable (%s)", exc)

    def _ensure_occluder(self):
        if self._occluder is not None:
            return
        path = os.path.join(self.models_path, "xseg_1.onnx")
        if not os.path.isfile(path):
            logger.info("occlusion model not found (%s) - occluder mask disabled", path)
            return
        try:
            from .parsing import XSegOccluder
            self._occluder = XSegOccluder(path, self._providers)
            logger.info("Occlusion masking ready: xseg_1")
        except Exception as exc:
            logger.warning("Occlusion masking unavailable (%s)", exc)

    def _ensure_restorer(self, key=None) -> FaceRestorer:
        key = key or self.config.restorer or DEFAULT_RESTORER
        if key not in RESTORERS:
            raise KeyError(f"Unknown restorer '{key}'")
        restorer = self._restorers.get(key)
        if restorer is None:
            logger.info("Loading restorer %s ...", RESTORERS[key].label)
            restorer = FaceRestorer(self.models_path, key, self._providers, self.config.codeformer_weight)
            self._restorers[key] = restorer
        restorer.weight = self.config.codeformer_weight
        return restorer

    def _ensure_codeformer(self):
        return self._ensure_restorer("codeformer")

    def _ensure_analysis(self):
        try:
            from insightface.app import FaceAnalysis
            from insightface.model_zoo import model_zoo
        except ImportError as exc:
            raise RuntimeError("Install insightface and onnxruntime in this application's environment") from exc
        from .runtime import session_options
        insight_root = os.path.join(self.models_path, "insightface")
        if self._providers is None:
            self._providers = preferred_providers()
            logger.info("ONNX providers: %s", provider_names(self._providers))
        if self._analysis is not None:
            return self._analysis
        logger.info("Loading FaceAnalysis (buffalo_l) from %s ...", insight_root)

        def _init_analysis(providers):
            # FaceAnalysis.__init__ silently downloads a missing model pack
            # and discards sess_options in InsightFace 0.7.3. Load the local
            # pack through its router so setup is explicit and CPU threads
            # stay bounded without patching the installed library.
            from pathlib import Path
            model_dir = Path(insight_root) / "models" / "buffalo_l"
            required = ("det_10g.onnx", "w600k_r50.onnx")
            missing = [name for name in required if not (model_dir / name).is_file()]
            if missing:
                raise FileNotFoundError("Install the buffalo_l analysis pack before swapping; missing: " + ", ".join(missing))
            analysis = FaceAnalysis.__new__(FaceAnalysis)
            analysis.model_dir = str(model_dir)
            analysis.models = {}
            for model_path in sorted(model_dir.glob("*.onnx")):
                model = model_zoo.ModelRouter(str(model_path)).get_model(
                    providers=providers, sess_options=session_options())
                if model is not None:
                    analysis.models.setdefault(model.taskname, model)
            if not {"detection", "recognition"} <= analysis.models.keys():
                raise RuntimeError("The buffalo_l pack must provide detection and recognition models")
            analysis.det_model = analysis.models["detection"]
            use_cuda = uses_cuda(analysis.det_model.session.get_providers())
            if not use_cuda:
                self._providers = ["CPUExecutionProvider"]
            analysis.prepare(ctx_id=0 if use_cuda else -1, det_size=(640, 640))
            logger.info("FaceAnalysis ready (CUDA=%s)", use_cuda)
            return analysis

        try:
            self._analysis = _init_analysis(self._providers)
        except Exception as exc:
            if self._fallback_providers_to_cpu(exc):
                self._analysis = _init_analysis(self._providers)
            else:
                raise
        self._ensure_parser()
        self._ensure_occluder()
        return self._analysis

    def _ensure_swapper(self, swapper_name, boost: int = 1):
        from insightface.model_zoo import model_zoo
        from .runtime import session_options
        self._ensure_analysis()
        insight_root = os.path.join(self.models_path, "insightface")
        candidates = [os.path.join(self.models_path, swapper_name),
                      os.path.join(insight_root, "models", swapper_name)]
        path = next((candidate for candidate in candidates if os.path.isfile(candidate)), None)
        if path is None:
            raise FileNotFoundError(f"Place {swapper_name} in {self.models_path}")
        if boost > 1 and os.path.basename(path) == "reswapper_256.onnx":
            raise ValueError("Pixel-boost (@256/@512/…) is not supported for reswapper_256.onnx")
        precision = self.config.inswapper_precision
        if self._swapper is None or path != self._swapper_path or (
                os.path.basename(path) == "inswapper_128.onnx" and precision != self._swapper_precision):
            logger.info("Loading swapper: %s", path)
            if swapper_name in SWAP_MODELS:
                missing = missing_assets(self.models_path, swapper_name)
                if missing:
                    raise FileNotFoundError("Missing swapper assets: " + ", ".join(missing))
            self._swapper = None
            self._swapper_path = None

            def _load_swapper(providers):
                name = os.path.basename(path)
                if name == "reswapper_256.onnx":
                    return Reswapper256(path, providers)
                spec = SWAP_MODELS.get(name)
                if spec and spec.family == "inswapper":
                    return InswapperBackend(path, providers, precision)
                if spec and spec.family == "facedancer":
                    return FaceDancerSwapper(path, providers)
                if spec and spec.family == "mobilefaceswap":
                    return MobileFaceSwapper(path, providers)
                if spec:
                    return OnnxSwapper(path, providers, asset_root=self.models_path)
                model = model_zoo.ModelRouter(path).get_model(providers=providers, sess_options=session_options())
                if model.__class__.__name__ != "INSwapper":
                    raise RuntimeError(f"{swapper_name} is not a supported swap model "
                                       f"(loaded as {model.__class__.__name__}). Use inswapper_128.onnx.")
                return InswapperBackend(path, providers, precision)

            try:
                self._swapper = _load_swapper(self._providers)
            except Exception as exc:
                if self._fallback_providers_to_cpu(exc):
                    self._swapper = _load_swapper(self._providers)
                else:
                    raise
            self._swapper_path = path
            self._swapper_precision = precision
            detail = f", {self._swapper.precision}" if isinstance(self._swapper, InswapperBackend) else ""
            logger.info("Swapper loaded: %s (%s%s)", os.path.basename(path),
                        self._swapper.__class__.__name__, detail)
        return self._swapper

    def _load(self, swapper_name, boost: int = 1):
        """Backwards-compatible: load analysis + swapper."""
        return self._ensure_swapper(swapper_name, boost)

    # -------------------------------------------------------------- detection
    def _detect(self, image, det_size=None):
        size = int(det_size or self.config.det_size)
        det = getattr(self._analysis, "det_model", None)
        if det is not None:
            # SCRFD re-reads this attribute on every detect call.
            det.input_size = (size, size)
        records = []
        for face in self._analysis.get(image):
            bbox = clamp_bbox(face, image.shape)
            if min(bbox[2] - bbox[0], bbox[3] - bbox[1]) < self.config.min_face_size:
                continue
            landmarks = dense_landmarks(face)
            if self.config.refine_landmarks and len(landmarks) == 106:
                try:
                    face.kps_raw = np.asarray(face.kps, np.float32).copy()
                    face.kps = refine_kps(face.kps, landmarks)
                except Exception:  # pragma: no cover - defensive
                    pass
            record = FaceRecord(face, bbox, landmarks,
                                getattr(face, "normed_embedding", None),
                                float(getattr(face, "det_score", 0)))
            record.quality = quality_score(image, record)
            record.gender = getattr(face, "gender", None)
            record.age = getattr(face, "age", None)
            records.append(record)
        return sorted(records, key=lambda record: record.bbox[0])

    def _detect_adaptive(self, image):
        """Detect faces; retry once at a higher detector resolution when every
        face is small, which sharpens landmarks and improves alignment."""
        records = self._detect(image)
        limit = int(self.config.det_size_max or 0)
        if limit <= self.config.det_size:
            return records
        if not records:
            return self._detect(image, det_size=min(limit, self.config.det_size * 2))
        best = max(min(record.bbox[2] - record.bbox[0],
                       record.bbox[3] - record.bbox[1]) for record in records)
        if best >= max(96, self.config.min_face_size * 2):
            return records
        bigger = min(limit, self.config.det_size * 2)
        logger.info("  detection: largest face %dpx < threshold, retrying at det_size=%d",
                    best, bigger)
        return self._detect(image, det_size=bigger)

    def detect_faces(self, image) -> list[FaceRecord]:
        """Public: detected faces (left to right) with embeddings, gender, age."""
        with self._lock:
            self._ensure_analysis()
            return self._detect_adaptive(image)

    def analyze_references(self, references, source_index=0, label="") -> ReferenceSet:
        """Detect, quality-gate and aggregate up to four reference images."""
        with self._lock:
            self._ensure_analysis()
            cfg = self.config
            records, images = [], []
            for i, image in enumerate(list(references)[:4]):
                faces = self._detect_adaptive(image)
                logger.info("  reference[%d]: found %d face(s), image %dx%d",
                            i, len(faces), image.shape[1], image.shape[0])
                if not faces:
                    continue
                if not 0 <= int(source_index) < len(faces):
                    raise ValueError(
                        f"Reference face index {source_index} is out of range; "
                        f"this reference has {len(faces)} face(s)")
                record = faces[int(source_index)]
                parse_face(image, record, self._parser)
                logger.info("    selected face %d: bbox=%s quality=%.3f score=%.3f gender=%s age=%s",
                            int(source_index), tuple(map(int, record.bbox)), record.quality,
                            record.score, gender_label(record.gender), record.age)
                if record.quality >= cfg.reference_quality:
                    records.append(record)
                    images.append(image)
                else:
                    logger.warning("    rejected: quality %.3f < threshold %.2f",
                                   record.quality, cfg.reference_quality)
            if not records:
                raise RuntimeError("No usable reference face passed the quality threshold")
            identity = weighted_identity(records)
            logger.info("  identity%s: aggregated from %d reference(s)",
                        f" '{label}'" if label else "", len(records))
            return ReferenceSet(identity, records, images, label)

    # ------------------------------------------------------------ embeddings
    def _embed_at(self, image, kps):
        """ArcFace embedding of `image` aligned at the given 5 keypoints —
        no detection pass, deterministic face association."""
        recognition = self._analysis.models.get("recognition")
        if recognition is None:
            return None
        try:
            return np.asarray(recognition.get(image, SimpleNamespace(kps=np.asarray(kps, np.float32))),
                              np.float32).reshape(-1)
        except Exception as exc:
            logger.debug("embedding failed: %s", exc)
            return None

    def _embed_aligned(self, crop, size):
        """Embed an already arcface-aligned crop of `size` px."""
        return self._embed_at(crop, arcface_kps(size))

    def _similarity_at(self, image, kps, identity):
        embedding = self._embed_at(image, kps)
        return cosine(identity, embedding) if embedding is not None else None

    def _reference_similarity(self, image, target_record, identity):
        """Reference similarity of the face at the target's landmarks."""
        return self._similarity_at(image, target_record.face.kps, identity)

    def _crop_sharpness(self, image, kps, size=256):
        """Sharpness of the face as it appears in the frame."""
        from insightface.utils import face_align
        try:
            crop, _ = face_align.norm_crop2(image, np.asarray(kps, np.float32), size)
            return sharpness(crop)
        except Exception:
            return 0.0

    # --------------------------------------------------------------- context
    def _target_context(self, frame, record, cfg) -> TargetContext:
        from insightface.utils import face_align
        from .parsing import aligned_crop_matrix

        raw_kps = getattr(getattr(record, "face", None), "kps", None)
        kps = None if raw_kps is None else np.asarray(raw_kps, np.float32)
        shape = frame.shape
        rois = []
        ffhq_matrix = None
        if kps is not None and kps.shape == (5, 2):
            swap_matrix = face_align.estimate_norm(kps, 128)
            ffhq_matrix = aligned_crop_matrix(kps, 512)
            rois.append(crop_roi(swap_matrix, 128, shape, pad=.25))
            if ffhq_matrix is not None:
                rois.append(crop_roi(ffhq_matrix, 512, shape, pad=.12))
        x1, y1, x2, y2 = record.bbox
        fw, fh = x2 - x1, y2 - y1
        rois.append((max(0, int(x1 - fw * .4)), max(0, int(y1 - fh * .5)),
                     min(shape[1], int(x2 + fw * .4)), min(shape[0], int(y2 + fh * .4))))
        roi = union_roi(*rois)
        rx1, ry1, rx2, ry2 = roi
        local = _shift_record(record, rx1, ry1)
        local_shape = (ry2 - ry1, rx2 - rx1)
        face_mask = build_face_mask(local.bbox, local_shape, landmarks=local.landmarks)
        interior = composite_region_mask(local, local_shape, ("skin", "eyebrows", "nose", "lips", "neck"))
        face_px = max(int(fw), 1)
        if interior is None:
            erode_k = max(3, int(face_px * .08)) | 1
            interior = cv2.erode(face_mask, np.ones((erode_k, erode_k), np.uint8))
        occlusion = None
        if cfg.occluder_enabled and self._occluder is not None:
            occlusion = self._occluder.map_to_frame(local, frame[ry1:ry2, rx1:rx2])
        parsing_available = bool(getattr(record, "parsing_available", False)) and self._parser is not None
        mouth_mode = str(getattr(cfg, "mouth_mode", "") or "").strip().lower()
        if not mouth_mode:
            mouth_mode = "keep_target_mouth" if bool(getattr(cfg, "keep_target_mouth", True)) else "swap_mouth"
        names = {"keep_target_mouth": ("lips", "teeth"), "swap_lips_keep_teeth": ("teeth",)}.get(mouth_mode, ())
        mouth = eyes = None
        if names and parsing_available:
            mouth = composite_region_mask(local, local_shape, names, dilate_frac=.006, feather_frac=.012)
            if mouth is not None and float(mouth.max()) <= 0:
                mouth = None
        elif names:
            logger.warning("  mouth preserve skipped: face parser unavailable")
        if str(getattr(cfg, "eye_mode", "swap")).lower() == "keep_target_eyes" and parsing_available:
            eyes = composite_region_mask(local, local_shape, ("eyes",), dilate_frac=.02, feather_frac=.02)
            if eyes is not None and float(eyes.max()) <= 0:
                eyes = None
        return TargetContext(record, kps, roi, face_mask, interior, occlusion, mouth, eyes,
                             face_px, ffhq_matrix, parsing_available)

    def _region_mask_for(self, aimg, fake, ctx):
        """Crop-space blend mask from the parser, or None for the box mask."""
        if self._parser is None or self.config.mask_mode != "region":
            return None
        try:
            labels_fake = self._parser(fake)
            labels_target = self._parser(aimg) if aimg.shape[:2] == fake.shape[:2] else None
            return region_blend_mask(labels_fake, labels_target)
        except Exception as exc:
            logger.warning("  region mask unavailable (%s); using box mask", exc)
            return None

    # ------------------------------------------------------------------ swap
    def _run_swap(self, swapper, image, record, source, boost, paste_back=True, ctx=None, info=None):
        """Run the loaded swapper; boost>1 routes through the pixel-boost path."""
        if not paste_back:
            return swapper.get(image, record.face, source, paste_back=False, factor=boost)
        fake, matrix = swapper.get(image, record.face, source, paste_back=False, factor=boost)
        from .boost import _paste_back
        from insightface.utils import face_align
        size = int(fake.shape[0])
        if isinstance(swapper, (InswapperBackend, Reswapper256)):
            aimg, _ = face_align.norm_crop2(image, record.face.kps, size)
        else:
            aimg = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        region = self._region_mask_for(aimg, fake, ctx) if ctx is not None else None
        return _paste_back(aimg, fake, matrix, image, region_mask=region,
                           blend=self.config.blend_mode, info=info)

    def _postprocess_roi(self, swapped_roi, target_roi, ctx: TargetContext, cfg, conservative=False):
        """Image-preserving corrections on the face ROI of one pasted-back swap."""
        interior, face_mask = ctx.interior, ctx.face_mask
        # In best-likeness mode this path is also run with cosmetic stages off.
        # That lets the similarity check reject a finish that drifts toward the
        # target's coloring or edge detail.
        color_strength = 0.0 if conservative else cfg.color_strength
        corrected = color_match(swapped_roi, target_roi, interior, color_strength)
        result = np.clip(corrected, 0, 255).astype(np.uint8)
        if ctx.occlusion is not None:
            occ = ctx.occlusion[:, :, None]
            result = (result.astype(np.float32) * (1 - occ) + target_roi.astype(np.float32) * occ).astype(np.uint8)
        result = recover_occlusions(target_roi, result, face_mask)
        if cfg.sharpen_strength > 0 and not conservative:
            amount = cfg.sharpen_strength * .5
            sigma = max(1.0, ctx.face_px * .006)
            result = sharpen_face_region(result, interior, amount=amount, sigma=sigma)
        for keep_mask in (ctx.mouth, ctx.eyes):
            if keep_mask is not None:
                keep = np.clip(keep_mask, 0, 1)[:, :, None]
                result = (result.astype(np.float32) * (1 - keep) + target_roi.astype(np.float32) * keep).astype(np.uint8)
        return result

    def _postprocess(self, swapped, target, target_record, cfg, conservative=False):
        """Backwards-compatible full-frame entry point."""
        ctx = self._target_context(target, target_record, cfg)
        x1, y1, x2, y2 = ctx.roi
        out = swapped.copy()
        out[y1:y2, x1:x2] = self._postprocess_roi(swapped[y1:y2, x1:x2], target[y1:y2, x1:x2], ctx, cfg, conservative)
        return out

    def _finish(self, swapped, frame, ctx, cfg, conservative=False):
        x1, y1, x2, y2 = ctx.roi
        out = swapped.copy()
        out[y1:y2, x1:x2] = self._postprocess_roi(swapped[y1:y2, x1:x2], frame[y1:y2, x1:x2], ctx, cfg, conservative)
        return out

    def _restore_candidate(self, candidate, frame, ctx: TargetContext, identity, cfg):
        """Restore through the FFHQ geometry, blend, then guard likeness."""
        if ctx.ffhq_matrix is None:
            logger.warning("  restoration skipped: unable to make FFHQ alignment")
            return candidate, False, "off"
        restorer = self._ensure_restorer(cfg.restorer)
        size = restorer.size
        matrix = ctx.ffhq_matrix if size == 512 else ctx.ffhq_matrix * (size / 512.0)
        aligned = cv2.warpAffine(candidate, matrix, (size, size), flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_REFLECT_101)
        restored = restorer.restore_aligned(aligned)
        blend = float(np.clip(cfg.restoration_blend, 0, 1))
        if blend < 1.0:
            restored = np.clip(restored.astype(np.float32) * blend +
                               aligned.astype(np.float32) * (1 - blend), 0, 255).astype(np.uint8)
        from .boost import shift_affine
        x1, y1, x2, y2 = ctx.roi
        shifted = shift_affine(matrix, x1, y1)
        local = _shift_record(ctx.record, x1, y1)
        roi_restored = paste_restored_back(restored, shifted, size, candidate[y1:y2, x1:x2], local.landmarks)
        restored_frame = candidate.copy()
        restored_frame[y1:y2, x1:x2] = roi_restored
        restored_frame = self._finish(restored_frame, frame, ctx, cfg)
        label = f"{RESTORERS[cfg.restorer].label}" + (f" w={cfg.codeformer_weight:.2f}" if RESTORERS[cfg.restorer].has_weight else "") + f" blend={blend:.2f}"
        if not cfg.codeformer_verify_identity:
            return restored_frame, True, label
        before = self._similarity_at(candidate, ctx.kps, identity)
        after = self._similarity_at(restored_frame, ctx.kps, identity)
        if after is None:
            logger.warning("  restoration rejected: enhanced candidate could not be verified")
            return candidate, False, "off (unverified)"
        if before is not None and after < before - cfg.likeness_tolerance:
            logger.warning("  restoration rejected: reference similarity %.3f -> %.3f", before, after)
            return candidate, False, f"off (rejected {before:.3f}->{after:.3f})"
        logger.info("  restoration accepted: reference similarity %s -> %.3f",
                    "unavailable" if before is None else f"{before:.3f}", after)
        return restored_frame, True, label

    def _swap_face(self, frame, target_record, refset: ReferenceSet, swapper, swapper_name,
                   model_file, boost, face_index, cfg) -> tuple[np.ndarray, FaceReport, dict]:
        started = time.perf_counter()
        self._check_cancel()
        identity = refset.identity
        parse_face(frame, target_record, self._parser)
        ctx = self._target_context(frame, target_record, cfg)
        source = virtual_face(identity, refset.records[0].face)
        extras = []
        if isinstance(swapper, OnnxSwapper):
            best = refset.best_index
            # Converters expect the original recognizer embedding magnitude.
            raw = np.asarray(refset.records[0].face.embedding, np.float32)
            if len(refset.records) > 1:
                weights = np.asarray([max(r.quality, 1e-8) for r in refset.records])
                magnitude = np.average([np.linalg.norm(r.face.embedding) for r in refset.records], weights=weights)
                raw = identity * magnitude
            pixel_conditioned = swapper.spec.family in ("blendswap", "uniface", "facedancer", "mobilefaceswap")
            if not refset.images:
                if pixel_conditioned:
                    raise ValueError(f"{swapper.spec.name} needs reference photos; a saved identity only "
                                     "stores the embedding. Upload a reference image for this model.")
                swapper.prepare_reference(None, refset.records[best].face, identity, raw)
            else:
                swapper.prepare_reference(refset.images[best], refset.records[best].face, identity, raw)
            if pixel_conditioned and len(refset.records) > 1:
                extras.append(f"source pixels: reference {best + 1}")

        quality_mode = str(getattr(cfg, "quality_mode", "manual")).strip().lower()
        is_inswapper = os.path.basename(model_file) == "inswapper_128.onnx"
        explicit_boost = bool(str(swapper_name).partition("@")[2])
        candidate_boosts = candidate_boosts_for(swapper_name, model_file, boost, quality_mode)
        if hasattr(swapper, "dealias"):
            swapper.dealias = bool(cfg.boost_dealias)
        if not explicit_boost and len(candidate_boosts) > 1 and ctx.kps is not None:
            from insightface.utils import face_align
            from .boost import max_useful_boost
            native = SWAP_MODELS[model_file].size if model_file in SWAP_MODELS else 128
            cap = max_useful_boost(face_align.estimate_norm(ctx.kps, native), native, cfg.boost_slack)
            capped = tuple(f for f in candidate_boosts if f <= cap) or (candidate_boosts[0],)
            if capped != candidate_boosts:
                logger.info("  pixel boost capped at x%d for a %dpx face (skipping %s)", cap, ctx.face_px,
                            ", ".join(f"x{f}" for f in candidate_boosts if f not in capped))
            candidate_boosts = capped
        profiles = (("default", False),)
        if quality_mode == "best_likeness" and is_inswapper:
            profiles += (("identity-preserving", True),)
        candidates: list[Candidate] = []
        debug = {}
        for candidate_boost in candidate_boosts:
            self._check_cancel()
            t0 = time.perf_counter()
            info = {} if cfg.debug_masks else None
            swapped = self._run_swap(swapper, frame, target_record, source, candidate_boost, ctx=ctx, info=info)
            if swapped is None:
                logger.warning("  candidate x%d returned no image", candidate_boost)
                continue
            if info:
                debug[f"blend_mask_x{candidate_boost}"] = info.get("mask")
            raw_similarity = self._similarity_at(swapped, ctx.kps, identity)
            for profile, conservative in profiles:
                candidate = self._finish(swapped, frame, ctx, cfg, conservative=conservative)
                similarity = self._similarity_at(candidate, ctx.kps, identity)
                sharp = self._crop_sharpness(candidate, ctx.kps)
                logger.info("  candidate x%d %s: raw=%s final=%s sharp=%.0f (%.2fs)",
                            candidate_boost, profile,
                            "unverified" if raw_similarity is None else f"{raw_similarity:.3f}",
                            "unverified" if similarity is None else f"{similarity:.3f}",
                            sharp, time.perf_counter() - t0)
                candidates.append(Candidate(candidate_boost, profile, candidate, similarity, raw_similarity, sharp))
        if not candidates:
            raise RuntimeError("Face swap model returned no image")
        chosen = select_likeness_candidate(candidates, cfg.likeness_tolerance)
        result, confidence = chosen.image, chosen.score
        logger.info("  selected candidate: x%d %s (raw=%s final=%s)", chosen.boost, chosen.profile,
                    "unverified" if chosen.raw_score is None else f"{chosen.raw_score:.3f}",
                    "unverified" if confidence is None else f"{confidence:.3f}")

        restoration = "off"
        if cfg.codeformer_enabled:
            self._check_cancel()
            t1 = time.perf_counter()
            try:
                result, used, restoration = self._restore_candidate(result, frame, ctx, identity, cfg)
            except Exception as exc:
                logger.warning("Optional restoration unavailable; keeping successful swap: %s", exc)
                restoration = "off (unavailable)"
                used = False
            logger.info("  restoration stage: %s (%.2fs)", "accepted" if used else "skipped/rejected",
                        time.perf_counter() - t1)
            if used:
                confidence = self._similarity_at(result, ctx.kps, identity)
        verified = confidence is not None and confidence >= cfg.verification_threshold
        if explicit_boost and chosen.boost > 1:
            native_size = SWAP_MODELS[model_file].size if model_file in SWAP_MODELS else 128
            extras.append(f"requested pixel-boost {chosen.boost * native_size}px (x{chosen.boost})")
        elif quality_mode == "best_likeness" and is_inswapper:
            extras.append(f"best-likeness x{chosen.boost} {chosen.profile} (compared 128/256/512)")
        elif chosen.boost > 1:
            extras.append(f"pixel-boost x{chosen.boost}")
        if isinstance(swapper, InswapperBackend) and swapper.precision == "fp16":
            extras.append("fp16")
        if self._parser is not None:
            extras.append("parsing")
        if ctx.occlusion is not None:
            extras.append("occluder")
        if cfg.mask_mode == "region" and self._parser is not None:
            extras.append("region-mask")
        if cfg.blend_mode == "multiband":
            extras.append("multiband")
        mouth_mode = str(getattr(cfg, "mouth_mode", "")).strip().lower()
        if mouth_mode == "swap_lips_keep_teeth":
            extras.append("target-teeth")
        elif mouth_mode == "keep_target_mouth":
            extras.append("target-mouth")
        if ctx.eyes is not None:
            extras.append("target-eyes")
        if cfg.debug_masks:
            x1, y1, x2, y2 = ctx.roi
            for name, mask in (("face_mask", ctx.face_mask), ("interior", ctx.interior),
                               ("occlusion", ctx.occlusion), ("mouth", ctx.mouth), ("eyes", ctx.eyes)):
                if mask is not None:
                    full = np.zeros(frame.shape[:2], np.float32)
                    full[y1:y2, x1:x2] = mask
                    debug[name] = full
        report = FaceReport(face_index, tuple(map(int, target_record.bbox)), chosen.boost, chosen.profile,
                            confidence, chosen.raw_score, verified, restoration,
                            time.perf_counter() - started, refset.label, extras)
        return result, report, debug

    # --------------------------------------------------------------- public
    def resolve_assignments(self, targets, assignments, cfg=None):
        """Map assignments onto detected target faces. Returns [(record, refset)]."""
        cfg = cfg or self.config
        plan, taken = [], []
        for assignment in assignments:
            mode = str(assignment.match_mode or "index").strip().lower()
            if mode not in MATCH_MODES:
                raise ValueError(f"match_mode must be one of {MATCH_MODES}")
            if mode == "all":
                for record in targets:
                    if not any(record is t for t in taken):
                        plan.append((record, assignment.reference))
                        taken.append(record)
                continue
            gender = assignment.reference.gender if mode == "gender" else None
            if mode == "gender" and gender is None:
                raise ValueError("Gender detection is unavailable for the reference face(s); "
                                 "the analysis pack appears to lack the genderage model")
            record = select_target_record(targets, mode, int(assignment.target_index), gender,
                                          anchor=assignment.anchor, threshold=cfg.similar_threshold,
                                          exclude=taken)
            if record is None:
                if mode == "gender":
                    counts = Counter(gender_label(r.gender) for r in targets)
                    detail = ", ".join(f"{n} {label}" for label, n in counts.most_common()) or "unknown"
                    raise ValueError(f"No target face matches the reference gender ({gender_label(gender)}); "
                                     f"the target has {len(targets)} face(s) ({detail})")
                raise ValueError("No target face is similar enough to the chosen face")
            if any(record is t for t in taken):
                index = next(i for i, r in enumerate(targets) if r is record)
                raise ValueError(f"Target face {index} was assigned twice")
            plan.append((record, assignment.reference))
            taken.append(record)
        return plan

    def swap(self, target, assignments, swapper_name="inswapper_128.onnx", save=None) -> SwapOutput:
        """Swap one or more faces of `target` according to `assignments`."""
        started = time.perf_counter()
        if target is None or not assignments:
            raise ValueError("A target and at least one face assignment are required")
        if not isinstance(target, np.ndarray) or target.ndim != 3 or target.shape[2] != 3 or target.size == 0 or target.dtype != np.uint8:
            raise ValueError("Target must be a non-empty uint8 BGR image")
        self.cancel_event.clear()
        with self._lock:
            cfg = self.config
            model_file, boost = parse_swapper_spec(swapper_name)
            swapper = self._ensure_swapper(model_file, boost)
            logger.info("=" * 60)
            logger.info("Visenlo pipeline started: %s, target %dx%d, %d assignment(s)",
                        swapper_name, target.shape[1], target.shape[0], len(assignments))
            logger.info("  config: restorer=%s (%s, w=%.2f, blend=%.2f), sharpen=%.2f, color=%.2f, parsing=%s, "
                        "occluder=%s, mask=%s, blend=%s%s",
                        cfg.restorer if cfg.codeformer_enabled else "off", cfg.codeformer_enabled,
                        cfg.codeformer_weight, cfg.restoration_blend, cfg.sharpen_strength, cfg.color_strength,
                        "bisenet" if self._parser else "geometric",
                        "xseg" if (cfg.occluder_enabled and self._occluder) else "off",
                        cfg.mask_mode, cfg.blend_mode, f", boost x{boost}" if boost > 1 else "")
            targets = self._detect_adaptive(target)
            logger.info("  target: found %d face(s)", len(targets))
            if not targets:
                raise RuntimeError("No target face passed the size threshold")
            plan = self.resolve_assignments(targets, assignments, cfg)
            frame = target.copy()
            reports, debug = [], {}
            for record, refset in plan:
                index = next(i for i, r in enumerate(targets) if r is record)
                self._progress(f"face {index}" + (f" ← {refset.label}" if refset.label else ""))
                logger.info("  target face %d: bbox=%s quality=%.3f gender=%s age=%s",
                            index, tuple(map(int, record.bbox)), record.quality,
                            gender_label(record.gender), record.age)
                frame, report, face_debug = self._swap_face(frame, record, refset, swapper, swapper_name,
                                                            model_file, boost, index, cfg)
                reports.append(report)
                for key, value in face_debug.items():
                    debug[f"face{index}_{key}"] = value
                logger.info("  %s", report.line())
            saved_path = None
            if cfg.save_swaps if save is None else save:
                saved_path = self._save(frame)
            elapsed = time.perf_counter() - started
            logger.info("  total time: %.2fs", elapsed)
            logger.info("=" * 60)
            status = self._status_line(reports, model_file, elapsed, saved_path, len(targets))
            return SwapOutput(frame, status, reports, debug, saved_path, targets)

    def _save(self, image):
        # Persist every completed swap as <date>_<NN>.png; a failed save
        # must never discard a successful swap, so only warn on error.
        out_path = None
        try:
            out_path = allocate_output_path(self.output_dir)
            ok, encoded = cv2.imencode(".png", image)
            if ok:
                encoded.tofile(out_path)
                logger.info("  saved swapped image: %s", out_path)
                return out_path
            logger.warning("  PNG encoding failed; swap not saved")
            os.unlink(out_path)
        except Exception as exc:
            logger.warning("  could not save swapped image: %s", exc)
            if out_path and os.path.exists(out_path):
                try:
                    os.unlink(out_path)
                except OSError:
                    pass
        return None

    @staticmethod
    def _status_line(reports, model_file, elapsed, saved_path, n_targets):
        if len(reports) == 1:
            r = reports[0]
            sim = 0.0 if r.similarity is None else r.similarity
            extras = " | ".join(r.extras)
            return (f"Visenlo complete | model: {model_file} | reference similarity: {sim:.3f} | "
                    f"{'verified' if r.verified else 'UNVERIFIED/LOW CONFIDENCE'} | restoration: {r.restoration}"
                    + (f" | {extras}" if extras else "")
                    + (f" | target face {r.face_index} of {n_targets}" if n_targets > 1 else "")
                    + f" | {elapsed:.2f}s" + (f" | saved {os.path.basename(saved_path)}" if saved_path else ""))
        lines = "; ".join(r.line() for r in reports)
        return (f"Visenlo complete | model: {model_file} | {len(reports)} face(s): {lines} | {elapsed:.2f}s"
                + (f" | saved {os.path.basename(saved_path)}" if saved_path else ""))

    def process(self, references: list[np.ndarray], target: np.ndarray, source_index=0,
                target_index=0, swapper_name="inswapper_128.onnx", match_mode="index"):
        """Single-identity entry point (kept for existing callers).

        Returns ``(image, status)`` exactly as before; ``match_mode`` accepts
        "index", "gender" or "all".
        """
        if target is None or not references:
            raise ValueError("A target and at least one reference are required")
        for label, image in [("Target", target)] + [(f"Reference {i + 1}", image) for i, image in enumerate(references)]:
            if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.size == 0 or image.dtype != np.uint8:
                raise ValueError(f"{label} must be a non-empty uint8 BGR image")
        match_mode = str(match_mode or "index").strip().lower()
        if match_mode not in ("index", "gender", "all"):
            raise ValueError("match_mode must be 'index', 'gender' or 'all'")
        with self._lock:
            refset = self.analyze_references(references, source_index)
            output = self.swap(target, [FaceAssignment(refset, match_mode, int(target_index))], swapper_name)
        return output.image, output.status


# Backward compatibility: old ReactorX name still resolves to the same pipeline.
ReactorXPipeline = VisenloPipeline
