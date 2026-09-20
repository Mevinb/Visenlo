"""Complete Visenlo v1 pipeline orchestration."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from .engine import (
    FaceRecord,
    build_face_mask,
    clamp_bbox,
    color_match,
    composite_region_mask,
    cosine,
    dense_landmarks,
    parse_face,
    quality_score,
    recover_occlusions,
    sharpen_face_region,
    virtual_face,
    weighted_identity,
)
from .restoration import CodeFormer
from .restoration.utils import paste_restored_back

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
        # UI suffixes name the desired aligned resolution; boost.py expects
        # the scale factor relative to inswapper's native 128px resolution.
        if requested in (256, 512, 1024, 2048):
            boost = requested // 128
        elif requested in (2, 4, 8, 16):
            boost = requested
        else:
            raise ValueError(f"Unsupported pixel-boost size: @{requested}")
    return model_file, boost


def candidate_boosts_for(swapper_name, model_file, boost, quality_mode):
    """Resolve the exact boost factors to run for this request.

    A suffix is an explicit user choice and always wins. Automatic comparison
    applies only to the unsuffixed inswapper_128 selection.
    """
    explicit = bool(str(swapper_name).partition("@")[2])
    is_inswapper = os.path.basename(model_file) == "inswapper_128.onnx"
    if str(quality_mode).strip().lower() == "best_likeness" and is_inswapper and not explicit:
        return (1, 2, 4)
    return (int(boost),)


def select_likeness_candidate(candidates, tolerance=.01):
    """Pick ``(boost, profile, image, final_score, raw_score)`` safely.

    Scores within ``tolerance`` are indistinguishable for this metric, so keep
    the normal visual finish first and then prefer the lower boost to reduce
    tiling artifacts.
    """
    verified = [item for item in candidates if item[3] is not None]
    if not verified:
        return candidates[0]
    best_score = max(item[3] for item in verified)
    eligible = [item for item in verified if item[3] >= best_score - tolerance]
    return min(eligible, key=lambda item: (item[1] != "default", item[0]))


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


def select_target_record(records, match_mode="index", target_index=0, source_gender=None):
    """Choose the single target face to swap.

    match_mode="index" (default): the face at `target_index`, bounds-checked.
    match_mode="gender": the leftmost face matching `source_gender` (records are
    already sorted left-to-right), or None when nothing matches.
    """
    if match_mode != "gender":
        if not 0 <= int(target_index) < len(records):
            raise ValueError(
                f"Target face index {target_index} is out of range; "
                f"the target has {len(records)} face(s)")
        return records[int(target_index)]
    if source_gender is None:
        return None
    for record in records:
        if record.gender == source_gender:
            return record
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


class Reswapper256:
    """Adapter for the two-input 256px Reswapper ONNX graph."""

    def __init__(self, model_path, providers):
        import onnx
        import onnxruntime as ort
        from insightface.utils import face_align
        from onnx import numpy_helper

        self.face_align = face_align
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name
        # Robust input_size: handle NCHW/NHWC and dynamic dims (None/str).
        inp_shape = self.session.get_inputs()[0].shape
        cand = None
        if len(inp_shape) == 4:
            # Prefer spatial dims, ignore channel dim (3).
            for idx in (2, 3, 1):
                v = inp_shape[idx] if idx < len(inp_shape) else None
                if isinstance(v, int) and v > 0 and v != 3:
                    cand = v
                    break
        if cand is None:
            cand = 256
        self.input_size = int(cand)
        graph = onnx.load(model_path).graph
        emap_init = next((item for item in graph.initializer if item.name == "emap"), None)
        if emap_init is None:
            raise RuntimeError(f"{model_path}: initializer 'emap' not found — incompatible reswapper_256.onnx")
        self.emap = numpy_helper.to_array(emap_init).astype(np.float32)

    def get(self, image, target_face, source_face, paste_back=True):
        size = self.input_size
        aligned, matrix = self.face_align.norm_crop2(image, target_face.kps, size)
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 255.0, (size, size), (0, 0, 0), swapRB=True)
        latent = np.asarray(source_face.normed_embedding, np.float32).reshape(1, -1)
        latent = np.dot(latent, self.emap)
        latent /= max(np.linalg.norm(latent), 1e-8)
        output = self.session.run([self.output_name], {
            self.input_names[0]: blob,
            self.input_names[1]: latent.astype(np.float32),
        })[0]
        fake = np.clip(output.transpose(0, 2, 3, 1)[0] * 255, 0, 255).astype(np.uint8)[:, :, ::-1]
        if not paste_back:
            return fake, matrix
        inverse = cv2.invertAffineTransform(matrix)
        warped = cv2.warpAffine(fake, inverse, (image.shape[1], image.shape[0]),
                                flags=cv2.INTER_LANCZOS4, borderValue=0)
        mask = cv2.warpAffine(np.full((size, size), 255, np.uint8), inverse,
                              (image.shape[1], image.shape[0]))
        mask = cv2.GaussianBlur(mask, (0, 0), max(3, int(size * .04))).astype(np.float32) / 255
        return (warped.astype(np.float32) * mask[:, :, None] +
                image.astype(np.float32) * (1 - mask[:, :, None])).astype(np.uint8)


@dataclass
class PipelineConfig:
    min_face_size: int = 48
    reference_quality: float = .20
    verification_threshold: float = .30
    color_strength: float = .25
    codeformer_enabled: bool = True
    codeformer_weight: float = .9
    codeformer_verify_identity: bool = True
    sharpen_strength: float = .5
    det_size: int = 640
    det_size_max: int = 1280
    occluder_enabled: bool = True
    # Best likeness compares native/256/512 pixel-boost candidates before the
    # optional restorer. Manual preserves the user-selected swapper setting.
    quality_mode: str = "best_likeness"
    # Swap the reference lips by default, but retain only target teeth/mouth
    # interior where the 128px generator is most likely to create artifacts.
    mouth_mode: str = "swap_lips_keep_teeth"
    # Deprecated compatibility setting for callers from v1.  New callers use
    # mouth_mode; this remains so existing integrations do not crash.
    keep_target_mouth: bool = True
    save_swaps: bool = True


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
        self._providers = None
        self._parser = None
        self._occluder = None
        self._codeformer = None
        os.makedirs(models_path, exist_ok=True)

    def _fallback_providers_to_cpu(self, exc: Exception) -> bool:
        """If CUDA provider failed (missing libcublas/cudnn), fall back to CPU once."""
        msg = str(exc).lower()
        if any(k in msg for k in ("cuda", "cublas", "cudnn", "provider")) and self._providers and any(
            (p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider" for p in self._providers
        ):
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

    def _ensure_codeformer(self):
        if self._codeformer is None:
            logger.info("Loading CodeFormer (weight=%.2f) ...", self.config.codeformer_weight)
            self._codeformer = CodeFormer(self.models_path, self._providers, self.config.codeformer_weight)
            logger.info("CodeFormer ready")
        self._codeformer.weight = self.config.codeformer_weight
        return self._codeformer

    def _load(self, swapper_name, boost: int = 1):
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis
            from insightface.model_zoo import model_zoo
        except ImportError as exc:
            raise RuntimeError("Install insightface and onnxruntime in this application's environment") from exc
        insight_root = os.path.join(self.models_path, "insightface")
        if self._providers is None:
            try:
                # Load cuDNN/CUBLAS from pip nvidia wheels when present.
                ort.preload_dlls()
            except Exception as exc:
                logger.debug("ort.preload_dlls() failed: %s", exc)
            available = ort.get_available_providers()
            # Fast conv-algo selection keeps session startup quick; a same-as-
            # requested arena avoids VRAM spikes when juggling several models.
            gpu_options = {"cudnn_conv_algo_search": "HEURISTIC",
                           "arena_extend_strategy": "kSameAsRequested",
                           "cudnn_conv_use_max_workspace": "1"}
            self._providers = []
            if "CUDAExecutionProvider" in available:
                self._providers.append(("CUDAExecutionProvider", gpu_options))
            if "CPUExecutionProvider" in available or not self._providers:
                self._providers.append("CPUExecutionProvider")
            logger.info("ONNX providers: %s", self._providers)
        if self._analysis is None:
            logger.info("Loading FaceAnalysis (buffalo_l) from %s ...", insight_root)

            def _init_analysis(providers):
                use_cuda = any(
                    (p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider" for p in providers
                )
                analysis = FaceAnalysis(name="buffalo_l", root=insight_root, providers=providers)
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
        candidates = [os.path.join(self.models_path, swapper_name),
                      os.path.join(insight_root, "models", swapper_name)]
        path = next((candidate for candidate in candidates if os.path.isfile(candidate)), None)
        if path is None:
            raise FileNotFoundError(f"Place {swapper_name} in {self.models_path}")
        # Pixel-boost is only implemented for the 128px model.
        if boost > 1 and os.path.basename(path) == "reswapper_256.onnx":
            raise ValueError("Pixel-boost (@256/@512/…) is not supported for reswapper_256.onnx")
        if self._swapper is None or path != self._swapper_path:
            logger.info("Loading swapper: %s", path)

            def _load_swapper(providers):
                if os.path.basename(path) == "reswapper_256.onnx":
                    return Reswapper256(path, providers)
                return model_zoo.get_model(path, providers=providers)

            try:
                self._swapper = _load_swapper(self._providers)
            except Exception as exc:
                if self._fallback_providers_to_cpu(exc):
                    self._swapper = _load_swapper(self._providers)
                else:
                    raise
            self._swapper_path = path
            logger.info("Swapper loaded: %s (%s)",
                       os.path.basename(path), self._swapper.__class__.__name__)
        if (os.path.basename(path) != "reswapper_256.onnx" and
                self._swapper.__class__.__name__ != "INSwapper"):
            raise RuntimeError(
                f"{swapper_name} is not compatible with InsightFace 0.7.3 "
                f"(loaded as {self._swapper.__class__.__name__}). "
                "Use inswapper_128.onnx."
            )
        self._ensure_parser()
        self._ensure_occluder()
        return self._swapper

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
            record = FaceRecord(face, bbox, dense_landmarks(face),
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
        if not records or limit <= self.config.det_size:
            return records
        best = max(min(record.bbox[2] - record.bbox[0],
                       record.bbox[3] - record.bbox[1]) for record in records)
        if best >= max(96, self.config.min_face_size * 2):
            return records
        bigger = min(limit, self.config.det_size * 2)
        logger.info("  detection: largest face %dpx < threshold, retrying at det_size=%d",
                    best, bigger)
        return self._detect(image, det_size=bigger)

    def _restoration_keeps_identity(self, aligned, restored, identity, _target_record=None):
        """Return True if CodeFormer restoration preserves the swapped identity.

        Embeds the aligned swapped crop and the restored crop with the recognition
        model directly (using the standard alignment template), and compares their
        reference-identity similarity. Falls back to accepting if embedding fails.
        """
        try:
            swap_embed = self._embed_aligned(aligned, aligned.shape[0])
            restored_embed = self._embed_aligned(restored, restored.shape[0])
        except Exception as exc:
            logger.warning("identity check embedding failed (%s) — accepting restoration", exc)
            return True
        if swap_embed is None or restored_embed is None:
            logger.warning("identity check could not embed — accepting restoration")
            return True
        swap_sim = cosine(identity, swap_embed)
        restored_sim = cosine(identity, restored_embed)
        # Clamp negative similarities — for very low swap_sim the 0.95x gate is too strict.
        threshold = max(0.0, swap_sim * 0.95, swap_sim - 0.10)
        return restored_sim >= threshold

    def _embed_aligned(self, crop, size):
        """Embed an aligned face crop using the recognition model directly."""
        from types import SimpleNamespace
        recognition = self._analysis.models.get("recognition")
        if recognition is None:
            return None
        face = SimpleNamespace()
        face.kps = arcface_kps(size)
        try:
            return np.asarray(recognition.get(crop, face), np.float32).reshape(-1)
        except Exception:
            return None

    def _run_swap(self, swapper, image, record, source, boost, paste_back=True):
        """Run the loaded swapper; boost>1 routes through the pixel-boost path."""
        if boost > 1:
            from .boost import inswapper_boost_get
            return inswapper_boost_get(swapper, image, record.face, source,
                                       boost, paste_back=paste_back)
        return swapper.get(image, record.face, source, paste_back=paste_back)

    @staticmethod
    def _face_overlap(a, b):
        """Intersection over union for two (x1, y1, x2, y2) face boxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
        area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
        return inter / float(area_a + area_b - inter)

    def _output_embedding(self, image, target_record):
        """Embed the generated face that best occupies the original target slot."""
        generated = self._detect(image)
        if not generated:
            return None, 0
        tx = (target_record.bbox[0] + target_record.bbox[2]) * .5
        ty = (target_record.bbox[1] + target_record.bbox[3]) * .5
        diagonal = max(1.0, np.hypot(target_record.bbox[2] - target_record.bbox[0],
                                     target_record.bbox[3] - target_record.bbox[1]))

        def score(item):
            cx = (item.bbox[0] + item.bbox[2]) * .5
            cy = (item.bbox[1] + item.bbox[3]) * .5
            distance = np.hypot(cx - tx, cy - ty) / diagonal
            return self._face_overlap(item.bbox, target_record.bbox) - distance * .15

        matched = max(generated, key=score)
        logger.info("  post-swap detect: %d face(s), matched bbox=%s score=%.3f",
                    len(generated), tuple(map(int, matched.bbox)), matched.score)
        return matched.embedding, len(generated)

    def _reference_similarity(self, image, target_record, identity):
        embedding, _ = self._output_embedding(image, target_record)
        return cosine(identity, embedding) if embedding is not None else None

    def _postprocess(self, swapped, target, target_record, cfg, conservative=False):
        """Apply image-preserving corrections to one already pasted-back swap."""
        shape = target.shape
        face_mask = build_face_mask(target_record.bbox, shape,
                                    landmarks=target_record.landmarks)
        # Color comes from contiguous skin, not high-contrast eyes/teeth or
        # target hair. This prevents LAB transfer from tinting the eye whites.
        interior = composite_region_mask(
            target_record, shape, ("skin", "eyebrows", "nose", "lips", "neck"))
        face_px = max(int(target_record.bbox[2] - target_record.bbox[0]), 1)
        if interior is None:
            erode_k = max(3, int(face_px * .08)) | 1
            interior = cv2.erode(face_mask, np.ones((erode_k, erode_k), np.uint8))

        # In best-likeness mode this path is also run with cosmetic stages off.
        # That lets the similarity check reject a finish that drifts toward the
        # target's coloring or edge detail.
        color_strength = 0.0 if conservative else cfg.color_strength
        corrected = color_match(swapped, target, interior, color_strength)
        result = np.clip(corrected, 0, 255).astype(np.uint8)
        logger.info("  color match: strength=%.2f (visible face interior)", color_strength)

        if cfg.occluder_enabled and self._occluder is not None:
            occ_full = self._occluder.map_to_frame(target_record, target)
            if occ_full is not None:
                result = (result.astype(np.float32) * (1 - occ_full[:, :, None]) +
                          target.astype(np.float32) * occ_full[:, :, None]).astype(np.uint8)
                logger.info("  occlusion mask: xseg applied")
        result = recover_occlusions(target, result, face_mask)

        if cfg.sharpen_strength > 0 and not conservative:
            amount = cfg.sharpen_strength * .5
            sigma = max(1.0, face_px * .006)
            result = sharpen_face_region(result, interior, amount=amount, sigma=sigma)
            logger.info("  face-region sharpen: amount=%.2f sigma=%.1f (face ~%dpx)",
                        amount, sigma, face_px)

        mouth_mode = str(getattr(cfg, "mouth_mode", "") or "").strip().lower()
        # Upgrade old API calls without changing their behavior. New UI calls
        # always provide mouth_mode.
        if not mouth_mode:
            mouth_mode = "keep_target_mouth" if bool(getattr(cfg, "keep_target_mouth", True)) else "swap_mouth"
        if mouth_mode == "keep_target_mouth":
            names = ("lips", "teeth")
        elif mouth_mode == "swap_lips_keep_teeth":
            names = ("teeth",)
        else:
            names = ()
        # Do not invent a teeth region from geometric fallback masks. A failed
        # parser means we keep the generated mouth intact and report why.
        if names and self._parser is not None:
            mouth = composite_region_mask(target_record, shape, names,
                                          dilate_frac=.006, feather_frac=.012)
            if mouth is not None and float(mouth.max()) > 0:
                keep = np.clip(mouth, 0, 1)[:, :, None]
                result = (result.astype(np.float32) * (1 - keep) +
                          target.astype(np.float32) * keep).astype(np.uint8)
                logger.info("  mouth preserve: target %s kept (cov=%.2f%%)",
                            "+".join(names), float(mouth.mean() * 100))
        elif names:
            logger.warning("  mouth preserve skipped: face parser unavailable")
        return result

    def _restore_candidate(self, candidate, target, target_record, identity, cfg):
        """Restore through CodeFormer's FFHQ geometry, then guard likeness."""
        from .parsing import aligned_crop_matrix

        kps = getattr(target_record.face, "kps", None)
        if kps is None:
            logger.warning("  CodeFormer skipped: target landmarks unavailable")
            return candidate, False
        matrix = aligned_crop_matrix(kps, 512)
        if matrix is None:
            logger.warning("  CodeFormer skipped: unable to make FFHQ alignment")
            return candidate, False
        aligned = cv2.warpAffine(candidate, matrix, (512, 512), flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_REFLECT_101)
        codeformer = self._ensure_codeformer()
        restored = codeformer.restore_aligned(aligned)
        # `matrix` maps full image -> 512px FFHQ crop. Passing swap_size=512
        # keeps the restored crop and mask in this same coordinate system.
        restored_frame = paste_restored_back(restored, matrix, 512, candidate,
                                             target_record.landmarks)
        restored_frame = self._postprocess(restored_frame, target, target_record, cfg)
        before = self._reference_similarity(candidate, target_record, identity)
        after = self._reference_similarity(restored_frame, target_record, identity)
        if after is None:
            logger.warning("  CodeFormer rejected: enhanced candidate could not be verified")
            return candidate, False
        if before is not None and after < before - .01:
            logger.warning("  CodeFormer rejected: reference similarity %.3f -> %.3f", before, after)
            return candidate, False
        logger.info("  CodeFormer accepted: reference similarity %s -> %.3f",
                    "unavailable" if before is None else f"{before:.3f}", after)
        return restored_frame, True

    def process(self, references: list[np.ndarray], target: np.ndarray, source_index=0,
                target_index=0, swapper_name="inswapper_128.onnx", match_mode="index"):
        started = time.perf_counter()
        match_mode = str(match_mode or "index").strip().lower()
        if match_mode not in ("index", "gender"):
            raise ValueError("match_mode must be 'index' or 'gender'")
        logger.info("=" * 60)
        logger.info("Visenlo pipeline started")
        logger.info("  swapper:   %s", swapper_name)
        logger.info("  target:    %dx%d", target.shape[1], target.shape[0])
        logger.info("  references: %d", len(references))
        logger.info("  face matching: %s", match_mode)
        with self._lock:
            cfg = self.config
            model_file, boost = parse_swapper_spec(swapper_name)
            swapper = self._load(model_file, boost)
            logger.info("  config:    CodeFormer=%s (w=%.2f), sharpen=%.2f, color=%.2f, "
                         "parsing=%s, occluder=%s%s",
                         cfg.codeformer_enabled, cfg.codeformer_weight,
                         cfg.sharpen_strength, cfg.color_strength,
                         "bisenet" if self._parser else "geometric",
                         "xseg" if (cfg.occluder_enabled and self._occluder) else "off",
                         f", boost x{boost}" if boost > 1 else "")
            if not references or target is None:
                raise ValueError("A target and at least one reference are required")
            reference_records = []
            for i, image in enumerate(references[:4]):
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
                logger.info("    selected face %d: bbox=%s quality=%.3f score=%.3f "
                            "gender=%s age=%s",
                           int(source_index),
                           tuple(map(int, record.bbox)), record.quality, record.score,
                           gender_label(record.gender), record.age)
                if record.quality >= cfg.reference_quality:
                    reference_records.append(record)
                else:
                    logger.warning("    rejected: quality %.3f < threshold %.2f",
                                   record.quality, cfg.reference_quality)
            targets = self._detect_adaptive(target)
            logger.info("  target: found %d face(s)", len(targets))
            if not reference_records:
                raise RuntimeError("No usable reference face passed the quality threshold")
            if not targets:
                raise RuntimeError("No target face passed the size threshold")
            identity = weighted_identity(reference_records)
            logger.info("  identity: aggregated from %d reference(s), norm=%.4f",
                       len(reference_records), float(np.linalg.norm(identity)))
            source_gender = source_gender_vote(reference_records) if match_mode == "gender" else None
            if match_mode == "gender" and source_gender is None:
                raise ValueError(
                    "Gender detection is unavailable for the reference face(s); "
                    "the analysis pack appears to lack the genderage model")
            target_record = select_target_record(targets, match_mode,
                                                 int(target_index), source_gender)
            if target_record is None:
                counts = Counter(gender_label(record.gender) for record in targets)
                detail = ", ".join(f"{n} {label}" for label, n in counts.most_common()) or "unknown"
                raise ValueError(
                    f"No target face matches the reference gender "
                    f"({gender_label(source_gender)}); the target has "
                    f"{len(targets)} face(s) ({detail})")
            face_idx = next(i for i, record in enumerate(targets)
                            if record is target_record)
            logger.info("  target face %d: bbox=%s quality=%.3f score=%.3f "
                        "gender=%s age=%s",
                       face_idx, tuple(map(int, target_record.bbox)),
                       target_record.quality, target_record.score,
                       gender_label(target_record.gender), target_record.age)
            parse_face(target, target_record, self._parser)
            source = virtual_face(identity, reference_records[0].face)

            quality_mode = str(getattr(cfg, "quality_mode", "manual")).strip().lower()
            is_inswapper = os.path.basename(model_file) == "inswapper_128.onnx"
            explicit_boost = bool(str(swapper_name).partition("@")[2])
            candidate_boosts = candidate_boosts_for(
                swapper_name, model_file, boost, quality_mode)
            candidates = []
            profiles = (("default", False),)
            if quality_mode == "best_likeness" and is_inswapper:
                profiles += (("identity-preserving", True),)
            for candidate_boost in candidate_boosts:
                t0 = time.perf_counter()
                swapped = self._run_swap(swapper, target.copy(), target_record, source,
                                         candidate_boost, paste_back=True)
                if swapped is None:
                    logger.warning("  candidate x%d returned no image", candidate_boost)
                    continue
                raw_similarity = self._reference_similarity(swapped, target_record, identity)
                for profile, conservative in profiles:
                    candidate = self._postprocess(
                        swapped, target, target_record, cfg, conservative=conservative)
                    similarity = self._reference_similarity(candidate, target_record, identity)
                    logger.info("  candidate x%d %s: raw=%s final=%s (%.2fs)",
                                candidate_boost, profile,
                                "unverified" if raw_similarity is None else f"{raw_similarity:.3f}",
                                "unverified" if similarity is None else f"{similarity:.3f}",
                                time.perf_counter() - t0)
                    candidates.append((candidate_boost, profile, candidate,
                                       similarity, raw_similarity))
            if not candidates:
                raise RuntimeError("Face swap model returned no image")

            selected_boost, selected_profile, result, confidence, raw_confidence = (
                select_likeness_candidate(candidates))
            logger.info("  selected candidate: x%d %s (raw=%s final=%s)",
                        selected_boost, selected_profile,
                        "unverified" if raw_confidence is None else f"{raw_confidence:.3f}",
                        "unverified" if confidence is None else f"{confidence:.3f}")

            use_codeformer = False
            if cfg.codeformer_enabled:
                t1 = time.perf_counter()
                result, use_codeformer = self._restore_candidate(
                    result, target, target_record, identity, cfg)
                logger.info("  CodeFormer stage: %s (%.2fs, w=%.2f)",
                            "accepted" if use_codeformer else "skipped/rejected",
                            time.perf_counter() - t1, cfg.codeformer_weight)
                confidence = self._reference_similarity(result, target_record, identity)
            confidence_value = 0.0 if confidence is None else confidence
            verdict = "verified" if confidence is not None and confidence >= cfg.verification_threshold else "UNVERIFIED/LOW CONFIDENCE"
            restoration = f"CodeFormer w={cfg.codeformer_weight:.2f}" if use_codeformer else "off"
            extras = []
            if explicit_boost and selected_boost > 1:
                extras.append(f"requested pixel-boost {selected_boost * 128}px (x{selected_boost})")
            elif quality_mode == "best_likeness" and is_inswapper:
                extras.append(f"best-likeness x{selected_boost} {selected_profile} (compared 128/256/512)")
            elif selected_boost > 1:
                extras.append(f"pixel-boost x{selected_boost}")
            if self._parser is not None:
                extras.append("parsing")
            if cfg.occluder_enabled and self._occluder is not None:
                extras.append("occluder")
            mouth_mode = str(getattr(cfg, "mouth_mode", "")).strip().lower()
            if mouth_mode == "swap_lips_keep_teeth":
                extras.append("target-teeth")
            elif mouth_mode == "keep_target_mouth":
                extras.append("target-mouth")
            extra_str = f" | {' | '.join(extras)}" if extras else ""
            match_str = (""
                         if match_mode != "gender" else
                         f" | gender-matched ({gender_label(source_gender)}): "
                         f"target face {face_idx} of {len(targets)}")
            logger.info("  identity: reference similarity=%s | %s | threshold=%.2f",
                        "unavailable" if confidence is None else f"{confidence:.3f}",
                        verdict, cfg.verification_threshold)
            logger.info("  total time: %.2fs", time.perf_counter() - started)

            # Persist every completed swap as <date>_<NN>.png; a failed save
            # must never discard a successful swap, so only warn on error.
            saved_note = ""
            if cfg.save_swaps:
                try:
                    out_path = allocate_output_path(self.output_dir)
                    ok, encoded = cv2.imencode(".png", result)
                    if ok:
                        encoded.tofile(out_path)
                        logger.info("  saved swapped image: %s", out_path)
                        saved_note = f" | saved {os.path.basename(out_path)}"
                    else:
                        logger.warning("  PNG encoding failed; swap not saved")
                except Exception as exc:
                    logger.warning("  could not save swapped image: %s", exc)

            logger.info("=" * 60)
            status = (f"Visenlo complete | references accepted: {len(reference_records)} | "
                      f"reference similarity: {confidence_value:.3f} | {verdict} | "
                      f"restoration: {restoration}{extra_str}{match_str} | "
                      f"{time.perf_counter() - started:.2f}s{saved_note}")
            return result, status


# Backward compatibility: old ReactorX name still resolves to the same pipeline.
ReactorXPipeline = VisenloPipeline
