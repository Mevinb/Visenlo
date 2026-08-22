"""Complete ReactorX v1 pipeline orchestration."""

from __future__ import annotations

import logging
import os
import threading
import time
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

logger = logging.getLogger("reactorx")
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
        self.input_size = int(self.session.get_inputs()[0].shape[2])
        graph = onnx.load(model_path).graph
        emap = next((item for item in graph.initializer if item.name == "emap"), graph.initializer[-1])
        self.emap = numpy_helper.to_array(emap).astype(np.float32)

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
    codeformer_enabled: bool = False
    codeformer_weight: float = .8
    codeformer_verify_identity: bool = True
    sharpen_strength: float = .5
    det_size: int = 640
    det_size_max: int = 1280
    occluder_enabled: bool = True


class ReactorXPipeline:
    def __init__(self, models_path: str, config: PipelineConfig | None = None):
        self.models_path = models_path
        self.config = config or PipelineConfig()
        self._lock = threading.RLock()
        self._analysis = None
        self._swapper = None
        self._swapper_path = None
        self._providers = None
        self._parser = None
        self._occluder = None
        self._codeformer = None
        os.makedirs(models_path, exist_ok=True)

    def update_config(self, config: PipelineConfig):
        """Swap configuration atomically; safe against a running process()."""
        with self._lock:
            self.config = config

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

    def _load(self, swapper_name):
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
            except Exception:
                pass
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
            use_cuda = any(isinstance(p, tuple) and p[0] == "CUDAExecutionProvider"
                           for p in self._providers) or \
                "CUDAExecutionProvider" in self._providers
            self._analysis = FaceAnalysis(
                name="buffalo_l",
                root=insight_root,
                providers=self._providers,
            )
            self._analysis.prepare(ctx_id=0 if use_cuda else -1, det_size=(640, 640))
            logger.info("FaceAnalysis ready (CUDA=%s)", use_cuda)
        candidates = [os.path.join(self.models_path, swapper_name),
                      os.path.join(insight_root, "models", swapper_name)]
        path = next((candidate for candidate in candidates if os.path.isfile(candidate)), None)
        if path is None:
            raise FileNotFoundError(f"Place {swapper_name} in {self.models_path}")
        if self._swapper is None or path != self._swapper_path:
            logger.info("Loading swapper: %s", path)
            if os.path.basename(path) == "reswapper_256.onnx":
                self._swapper = Reswapper256(path, self._providers)
            else:
                self._swapper = model_zoo.get_model(path, providers=self._providers)
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

    def _restoration_keeps_identity(self, aligned, restored, identity, target_record):
        """Return True if CodeFormer restoration preserves the swapped identity.

        Embeds the aligned swapped crop and the restored crop with the recognition
        model directly (using the standard alignment template), and compares their
        reference-identity similarity. Falls back to accepting if embedding fails.
        """
        try:
            swap_embed = self._embed_aligned(aligned, aligned.shape[0])
            restored_embed = self._embed_aligned(restored, restored.shape[0])
        except Exception:
            return True
        if swap_embed is None or restored_embed is None:
            return True
        swap_sim = cosine(identity, swap_embed)
        restored_sim = cosine(identity, restored_embed)
        return restored_sim >= max(swap_sim * .95, swap_sim - .10)

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

    def process(self, references: list[np.ndarray], target: np.ndarray, source_index=0,
                target_index=0, swapper_name="inswapper_128.onnx"):
        started = time.perf_counter()
        logger.info("=" * 60)
        logger.info("ReactorX pipeline started")
        logger.info("  swapper:   %s", swapper_name)
        logger.info("  target:    %dx%d", target.shape[1], target.shape[0])
        logger.info("  references: %d", len(references))
        with self._lock:
            cfg = self.config
            model_file, _, boost_raw = swapper_name.partition("@")
            boost = int(boost_raw) if boost_raw.isdigit() and int(boost_raw) > 1 else 1
            swapper = self._load(model_file)
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
                logger.info("    selected face %d: bbox=%s quality=%.3f score=%.3f",
                           int(source_index),
                           tuple(map(int, record.bbox)), record.quality, record.score)
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
            if not 0 <= int(target_index) < len(targets):
                raise ValueError(
                    f"Target face index {target_index} is out of range; "
                    f"the target has {len(targets)} face(s)")

            identity = weighted_identity(reference_records)
            logger.info("  identity: aggregated from %d reference(s), norm=%.4f",
                       len(reference_records), float(np.linalg.norm(identity)))
            target_record = targets[int(target_index)]
            logger.info("  target face %d: bbox=%s quality=%.3f score=%.3f",
                       int(target_index),
                       tuple(map(int, target_record.bbox)),
                       target_record.quality, target_record.score)
            parse_face(target, target_record, self._parser)
            source = virtual_face(identity, reference_records[0].face)

            # The swapper's paste-back already produces a complete full-face swap
            # with its own blend mask. With CodeFormer enabled we instead extract the
            # aligned swapped face, restore it, and paste the restored face back.
            use_codeformer = cfg.codeformer_enabled
            t0 = time.perf_counter()
            if use_codeformer:
                aligned, matrix = self._run_swap(swapper, target.copy(), target_record,
                                                 source, boost, paste_back=False)
                if aligned is None:
                    raise RuntimeError("Face swap model returned no image")
                logger.info("  swap: aligned crop %dx%d (took %.2fs)",
                           aligned.shape[1], aligned.shape[0], time.perf_counter() - t0)
                codeformer = self._ensure_codeformer()
                t1 = time.perf_counter()
                restored = codeformer.restore_aligned(aligned)
                logger.info("  CodeFormer: restored %dx%d (took %.2fs, weight=%.2f)",
                           restored.shape[1], restored.shape[0],
                           time.perf_counter() - t1, cfg.codeformer_weight)
                if cfg.codeformer_verify_identity and not self._restoration_keeps_identity(
                        aligned, restored, identity, target_record):
                    logger.warning("  identity check failed: CodeFormer altered identity, falling back to plain swap")
                    swapped = self._run_swap(swapper, target.copy(), target_record,
                                             source, boost, paste_back=True)
                    use_codeformer = False
                else:
                    logger.info("  identity check: PASSED (CodeFormer preserves identity)")
                    swapped = paste_restored_back(restored, matrix, aligned.shape[0], target,
                                                  target_record.landmarks)
            else:
                swapped = self._run_swap(swapper, target.copy(), target_record,
                                         source, boost, paste_back=True)
                if swapped is None:
                    raise RuntimeError("Face swap model returned no image")
                logger.info("  swap: plain paste-back %dx%d (took %.2fs)",
                            swapped.shape[1], swapped.shape[0], time.perf_counter() - t0)

            shape = target.shape
            face_mask = build_face_mask(target_record.bbox, shape,
                                        landmarks=target_record.landmarks)
            # Tight interior (skin + features) keeps color/sharpen statistics on
            # actual face pixels instead of hair/background diluting them.
            interior = composite_region_mask(
                target_record, shape, ("skin", "eyebrows", "nose", "lips", "eyes", "neck"))
            face_px = max(int(target_record.bbox[2] - target_record.bbox[0]), 1)
            if interior is None:
                erode_k = max(3, int(face_px * .08)) | 1
                interior = cv2.erode(face_mask, np.ones((erode_k, erode_k), np.float32))

            corrected = color_match(swapped, target, interior, cfg.color_strength)
            result = np.clip(corrected, 0, 255).astype(np.uint8)
            logger.info("  color match: strength=%.2f (interior mask)", cfg.color_strength)

            # Model-based occluders first (glasses, hands, hair over the face).
            if cfg.occluder_enabled and self._occluder is not None:
                occ_full = self._occluder.map_to_frame(target_record, target)
                if occ_full is not None:
                    result = (result.astype(np.float32) * (1 - occ_full[:, :, None]) +
                              target.astype(np.float32) * occ_full[:, :, None]).astype(np.uint8)
                    logger.info("  occlusion mask: xseg applied")

            # Restore only occluders (hair strands, glasses) that intrude from the
            # face boundary. The interior of the swapped face is left intact.
            result = recover_occlusions(target, result, face_mask)
            logger.info("  occlusion recovery: complete")

            if cfg.sharpen_strength > 0:
                amount = cfg.sharpen_strength * .5
                sigma = max(1.0, face_px * .006)
                result = sharpen_face_region(result, interior, amount=amount, sigma=sigma)
                logger.info("  face-region sharpen: amount=%.2f sigma=%.1f (face ~%dpx)",
                            amount, sigma, face_px)

            # Detect and embed the generated face for actual post-swap verification.
            generated = self._detect(result)
            generated_embedding = None
            if generated:
                target_center = (target_record.bbox[0] + target_record.bbox[2]) * .5
                matched = min(generated, key=lambda item: abs((item.bbox[0] + item.bbox[2]) * .5 - target_center))
                generated_embedding = matched.embedding
                logger.info("  post-swap detect: %d face(s), matched bbox=%s score=%.3f",
                           len(generated), tuple(map(int, matched.bbox)), matched.score)
            else:
                logger.warning("  post-swap detect: no faces found")
            confidence = cosine(identity, generated_embedding) if generated_embedding is not None else 0.0
            verdict = "verified" if confidence >= cfg.verification_threshold else "LOW CONFIDENCE"
            restoration = (f"CodeFormer w={cfg.codeformer_weight:.2f}"
                           if use_codeformer else "off")
            extras = []
            if boost > 1:
                extras.append(f"pixel-boost x{boost}")
            if self._parser is not None:
                extras.append("parsing")
            if cfg.occluder_enabled and self._occluder is not None:
                extras.append("occluder")
            extra_str = f" | {' | '.join(extras)}" if extras else ""
            logger.info("  identity: confidence=%.3f | %s | threshold=%.2f",
                        confidence, verdict, cfg.verification_threshold)
            logger.info("  total time: %.2fs", time.perf_counter() - started)
            logger.info("=" * 60)
            status = (f"ReactorX complete | references accepted: {len(reference_records)} | "
                      f"identity confidence: {confidence:.3f} | {verdict} | "
                      f"restoration: {restoration}{extra_str} | "
                      f"{time.perf_counter() - started:.2f}s")
            return result, status
