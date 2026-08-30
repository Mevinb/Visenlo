"""Face parsing (BiSeNet) and occlusion (XSeg) ONNX adapters.

Models follow the FaceFusion conventions:
- bisenet_resnet_34.onnx: 512x512 RGB input, ImageNet-normalized, NCHW;
  outputs a 19-class CelebAMask-HQ label map.
- xseg_1.onnx: 256x256 RGB input scaled to [0,1], NHWC; outputs a face keep
  probability map, which this adapter converts to an occluder probability map.
"""

from __future__ import annotations

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# Normalized ffhq_512 template (left eye, right eye, nose, mouth corners),
# matching the alignment used by GFPGAN/CodeFormer-era models.
FFHQ_TEMPLATE_NORM = np.array([
    [0.37691676, 0.46864664],
    [0.62285697, 0.46912813],
    [0.50123859, 0.61331904],
    [0.39308822, 0.72541100],
    [0.61150205, 0.72490465],
], np.float32)


def ffhq_template(size: int) -> np.ndarray:
    return (FFHQ_TEMPLATE_NORM * float(size)).astype(np.float32)


def aligned_crop_matrix(kps: np.ndarray, size: int):
    """Similarity transform (image -> aligned crop) fitted robustly with RANSAC.

    Unlike a plain least-squares fit, a single bad landmark cannot skew the
    whole crop.
    """
    dst = ffhq_template(size)
    kps = np.asarray(kps, np.float32).reshape(-1, 2)
    if kps.shape[0] < 2:
        return None
    matrix, _ = cv2.estimateAffinePartial2D(
        kps, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0,
        confidence=0.99, refineIters=10)
    return None if matrix is None else matrix.astype(np.float32)


class BisenetParser:
    """Callable: BGR face crop -> int label map (CelebAMask-HQ classes)."""

    def __init__(self, model_path: str, providers=None):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.output_name = self.session.get_outputs()[0].name
        shape = inp.shape
        size = 512
        if len(shape) == 4:
            for idx in (2, 3):
                v = shape[idx] if idx < len(shape) else None
                if isinstance(v, int) and v > 0 and v != 3:
                    size = v
                    break
        self.size = int(size)

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        img = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        blob = np.ascontiguousarray(img.transpose(2, 0, 1))[None]
        logits = self.session.run([self.output_name], {self.input_name: blob})[0]
        labels = logits[0].argmax(axis=0).astype(np.int32)
        if labels.shape != (h, w):
            # Labels are discrete class ids 0..18 — use nearest on uint8 to avoid float rounding.
            labels = cv2.resize(labels.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST).astype(np.int32)
        return labels


class XSegOccluder:
    """Callable-ish occlusion detector producing full-frame occluder masks."""

    def __init__(self, model_path: str, providers=None):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.output_name = self.session.get_outputs()[0].name
        shape = inp.shape
        size = 256
        if len(shape) == 4:
            # Handle both NHWC (1,H,W,3) and NCHW (1,3,H,W); skip channel dim (3).
            candidates = []
            for idx in range(1, 4):
                v = shape[idx] if idx < len(shape) else None
                if isinstance(v, int) and v > 0 and v != 3:
                    candidates.append(v)
            if candidates:
                # Prefer the largest spatial dimension (covers both layouts).
                size = max(candidates) if len(candidates) > 1 else candidates[0]
                # For NHWC the spatial dims are [1,2], for NCHW [2,3]; if both present they are equal.
                if len(shape) == 4 and isinstance(shape[3], int) and shape[3] == 3:
                    # NHWC: shape[1] is H
                    if isinstance(shape[1], int) and shape[1] > 0 and shape[1] != 3:
                        size = shape[1]
                elif len(shape) == 4 and isinstance(shape[1], int) and shape[1] == 3:
                    # NCHW: shape[2] is H
                    if isinstance(shape[2], int) and shape[2] > 0 and shape[2] != 3:
                        size = shape[2]
        self.size = int(size)

    def detect(self, crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        blob = (img[:, :, ::-1].astype(np.float32) / 255.0)[None]
        out = self.session.run([self.output_name], {self.input_name: blob})[0]
        # squeeze() handles both NCHW (1,1,H,W) and NHWC (1,H,W,1) exports;
        # anything else would silently mis-index under [0]-stripping.
        keep_mask = np.squeeze(np.asarray(out, np.float32))
        if keep_mask.ndim != 2:
            raise ValueError(f"xseg_1 output has unsupported layout {np.asarray(out).shape}")
        # XSeg emits high values for valid face pixels. ReactorX consumes an
        # occluder mask (high = restore the original target), so invert in crop
        # space before warping. Inverting after warp would mark the entire area
        # outside the aligned crop as occluded.
        return 1.0 - np.clip(keep_mask, 0, 1)

    def map_to_frame(self, record, target: np.ndarray):
        """Return a full-frame occluder mask for this face, or None."""
        kps = getattr(record.face, "kps", None)
        if kps is None:
            return None
        matrix = aligned_crop_matrix(kps, self.size)
        if matrix is None:
            return None
        h, w = target.shape[:2]
        crop = cv2.warpAffine(target, matrix, (self.size, self.size),
                              flags=cv2.INTER_LINEAR, borderValue=0)
        occ = self.detect(crop)
        inverse = cv2.invertAffineTransform(matrix)
        full = cv2.warpAffine(occ, inverse, (w, h), flags=cv2.INTER_LINEAR)
        sigma = max(2.0, self.size * .02)
        full = cv2.GaussianBlur(full, (int(sigma * 4) | 1,) * 2, sigma)
        return np.clip(full, 0, 1)
