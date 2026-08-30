"""Model-independent stages used by ReactorX Swap Engine v1."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import cv2
import numpy as np

logger = logging.getLogger("reactorx.engine")


@dataclass
class FaceRecord:
    face: object
    bbox: tuple[int, int, int, int]
    landmarks: np.ndarray
    embedding: np.ndarray | None
    score: float
    quality: float = 0.0
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    # 0 = female, 1 = male (from insightface's genderage model); None when the
    # analysis pack does not provide it. Age is captured for future features.
    gender: int | None = None
    age: int | None = None


MASK_NAMES = ("skin", "eyes", "eyebrows", "nose", "lips", "teeth",
              "hair", "neck", "ear", "glasses", "hat", "background")


def dense_landmarks(face) -> np.ndarray:
    points = getattr(face, "landmark_2d_106", None)
    if points is None:
        points = getattr(face, "kps", None)
    points = np.asarray(points, dtype=np.float32) if points is not None else np.empty((0, 2), np.float32)
    return points.reshape((-1, 2)) if points.size else np.empty((0, 2), np.float32)


def clamp_bbox(face, shape, padding=0.18):
    h, w = shape[:2]
    x1, y1, x2, y2 = [float(v) for v in face.bbox]
    px, py = (x2 - x1) * padding, (y2 - y1) * padding
    return (max(0, int(x1 - px)), max(0, int(y1 - py)),
            min(w, int(x2 + px)), min(h, int(y2 + py)))


def quality_score(image, record: FaceRecord) -> float:
    """Sharpness (normalized to a fixed crop size so it is resolution-independent)
    plus a yaw estimate from the detector's five keypoints: the nose should sit
    near the horizontal midpoint between the eyes for a frontal face."""
    x1, y1, x2, y2 = record.bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    sharp = min(1.0, cv2.Laplacian(gray, cv2.CV_32F).var() / 180.0)
    kps = getattr(record.face, "kps", None)
    pose = 0.5
    if kps is not None and len(kps) >= 3:
        left_eye, right_eye, nose = (np.asarray(kps[i], np.float32) for i in range(3))
        span = float(right_eye[0] - left_eye[0])
        if abs(span) > 1.0:
            ratio = float(nose[0] - left_eye[0]) / span
            pose = float(np.clip(1.0 - abs(ratio - .5) / .30, 0, 1))
    return float(np.clip(.60 * sharp + .40 * pose, 0, 1))


def _ellipse(mask, center, axes, value=1.0):
    cv2.ellipse(mask, tuple(map(int, center)), tuple(map(int, axes)), 0, 0, 360, value, -1)


def fallback_masks(shape):
    h, w = shape[:2]
    masks = {name: np.zeros((h, w), np.float32) for name in MASK_NAMES}
    _ellipse(masks["skin"], (w * .5, h * .53), (w * .40, h * .44))
    _ellipse(masks["hair"], (w * .5, h * .10), (w * .48, h * .20))
    _ellipse(masks["eyes"], (w * .5, h * .37), (w * .32, h * .09))
    _ellipse(masks["eyebrows"], (w * .5, h * .28), (w * .30, h * .05))
    _ellipse(masks["nose"], (w * .5, h * .52), (w * .13, h * .19))
    _ellipse(masks["lips"], (w * .5, h * .72), (w * .20, h * .09))
    masks["background"] = 1.0 - np.clip(masks["skin"] + masks["hair"], 0, 1)
    for name in masks:
        masks[name] = cv2.GaussianBlur(masks[name], (0, 0), max(1.0, min(h, w) * .012))
    return masks


def soften_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Feather a binary mask the FaceFusion way: blur, then clip to [0.5, 1] and
    rescale. The interior stays a hard 1.0 (no ghosting under the face) and the
    falloff is confined to the true boundary instead of bleeding outward."""
    if sigma <= 0:
        return np.clip(mask, 0, 1)
    h, w = mask.shape[:2]
    k = int(sigma * 4) | 1
    # Clamp kernel to image size to avoid cv2.error on huge sigma / small masks.
    max_k = (min(h, w) // 2) | 1
    if max_k >= 3 and k > max_k:
        k = max_k if max_k % 2 == 1 else max_k - 1
        sigma = max(1.0, k / 4.0)
    blurred = cv2.GaussianBlur(np.clip(mask, 0, 1).astype(np.float32), (k, k), sigma)
    return np.clip((np.clip(blurred, 0.5, 1.0) - 0.5) * 2.0, 0, 1)


def build_face_mask(bbox, shape, landmarks=None, feather=.25):
    """Face-region mask for color/sharpen/occlusion stages.

    Uses the landmark convex hull (expanded ~12% so it reaches the hairline)
    when dense landmarks are available, which follows head roll; falls back to
    an axis-aligned ellipse from the padded bbox otherwise.
    """
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    mask = np.zeros((h, w), np.float32)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    ax, ay = max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2)
    pts = None if landmarks is None else np.asarray(landmarks, np.float32).reshape(-1, 2)
    if pts is not None and len(pts) >= 8:
        center = pts.mean(axis=0)
        grown = center + (pts - center) * 1.12
        hull = cv2.convexHull(grown.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1.0)
    else:
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
    sigma = max(3.0, min(ax, ay) * feather * .5)
    return soften_mask(mask, sigma)


def recover_occlusions(target, result, face_mask, sensitivity=.6):
    """Restore only thin intrusions (hair strands, glasses arms) at the face
    boundary. Reverted pixels are blended at `sensitivity` so mistakes wash out
    instead of hard-replacing swapped skin with uncorrected target pixels."""
    h, w = target.shape[:2]
    roi = face_mask > .1
    if not np.any(roi) or sensitivity <= 0:
        return result
    gray_t = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_r = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(gray_t - gray_r)
    gx_t = cv2.Sobel(gray_t, cv2.CV_32F, 1, 0)
    gy_t = cv2.Sobel(gray_t, cv2.CV_32F, 0, 1)
    edges_t = np.abs(gx_t) + np.abs(gy_t)
    gx_r = cv2.Sobel(gray_r, cv2.CV_32F, 1, 0)
    gy_r = cv2.Sobel(gray_r, cv2.CV_32F, 0, 1)
    edges_r = np.abs(gx_r) + np.abs(gy_r)
    lost = (edges_t > 60) & (edges_r < edges_t * .6)
    thr = max(20.0, float(np.percentile(diff[roi], 85)))
    candidates = (lost & (diff > thr) & roi).astype(np.uint8)
    face_bin = (face_mask > .5).astype(np.uint8)
    k = max(3, int(min(h, w) * .008))
    kernel = np.ones((k, k), np.uint8)
    inner = cv2.erode(face_bin, kernel)
    boundary = np.clip(face_bin - inner, 0, 1).astype(np.uint8)
    # Only consider occluders within a narrow band around the boundary; deep
    # interior edge loss is almost always legitimate swap smoothing.
    band_kernel = max(3, k * 3) | 1
    band = cv2.dilate(boundary, np.ones((band_kernel, band_kernel), np.uint8))
    candidates &= band
    num, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    occ = np.zeros_like(candidates)
    min_area = max(6, int(face_bin.sum() * .0008))
    max_area = int(face_bin.sum() * .12)
    for comp in range(1, num):
        area = int(stats[comp, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        region = labels == comp
        if np.any(boundary[region]):
            occ[region] = 1
    if not np.any(occ):
        return result
    fk = max(5, k * 2) | 1
    occ = cv2.GaussianBlur(occ.astype(np.float32), (fk, fk), fk * .35)
    occ = np.clip(occ, 0, 1) * float(sensitivity)
    return (result.astype(np.float32) * (1 - occ[:, :, None]) +
            target.astype(np.float32) * occ[:, :, None]).astype(np.uint8)


def parse_face(image, record: FaceRecord, parser=None):
    x1, y1, x2, y2 = record.bbox
    crop = image[y1:y2, x1:x2]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    masks = None
    if parser is not None:
        try:
            labels = parser(crop)
            # CelebAMask-HQ label ids: 1 skin, 2/3 brows, 4/5 eyes, 6 glasses,
            # 7/8 ears, 10 nose, 12/13 lips, 14 neck, 17 hair, 18 hat.
            ids = {"skin": [1], "eyes": [4, 5], "eyebrows": [2, 3], "nose": [10],
                   "lips": [12, 13], "neck": [14], "hair": [17], "ear": [7, 8],
                   "glasses": [6], "hat": [18]}
            masks = {name: np.zeros(labels.shape, np.float32) for name in MASK_NAMES}
            for name, values in ids.items():
                masks[name] = np.isin(labels, values).astype(np.float32)
            masks["background"] = (labels == 0).astype(np.float32)
            for name in MASK_NAMES:
                # Parser already resizes labels to (bh,bw); only resize if shape mismatches.
                if masks[name].shape[1] != bw or masks[name].shape[0] != bh:
                    masks[name] = cv2.resize(masks[name], (bw, bh), interpolation=cv2.INTER_NEAREST)
        except Exception as exc:
            logger.warning("face parsing failed (%s); using geometric masks", exc)
            masks = None
    record.masks = masks if masks is not None else fallback_masks(crop.shape)
    return record


def weighted_identity(records: list[FaceRecord]) -> np.ndarray:
    usable = [record for record in records if record.embedding is not None and record.quality > 0]
    if not usable:
        raise RuntimeError("No usable reference face embeddings")
    vectors = np.asarray([record.embedding for record in usable], np.float32)
    weights = np.asarray([max(record.quality, .05) for record in usable], np.float32)
    vector = (vectors * (weights / weights.sum())[:, None]).sum(axis=0)
    return vector / max(np.linalg.norm(vector), 1e-8)


def virtual_face(embedding, source_face):
    """Provide inswapper's expected face object with an aggregate identity."""
    result = SimpleNamespace(**getattr(source_face, "__dict__", {}))
    result.normed_embedding = np.asarray(embedding, np.float32)
    return result


def unsharp_mask(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0,
                 amount: float = 0.5) -> np.ndarray:
    """Apply unsharp masking to a full BGR image."""
    if amount <= 0:
        return image
    blurred = cv2.GaussianBlur(image, (kernel_size | 1, kernel_size | 1), sigma)
    detail = image.astype(np.float32) - blurred.astype(np.float32)
    result = image.astype(np.float32) + detail * amount
    return np.clip(result, 0, 255).astype(np.uint8)


def sharpen_face_region(image: np.ndarray, face_mask: np.ndarray,
                        amount: float = 0.5, sigma: float | None = None) -> np.ndarray:
    """Apply unsharp masking only within the face mask region.

    The mask is a float32 [0,1] array matching the image dimensions. `sigma`
    sets the detail radius in pixels; callers should scale it with the face
    size so small and large faces get proportionate sharpening (defaults to a
    strength-coupled radius for backwards compatibility).
    """
    if amount <= 0:
        return image
    sigma = max(1.0, amount * 2.0) if sigma is None else max(1.0, float(sigma))
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    detail = image.astype(np.float32) - blurred.astype(np.float32)
    sharpened = image.astype(np.float32) + detail * amount
    alpha = np.clip(face_mask, 0, 1)[:, :, None]
    result = sharpened * alpha + image.astype(np.float32) * (1 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def color_match(swapped, target, mask, strength=.75):
    """Reinhard-style LAB statistics transfer toward the target, restricted to
    `mask`. Returns a float32 frame so callers quantize to uint8 only once."""
    if strength <= 0:
        return swapped.astype(np.float32)
    lab_s = cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_t = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    active = mask > .25
    if int(active.sum()) < 32:
        return swapped.astype(np.float32)
    for channel in range(3):
        plane = lab_s[:, :, channel]
        source_values = plane[active]
        target_values = lab_t[:, :, channel][active]
        ratio = 1 + (target_values.std() / max(source_values.std(), 1e-3) - 1) * strength
        plane = (plane - source_values.mean()) * ratio + (
            source_values.mean() + (target_values.mean() - source_values.mean()) * strength
        )
        lab_s[:, :, channel] = np.clip(plane, 0, 255)
    matched = cv2.cvtColor(lab_s.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    return matched * mask[:, :, None] + swapped.astype(np.float32) * (1 - mask[:, :, None])


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8))


def composite_region_mask(record: FaceRecord, shape, names, dilate_frac=.02,
                          feather_frac=.02):
    """Lift the record's parsed (crop-space) region masks into a full-frame,
    softly feathered union. Returns None when none of `names` were parsed so
    callers can fall back to a geometric mask."""
    masks = getattr(record, "masks", None) or {}
    planes = [np.asarray(masks[name], np.float32) for name in names
              if masks.get(name) is not None]
    if not planes:
        return None
    h, w = shape[:2]
    x1, y1, x2, y2 = record.bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    union = np.clip(sum(planes), 0, 1)
    if union.shape[1] != bw or union.shape[0] != bh:
        union = cv2.resize(union, (bw, bh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((h, w), np.float32)
    canvas[y1:y2, x1:x2] = union
    k = max(3, int(min(bw, bh) * dilate_frac)) | 1
    # dilate expects uint8; convert slice, dilate, then back to float.
    roi_u8 = (np.clip(canvas[y1:y2, x1:x2], 0, 1) * 255).astype(np.uint8)
    region = cv2.dilate(roi_u8, np.ones((k, k), np.uint8)).astype(np.float32) / 255.0
    canvas[y1:y2, x1:x2] = region
    sigma = max(2.0, min(bw, bh) * feather_frac)
    return soften_mask(canvas, sigma)
