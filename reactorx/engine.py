"""Model-independent stages used by ReactorX Swap Engine v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FaceRecord:
    face: object
    bbox: Tuple[int, int, int, int]
    landmarks: np.ndarray
    embedding: Optional[np.ndarray]
    score: float
    quality: float = 0.0
    masks: Dict[str, np.ndarray] = field(default_factory=dict)


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


