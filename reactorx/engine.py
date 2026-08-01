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


