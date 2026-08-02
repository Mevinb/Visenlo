"""Face parsing (BiSeNet) and occlusion (XSeg) ONNX adapters.

Models follow the FaceFusion conventions:
- bisenet_resnet_34.onnx: 512x512 RGB input, ImageNet-normalized, NCHW;
  outputs a 19-class CelebAMask-HQ label map.
- xseg_1.onnx: 256x256 RGB input scaled to [0,1], NHWC; outputs an occluder
  probability map (high = something in front of the face).
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


