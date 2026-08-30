"""Pre/post-processing helpers for the CodeFormer restoration stage."""

from __future__ import annotations

import cv2
import numpy as np

CODEFORMER_SIZE = 512


def to_onnx_blob(bgr_face: np.ndarray) -> np.ndarray:
    """Convert a BGR face crop to the model's [1,3,512,512] float32 input in [-1,1]."""
    rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB).astype(np.float32)
    return (rgb / 127.5 - 1.0)[None].transpose(0, 3, 1, 2)


def from_onnx(output: np.ndarray) -> np.ndarray:
    """Decode the model's [1,3,512,512] float32 output back to a BGR uint8 face."""
    restored = np.clip((output[0].transpose(1, 2, 0) + 1.0) * 0.5 * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)


def scale_affine(matrix: np.ndarray, from_size: int, to_size: int) -> np.ndarray:
    """Rescale a 2x3 affine so crop coords change from from_size to to_size space.

    The affine maps image -> crop. A crop of size to_size spans the same normalized
    content as from_size with coordinates scaled by to_size/from_size, so the affine
    columns are scaled by that factor.
    """
    factor = to_size / float(from_size)
    scaled = matrix.copy()
    scaled[:, 0] *= factor
    scaled[:, 1] *= factor
    scaled[:, 2] *= factor
    return scaled


def build_aligned_face_mask(landmarks: np.ndarray, swap_matrix: np.ndarray,
                            swap_size: int, size: int = CODEFORMER_SIZE) -> np.ndarray:
    """Build a feathered face-region mask in the aligned crop space.

    Warps the detected landmarks into the aligned crop via the swap affine and
    fills their convex hull. Falls back to a face ellipse when landmarks are too
    sparse. The result only replaces the face interior so the target's native
    hair/background are preserved and no square edge is visible.
    """
    mask = np.zeros((size, size), np.float32)
    if landmarks is not None and len(landmarks) >= 8:
        matrix = scale_affine(swap_matrix, swap_size, size)
        pts = np.hstack([np.asarray(landmarks, np.float32),
                         np.ones((len(landmarks), 1), np.float32)]) @ matrix.T
        cv2.fillConvexPoly(mask, cv2.convexHull(pts.astype(np.int32)), 1.0)
        dilate = max(5, int(size * .04))
        mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))
    else:
        cv2.ellipse(mask, (int(size * .5), int(size * .60)),
                    (int(size * .34), int(size * .40)), 0, 0, 360, 1.0, -1)
    feather = max(15, int(size * .06)) | 1
    mask = cv2.GaussianBlur(mask, (feather, feather), feather * .4)
    return np.clip(mask, 0, 1)


def paste_restored_back(restored: np.ndarray, swap_matrix: np.ndarray,
                        swap_size: int, target: np.ndarray, landmarks=None):
    """Warp a 512px restored aligned face back onto the target using the swap's affine.

    Replaces only the face region (from the detected landmarks) so the restored
    face is blended in without a visible square edge and the target's native
    hair/background are preserved.
    """
    h_r, w_r = restored.shape[:2]
    assert h_r == w_r, f"restored crop must be square, got {restored.shape[:2]}"
    size = int(h_r)
    h, w = target.shape[:2]
    matrix = scale_affine(swap_matrix, swap_size, size)
    inverse = cv2.invertAffineTransform(matrix)
    warped = cv2.warpAffine(restored, inverse, (w, h), borderValue=0, flags=cv2.INTER_LANCZOS4)
    aligned_mask = build_aligned_face_mask(landmarks, swap_matrix, swap_size, size)
    mask = cv2.warpAffine(aligned_mask, inverse, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return (warped.astype(np.float32) * mask[:, :, None] +
            target.astype(np.float32) * (1 - mask[:, :, None])).astype(np.uint8)
