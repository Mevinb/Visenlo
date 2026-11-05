"""Pixel-boost for inswapper_128: 256/512/1024/2048 output from the 128px model.

Polyphase decomposition (FaceFusion technique): an aligned crop of
128*factor px is split into factor^2 whole-face views downsampled by
`factor` at sub-pixel phase offsets. Every tile keeps the model's expected
receptive field, so there are no seams; outputs are interleaved back.

This module also owns the shared paste-back used by every swapper backend:
the blend is computed only inside the region of the frame the aligned crop
actually touches, optionally restricted to a parsed face region and blended
with a Laplacian pyramid instead of a single alpha feather.
"""

from __future__ import annotations

import cv2
import numpy as np

from .engine import crop_roi, laplacian_blend


def implode_tiles(image: np.ndarray, factor: int) -> np.ndarray:
    h, w = image.shape[:2]
    th, tw = h // factor, w // factor
    channels = image.shape[2]
    tiles = image.reshape(th, factor, tw, factor, channels)
    tiles = tiles.transpose(1, 3, 0, 2, 4)
    return np.ascontiguousarray(tiles.reshape(factor * factor, th, tw, channels))


def explode_tiles(tiles: np.ndarray, factor: int) -> np.ndarray:
    _, th, tw, channels = tiles.shape
    out = tiles.reshape(factor, factor, th, tw, channels)
    out = out.transpose(2, 0, 3, 1, 4)
    return np.ascontiguousarray(out.reshape(th * factor, tw * factor, channels))


def dealias_boost(fake: np.ndarray, factor: int) -> np.ndarray:
    """Remove the period-`factor` interleave grid left by polyphase tiles.

    Each tile hallucinates slightly different high-frequency texture; after
    `explode_tiles` that shows up as a checkerboard with period `factor`. A
    median of that footprint removes the grid while keeping edges, which a
    Gaussian of the same footprint would blur away.
    """
    if factor <= 1:
        return fake
    k = int(factor) | 1
    if k < 3:
        k = 3
    return cv2.medianBlur(fake, min(k, 17))


def max_useful_boost(matrix, native=128, slack=1.5, choices=(1, 2, 4, 8, 16)):
    """Largest boost factor that does not upsample the source face pixels.

    `matrix` is the image->crop affine for the `native` crop; its scale says
    how many source pixels feed one crop pixel. `slack` allows a modest
    upsample before the extra tiles stop carrying information.
    """
    scale = float(np.sqrt(abs(np.linalg.det(np.asarray(matrix, np.float32)[:, :2]))))
    useful = (1.0 / scale) * slack if scale > 1e-6 else 1.0
    allowed = [f for f in choices if f <= useful]
    return max(allowed) if allowed else 1


def inswapper_latent(swapper, source_face):
    latent = np.asarray(source_face.normed_embedding, np.float32).reshape(1, -1)
    latent = np.dot(latent, np.asarray(swapper.emap, np.float32))
    latent /= max(float(np.linalg.norm(latent)), 1e-8)
    return latent.astype(np.float32)


def run_inswapper_tiles(session, input_names, output_name, tiles, latent):
    """Run BGR uint8 tiles through an inswapper-layout session one by one
    (the graph declares a fixed batch of 1). Returns BGR uint8 tiles."""
    outputs = []
    latent = np.ascontiguousarray(latent, dtype=np.float32)
    for tile in tiles:
        blob = tile[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        pred = session.run([output_name], {input_names[0]: np.ascontiguousarray(blob[None]),
                                           input_names[1]: latent})[0]
        outputs.append(pred[0])
    stacked = np.stack(outputs, axis=0)  # (N,3,size,size)
    return np.clip(stacked.transpose(0, 2, 3, 1) * 255, 0, 255).astype(np.uint8)[:, :, :, ::-1]


def inswapper_boost_get(swapper, image, target_face, source_face, factor,
                        paste_back=True, **paste_kwargs):
    """INSwapper-compatible get() with pixel boost. Returns either
    (fake, matrix) when paste_back=False or the blended full frame."""
    from insightface.utils import face_align

    # Support both tuple (128,128) and int (256) layouts.
    raw = getattr(swapper, "input_size", 128)
    base = int(raw[0]) if isinstance(raw, (list, tuple)) else int(raw)
    big = base * factor
    aimg, matrix = face_align.norm_crop2(image, target_face.kps, big)
    assert aimg.shape[0] == big and aimg.shape[1] == big, f"norm_crop2 returned {aimg.shape[:2]} != {big}"

    tiles = implode_tiles(aimg, factor)
    latent = inswapper_latent(swapper, source_face)
    fakes = run_inswapper_tiles(swapper.session, swapper.input_names, swapper.output_names[0], tiles, latent)
    fake = explode_tiles(fakes, factor)
    if getattr(swapper, "dealias", True):
        fake = dealias_boost(fake, factor)
    if not paste_back:
        return fake, matrix
    return _paste_back(aimg, fake, matrix, image, **paste_kwargs)


def shift_affine(matrix, x1, y1):
    """Re-express an image->crop affine for a frame ROI whose origin is (x1, y1):
    crop = A·img + t with img = roi + (x1, y1)  =>  t' = t + A·(x1, y1)."""
    shifted = np.asarray(matrix, np.float32).copy()
    shifted[:, 2] += shifted[:, :2] @ np.array([x1, y1], np.float32)
    return shifted


def crop_box_mask(size: int) -> np.ndarray:
    """INSwapper's erode + feather box mask, expressed in aligned-crop space."""
    mask = np.full((size, size), 255, np.float32)
    k = max(size // 10, 10)
    # Outside the crop is "not face": erode/blur against a zero border so the
    # edge actually shrinks (OpenCV's default erode border would keep it 255).
    mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1,
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    kb = max(size // 20, 5)
    mask = cv2.GaussianBlur(mask, (2 * kb + 1, 2 * kb + 1), 0, borderType=cv2.BORDER_CONSTANT)
    return mask / 255.0


def _paste_back(aimg, bgr_fake, matrix, target_img, region_mask=None, blend="alpha",
                info=None):
    """Blend the aligned fake back onto the frame.

    Only the frame region covered by the crop is touched. `region_mask` is an
    optional crop-space float mask (eg. from face parsing) multiplied into
    the box mask; `blend` is "alpha" (feathered) or "multiband" (Laplacian
    pyramid). When `info` is a dict it receives the frame-space mask and ROI.
    """
    size = int(aimg.shape[0])
    ih, iw = target_img.shape[:2]
    x1, y1, x2, y2 = crop_roi(matrix, size, target_img.shape, pad=.08)
    if x2 <= x1 or y2 <= y1:
        return target_img.copy()
    rw, rh = x2 - x1, y2 - y1
    inverse = cv2.invertAffineTransform(shift_affine(matrix, x1, y1))

    mask = crop_box_mask(size)
    if region_mask is not None:
        rm = np.asarray(region_mask, np.float32)
        if rm.shape[:2] != (size, size):
            rm = cv2.resize(rm, (size, size), interpolation=cv2.INTER_LINEAR)
        mask = mask * np.clip(rm, 0, 1)
    warped_fake = cv2.warpAffine(bgr_fake, inverse, (rw, rh), flags=cv2.INTER_LANCZOS4, borderValue=0)
    warped_mask = cv2.warpAffine(mask, inverse, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=0.0)
    warped_mask = np.clip(warped_mask, 0, 1)

    roi = target_img[y1:y2, x1:x2]
    if blend == "multiband":
        merged = laplacian_blend(warped_fake, roi, warped_mask)
    else:
        merged = (warped_mask[:, :, None] * warped_fake.astype(np.float32) +
                  (1 - warped_mask[:, :, None]) * roi.astype(np.float32))
    out = target_img.copy()
    out[y1:y2, x1:x2] = np.clip(merged, 0, 255).astype(np.uint8)
    if info is not None:
        full = np.zeros((ih, iw), np.float32)
        full[y1:y2, x1:x2] = warped_mask
        info["mask"] = full
        info["roi"] = (x1, y1, x2, y2)
    return out
