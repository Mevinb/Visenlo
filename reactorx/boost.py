"""Pixel-boost for inswapper_128: 256/512 output from the 128px model.

Polyphase decomposition (FaceFusion technique): an aligned crop of
128*factor px is split into factor^2 whole-face views downsampled by
`factor` at sub-pixel phase offsets. Every tile keeps the model's expected
receptive field, so there are no seams; outputs are interleaved back.
"""

from __future__ import annotations

import cv2
import numpy as np


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


def inswapper_boost_get(swapper, image, target_face, source_face, factor,
                        paste_back=True):
    """INSwapper-compatible get() with pixel boost. Returns either
    (fake, matrix) when paste_back=False or the blended full frame."""
    from insightface.utils import face_align

    base = int(swapper.input_size[0])
    big = base * factor
    aimg, matrix = face_align.norm_crop2(image, target_face.kps, big)

    tiles = implode_tiles(aimg, factor)

    latent = np.asarray(source_face.normed_embedding, np.float32).reshape(1, -1)
    latent = np.dot(latent, np.asarray(swapper.emap, np.float32))
    latent /= max(float(np.linalg.norm(latent)), 1e-8)
    latent = latent.astype(np.float32)

    # inswapper_128.onnx declares a fixed batch of 1, so tiles run sequentially.
    input_name = swapper.input_names[0]
    latent_name = swapper.input_names[1]
    output_name = swapper.output_names[0]
    outputs = []
    for tile in tiles:
        blob = tile[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        pred = swapper.session.run(
            [output_name], {input_name: np.ascontiguousarray(blob[None]),
                            latent_name: latent.copy()})[0]
        outputs.append(pred[0])
    stacked = np.stack(outputs, axis=0)  # (N,3,size,size)
    fakes = np.clip(stacked.transpose(0, 2, 3, 1) * 255, 0, 255).astype(np.uint8)[:, :, :, ::-1]
    fake = explode_tiles(fakes, factor)

    if not paste_back:
        return fake, matrix
    return _paste_back(aimg, fake, matrix, image)


def _paste_back(aimg, bgr_fake, matrix, target_img):
    """Blend the boosted aligned fake back, mirroring INSwapper's mask logic
    (erode + feather) but upscaling with Lanczos instead of bilinear."""
    ih, iw = target_img.shape[:2]
    inverse = cv2.invertAffineTransform(matrix)
    warped_fake = cv2.warpAffine(bgr_fake, inverse, (iw, ih),
                                 flags=cv2.INTER_LANCZOS4, borderValue=0.0)
    img_white = cv2.warpAffine(np.full(aimg.shape[:2], 255, np.float32),
                               inverse, (iw, ih), borderValue=0.0)
    img_white[img_white > 20] = 255

    ys, xs = np.where(img_white == 255)
    if ys.size == 0 or xs.size == 0:
        return target_img.astype(np.uint8) if target_img.dtype != np.uint8 else target_img.copy()
    mask_h = int(ys.max() - ys.min())
    mask_w = int(xs.max() - xs.min())
    mask_size = max(int(np.sqrt(mask_h * mask_w)), 10)

    k = max(mask_size // 10, 10)
    img_mask = cv2.erode(img_white, np.ones((k, k), np.uint8), iterations=1)
    kb = max(mask_size // 20, 5)
    img_mask = cv2.GaussianBlur(img_mask, (2 * kb + 1, 2 * kb + 1), 0)
    img_mask /= 255.0
    merged = img_mask[:, :, None] * warped_fake.astype(np.float32) + \
        (1 - img_mask[:, :, None]) * target_img.astype(np.float32)
    return np.clip(merged, 0, 255).astype(np.uint8)
