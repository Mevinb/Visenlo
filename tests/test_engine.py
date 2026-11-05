"""Synthetic unit tests for engine helpers (no model weights needed)."""
import cv2
import numpy as np
import pytest

from visenlo.boost import (_paste_back, crop_box_mask, dealias_boost, explode_tiles, implode_tiles,
                           max_useful_boost, shift_affine)
from visenlo.engine import (crop_roi, laplacian_blend, refine_kps, region_blend_mask, sharpness,
                            soften_mask, union_roi)


def test_tiles_roundtrip(rng):
    frame = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    for factor in (2, 4, 8):
        tiles = implode_tiles(frame, factor)
        assert tiles.shape == (factor * factor, 256 // factor, 256 // factor, 3)
        assert np.array_equal(explode_tiles(tiles, factor), frame)


def test_dealias_removes_period_grid():
    base = np.full((128, 128, 3), 120, np.uint8)
    noisy = base.copy()
    noisy[::2, ::2] = 140  # period-2 interleave grid
    cleaned = dealias_boost(noisy, 2)
    assert np.abs(cleaned.astype(int) - 120).mean() < np.abs(noisy.astype(int) - 120).mean()
    assert dealias_boost(noisy, 1) is noisy


def test_max_useful_boost_scales_with_face_size():
    small = np.array([[.5, 0, 0], [0, .5, 0]], np.float32)   # 2 source px per crop px
    large = np.array([[2.0, 0, 0], [0, 2.0, 0]], np.float32)  # crop already upsamples
    assert max_useful_boost(small) == 2
    assert max_useful_boost(large) == 1
    tiny = np.array([[.1, 0, 0], [0, .1, 0]], np.float32)
    assert max_useful_boost(tiny) == 8


def test_shift_affine_matches_cropping():
    matrix = np.array([[.5, .1, 20], [-.1, .5, 30]], np.float32)
    point = np.array([200, 150, 1], np.float32)
    shifted = shift_affine(matrix, 100, 50)
    local_point = np.array([100, 100, 1], np.float32)
    assert np.allclose(matrix @ point, shifted @ local_point, atol=1e-4)


def test_crop_roi_covers_crop_and_clamps():
    matrix = np.array([[1, 0, -100], [0, 1, -50]], np.float32)  # crop = frame shifted
    roi = crop_roi(matrix, 128, (300, 400), pad=0)
    assert roi == (100, 50, 228, 178)
    assert union_roi(roi, (0, 0, 10, 10)) == (0, 0, 228, 178)
    assert union_roi() is None


def test_paste_back_only_touches_roi(rng):
    frame = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    matrix = np.array([[1, 0, -100], [0, 1, -50]], np.float32)
    aimg = frame[50:178, 100:228].copy()
    fake = np.zeros_like(aimg)
    info = {}
    out = _paste_back(aimg, fake, matrix, frame, info=info)
    assert out.shape == frame.shape
    assert np.array_equal(out[:40], frame[:40]) and np.array_equal(out[:, 260:], frame[:, 260:])
    assert out[114, 164].sum() < frame[114, 164].sum() or out[114, 164].sum() == 0
    assert info["mask"].shape == frame.shape[:2] and 0 < info["mask"].max() <= 1
    multiband = _paste_back(aimg, fake, matrix, frame, blend="multiband")
    assert multiband.shape == frame.shape and multiband.dtype == np.uint8


def test_paste_back_region_mask_restricts_blend(rng):
    frame = np.full((300, 400, 3), 200, np.uint8)
    matrix = np.array([[1, 0, -100], [0, 1, -50]], np.float32)
    aimg = frame[50:178, 100:228].copy()
    fake = np.zeros_like(aimg)
    region = np.zeros((128, 128), np.float32)
    region[32:96, 32:96] = 1
    out = _paste_back(aimg, fake, matrix, frame, region_mask=region)
    assert out[114, 164, 0] < 60          # inside region: replaced
    assert out[60, 110, 0] > 180          # outside region: untouched


def test_box_mask_hard_center_soft_edge():
    mask = crop_box_mask(128)
    assert mask[64, 64] == pytest.approx(1.0, abs=1e-5)
    assert mask[2, 2] < .05 and mask[0, 64] < .05
    assert 0.3 < mask[8, 8] < 1.0


def test_laplacian_blend_identity_and_mixing(rng):
    a = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    b = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    ones = np.ones((64, 64), np.float32)
    assert np.abs(laplacian_blend(a, b, ones) - a).max() < 2
    assert np.abs(laplacian_blend(a, b, np.zeros_like(ones)) - b).max() < 2
    half = np.zeros_like(ones)
    half[:, 32:] = 1
    mixed = laplacian_blend(a, b, half)
    assert np.abs(mixed[:, :8] - b[:, :8]).mean() < np.abs(mixed[:, :8] - a[:, :8]).mean()


def test_region_blend_mask_keeps_face_drops_hair():
    labels = np.zeros((128, 128), np.int32)
    labels[20:40, :] = 17          # hair band
    labels[40:120, 24:104] = 1     # skin
    labels[60:70, 40:60] = 4       # eye
    mask = region_blend_mask(labels)
    assert mask[80, 64] == pytest.approx(1.0)
    assert mask[25, 64] < .5
    assert mask[5, 5] == pytest.approx(0.0)


def test_refine_kps_uses_contours_and_rejects_outliers():
    kps = np.array([[40, 40], [88, 40], [64, 64], [44, 92], [84, 92]], np.float32)
    pts = np.zeros((106, 2), np.float32)
    pts[33:43] = [42, 41]
    pts[87:97] = [86, 41]
    pts[52] = [45, 93]
    pts[61] = [500, 500]  # bogus right mouth corner
    refined = refine_kps(kps, pts)
    assert np.allclose(refined[0], [42, 41]) and np.allclose(refined[1], [86, 41])
    assert np.allclose(refined[3], [45, 93])
    assert np.allclose(refined[4], kps[4])  # outlier ignored
    assert np.array_equal(refine_kps(kps, None), kps)
    assert np.array_equal(refine_kps(kps, pts[:5]), kps)


def test_sharpness_orders_blur():
    img = np.zeros((128, 128, 3), np.uint8)
    img[::4] = 255
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    assert sharpness(img) > sharpness(blurred) > 0
    assert sharpness(None) == 0.0


def test_soften_mask_interior_hard():
    mask = np.zeros((100, 100), np.float32)
    mask[40:60, 40:60] = 1
    soft = soften_mask(mask, 3)
    assert soft[50, 50] == 1.0 and soft[5, 5] == 0.0
