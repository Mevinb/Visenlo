"""Static + unit self-check for ReactorX quality upgrades.

Run with the project venv: .venv/bin/python scripts/selfcheck.py
Uses synthetic data only - no real faces or model weights required
(model adapters are exercised when present in models/).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILURES = []


def check(name, condition, detail=""):
    status = "ok" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def main():
    import cv2

    from reactorx.boost import explode_tiles, implode_tiles
    from reactorx.engine import (
        MASK_NAMES,
        build_face_mask,
        color_match,
        composite_region_mask,
        parse_face,
        quality_score,
        recover_occlusions,
        sharpen_face_region,
        soften_mask,
        weighted_identity,
    )
    from reactorx.pipeline import (
        PipelineConfig,
        ReactorXPipeline,
        allocate_output_path,
        arcface_kps,
        gender_label,
        parse_swapper_spec,
        select_target_record,
        source_gender_vote,
    )

    # --- arcface template matches insightface's estimate_norm destination ---
    try:
        from insightface.utils.face_align import estimate_norm
        for size in (128, 512):
            # If kps == arcface_kps(size), alignment must be the identity map.
            matrix = estimate_norm(arcface_kps(size), size)
            identity = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
            check(f"arcface_kps({size}) reproduces insightface template",
                  bool(np.allclose(matrix, identity, atol=1e-3)),
                  f"matrix={matrix!r}")
    except ImportError:
        check("insightface available", False)

    # --- pixel boost roundtrip ---
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    for factor in (2, 4):
        tiles = implode_tiles(frame, factor)
        back = explode_tiles(tiles, factor)
        check(f"implode/explode roundtrip x{factor}",
              tiles.shape[0] == factor * factor and np.array_equal(back, frame))
        phase_ok = True
        for b in range(factor):
            for d in range(factor):
                tile = tiles[b * factor + d]
                if not np.array_equal(tile, frame[b::factor, d::factor]):
                    phase_ok = False
        check(f"tile phases are strided views x{factor}", phase_ok)

    # --- mask helpers ---
    mask = np.zeros((100, 100), np.float32)
    mask[40:60, 40:60] = 1
    soft = soften_mask(mask, 3)
    check("soften_mask keeps interior hard", soft[50, 50] == 1.0)
    check("soften_mask keeps far exterior zero", soft[5, 5] == 0.0)

    lmk = np.array([[30, 40], [70, 40], [50, 55], [38, 70], [62, 70],
                    [35, 45], [65, 45], [50, 35]], np.float32)
    fm = build_face_mask((20, 20, 80, 90), (120, 120), landmarks=lmk)
    check("landmark face mask covers eye midpoint",
          fm[45, 50] == 1.0 and fm.shape == (120, 120))
    fm_ell = build_face_mask((20, 20, 80, 90), (120, 120), landmarks=None)
    check("ellipse fallback still works", fm_ell.max() == 1.0)

    # --- color match ---
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    m = np.ones((64, 64), np.float32)
    same = color_match(img, img.copy(), m, .8)
    # Identity transfer equals the irreducible uint8-LAB roundtrip of OpenCV.
    baseline = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2LAB),
                            cv2.COLOR_LAB2BGR).astype(np.float32)
    diff_max = int(np.abs(same - baseline).max())
    check("color_match identity matches LAB-roundtrip baseline", diff_max <= 4,
          f"max diff {diff_max}")
    check("color_match returns float32", same.dtype == np.float32)
    check("color_match no-op at strength 0",
          np.array_equal(color_match(img, np.flip(img, 1).copy(), m, 0), img.astype(np.float32)))

    # --- occlusion recovery is bounded by sensitivity ---
    target = np.full((64, 64, 3), 10, np.uint8)
    result = np.full((64, 64, 3), 200, np.uint8)
    fmask = soften_mask(np.pad(np.ones((28, 28), np.float32), 18), 1.0)
    out = recover_occlusions(target, result, fmask, sensitivity=.6)
    check("recover_occlusions returns uint8 frame", out.dtype == np.uint8)

    # --- sharpen sigma decoupling ---
    small = np.full((32, 32, 3), 128, np.uint8)
    small[16, 16] = 255
    s1 = sharpen_face_region(small, np.ones((32, 32), np.float32), .5, sigma=1.0)
    s9 = sharpen_face_region(small, np.ones((32, 32), np.float32), .5, sigma=9.0)
    check("sigma controls sharpen radius",
          not np.array_equal(s1, s9) and s1[15, 16, 0] < s9[15, 16, 0])

    # --- parsing label map fix ---
    rec = SimpleNamespace(bbox=(0, 0, 40, 40), masks=None)
    labels = np.full((40, 40), 17, np.int32)  # hair everywhere
    labels[10:20, 10:20] = 4  # eyes
    parse_face(np.zeros((40, 40, 3), np.uint8), rec, None)
    check("fallback masks cover all names", set(MASK_NAMES) <= set(rec.masks.keys()))

    class _EyeParser:
        def __call__(self, crop):
            return labels

    rec2 = SimpleNamespace(bbox=(0, 0, 40, 40))
    parse_face(np.zeros((40, 40, 3), np.uint8), rec2, _EyeParser())
    check("eyes parsed from CelebAMask id 4", float(rec2.masks["eyes"][12:18, 12:18].max()) == 1.0)
    check("hair parsed from id 17", float(rec2.masks["hair"][0:5, 0:5].max()) == 1.0)
    cm = composite_region_mask(rec2, (40, 40), ("eyes",))
    check("composite_region_mask lifts crop mask to frame",
          cm is not None and cm[12:18, 12:18].max() == 1.0)
    empty = SimpleNamespace(bbox=(0, 0, 40, 40), masks={})
    check("composite_region_mask returns None without planes",
          composite_region_mask(empty, (40, 40), ("skin",)) is None)

    # --- quality score behavior ---
    def make_record(sharp_img, kps):
        face = SimpleNamespace(kps=kps, bbox=None)
        return SimpleNamespace(face=face, bbox=(0, 0, sharp_img.shape[1], sharp_img.shape[0]))

    smooth = np.zeros((128, 128, 3), np.uint8)
    smooth[...] = 128
    textured = np.zeros((128, 128, 3), np.uint8)
    textured[::2, ::2] = 255
    frontal_kps = np.array([[40, 40], [88, 40], [64, 64], [44, 92], [84, 92]], np.float32)
    profile_kps = np.array([[20, 40], [68, 40], [58, 64], [30, 92], [66, 92]], np.float32)
    q_sharp_frontal = quality_score(textured, make_record(textured, frontal_kps))
    q_blur_frontal = quality_score(smooth, make_record(smooth, frontal_kps))
    q_sharp_profile = quality_score(textured, make_record(textured, profile_kps))
    check("quality prefers sharp over blurry", q_sharp_frontal > q_blur_frontal)
    check("quality prefers frontal over profile", q_sharp_frontal > q_sharp_profile)
    check("quality gate (.20 default) accepts sharp frontal", q_sharp_frontal >= .20,
          f"score={q_sharp_frontal:.3f}")

    # --- weighted identity uses quality ---
    e1 = np.zeros(8, np.float32); e1[0] = 1
    e2 = np.zeros(8, np.float32); e2[1] = 1
    r_hi = SimpleNamespace(embedding=e1, quality=.9)
    r_lo = SimpleNamespace(embedding=e2, quality=.1)
    agg = weighted_identity([r_hi, r_lo])
    check("identity aggregation dominated by high-quality ref", agg[0] > agg[1])

    # --- gender-based target selection ---
    from reactorx.engine import FaceRecord

    def gface(gender):
        rec = FaceRecord(face=None, bbox=(0, 0, 10, 10),
                         landmarks=np.zeros((5, 2), np.float32),
                         embedding=None, score=.9)
        rec.gender = gender
        return rec

    f1, f2, m1 = gface(0), gface(0), gface(1)
    u = gface(None)
    plain = FaceRecord(face=None, bbox=(0, 0, 10, 10),
                       landmarks=np.zeros((5, 2), np.float32),
                       embedding=None, score=.9)
    check("FaceRecord gender/age default to None",
          plain.gender is None and plain.age is None)
    check("gender_label maps ids", gender_label(0) == "F" and gender_label(1) == "M"
          and gender_label(None) == "?")
    check("source_gender_vote majority female", source_gender_vote([f1, f2, m1]) == 0)
    check("source_gender_vote majority male", source_gender_vote([f1, m1, gface(1)]) == 1)
    check("source_gender_vote tie falls back to first",
          source_gender_vote([f1, m1]) == 0)
    check("source_gender_vote all-unknown is None", source_gender_vote([u, u]) is None)
    sel = select_target_record([f1, f2, m1], "gender", 0, 0)
    check("gender mode picks leftmost match", sel is f1)
    sel = select_target_record([m1, f1], "gender", 0, 0)
    check("gender mode skips non-matching left faces", sel is f1)
    check("gender mode no-match returns None",
          select_target_record([m1, gface(1)], "gender", 0, 0) is None)
    check("gender mode unknown source gender returns None",
          select_target_record([f1], "gender", 0, None) is None)
    check("index mode still bounds-checks",
          select_target_record([f1, m1], "index", 1, None) is m1)
    try:
        select_target_record([f1], "index", 5, None)
        check("index mode out-of-range raises", False)
    except ValueError:
        check("index mode out-of-range raises", True)

    # --- config defaults ---
    cfg = PipelineConfig()
    check("reference_quality raised to .20", abs(cfg.reference_quality - .20) < 1e-9)
    check("occluder enabled by default", cfg.occluder_enabled)
    check("det_size ceiling configured", cfg.det_size_max >= cfg.det_size)
    check("swap auto-save enabled by default", cfg.save_swaps)

    # --- auto-save output naming (date + zero-padded counter) ---
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        first = Path(allocate_output_path(tmp, stamp="2026-08-26"))
        first.write_bytes(b"x")
        second = Path(allocate_output_path(tmp, stamp="2026-08-26"))
        second.write_bytes(b"x")
        third = Path(allocate_output_path(tmp, stamp="2026-08-26"))
        next_day = Path(allocate_output_path(tmp, stamp="2026-08-27"))
    check("output naming starts at 00", first.name == "2026-08-26_00.png",
          f"got {first.name}")
    check("output naming keeps counting", second.name == "2026-08-26_01.png"
          and third.name == "2026-08-26_02.png",
          f"got {second.name}, {third.name}")
    check("output naming resets per date", next_day.name == "2026-08-27_00.png",
          f"got {next_day.name}")
    check("standard swapper spec keeps model name",
          parse_swapper_spec("inswapper_128.onnx") == ("inswapper_128.onnx", 1))
    check("boost swapper spec resolves installed model",
          parse_swapper_spec("inswapper_128.onnx@512") == ("inswapper_128.onnx", 4))
    check("legacy boost spec gains onnx extension",
          parse_swapper_spec("inswapper_128@256") == ("inswapper_128.onnx", 2))

    # --- pipeline object constructs without loading models ---
    pipe = ReactorXPipeline(str(ROOT / "models"), cfg)
    new_cfg = PipelineConfig(color_strength=.5)
    pipe.update_config(new_cfg)
    check("update_config swaps atomically", pipe.config.color_strength == .5)
    check("outputs directory sits beside the project root",
          Path(pipe.output_dir).resolve() == (ROOT / "outputs").resolve(),
          f"output_dir={pipe.output_dir}")

    # --- model adapters load and run when present ---
    bisenet_path = ROOT / "models" / "bisenet_resnet_34.onnx"
    xseg_path = ROOT / "models" / "xseg_1.onnx"
    codeformer_path = ROOT / "models" / "codeformer.onnx"
    if bisenet_path.is_file():
        from reactorx.parsing import BisenetParser
        parser = BisenetParser(str(bisenet_path))
        out_labels = parser(np.zeros((100, 120, 3), np.uint8))
        check("bisenet parser output shape", out_labels.shape == (100, 120)
              and out_labels.dtype == np.int32 and out_labels.max() < 19)
    else:
        print("[skip] bisenet model not present")
    if xseg_path.is_file():
        from reactorx.parsing import XSegOccluder, aligned_crop_matrix
        occ = XSegOccluder(str(xseg_path))
        ffhq_like = np.array([[.3769, .4686], [.6229, .4691], [.5012, .6133],
                              [.3931, .7254], [.6115, .7249]], np.float32) * 300 \
            + np.array([50, 40], np.float32)
        mat = aligned_crop_matrix(ffhq_like, 256)
        check("RANSAC alignment matrix valid", mat is not None and mat.shape == (2, 3))
        occ_mask = occ.map_to_frame(
            SimpleNamespace(face=SimpleNamespace(kps=ffhq_like)),
            np.zeros((400, 400, 3), np.uint8))
        check("xseg full-frame mask", occ_mask is None or occ_mask.shape == (400, 400))

        class _KeepMaskSession:
            def run(self, *_args, **_kwargs):
                return [np.ones((1, 256, 256, 1), np.float32)]

        polarity_probe = object.__new__(XSegOccluder)
        polarity_probe.size = 256
        polarity_probe.session = _KeepMaskSession()
        polarity_probe.input_name = "input"
        polarity_probe.output_name = "output"
        occ = polarity_probe.detect(np.zeros((256, 256, 3), np.uint8))
        check("xseg face keep mask is inverted to occlusion",
              occ.shape == (256, 256) and float(occ.max()) == 0.0)
    else:
        print("[skip] xseg model not present")
    if codeformer_path.is_file():
        from reactorx.restoration import CodeFormer
        restorer = CodeFormer(str(ROOT / "models"))
        synthetic_face = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
        restored = restorer.restore_aligned(synthetic_face)
        check("codeformer restores aligned crop",
              restored.shape == (512, 512, 3) and restored.dtype == np.uint8,
              f"shape={restored.shape} dtype={restored.dtype}")
    else:
        print("[skip] codeformer model not present")

    # --- app imports ---
    import app  # noqa: F401
    check("app module imports", True)

    import onnxruntime as ort
    print("\nonnxruntime:", ort.__version__, ort.get_available_providers())
    print("opencv:", cv2.__version__)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
