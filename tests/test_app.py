"""App boundary tests (no inference, no network)."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

import app


def test_readiness_requires_analysis_but_not_optional_finishers(tmp_path):
    with patch.object(app, "MODELS", tmp_path):
        (tmp_path / "inswapper_128.onnx").write_bytes(b"present")
        assert not app._check_models()[1]
        analysis = tmp_path / "insightface/models/buffalo_l"
        analysis.mkdir(parents=True)
        for name in app.MODEL_SPECS[0][2]:
            (analysis / name).write_bytes(b"present")
        assert app._check_models()[1]
        (tmp_path / "inswapper_128.onnx").write_bytes(b"")
        assert not app._check_models()[1]
    assert not app._check_models("not-a-model")[1]
    assert not app._check_models(None)[1]


def test_orientation_and_paths():
    image = Image.new("RGB", (30, 10), "red")
    image.getexif()[274] = 6
    assert app._bgr(image).shape == (30, 10, 3)
    assert app._target_path(({"path": "/tmp/image.png"}, "caption")) == "/tmp/image.png"
    assert app._target_path([]) == ""
    assert app._upload_path(Path("/tmp/image.png")) == "/tmp/image.png"
    assert app._paths([("/a.png", None), {"path": "/b.png"}, ""]) == ["/a.png", "/b.png"]


def test_settings_to_config_maps_labels():
    cfg = app.settings_to_config({**app.SETTING_DEFAULTS, "restorer": "GFPGAN 1.4 (not installed)",
                                  "quality_mode": "Manual", "eye_mode": "Keep target eyes / gaze",
                                  "mask_mode": "Classic box", "precision": "FP32 only"})
    assert cfg.restorer == "gfpgan_1.4" and cfg.codeformer_enabled
    assert cfg.quality_mode == "manual" and cfg.eye_mode == "keep_target_eyes"
    assert cfg.mask_mode == "box" and cfg.inswapper_precision == "fp32"
    assert not app.settings_to_config({**app.SETTING_DEFAULTS, "restorer": "Off"}).codeformer_enabled


def test_batch_reports_failed_input_and_keeps_later_success(tmp_path, tmp_config_dir):
    class Output:
        def __init__(self, image):
            self.image, self.status, self.saved_path, self.debug = image, "Visenlo complete", None, {}

    class Pipeline:
        cancel_event = __import__("threading").Event()

        def analyze_references(self, images, idx, label=""):
            return object()

        def swap(self, target, assignments, model):
            return Output(target.copy())

    good, bad = tmp_path / "valid.png", tmp_path / "broken.png"
    Image.new("RGB", (32, 32), "red").save(good)
    bad.write_text("not an image")
    settings = [app.SETTING_DEFAULTS[k] for k in app.SETTING_KEYS]
    with patch.object(app, "_check_models", return_value=("", True)), patch.object(app, "get_pipeline", return_value=Pipeline()):
        updates = list(app.run_images([str(bad), str(good)], [str(good)], None, None, None, "(none)", "(none)", "(none)", "(none)",
                                      None, "Single face (index)", 0, 0, *(["Keep"] * app.MAX_MAP_FACES), *settings))
    assert updates[0][0] == []
    gallery, report = updates[-1][0], updates[-1][1]
    assert len(gallery) == 1
    assert "1 of 2 images completed · 1 failed" in report
    assert "broken.png: FAILED" in report and "valid.png: Visenlo complete" in report
    assert str(tmp_path) not in report
    assert len(updates[-1][2]) == 1  # before/after pair recorded


def test_missing_input_is_actionable():
    settings = [app.SETTING_DEFAULTS[k] for k in app.SETTING_KEYS]
    with pytest.raises(Exception, match="Select at least one target image"):
        list(app.run_images([], None, None, None, None, "(none)", "(none)", "(none)", "(none)", None,
                            "Single face (index)", 0, 0, *(["Keep"] * app.MAX_MAP_FACES), *settings))


def test_build_ui_and_png_gallery():
    with patch.object(app, "_check_models", return_value=("", True)):
        ui = app.build_ui()
    gallery = next(c for c in ui.blocks.values() if isinstance(c, app.gr.Gallery) and c.label == "Swapped images")
    pixels = np.random.default_rng(7).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    data = gallery.postprocess([Image.fromarray(pixels)])
    with Image.open(data.root[0].image.path) as downloaded:
        assert downloaded.format == "PNG" and np.array_equal(np.asarray(downloaded), pixels)
