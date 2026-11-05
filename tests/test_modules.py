"""Identity bank, presets, video helpers, CLI parsing and restorer registry."""
from types import SimpleNamespace

import numpy as np
import pytest

from visenlo.engine import FaceRecord
from visenlo.identity import IdentityBank
from visenlo.pipeline import ReferenceSet
from visenlo.presets import delete_preset, list_presets, load_preset, save_preset
from visenlo.restoration import RESTORERS, available_restorers, restorer_path
from visenlo.video import VideoProgress, smooth_kps


def _refset(label="Maya"):
    identity = np.zeros(512, np.float32)
    identity[3] = 1
    face = SimpleNamespace(embedding=identity * 20, normed_embedding=identity, kps=np.zeros((5, 2), np.float32),
                           gender=0, age=31)
    record = FaceRecord(face, (10, 10, 60, 60), np.zeros((106, 2), np.float32), identity, .9, quality=.8)
    record.gender, record.age = 0, 31
    image = np.full((100, 100, 3), 90, np.uint8)
    return ReferenceSet(identity, [record], [image], label)


def test_identity_bank_roundtrip(tmp_path):
    bank = IdentityBank(tmp_path / "ids")
    assert bank.names() == []
    saved = bank.save("Maya Test", _refset())
    assert bank.names() == ["Maya_Test"]
    assert saved.gender_label == "F" and saved.age == 31 and saved.thumbnail.shape == (160, 160, 3)
    assert saved.raw_magnitude == pytest.approx(20.0)
    refset = bank.as_reference_set("Maya_Test")
    assert np.allclose(refset.identity, saved.identity)
    assert refset.records[0].face.embedding[3] == pytest.approx(20.0)
    assert refset.images == [] and refset.label == "Maya Test"
    assert bank.delete("Maya_Test") and not bank.delete("Maya_Test")
    with pytest.raises(FileNotFoundError):
        bank.load("Maya_Test")


def test_presets_roundtrip(tmp_config_dir):
    assert list_presets() == []
    name = save_preset("My Portraits!", {"quality_mode": "Manual", "sharpen_strength": .3})
    assert name == "My_Portraits" and list_presets() == [name]
    assert load_preset(name)["sharpen_strength"] == .3
    assert delete_preset(name) and list_presets() == []
    with pytest.raises(FileNotFoundError):
        load_preset("missing")


def test_smooth_kps_filters_small_motion_and_passes_jumps():
    prev = np.array([[40, 40], [88, 40], [64, 64], [44, 92], [84, 92]], np.float32)
    cur = prev + 2
    smoothed = smooth_kps(prev, cur, alpha=.5)
    assert np.allclose(smoothed, prev + 1)
    jump = prev + 200
    assert np.array_equal(smooth_kps(prev, jump, alpha=.5), jump)
    assert np.array_equal(smooth_kps(None, cur), cur)
    assert VideoProgress(5, 10, 3, 1.0).fraction == .5


def test_restorer_registry(tmp_path):
    assert set(RESTORERS) >= {"codeformer", "gfpgan_1.4", "gpen_bfr_512", "restoreformer_plus_plus"}
    assert RESTORERS["codeformer"].has_weight and not RESTORERS["gfpgan_1.4"].has_weight
    assert available_restorers(str(tmp_path)) == []
    (tmp_path / "restoration").mkdir()
    (tmp_path / "restoration" / "codeformer.onnx").write_bytes(b"x")
    assert restorer_path(str(tmp_path), "codeformer").endswith("restoration/codeformer.onnx")
    assert available_restorers(str(tmp_path)) == ["codeformer"]


def test_cli_parses_commands(monkeypatch):
    from visenlo import cli
    calls = {}
    monkeypatch.setattr(cli, "cmd_swap", lambda args: calls.__setitem__("swap", args) or 0)
    monkeypatch.setattr(cli, "cmd_video", lambda args: calls.__setitem__("video", args) or 0)
    assert cli.main(["swap", "--ref", "a.jpg", "--target", "t.jpg", "--model", "inswapper_128.onnx@512",
                     "--match", "all", "--restorer", "gfpgan_1.4"]) == 0
    assert calls["swap"].model == "inswapper_128.onnx@512" and calls["swap"].match == "all"
    assert cli.main(["video", "--identity", "maya", "--input", "in.mp4", "--every", "2"]) == 0
    assert calls["video"].every == 2
    with pytest.raises(SystemExit):
        cli.main(["identity", "save"])  # missing name


def test_model_choices_hide_2048_unless_expert(monkeypatch):
    from visenlo.swappers import model_choices
    monkeypatch.delenv("VISENLO_EXPERT", raising=False)
    assert "inswapper_128.onnx@1024" in model_choices() and "inswapper_128.onnx@2048" not in model_choices()
    assert "inswapper_128.onnx@2048" in model_choices(expert=True)
