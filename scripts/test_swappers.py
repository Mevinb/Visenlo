"""Contract regressions independent of large model downloads."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visenlo.swappers import MODELS, OnnxSwapper, TEMPLATES, align, missing_assets, model_choices
from visenlo.pipeline import parse_swapper_spec, candidate_boosts_for


class SwapperTests(unittest.TestCase):
    def test_choices_resolve(self):
        for choice in model_choices():
            name, factor = parse_swapper_spec(choice)
            self.assertIn(name, MODELS)
            if "@" in choice:
                self.assertEqual(MODELS[name].size * factor, int(choice.split("@")[1]))
                self.assertEqual(candidate_boosts_for(choice, name, factor, "best_likeness"), (factor,))

    def test_native_resolution(self):
        self.assertEqual(parse_swapper_spec("hyperswap_1a_256.onnx@512")[1], 2)
        self.assertEqual(parse_swapper_spec("simswap_unofficial_512.onnx@1024")[1], 2)
        with self.assertRaises(ValueError):
            parse_swapper_spec("simswap_unofficial_512.onnx@256")
        with self.assertRaises(ValueError):
            parse_swapper_spec("MobileFaceSwap_224.pdparams@256")
        self.assertEqual(candidate_boosts_for("MobileFaceSwap_224.pdparams", "MobileFaceSwap_224.pdparams", 1, "best_likeness"), (1,))
        self.assertEqual(candidate_boosts_for("hyperswap_1a_256.onnx", "hyperswap_1a_256.onnx", 1, "best_likeness"), (1, 2))

    def test_converter_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "ghost_1_256.onnx").touch()
            self.assertEqual(missing_assets(tmp, "ghost_1_256.onnx"), ["crossface_ghost.onnx"])

    def test_alignment(self):
        for name, template in TEMPLATES.items():
            _, matrix = align(np.zeros((256, 256, 3), np.uint8), np.array(template) * 256, name, 256)
            np.testing.assert_allclose(matrix, [[1, 0, 0], [0, 1, 0]], atol=1e-5)

    def test_boost_preserves_spatial_phases(self):
        backend = OnnxSwapper.__new__(OnnxSwapper)
        backend.spec = MODELS["hyperswap_1a_256.onnx"]
        backend.input_size = 256
        backend._source = np.zeros((1, 512))
        backend.dealias = False  # raw polyphase roundtrip must be exact
        backend.infer_crop = lambda tile: tile.copy()
        image = np.random.default_rng(7).integers(0, 256, (512, 512, 3), np.uint8)
        face = SimpleNamespace(kps=np.array(TEMPLATES["arcface_128"]) * 512)
        output, _ = backend.get(image, face, None, paste_back=False, factor=2)
        np.testing.assert_array_equal(output, image)

    def test_tensor_conventions(self):
        for name in ("hyperswap_1a_256.onnx", "ghost_1_256.onnx", "simswap_256.onnx", "uniface_256.onnx"):
            backend = OnnxSwapper.__new__(OnnxSwapper)
            backend.spec = MODELS[name]
            backend._source = np.ones((1, 512), np.float32)
            feed = {}
            def run(_, values):
                feed.update(values)
                return [np.zeros((1, 3, 256, 256), np.float32)]
            backend.session = SimpleNamespace(run=run, get_inputs=lambda: [
                SimpleNamespace(name="source", type="tensor(float16)"),
                SimpleNamespace(name="target", type="tensor(float)")])
            output = backend.infer_crop(np.full((256, 256, 3), 255, np.uint8))
            self.assertEqual(feed["source"].dtype, np.float16)
            self.assertEqual(feed["target"].dtype, np.float32)
            self.assertEqual(output.dtype, np.uint8)
            self.assertEqual(int(output[0, 0, 0]), 0 if name == "simswap_256.onnx" else 127)

    def test_embedding_conditioning(self):
        identity = np.ones(512, np.float32) / np.sqrt(512)
        raw = identity * 20
        for name in ("alphaface_256.onnx", "hyperswap_1a_256.onnx", "ghost_1_256.onnx", "simswap_256.onnx"):
            backend = OnnxSwapper.__new__(OnnxSwapper)
            backend.spec = MODELS[name]
            backend.converter = None
            seen = []
            if backend.spec.converter:
                def run(_, feed):
                    seen.append(feed["input"].copy())
                    return [feed["input"] * 2]
                backend.converter = SimpleNamespace(run=run, get_inputs=lambda: [SimpleNamespace(name="input")])
            backend.prepare_reference(None, SimpleNamespace(kps=np.zeros((5, 2))), identity, raw)
            if backend.spec.converter:
                np.testing.assert_allclose(seen[0], raw[None], rtol=1e-6)
            expected_norm = 20 if backend.spec.family == "alphaface" else 40 if backend.spec.family == "ghost" else 1
            self.assertAlmostEqual(float(np.linalg.norm(backend._source)), expected_norm, places=4)

    def test_selected_model_readiness(self):
        import app
        with tempfile.TemporaryDirectory() as tmp, patch.object(app, "MODELS", Path(tmp)):
            analysis = Path(tmp, "insightface/models/buffalo_l")
            analysis.mkdir(parents=True)
            for name in app.MODEL_SPECS[0][2]:
                (analysis / name).write_bytes(b"present")
            Path(tmp, "hyperswap_1a_256.onnx").write_bytes(b"present")
            self.assertTrue(app._check_models("hyperswap_1a_256.onnx")[1])
            self.assertFalse(app._check_models("ghost_1_256.onnx")[1])
            self.assertFalse(app._check_models()[1])


if __name__ == "__main__":
    unittest.main()
