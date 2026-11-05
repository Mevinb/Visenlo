"""Local swap-model contracts; model-specific tensor conventions stay here.

Conventions verified against FaceFusion's face_swapper/core.py (2026-09-21).
No FaceFusion runtime or finishing pipeline is imported.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .boost import (_paste_back, dealias_boost, explode_tiles, implode_tiles, inswapper_latent,
                    run_inswapper_tiles)
from .runtime import create_session, uses_cuda


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    size: int = 256
    release: str = "3.0.0"
    template: str = "arcface_128"
    license: str = "Non-Commercial"
    converter: str = ""

    @property
    def filename(self):
        if self.family == "mobilefaceswap":
            return self.name + ".pdparams"
        return self.name + (".h5" if self.family == "facedancer" else ".onnx")

    @property
    def assets(self):
        if self.family == "mobilefaceswap":
            return []
        if self.family == "facedancer":
            return [(self.filename, "felixrosberg/FaceDancer"), ("ArcFace-Res50.h5", "felixrosberg/ArcFace")]
        assets = [(self.filename, self.release)] if self.family != "reswapper" else []
        if self.converter:
            assets.append((self.converter + ".onnx", "3.4.0"))
        return assets


_specs = [
    ModelSpec("inswapper_128", "inswapper", 128),
    ModelSpec("reswapper_256", "reswapper"),
    ModelSpec("inswapper_128_fp16", "inswapper", 128),
    ModelSpec("alphaface_256", "alphaface", release="3.9.0"),
    ModelSpec("blendswap_256", "blendswap", template="ffhq_512"),
    *[ModelSpec(f"ghost_{i}_256", "ghost", template="arcface_112_v1",
                license="Apache-2.0", converter="crossface_ghost") for i in (1, 2, 3)],
    ModelSpec("hififace_unofficial_256", "hififace", release="3.1.0",
              template="mtcnn_512", license="Unknown", converter="crossface_hififace"),
    *[ModelSpec(f"hyperswap_1{i}_256", "hyperswap", release="3.3.0",
                license="ResearchRAIL") for i in "abc"],
    ModelSpec("simswap_256", "simswap", template="arcface_112_v1", converter="crossface_simswap"),
    ModelSpec("simswap_unofficial_512", "simswap", 512, template="arcface_112_v1",
              converter="crossface_simswap"),
    ModelSpec("uniface_256", "uniface", template="ffhq_512", license="Unknown"),
    ModelSpec("MobileFaceSwap_224", "mobilefaceswap", 224, template="arcface_112_v2", license="See upstream license"),
    *[ModelSpec("FaceDancer_config_" + variant, "facedancer", template="arcface_112_v2",
                license="CC-BY-NC-SA-4.0") for variant in
      ("A", "B", "B_tuned", "C", "C_tuned", "D", "D_tuned", "E", "c_HQ")],
]
MODELS = {spec.filename: spec for spec in _specs}


BOOST_SIZES = (256, 512, 1024, 2048)


def model_choices(expert=None):
    """Dropdown entries. The 256-tile `@2048` boost is hidden unless
    VISENLO_EXPERT=1 (or expert=True): its gain over @1024 is marginal and it
    costs minutes on CPU."""
    import os
    if expert is None:
        expert = os.environ.get("VISENLO_EXPERT", "") == "1"
    sizes = BOOST_SIZES if expert else BOOST_SIZES[:-1]
    choices = []
    for spec in _specs:
        choices.append(spec.filename)
        if spec.family != "reswapper":
            choices.extend(f"{spec.filename}@{size}" for size in sizes
                           if size > spec.size and size % spec.size == 0)
    return choices


def missing_assets(root, filename):
    spec = MODELS[filename]
    names = [filename] + ([spec.converter + ".onnx"] if spec.converter else [])
    if spec.family == "facedancer":
        names += ["ArcFace-Res50.h5"]
    if spec.family == "mobilefaceswap":
        names += ["arcface.pdparams"]
    return [name for name in names if not (Path(root) / name).is_file()
            and not (Path(root) / "insightface/models" / name).is_file()]


def runtime_ready(filename):
    if MODELS[filename].family == "mobilefaceswap":
        root = Path(__file__).resolve().parents[1]
        return ((root / ".venvs/mobilefaceswap/bin/python").is_file()
                and (root / "models/runtime-source/MobileFaceSwap/models/model.py").is_file())
    if MODELS[filename].family == "facedancer":
        root = Path(__file__).resolve().parents[1]
        return ((root / ".venvs/facedancer/bin/python").is_file()
                and (root / "models/runtime-source/FaceDancer/networks/layers.py").is_file())
    return True


TEMPLATES = {
    "arcface_112_v1": [(0.35473214, .45658929), (.64526786, .45658929), (.5, .61154464), (.37913393, .776875), (.62086607, .776875)],
    "arcface_112_v2": [(.34191607, .46157411), (.65653393, .45983393), (.500225, .64050536), (.37097589, .82469196), (.63151696, .82325089)],
    "arcface_128": [(.36167656, .40387734), (.63696719, .40235469), (.50019687, .56044219), (.38710391, .72160547), (.61507734, .72034453)],
    "ffhq_512": [(.37691676, .46864664), (.62285697, .46912813), (.50123859, .61331904), (.39308822, .725411), (.61150205, .72490465)],
    "mtcnn_512": [(.36562865, .46733799), (.63305391, .46585885), (.50019127, .61942959), (.39032951, .77598822), (.61178945, .77476328)],
}


def align(image, landmarks, template, size):
    points = np.asarray(landmarks, np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Swapper requires five finite face landmarks")
    matrix, _ = cv2.estimateAffinePartial2D(points, np.asarray(TEMPLATES[template], np.float32) * size,
                                          method=cv2.RANSAC, ransacReprojThreshold=100)
    if matrix is None:
        raise ValueError("Cannot align selected face")
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REPLICATE), matrix


class OnnxSwapper:
    """One session per model, source conditioning cached per swap request."""
    def __init__(self, path, providers, asset_root=None):
        self.spec = MODELS[Path(path).name]
        self.input_size = self.spec.size
        self.session = create_session(path, providers)
        self.converter = None
        if self.spec.converter:
            converter_path = Path(path).parent / (self.spec.converter + ".onnx")
            if not converter_path.is_file() and asset_root:
                converter_path = Path(asset_root) / (self.spec.converter + ".onnx")
            self.converter = create_session(converter_path, providers)
        self._source = None
        self._source_image = None
        self._source_kps = None
        self.dealias = False
        if {item.name for item in self.session.get_inputs()} != {"source", "target"}:
            raise ValueError(f"Unexpected inputs in {path}")

    def prepare_reference(self, image, face, identity, raw_embedding):
        self._source_image, self._source_kps = image, face.kps
        family = self.spec.family
        if family in ("blendswap", "uniface"):
            template, size = ("arcface_112_v2", 112) if family == "blendswap" else ("ffhq_512", 256)
            crop, _ = align(image, face.kps, template, size)
            value = crop[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
        elif family == "hyperswap":
            value = np.asarray(identity, np.float32).reshape(1, -1)
        else:
            value = np.asarray(raw_embedding, np.float32).reshape(1, -1)
            if self.converter:
                value = self.converter.run(None, {self.converter.get_inputs()[0].name: value})[0]
                if family != "ghost":
                    value = value / max(float(np.linalg.norm(value)), 1e-8)
        self._source = np.ascontiguousarray(value)

    def infer_crop(self, crop):
        family = self.spec.family
        value = crop[:, :, ::-1].astype(np.float32) / 255
        if family in ("ghost", "hififace", "hyperswap", "uniface"):
            value = (value - .5) / .5
        elif self.spec.name == "simswap_256":
            value = (value - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        feed = {"target": value.transpose(2, 0, 1)[None], "source": self._source}
        for item in self.session.get_inputs():
            feed[item.name] = np.ascontiguousarray(feed[item.name], dtype=np.float16 if item.type == "tensor(float16)" else np.float32)
        output = self.session.run(None, feed)[0][0].transpose(1, 2, 0)
        if output.shape != crop.shape or not np.isfinite(output).all():
            raise RuntimeError(f"Invalid output from {self.spec.name}: {output.shape}")
        if family in ("ghost", "hififace", "hyperswap", "uniface"):
            output = output * .5 + .5
        return (np.clip(output, 0, 1)[:, :, ::-1] * 255).astype(np.uint8)

    def get(self, image, target_face, source_face, paste_back=True, factor=1, **paste_kwargs):
        if self._source is None:
            raise RuntimeError("Source reference has not been prepared")
        crop, matrix = align(image, target_face.kps, self.spec.template, self.input_size * factor)
        outputs = [self.infer_crop(tile) for tile in implode_tiles(crop, factor)]
        fake = explode_tiles(np.stack(outputs), factor)
        if getattr(self, "dealias", False):
            fake = dealias_boost(fake, factor)
        return _paste_back(crop, fake, matrix, image, **paste_kwargs) if paste_back else (fake, matrix)


def _load_emap(model_path):
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(model_path, load_external_data=False).graph
    init = next((item for item in graph.initializer if item.name == "emap"), None)
    if init is None:
        init = graph.initializer[-1]
    return numpy_helper.to_array(init).astype(np.float32)


class InswapperBackend:
    """inswapper_128 with pixel boost and an optional FP16 twin.

    `precision` is "auto" (use inswapper_128_fp16.onnx when it exists and the
    session runs on CUDA — same weights, ~2x faster tiles), "fp32" or "fp16".
    """

    def __init__(self, path, providers, precision="auto"):
        from insightface.utils import face_align

        self.face_align = face_align
        self.path = Path(path)
        self.spec = MODELS.get(self.path.name, MODELS["inswapper_128.onnx"])
        self.emap = _load_emap(str(self.path))
        self.dealias = False
        fp16 = self.path.with_name("inswapper_128_fp16.onnx")
        chosen = self.path
        if precision == "fp16" or (precision == "auto" and uses_cuda(providers)):
            if fp16.is_file() and self.path.name == "inswapper_128.onnx":
                chosen = fp16
        self.session = create_session(str(chosen), providers)
        if chosen != self.path and not uses_cuda(self.session.get_providers()):
            # FP16 graphs are slow on CPU; fall back to the fp32 file.
            chosen = self.path
            self.session = create_session(str(chosen), providers)
        self.active_path = chosen
        self.precision = "fp16" if chosen != self.path or "fp16" in chosen.name else "fp32"
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]
        shape = self.session.get_inputs()[0].shape
        spatial = [v for v in shape[1:] if isinstance(v, int) and v > 0 and v != 3]
        self.input_size = (spatial[0], spatial[0]) if spatial else (128, 128)

    @property
    def size(self):
        return int(self.input_size[0])

    def swap_crop(self, aimg, source_face, factor=1):
        tiles = implode_tiles(aimg, factor) if factor > 1 else aimg[None]
        fakes = run_inswapper_tiles(self.session, self.input_names, self.output_names[0],
                                    tiles, inswapper_latent(self, source_face))
        if factor <= 1:
            return fakes[0]
        fake = explode_tiles(fakes, factor)
        return dealias_boost(fake, factor) if self.dealias else fake

    def get(self, image, target_face, source_face, paste_back=True, factor=1, **paste_kwargs):
        big = self.size * int(factor)
        aimg, matrix = self.face_align.norm_crop2(image, target_face.kps, big)
        fake = self.swap_crop(aimg, source_face, int(factor))
        if not paste_back:
            return fake, matrix
        return _paste_back(aimg, fake, matrix, image, **paste_kwargs)


class Reswapper256:
    """Adapter for the two-input 256px Reswapper ONNX graph."""

    def __init__(self, model_path, providers):
        from insightface.utils import face_align

        self.face_align = face_align
        self.path = Path(model_path)
        self.spec = MODELS["reswapper_256.onnx"]
        self.session = create_session(model_path, providers)
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name
        shape = self.session.get_inputs()[0].shape
        spatial = [v for v in shape[1:] if isinstance(v, int) and v > 0 and v != 3] if len(shape) == 4 else []
        self.input_size = int(spatial[0]) if spatial else 256
        self.emap = _load_emap(model_path)

    def get(self, image, target_face, source_face, paste_back=True, factor=1, **paste_kwargs):
        if factor != 1:
            raise ValueError("Pixel-boost (@256/@512/…) is not supported for reswapper_256.onnx")
        size = self.input_size
        aligned, matrix = self.face_align.norm_crop2(image, target_face.kps, size)
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 255.0, (size, size), (0, 0, 0), swapRB=True)
        latent = inswapper_latent(self, source_face)
        output = self.session.run([self.output_name], {self.input_names[0]: blob,
                                                       self.input_names[1]: latent})[0]
        fake = np.clip(output.transpose(0, 2, 3, 1)[0] * 255, 0, 255).astype(np.uint8)[:, :, ::-1]
        if not paste_back:
            return fake, matrix
        return _paste_back(aligned, fake, matrix, image, **paste_kwargs)


class FaceDancerSwapper(OnnxSwapper):
    """TensorFlow lives only in a child process, receiving aligned crops locally."""
    def __init__(self, path, providers):
        self.path = Path(path)
        self.spec = MODELS[self.path.name]
        self.input_size = 256
        self._source = None
        root = Path(__file__).resolve().parents[1]
        self.python = root / ".venvs/facedancer/bin/python"
        self.worker = root / "scripts/facedancer_worker.py"
        if not runtime_ready(self.spec.filename):
            raise RuntimeError("Install optional runtime: .venv/bin/python scripts/setup_facedancer.py")

    def prepare_reference(self, image, face, identity, raw_embedding):
        self._source, _ = align(image, face.kps, "arcface_112_v2", 112)

    def get(self, image, target_face, source_face, paste_back=True, factor=1, **paste_kwargs):
        import os
        import subprocess
        import tempfile
        crop, matrix = align(image, target_face.kps, self.spec.template, self.input_size * factor)
        with tempfile.TemporaryDirectory(prefix="visenlo-facedancer-") as temp:
            request = Path(temp) / "input.npz"
            response = Path(temp) / "output.npy"
            np.savez(request, source=self._source, tiles=implode_tiles(crop, factor))
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="2",
                               TF_NUM_INTRAOP_THREADS="4", TF_NUM_INTEROP_THREADS="1")
            subprocess.run([str(self.python), str(self.worker), str(self.path), str(request), str(response)],
                           check=True, timeout=1800, env=environment)
            tiles = np.load(response, allow_pickle=False)
        if tiles.shape != (factor * factor, 256, 256, 3) or tiles.dtype != np.uint8:
            raise RuntimeError("Invalid FaceDancer output")
        fake = explode_tiles(tiles, factor)
        return _paste_back(crop, fake, matrix, image, **paste_kwargs) if paste_back else (fake, matrix)


class MobileFaceSwapper(FaceDancerSwapper):
    def __init__(self, path, providers):
        self.path = Path(path)
        self.spec = MODELS[self.path.name]
        self.input_size = 224
        self._source = None
        root = Path(__file__).resolve().parents[1]
        self.python = root / ".venvs/mobilefaceswap/bin/python"
        self.worker = root / "scripts/mobilefaceswap_worker.py"
        if not runtime_ready(self.spec.filename):
            raise RuntimeError("Install optional runtime: .venv/bin/python scripts/setup_mobilefaceswap.py")

    @staticmethod
    def _align(image, landmarks, size):
        # Match the original Paddle implementation's least-squares affine warp.
        points = np.asarray(landmarks, np.float64)
        destination = np.asarray(TEMPLATES["arcface_112_v2"]) * size
        matrix = np.linalg.lstsq(np.c_[points, np.ones(5)], destination, rcond=None)[0].T.astype(np.float32)
        return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR), matrix

    def prepare_reference(self, image, face, identity, raw_embedding):
        # Upstream aligns source to 224, then resizes to the 112px recognizer.
        crop, _ = self._align(image, face.kps, 224)
        self._source = cv2.resize(crop, (112, 112))

    def get(self, image, target_face, source_face, paste_back=True, factor=1, **paste_kwargs):
        import subprocess
        import tempfile
        if factor != 1:
            raise ValueError("MobileFaceSwap supports native 224px only; existing boost sizes are not multiples of 224")
        crop, matrix = self._align(image, target_face.kps, 224)
        with tempfile.TemporaryDirectory(prefix="visenlo-mobilefaceswap-") as temp:
            request, response = Path(temp) / "input.npz", Path(temp) / "output.npy"
            np.savez(request, source=self._source, target=crop)
            subprocess.run([str(self.python), str(self.worker), str(self.path), str(request), str(response)],
                           check=True, timeout=300)
            fake = np.load(response, allow_pickle=False)
        if fake.shape != crop.shape or fake.dtype != np.uint8:
            raise RuntimeError("Invalid MobileFaceSwap output")
        return _paste_back(crop, fake, matrix, image, **paste_kwargs) if paste_back else (fake, matrix)
