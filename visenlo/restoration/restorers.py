"""Face restorer registry: CodeFormer, GFPGAN, GPEN and RestoreFormer++ ONNX graphs.

All models follow the FaceFusion export conventions: RGB input in [-1, 1]
(NCHW, square), output in [-1, 1]; CodeFormer additionally takes a scalar
fidelity `weight`. Each model restores an FFHQ-aligned crop of `size` px.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from ..runtime import create_session

BASE_300 = "https://huggingface.co/facefusion/models-3.0.0/resolve/main"


@dataclass(frozen=True)
class RestorerSpec:
    key: str
    label: str
    filename: str
    size: int = 512
    has_weight: bool = False
    url: str = ""
    license: str = ""
    note: str = ""


RESTORERS = {
    "codeformer": RestorerSpec("codeformer", "CodeFormer", "codeformer.onnx", 512, True,
                               f"{BASE_300}/codeformer.onnx", "CC BY-NC 4.0",
                               "Strongest cleanup; can beautify away identity (guarded)."),
    "gfpgan_1.4": RestorerSpec("gfpgan_1.4", "GFPGAN 1.4", "gfpgan_1.4.onnx", 512, False,
                               f"{BASE_300}/gfpgan_1.4.onnx", "Apache-2.0",
                               "Balanced; keeps identity better than CodeFormer."),
    "gpen_bfr_512": RestorerSpec("gpen_bfr_512", "GPEN BFR 512", "gpen_bfr_512.onnx", 512, False,
                                 f"{BASE_300}/gpen_bfr_512.onnx", "Apache-2.0",
                                 "Most faithful to the swap; mild texture recovery."),
    "restoreformer_plus_plus": RestorerSpec("restoreformer_plus_plus", "RestoreFormer++",
                                            "restoreformer_plus_plus.onnx", 512, False,
                                            f"{BASE_300}/restoreformer_plus_plus.onnx", "Apache-2.0",
                                            "Transformer prior; good on heavy blur."),
}
DEFAULT_RESTORER = "codeformer"


def restorer_path(models_path: str, key: str):
    spec = RESTORERS[key]
    candidates = [os.path.join(models_path, spec.filename),
                  os.path.join(models_path, "restoration", spec.filename)]
    return next((c for c in candidates if os.path.isfile(c) and os.path.getsize(c) > 0), None)


def available_restorers(models_path: str) -> list[str]:
    return [key for key in RESTORERS if restorer_path(models_path, key) is not None]


class FaceRestorer:
    """Lazily loaded single-model restorer; one instance per model key."""

    def __init__(self, models_path: str, key: str = DEFAULT_RESTORER, providers=None, weight: float = .8):
        if key not in RESTORERS:
            raise KeyError(f"Unknown restorer '{key}'; choose from {sorted(RESTORERS)}")
        self.spec = RESTORERS[key]
        self.models_path = models_path
        self.providers = providers
        self.weight = float(weight)
        self._session = None
        self._inputs = None
        self._output = None

    @property
    def key(self):
        return self.spec.key

    @property
    def size(self):
        return self.spec.size

    def _ensure(self):
        if self._session is not None:
            return
        path = restorer_path(self.models_path, self.spec.key)
        if path is None:
            raise FileNotFoundError(
                f"Place {self.spec.filename} in {self.models_path} to use {self.spec.label} restoration")
        self._session = create_session(path, self.providers)
        self._inputs = self._session.get_inputs()
        self._output = self._session.get_outputs()[0].name

    def restore_aligned(self, bgr_face: np.ndarray, weight: float | None = None) -> np.ndarray:
        """Restore an FFHQ-aligned BGR crop; returns a `size` px BGR uint8 face."""
        self._ensure()
        size = self.spec.size
        if bgr_face.shape[0] != size or bgr_face.shape[1] != size:
            bgr_face = cv2.resize(bgr_face, (size, size), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (rgb / 127.5 - 1.0)[None].transpose(0, 3, 1, 2)
        feed = {self._inputs[0].name: np.ascontiguousarray(blob)}
        if len(self._inputs) > 1:
            extra = self._inputs[1]
            value = float(self.weight if weight is None else weight)
            dtype = np.float64 if "double" in (extra.type or "").lower() else np.float32
            feed[extra.name] = (np.array([value], dtype=dtype) if extra.shape and len(extra.shape) == 1
                                else np.array(value, dtype=dtype))
        output = self._session.run([self._output], feed)[0]
        restored = np.clip((output[0].transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
