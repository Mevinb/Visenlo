"""ONNX session for the CodeFormer face restoration graph."""

from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort

from .utils import CODEFORMER_SIZE, from_onnx, to_onnx_blob


class CodeFormerModel:
    """Holds the CodeFormer ONNX session; loaded once and kept in memory."""

    def __init__(self, model_path: str, providers=None):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Place codeformer.onnx in {os.path.dirname(model_path)}")
        available = ort.get_available_providers()
        self.providers = providers or [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                                       if p in available]
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.weight_name = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name

    def restore(self, bgr_face: np.ndarray, weight: float) -> np.ndarray:
        """Restore a face crop (any size, upscaled to 512) and return a 512 BGR face."""
        if bgr_face.shape[0] != CODEFORMER_SIZE or bgr_face.shape[1] != CODEFORMER_SIZE:
            face = cv2_resize(bgr_face, CODEFORMER_SIZE)
        else:
            face = bgr_face
        blob = to_onnx_blob(face)
        # Weight dtype/layout varies by export: some are float32, this repo's is double scalar [].
        inp1 = self.session.get_inputs()[1]
        is_double = "double" in (inp1.type or "").lower()
        dtype = np.float64 if is_double else np.float32
        w_shape = inp1.shape
        if w_shape and len(w_shape) == 1:
            weight_arr = np.array([float(weight)], dtype=dtype)
        else:
            weight_arr = np.array(float(weight), dtype=dtype)
        outputs = self.session.run([self.output_name], {
            self.input_name: blob,
            self.weight_name: weight_arr,
        })
        return from_onnx(outputs[0])


def cv2_resize(image, size):
    import cv2
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)
