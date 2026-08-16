"""CodeFormer restoration facade for the ReactorX pipeline."""

from __future__ import annotations

import os

import numpy as np

from .model import CodeFormerModel
from .utils import paste_restored_back


class CodeFormer:
    """Single-instance face restorer. Loaded lazily once and reused across swaps."""

    def __init__(self, models_path: str, providers=None, weight: float = 0.8):
        self.models_path = models_path
        self.providers = providers
        self.weight = float(weight)
        self._model = None
        self._model_path = None

    def _ensure_model(self):
        if self._model is not None:
            return
        candidates = [os.path.join(self.models_path, "codeformer.onnx"),
                      os.path.join(self.models_path, "restoration", "codeformer.onnx")]
        path = next((candidate for candidate in candidates if os.path.isfile(candidate)), None)
        if path is None:
            raise FileNotFoundError(
                f"Place codeformer.onnx in {self.models_path} to use face restoration")
        self._model = CodeFormerModel(path, self.providers)
        self._model_path = path

    def restore_aligned(self, swapped_crop: np.ndarray) -> np.ndarray:
        """Restore an aligned swapped face crop. Returns a 512x512 BGR restored face."""
        self._ensure_model()
        return self._model.restore(swapped_crop, self.weight)

    def enhance(self, swapped_crop: np.ndarray, swap_matrix: np.ndarray,
                swap_size: int, target: np.ndarray) -> np.ndarray:
        """Restore the aligned swapped crop and paste it back over the target."""
        restored = self.restore_aligned(swapped_crop)
        return paste_restored_back(restored, swap_matrix, swap_size, target)
