"""CodeFormer restoration facade (kept for backwards compatibility).

New code should use `FaceRestorer(models_path, key)` from `.restorers`.
"""

from __future__ import annotations

import numpy as np

from .restorers import FaceRestorer
from .utils import paste_restored_back


class CodeFormer(FaceRestorer):
    """Single-instance CodeFormer restorer. Loaded lazily once and reused across swaps."""

    def __init__(self, models_path: str, providers=None, weight: float = 0.8):
        super().__init__(models_path, "codeformer", providers, weight)

    def enhance(self, swapped_crop: np.ndarray, swap_matrix: np.ndarray,
                swap_size: int, target: np.ndarray, landmarks=None) -> np.ndarray:
        """Restore the aligned swapped crop and paste it back over the target."""
        restored = self.restore_aligned(swapped_crop)
        return paste_restored_back(restored, swap_matrix, swap_size, target, landmarks)
