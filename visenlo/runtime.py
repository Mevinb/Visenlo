"""Shared ONNX execution settings for predictable local CPU/GPU operation.

Environment knobs (all optional):
- VISENLO_DEVICE=cpu        force CPU even when a CUDA build is installed
- VISENLO_THREADS=N         intra-op threads per session (default: min(8, cores))
- VISENLO_TRT=1             try TensorRT EP (with an engine cache under models/)
                            before CUDA; falls back silently when unavailable
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from functools import lru_cache

logger = logging.getLogger("visenlo")


def thread_budget() -> int:
    raw = os.environ.get("VISENLO_THREADS", "").strip()
    cores = os.cpu_count() or 1
    if raw.isdigit() and int(raw) > 0:
        return min(int(raw), cores)
    # Several sessions (detector, parser, swapper, restorer) run back to back;
    # a moderate per-session pool keeps the desktop responsive on CPU boxes
    # while still using most of a modern core count.
    return max(1, min(8, cores))


def session_options():
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = thread_budget()
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def _cuda_driver_ok() -> bool:
    try:
        driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        return driver.cuInit(0) == 0
    except (OSError, AttributeError):
        return False


def _trt_cache_dir() -> str:
    root = os.environ.get("VISENLO_MODELS", os.environ.get("REACTORX_MODELS", "models"))
    path = os.path.join(root, "trt_cache")
    os.makedirs(path, exist_ok=True)
    return path


@lru_cache(maxsize=1)
def preferred_providers():
    import onnxruntime as ort

    if os.environ.get("VISENLO_DEVICE", "").lower() == "cpu":
        return ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        return ["CPUExecutionProvider"]
    # An installed GPU wheel advertises CUDA even without a working driver.
    # Check the driver before loading every large model and emitting errors.
    if not _cuda_driver_ok():
        return ["CPUExecutionProvider"]
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as exc:  # pragma: no cover - platform specific
            logger.debug("preload_dlls failed: %s", exc)
    providers = []
    if os.environ.get("VISENLO_TRT", "") == "1" and "TensorrtExecutionProvider" in available:
        providers.append(("TensorrtExecutionProvider", {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": _trt_cache_dir(),
            "trt_fp16_enable": True,
            "trt_timing_cache_enable": True,
        }))
    providers.append(("CUDAExecutionProvider", {
        "cudnn_conv_algo_search": "HEURISTIC",
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_use_max_workspace": "1",
    }))
    providers.append("CPUExecutionProvider")
    return providers


def provider_names(providers) -> list[str]:
    return [p[0] if isinstance(p, tuple) else p for p in (providers or [])]


def uses_cuda(providers) -> bool:
    names = provider_names(providers)
    return "CUDAExecutionProvider" in names or "TensorrtExecutionProvider" in names


def create_session(path, providers=None):
    import onnxruntime as ort

    requested = list(providers or preferred_providers())
    attempts = [requested]
    names = provider_names(requested)
    if "TensorrtExecutionProvider" in names:
        attempts.append([p for p in requested if provider_names([p])[0] != "TensorrtExecutionProvider"])
    if uses_cuda(requested):
        attempts.append(["CPUExecutionProvider"])
    last = None
    for index, attempt in enumerate(attempts):
        try:
            return ort.InferenceSession(str(path), sess_options=session_options(), providers=attempt)
        except Exception as exc:
            last = exc
            message = str(exc).lower()
            if not any(word in message for word in ("cuda", "cublas", "cudnn", "provider", "tensorrt", "trt")):
                raise
            if index + 1 < len(attempts):
                logger.warning("%s unavailable for %s; retrying with %s: %s",
                               provider_names(attempt)[0], os.path.basename(str(path)),
                               provider_names(attempts[index + 1])[0], str(exc).splitlines()[0][:160])
    raise last
