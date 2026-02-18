"""
Platform-aware device detection for Facet.

Provides utilities for detecting the best available compute device
(CUDA, MPS, or CPU) and safe wrappers for device-specific operations.
Supports NVIDIA GPUs (CUDA), Apple Silicon (MPS), and CPU fallback.
"""

import sys


def get_best_device() -> str:
    """Detect the best available compute device.

    Priority: CUDA > MPS > CPU

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


def has_gpu() -> bool:
    """Check if any GPU backend is available (CUDA or MPS)."""
    device = get_best_device()
    return device in ('cuda', 'mps')


def is_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def is_mps() -> bool:
    """Check if Apple MPS is available."""
    try:
        import torch
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except ImportError:
        return False


def safe_empty_cache():
    """Safely clear GPU memory cache on any platform.

    No-op if no GPU is available or if the backend doesn't support cache clearing.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # MPS does not have an empty_cache equivalent
    except ImportError:
        pass


def get_gpu_memory_bytes() -> int:
    """Get total GPU memory in bytes.

    Returns:
        Total GPU memory in bytes, or 0 if no GPU is available.
        On MPS, returns system unified memory as an approximation.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Silicon uses unified memory; report system RAM
            # as the shared GPU/CPU pool
            return _get_system_memory_bytes()
    except (ImportError, RuntimeError):
        pass
    return 0


def get_gpu_memory_gb() -> float:
    """Get total GPU memory in GB.

    Returns:
        Total GPU memory in GB, or 0.0 if no GPU available.
    """
    mem = get_gpu_memory_bytes()
    return mem / (1024 ** 3) if mem > 0 else 0.0


def get_gpu_allocated_bytes() -> int:
    """Get currently allocated GPU memory in bytes.

    Returns:
        Allocated GPU memory in bytes, or 0 if not available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory()
    except (ImportError, RuntimeError, AttributeError):
        pass
    return 0


def get_gpu_reserved_bytes() -> int:
    """Get reserved (cached) GPU memory in bytes.

    Returns:
        Reserved GPU memory in bytes, or 0 if not available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.mps.driver_allocated_memory()
    except (ImportError, RuntimeError, AttributeError):
        pass
    return 0


def get_onnx_providers() -> list:
    """Get ONNX Runtime execution providers appropriate for the platform.

    Returns:
        List of provider strings in priority order.
    """
    requested = []
    try:
        import torch
        if torch.cuda.is_available():
            requested.append('CUDAExecutionProvider')
    except ImportError:
        pass

    if sys.platform == 'darwin':
        requested.append('CoreMLExecutionProvider')

    requested.append('CPUExecutionProvider')

    # Filter against actually available providers to avoid failures
    # (e.g. CoreML not compiled into the onnxruntime build)
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
        filtered = [p for p in requested if p in available]
        return filtered if filtered else requested
    except (ImportError, Exception):
        return requested


def get_insightface_ctx_id() -> int:
    """Get the InsightFace context ID for the current platform.

    Returns:
        0 for CUDA (GPU 0), -1 for CPU/MPS (InsightFace doesn't support MPS).
    """
    return 0 if is_cuda() else -1


def get_device_map_or_device(device: str):
    """Get the appropriate loading strategy for HuggingFace models.

    For CUDA, uses device_map="auto" for multi-GPU support.
    For MPS/CPU, returns the device string directly.

    Args:
        device: The target device ('cuda', 'mps', 'cpu')

    Returns:
        Tuple of (device_map, device_for_to):
        - device_map: "auto" for CUDA, None otherwise
        - device_for_to: None for CUDA (handled by device_map), device string otherwise
    """
    if device == 'cuda':
        return "auto", None
    return None, device


def is_oom_error(error: Exception) -> bool:
    """Check if an exception is an out-of-memory error on any platform.

    Handles CUDA OutOfMemoryError and MPS RuntimeError with OOM message.

    Args:
        error: The caught exception

    Returns:
        True if the error is an OOM error
    """
    try:
        import torch
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    if isinstance(error, RuntimeError) and 'out of memory' in str(error).lower():
        return True
    return False


def _get_system_memory_bytes() -> int:
    """Get total system memory in bytes."""
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        # Fallback: try reading from sysctl on macOS
        if sys.platform == 'darwin':
            try:
                import subprocess
                result = subprocess.run(
                    ['sysctl', '-n', 'hw.memsize'],
                    capture_output=True, text=True, timeout=5
                )
                return int(result.stdout.strip())
            except Exception:
                pass
        return 0
