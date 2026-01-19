import gc
from typing import Optional

import torch


def get_best_device() -> torch.device:
    """Return the optimal device based on platform and available hardware.

    Priority order: CUDA > XPU > MPS > CPU

    Returns:
        torch.device: The best available device for the current system.

    Examples:
        >>> device = get_best_device()
        >>> model.to(device)
    """
    # CUDA: NVIDIA and AMD (via ROCm)
    if torch.cuda.is_available():
        return torch.device("cuda")

    # XPU: Intel ARC GPUs
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    # MPS: Apple Silicon and Intel Macs with AMD GPUs
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_best_dtype(device: Optional[torch.device] = None) -> torch.dtype:
    """Return the optimal dtype for the given device.

    Args:
        device: The target device. If None, uses get_best_device().

    Returns:
        torch.dtype: The optimal dtype for the device.
            - CUDA: bfloat16 if supported, else float16
            - XPU: bfloat16 if supported, else float16
            - MPS: float16
            - CPU: float32

    Examples:
        >>> device = get_best_device()
        >>> dtype = get_best_dtype(device)
        >>> model.to(device, dtype=dtype)
    """
    if device is None:
        device = get_best_device()

    device_type = device.type

    if device_type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    if device_type == "xpu":
        # Intel XPU supports bfloat16 on most ARC GPUs
        try:
            if (
                hasattr(torch.xpu, "is_bf16_supported")
                and torch.xpu.is_bf16_supported()
            ):
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    if device_type == "mps":
        return torch.float16

    # CPU fallback
    return torch.float32


def empty_cache(device: Optional[torch.device] = None) -> None:
    """Clear GPU memory cache in a platform-aware manner.

    Args:
        device: The device to clear cache for. If None, clears all available backends.

    Examples:
        >>> empty_cache()  # Clears cache for all available backends
        >>> empty_cache(torch.device("cuda"))  # Clears CUDA cache only
    """
    gc.collect()

    if device is not None:
        device_type = device.type
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif (
            device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available()
        ):
            torch.xpu.empty_cache()
        elif device_type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return

    # Clear all available backends
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_device_info(device: Optional[torch.device] = None) -> dict:
    """Get memory and device information for any backend.

    Args:
        device: The device to get info for. If None, uses get_best_device().

    Returns:
        dict: Device information including:
            - device: Device name or type
            - allocated_gb: Allocated memory in GB (if available)
            - reserved_gb: Reserved memory in GB (if available)
            - memory: "N/A" if memory info unavailable

    Examples:
        >>> info = get_device_info()
        >>> print(f"Device: {info['device']}")
    """
    if device is None:
        device = get_best_device()

    device_type = device.type

    if device_type == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return {
            "device": torch.cuda.get_device_name(0),
            "allocated_gb": f"{allocated:.2f}",
            "reserved_gb": f"{reserved:.2f}",
        }

    if device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            allocated = torch.xpu.memory_allocated() / 1024**3
            reserved = torch.xpu.memory_reserved() / 1024**3
            device_name = torch.xpu.get_device_name(0)
            return {
                "device": device_name,
                "allocated_gb": f"{allocated:.2f}",
                "reserved_gb": f"{reserved:.2f}",
            }
        except Exception:
            return {"device": "Intel XPU", "memory": "N/A"}

    if device_type == "mps" and torch.backends.mps.is_available():
        try:
            # MPS has limited memory reporting
            allocated = torch.mps.current_allocated_memory() / 1024**3
            return {
                "device": "Apple MPS",
                "allocated_gb": f"{allocated:.2f}",
                "reserved_gb": "N/A",
            }
        except Exception:
            return {"device": "Apple MPS", "memory": "N/A"}

    return {"device": "cpu", "memory": "N/A"}


def is_gpu_available() -> bool:
    """Check if any GPU backend is available.

    Returns:
        bool: True if CUDA, XPU, or MPS is available.

    Examples:
        >>> if is_gpu_available():
        ...     print("GPU acceleration available")
    """
    if torch.cuda.is_available():
        return True

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return True

    if torch.backends.mps.is_available():
        return True

    return False


def synchronize(device: Optional[torch.device] = None) -> None:
    """Synchronize the given device or all available GPU backends.

    Useful for accurate timing measurements and ensuring operations complete.

    Args:
        device: The device to synchronize. If None, synchronizes all available backends.
    """
    if device is not None:
        device_type = device.type
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif (
            device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available()
        ):
            torch.xpu.synchronize()
        elif device_type == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()
        return

    # Synchronize all available backends
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()

    if torch.backends.mps.is_available():
        torch.mps.synchronize()
