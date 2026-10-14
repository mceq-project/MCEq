"""Platform, MKL, CUDA and Accelerate detection and kernel resolution, probed on first use and cached."""

from __future__ import annotations

import functools
import importlib.util
import os
import pathlib
import platform
import sys

#: Solver kernels and the capability each needs.
KERNELS = {
    "accelerate_etd2": "has_accelerate",
    "mkl_etd2": "has_mkl",
    "numpy_etd2": None,
    "cuda_etd2": "has_cuda",
}

#: Short names accepted for the kernels above.
ALIASES = {
    "accelerate": "accelerate_etd2",
    "mkl": "mkl_etd2",
    "cuda": "cuda_etd2",
    "numpy": "numpy_etd2",
}

#: Order "auto" tries. CUDA is never chosen automatically: creating a GPU
#: context is slow and a matching cupy build may be missing.
AUTO_ORDER = ("accelerate_etd2", "mkl_etd2", "numpy_etd2")

_UNAVAILABLE = {
    "has_cuda": "CUDA unavailable. Make sure cupy is installed and a device is visible.",
    "has_mkl": "MKL unavailable. Make sure Intel MKL is installed.",
    "has_accelerate": "Accelerate unavailable. Only on MacOS.",
}


@functools.cache
def mkl_library_path():
    """Path `libmkl_rt` would be loaded from, whether or not it exists."""
    prefix = pathlib.Path(sys.prefix)
    tag = platform.platform()
    if "Linux" in tag:
        found = sorted((prefix / "lib").glob("libmkl_rt*"))
        return found[0] if found else prefix / "lib" / "libmkl_rt.so"
    if "macOS" in tag:
        return prefix / "lib" / "libmkl_rt.dylib"
    for directory in (prefix / "Library" / "bin", prefix / "lib"):
        if directory.exists():
            found = sorted(directory.glob("mkl_rt*.dll"))
            if found:
                return found[0]
    return pathlib.Path(os.fspath(prefix / "Library" / "bin" / "mkl_rt.dll"))


@functools.cache
def has_mkl():
    return mkl_library_path().is_file()


@functools.cache
def has_accelerate():
    return "macOS" in platform.platform()


@functools.cache
def has_cuda():
    """True if cupy imports and sees at least one device."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _available(capability):
    return capability is None or globals()[capability]()


def resolve_kernel(requested):
    """Map a `kernel_config` value onto the kernel that will actually run.

    Raises when a specific kernel is named and its capability is missing.
    """
    name = ALIASES.get(str(requested).lower(), str(requested).lower())
    if name == "auto":
        for candidate in AUTO_ORDER:
            if _available(KERNELS[candidate]):
                return candidate
        return "numpy_etd2"
    if name not in KERNELS:
        raise Exception(
            f"Unsupported integrator setting '{requested}'. "
            f"Choose one of: {', '.join(KERNELS)}."
        )
    if not _available(KERNELS[name]):
        raise Exception(_UNAVAILABLE[KERNELS[name]])
    return name
