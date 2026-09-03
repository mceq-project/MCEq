"""Platform, MKL, CUDA and Accelerate detection, resolved on first use.

Importing MCEq must not dlopen a BLAS, probe a GPU or decide which solver
kernel will run: those are properties of the machine that only matter once a
solve is set up, and paying for them at import made `import MCEq.config` a
half-second affair. Every probe here is cached after its first call;
:func:`reset_cache` exists for tests that need to re-probe.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import pathlib
import platform
import sys

#: Solver kernels, and whether ``kernel_config = "auto"`` may choose them.
#: CUDA is never auto-selected: a GPU context costs real time to create and a
#: matching cupy build is not always present on a machine that has a device.
KERNELS = {
    "accelerate_etd2": ("has_accelerate", True),
    "mkl_etd2": ("has_mkl", True),
    "numpy_etd2": (None, True),
    "cuda_etd2": ("has_cuda", False),
}

#: Names the dispatchers still accept for the kernels above.
ALIASES = {
    "accelerate": "accelerate_etd2",
    "mkl": "mkl_etd2",
    "cuda": "cuda_etd2",
    "numpy": "numpy_etd2",
}

#: Order "auto" tries. Accelerate wins on macOS, MKL on x86 Linux/Windows.
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
    """A usable device, not merely an importable cupy.

    `find_spec("cupy")` alone is True on a machine that has the wheel and no
    driver, where every CUDA route then fails at run time instead of being
    skipped.
    """
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


PROBES = {"has_mkl": has_mkl, "has_cuda": has_cuda, "has_accelerate": has_accelerate}


def available(capability):
    """`True` when the named capability (`has_mkl`, `has_cuda`, ...) is present."""
    return PROBES[capability]()


def resolve_kernel(requested):
    """Map a `kernel_config` value onto the kernel that will actually run.

    Raises when a specific kernel is named and its capability is missing, so a
    value assigned after import fails the same way one set before it does.
    """
    name = ALIASES.get(str(requested).lower(), str(requested).lower())
    if name == "auto":
        for candidate in AUTO_ORDER:
            capability = KERNELS[candidate][0]
            if capability is None or available(capability):
                return candidate
        return "numpy_etd2"
    if name not in KERNELS:
        raise Exception(
            f"Unsupported integrator setting '{requested}'. "
            f"Choose one of: {', '.join(KERNELS)}."
        )
    capability = KERNELS[name][0]
    if capability is not None and not available(capability):
        raise Exception(_UNAVAILABLE[capability])
    return name


def reset_cache():
    """Forget every probe result. For tests that fake a platform."""
    for probe in (mkl_library_path, has_mkl, has_cuda, has_accelerate):
        probe.cache_clear()
