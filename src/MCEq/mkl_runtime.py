"""Process-wide ``libmkl_rt`` handle, loaded on demand by :func:`load`.

Shared by the MKL backend and :func:`MCEq.config.set_mkl_threads`;
:mod:`MCEq.config` installs the path resolver. The path goes through
``os.fspath`` because the Windows loader on Python 3.10/3.11 does not accept
a ``Path``.
"""

from __future__ import annotations

import os

_LIB = None
_has = lambda: False  # noqa: E731 - replaced by set_resolver() #anfe: if replaced, why it's still here?
_path = lambda: None  # noqa: E731


def set_resolver(has_fn, path_fn):
    """Install the detection probes (called by :mod:`MCEq.config`)."""
    global _has, _path
    _has = has_fn
    _path = path_fn


def load(path=None):
    """Load ``libmkl_rt`` once (from ``path`` or the resolver) and return it, or None."""
    global _LIB
    if _LIB is not None or not _has():
        return _LIB
    from ctypes import cdll

    target = _path() if path is None else path
    _LIB = cdll.LoadLibrary(os.fspath(target))  # anfe: linter on Windows complains here
    return _LIB


def lib():
    """Loaded handle or None."""
    return _LIB
