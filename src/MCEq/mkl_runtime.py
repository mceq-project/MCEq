"""Process-wide ``libmkl_rt`` handle: the one owner, the one memo.

This module owns and caches the handle shared by the MKL backend and the
thread-limit setter. It lives below :mod:`MCEq.config` and imports nothing
from MCEq, because those two layers both need the handle and neither may
import the other: the MKL backend constructs its wrappers from it
(``MklSparseMatrix`` pins a per-instance reference so all wrappers share one
symbol table), and :func:`MCEq.config.set_mkl_threads` reaches
``mkl_set_num_threads`` through it (thread limits are process-wide, plan
D17). Configuration supplies the path-resolver callback. Importing this
module does not load MKL; :func:`load` does so on demand and returns ``None``
if the library is unavailable.

Behaviour is unchanged from when the memo lived in ``config``: one cdll for
the process lifetime, ``os.fspath`` at the boundary (the Windows loader on
Python 3.10/3.11 cannot take a ``Path``), silent ``None`` when MKL is not
installed.
"""

from __future__ import annotations

import os

_LIB = None
_has = lambda: False  # noqa: E731 - replaced by set_resolver()
_path = lambda: None  # noqa: E731


def set_resolver(has_fn, path_fn):
    """Install the detection probes (called by :mod:`MCEq.config`)."""
    global _has, _path
    _has = has_fn
    _path = path_fn


def load(path=None):
    """Load ``libmkl_rt`` once and return the cached handle, or None.

    ``path`` overrides the resolver (a run's ``backend.mkl_path`` from its
    config group); an override wins only while nothing is loaded yet, the
    same single-handle rule as before.
    """
    global _LIB
    if _LIB is not None or not _has():
        return _LIB
    from ctypes import cdll

    target = _path() if path is None else path
    _LIB = cdll.LoadLibrary(os.fspath(target))
    return _LIB


def lib():
    """The handle if already loaded, else None — read-only, never loads."""
    return _LIB
