"""Compatibility re-export of ``inverse_hankel_legacy`` from
:mod:`MCEq.driver.results`.

``MCEq.hankel`` stays importable for user code.
"""

from MCEq.driver.results import inverse_hankel_legacy  # noqa: F401

__all__ = ["inverse_hankel_legacy"]
