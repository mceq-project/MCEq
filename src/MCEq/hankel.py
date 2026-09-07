"""Compat shim: the Hankel transform moved to ``MCEq.driver.results``.

Kept as the permanent-ish re-export (§8.6); ``MCEq.hankel`` stays
importable for user code until Phase 7 deletes the flat shims (D6).
"""

from MCEq.driver.results import inverse_hankel_legacy  # noqa: F401

__all__ = ["inverse_hankel_legacy"]
