"""Detached configuration snapshot for one :class:`MCEqRun` (plan D3).

``MCEqRun(..., config=RunConfig())`` freezes the process-wide settings at
construction time: the run reads its groups and flat names through the
returned object afterwards, and later writes to :mod:`MCEq.config` (or the
``mceq_config`` shim) have no effect on it. ``config=None`` (the default)
keeps the documented tier-2 D3 semantics — live read-through, where a
post-construction ``config.x = ...`` is seen at the next ``solve()`` —
which is exactly what every existing caller relies on.

The snapshot holds copies, not views: every settings group is deep-copied
from the resolved values (dict-valued settings deep-copied, so mutating
``cfg.physics.filters`` cannot reach the global ``adv_set``), and the
lazily detected runtime facts are resolved once at snapshot time. The
object is shaped like the config module for the subset the driver reads —
group attributes (``cfg.physics``), flat names (``cfg.e_min``), and the
two derived helpers (``cfg.secant_mode(is_2d)``,
``cfg.secant_theta_cap()``) — so driver code reads ``cfg.X`` uniformly
whether ``cfg`` is the live module or a detached snapshot.

Two process-level exceptions are documented, not accidental: library
handles (the MKL ``cdll``, the CUDA context) and kernel detection
side effects belong to the process, not to a run; a snapshot freezes
``kernel_config`` as a string but the libraries still load globally.
"""

from __future__ import annotations

import copy
import sys
from types import SimpleNamespace


def _is_copyable(value):
    return isinstance(value, (dict, list, set))


class FrozenGroup(SimpleNamespace):
    """A resolved settings group, detached from the live config.

    Detached means deep-copied at snapshot time and **write-refusing**:
    attribute writes raise, mirroring :class:`RunConfig`, so a frozen
    group can never silently diverge from its own snapshot. Payloads
    inside dict-valued fields (``filters``, ``etd2_path``) stay ordinary
    mutable copies — a snapshot promises no immutability of contents,
    only of which value a name points at (M5, ruling D-2: detached
    copies, documented, not immutable payloads).

    Carries the one method the driver's §9 validator needs from a group —
    :meth:`with_overrides` — returning another frozen copy, so the
    validator works identically on a live ``GroupView`` and on a snapshot
    group without either type importing the other.
    """

    def __init__(self, **fields):
        # bypass the write guard below; construction is the only writer
        self.__dict__.update(fields)

    def __setattr__(self, attr, value):
        raise AttributeError(
            f"{attr!r} is read-only: a FrozenGroup is a detached snapshot "
            "(build a new RunConfig or change the global config before "
            "construction instead)"
        )

    def __delattr__(self, attr):
        raise AttributeError(f"{attr!r} cannot be deleted from a snapshot")

    def with_overrides(self, **values):
        unknown = set(values) - set(self.__dict__)
        if unknown:
            raise AttributeError(f"no setting {sorted(unknown)[0]!r}")
        merged = dict(self.__dict__)
        merged.update(values)
        return FrozenGroup(**merged)


class RunConfig:
    """A detached, config-module-shaped copy of the process settings."""

    def __init__(self):
        config = sys.modules["MCEq.config"]
        from MCEq.config import groups as _groups

        for name, fields in _groups.GROUPS.items():
            frozen = {}
            for attr, flat in fields.items():
                value = getattr(config, flat)
                frozen[attr] = copy.deepcopy(value) if _is_copyable(value) else value
            object.__setattr__(self, "_" + name, FrozenGroup(**frozen))
        for fact in ("has_mkl", "has_cuda", "has_accelerate"):
            setattr(self, fact, getattr(config, fact))

    @classmethod
    def snapshot(cls):
        """Build the snapshot of the current process-wide settings."""
        return cls()

    def __getattr__(self, attr):
        # Group names and flat names both resolve through the frozen copy.
        from MCEq.config.groups import FLAT_TO_GROUP, GROUPS

        if attr in GROUPS:
            return object.__getattribute__(self, "_" + attr)
        try:
            group, field = FLAT_TO_GROUP[attr]
        except KeyError:
            raise AttributeError(f"RunConfig has no setting {attr!r}") from None
        return getattr(object.__getattribute__(self, "_" + group), field)

    def __setattr__(self, attr, value):
        from MCEq.config.groups import FLAT_TO_GROUP

        if attr in FLAT_TO_GROUP or attr.startswith("_"):
            raise AttributeError(
                f"RunConfig is a detached snapshot; {attr!r} is read-only "
                "(build a new snapshot or change the global config before "
                "construction instead)"
            )
        object.__setattr__(self, attr, value)

    def __dir__(self):
        from MCEq.config.groups import GROUPS

        return sorted(GROUPS) + ["has_mkl", "has_cuda", "has_accelerate"]

    # --- derived helpers, re-computed from this snapshot's own fields --------

    def secant_theta_cap(self):
        return _cap_or_raise(self.secant.cap_deg)

    def secant_mode(self, is_2d):
        flag = self.secant.transport
        if not is_2d:
            return "off"
        if isinstance(flag, str) and flag.lower() == "auto":
            return "auto"
        return "require" if flag else "off"


def _cap_or_raise(cap_deg):
    cap = float(cap_deg)
    if not 50.0 <= cap < 90.0:
        raise ValueError(
            f"config.secant_theta_cap_deg = {cap:g} is outside the "
            "supported range [50, 90)."
        )
    return cap
