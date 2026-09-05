"""EM-database selection: the rho-stack slice, group paths, helicity copies.

Selection policy for the electromagnetic file, on top of the materialized
reads of :mod:`MCEq.data.hdf5_store` -- nothing here opens a file itself.
The module imports ``numpy``, ``MCEq.misc.info`` and nothing else from
``MCEq`` (data -> misc is the permitted direction); it reads no
configuration -- the requested air density is passed in as a value, and
where it comes from is the caller's business.
"""

from collections import defaultdict
from itertools import product

import numpy as np

from MCEq.misc import info


def medium_for_em(medium):
    """The medium the EM file should be read for, given the hadronic one.

    The historical ice -> water substitution, with its log line. B4
    (``tests/test_data_bug_pins.py``, two pins): ``interaction_db`` calls
    this and throws the result away -- it rewrites its local ``medium``
    but then reads the EM group under ``self.medium`` -- and
    ``_cs_db_single`` never substitutes at all, so an ice run reads the
    ice EM file on both paths and only the log line claims water. Until
    the fix ruling lands the callers keep reading what they always read;
    a fix must pick one path and update both pins (they pin ice-on-both).
    """
    if medium == "ice":
        info(5, "Electromagnetic cross sections for ice replaced by water.")
        return "water"
    return medium


def em_group_path(medium, idx=None):
    """Group path of the EM tables of ``medium``, slice ``idx`` if given."""
    if idx is None:
        return f"electromagnetic/{medium}"
    return f"electromagnetic/{medium}/rho_{idx:02d}"


def select_em_rho_slice(store, medium, air_density):
    """Return the index into ``rho_grid`` of ``electromagnetic/<medium>``
    whose stored density is closest in log10 to ``air_density`` (g/cm³), or
    ``None`` when no ρ stack is present / no override is requested.

    Only the air medium has a stacked layout in the current pipeline; for
    other media this returns ``None`` silently.  See
    ``mceq-em-integration/wiki/methods/lpm-density-factorization.md`` for
    the design.
    """
    if air_density is None:
        return None
    if medium != "air":
        return None
    rho_path = em_group_path(medium) + "/rho_grid"
    if not store.has(rho_path):
        info(2, "EM DB has no rho_grid; ignoring config.em_air_density.")
        return None
    rho_grid = np.asarray(store.read_dataset(rho_path), dtype=float)
    if rho_grid.size == 0:
        return None
    log_target = np.log10(float(air_density))
    idx = int(np.argmin(np.abs(np.log10(rho_grid) - log_target)))
    info(
        2,
        f"EM ρ-stack: selecting slice {idx} (ρ={rho_grid[idx]:.3e} g/cm³) "
        f"for requested ρ={air_density:.3e} g/cm³.",
    )
    return idx


def emca_group_path(store, medium, caller, air_density):
    """Group path holding the ``emca_mats`` pack, honoring the ρ-stack."""
    idx = select_em_rho_slice(store, medium, air_density)
    if idx is None:
        return em_group_path(medium)
    info(3, f"[{caller}] using ρ slice {idx} of /electromagnetic/{medium}/")
    return em_group_path(medium, idx)


def em_cs_group_path(store, medium, caller, air_density):
    """Group path holding the ``cs`` table, honoring the ρ-stack."""
    idx = select_em_rho_slice(store, medium, air_density)
    if idx is None:
        return em_group_path(medium)
    info(3, f"[{caller}] using ρ slice {idx} of /electromagnetic/{medium}/")
    return em_group_path(medium, idx)


def helicity_duplicate(em_index):
    """Copy bremsstrahlung and photon emission to polarised e± and μ±.

    Verbatim extraction of the block ``interaction_db`` runs under
    ``physics.muon_helicity_dependence``: for each charged lepton and each
    helicity the ``(h, h)`` and ``(h, γ)`` channels take the value of the
    unpolarised entry, ``parents`` gains the polarised pair, and
    ``relations`` / ``particles`` are rebuilt from the resulting index.

    This is only approximately valid and is done for consistency. Typically
    electrons would quickly depolarize due to multiple scattering but this
    requires additional matrices for ``(-11,1) -> (-11,0)`` etc. that are
    not available now.

    Mutates and returns the same dict.
    """
    info(
        5,
        "Copy bremsstrahlung and photon emission to "
        + "polarised electrons and muons.",
    )
    for pid, h in product([11, -11, 13, -13], [-1, 1]):
        index_d = em_index["index_d"]
        index_d[((pid, h), (pid, h))] = index_d[((pid, 0), (pid, 0))]
        index_d[((pid, h), (22, 0))] = index_d[((pid, 0), (22, 0))]
        em_index["parents"].append((pid, h))

    em_index["relations"] = defaultdict(list)
    em_index["particles"] = []

    for parent, child in em_index["index_d"]:
        em_index["relations"][parent].append(child)
        em_index["particles"].append(parent)
        em_index["particles"].append(child)
    return em_index
