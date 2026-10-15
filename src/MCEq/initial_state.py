"""Initial states: nucleon fluxes of a primary model and single primary particles."""

import warnings

import numpy as np
from scipy.linalg import solve

from MCEq.bin_average import delta_deposition
from MCEq.misc import getAZN, info


def fill_nucleons(phi0, pman, min_idx, p_flux, n_flux):
    """Write the nucleon fluxes ``p_flux``/``n_flux`` (bins ``min_idx`` and up,
    per m²) into ``phi0`` (per cm²); with no propagated neutron the neutron
    flux goes onto the protons. ``phi0`` may carry a trailing column axis."""
    p = pman[(2212, 0)]
    n = pman.pdg2pref.get((2112, 0))
    phi0[p.lidx + min_idx : p.uidx] = 1e-4 * p_flux
    if n is not None and not n.is_resonance:
        phi0[n.lidx + min_idx : n.uidx] = 1e-4 * n_flux
    else:
        info(2, "Neutrons not part of equation system, adding their flux to protons.")
        phi0[p.lidx + min_idx : p.uidx] += 1e-4 * n_flux


def nucleon_content(E, corsika_id=None, pdg_id=None):
    """``(A, Z, N)`` of a single primary: a CORSIKA nucleus ``100 * A + Z``
    (14 is the proton), or a particle given by PDG ID."""
    if corsika_id and pdg_id:
        raise Exception("Provide either corsika or PDG ID")
    if corsika_id:
        n_nucleons, n_protons = (1, 1) if corsika_id == 14 else divmod(corsika_id, 100)
        return n_nucleons, n_protons, n_nucleons - n_protons
    return getAZN(pdg_id)


def deposit_single_primary(phi0, pman, grid, E, corsika_id=None, pdg_id=None):
    """Add one primary of kinetic energy ``E`` to ``phi0``.

    A nucleus is ``Z`` protons and ``A - Z`` neutrons at ``E / A``. The
    particle is spread over three bins whose first three energy moments are
    those of a delta at ``E``, or, for stencil-averaged tables, over the
    ``2r + 1`` bins of :func:`MCEq.bin_average.delta_deposition`; the outermost
    bins take it whole.
    """
    n_nucleons, n_protons, n_neutrons = nucleon_content(E, corsika_id, pdg_id)
    En = E / float(n_nucleons) if n_nucleons > 0 else E
    if En < np.min(grid.c):
        raise Exception("energy per nucleon too low for primary " + str(corsika_id))
    info(
        3,
        f"superposition: n_protons={n_protons}, n_neutrons={n_neutrons}, "
        + f"energy per nucleon={En:5.3g} GeV",
    )
    if n_nucleons == 0:
        parts = [(pman[pdg_id], 1.0)]
    else:
        parts = [(pman[2212], n_protons), (pman[2112], n_neutrons)]
        parts = [(p, k) for p, k in parts if k > 0]

    if _deposit_for_stencil(phi0, grid, En, parts):
        return

    cenbin = np.argwhere(En < grid.b)[0][0] - 1
    if cenbin == 0 or cenbin == len(grid.c) - 1:
        for p, k in parts:
            phi0[p.lidx + cenbin] += k / grid.w[cenbin]
        return

    # Equalize the first three moments for 3 normalizations around the central bin
    sl = slice(cenbin - 1, cenbin + 2)
    emat = np.vstack(
        (grid.w[sl], grid.w[sl] * grid.c[sl], grid.w[sl] * grid.c[sl] ** 2)
    )
    for p, k in parts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            phi0[p.lidx + cenbin - 1 : p.lidx + cenbin + 2] += solve(
                emat, np.array([k, En * k, En**2 * k])
            )


def _deposit_for_stencil(phi0, grid, En, parts):
    """Deposit on ``2r + 1`` bins if the species' tables are stencil-averaged;
    False (nothing deposited) without marks or within ``r`` bins of the grid edge."""
    radius = {max(p._int_stencil, p._dec_stencil) for p, _ in parts}
    if radius == {0}:
        return False
    if len(radius) != 1 or 0 in radius:
        raise ValueError(
            "single primary: deposited species disagree on the bin-average "
            f"stencil ({sorted(radius)}); the database marks are inconsistent"
        )
    r = radius.pop()
    c = int(np.searchsorted(grid.b, En, side="right")) - 1
    if c < r or c > len(grid.c) - 1 - r:
        # A one-sided moment match oscillates; the tables read a constant
        # density near the edge anyway.
        info(
            2,
            f"single primary at {En:5.3g} GeV is within {r} bins of the grid "
            "edge; using the three-bin deposition",
        )
        return False
    first, contents = delta_deposition(grid.b, En, r)
    sl = slice(first, first + len(contents))
    for p, k in parts:
        phi0[p.lidx + sl.start : p.lidx + sl.stop] += k * contents / grid.w[sl]
    return True
