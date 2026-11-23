r"""Multiple Coulomb scattering of muons in the 2D transport, from the database.

A 2D state is a set of ``n_k`` angular modes, the Legendre moments
``<P_l(cos theta)>``. Scattering couples neither modes nor energies: it adds a
rate ``-Gamma(E, l)`` to the muon diagonal of each mode, which ETD2RK
integrates exactly through ``exp(h D)``. ``Gamma`` is the single-scattering
rate in mode space; multiple scattering follows from the exponential
(Goudsmit-Saunderson). ``Gamma(E, 0) = 0``, so the angle-integrated flux is
unchanged.

The rates are physics input like the cross sections. mceq-EM computes them
(``mceq_em.coulomb_scattering``: screened Coulomb, Wentzel kernel with the
nuclear form factor and the electron kinematic limit of Geant4's WentzelVI),
and the 2D databases store them at ``material_scattering/<medium>/screened_coulomb_rates``, one
row per mode and one column per energy bin of the file, with the muon mass,
the model and the grids as attributes. MCEq cuts the table to the run's energy
window; a database without it, or one made for another muon mass or other
modes, raises ``ValueError``.

``config.muon_multiple_scattering = False`` switches scattering off. One-
dimensional runs have no angular scattering.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MUON_KEYS", "RATE_TABLE", "mode_rates", "muon_state_offsets"]

#: Name of the rate table in ``material_scattering/<medium>/``.
RATE_TABLE = "screened_coulomb_rates"

#: PDG ids and helicity states of every muon species the cascade can carry.
#: Helicity 0 is the unpolarised species; +-1 appear with
#: ``config.muon_helicity_dependence``.
MUON_KEYS = tuple((pdg, hel) for pdg in (13, -13) for hel in (0, 1, -1))


def mode_rates(e_kin, l_grid, layout, muon_mass):
    """Rate ``Gamma >= 0`` [cm^2/g] from the database, shape ``(n_l, n_energy)``.

    Parameters: ``e_kin`` the run's kinetic energy grid [GeV]; ``l_grid`` its
    Legendre degrees; ``layout`` the backend (``scattering_rate_table()``);
    ``muon_mass`` [GeV] of the run's particle table.

    Returns: the stored table on the run's window. Raises ``ValueError`` when
    the database has none or it was made for other inputs.
    """
    stored = layout.scattering_rate_table()
    fix = (
        "; use a 2D database that carries it (the v2 2D packages do) "
        "or set config.muon_multiple_scattering = False."
    )
    if stored is None:
        raise ValueError(
            f"The database has no muon scattering rates ({RATE_TABLE})" + fix
        )
    rates, table_l, table_e, attrs = stored
    ell = np.asarray(l_grid, dtype=float)
    if not np.isclose(
        float(attrs.get("muon_mass", np.nan)), muon_mass, rtol=1e-12, atol=0
    ):
        raise ValueError(
            f"{RATE_TABLE} was made for muon mass {attrs.get('muon_mass')} GeV, "
            f"the run uses {muon_mass}" + fix
        )
    if (
        not np.array_equal(np.asarray(table_l, dtype=float), ell)
        or np.shape(table_e) != np.shape(e_kin)
        or not np.allclose(table_e, e_kin, rtol=1e-12, atol=0)
    ):
        raise ValueError(
            f"{RATE_TABLE} is not on the run's modes and energy grid" + fix
        )
    return np.array(rates, dtype=float)


def muon_state_offsets(pdg2pref):
    """State-vector row offsets of the physical muon species in ``pdg2pref``.

    ``pdg2pref`` is a particle manager's ``(pdg, helicity) -> particle`` map.
    Only the physical ``MUON_KEYS`` are considered, so tracking aliases are
    excluded. A species is taken when it has been assigned a slot in the
    state vector, which is what ``lidx`` and a non-negative ``mceqidx`` say;
    disabled species have neither.
    """
    offsets = []
    for key in MUON_KEYS:
        if key in pdg2pref:
            particle = pdg2pref[key]
            if hasattr(particle, "lidx") and getattr(particle, "mceqidx", -1) >= 0:
                offsets.append(particle.lidx)
    return offsets
