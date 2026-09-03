r"""Gaussian multiple Coulomb scattering of muons in the 2D transport.

In the 2D (Hankel) transport a state is a set of ``n_k`` modes, mode ``k``
carrying the angular scale ``kappa_k``; ``kappa = 0`` is the axis-integrated
(paraxial) mode. Small-angle diffusion of the angular distribution is a
Laplacian in the transverse angle, and the Hankel transform turns that into a
diagonal multiplier: with a Gaussian angular spread growing as
``<theta^2> = theta_s^2(E) X`` per unit slant depth,

    ``dF_k/dX |_ms = -(kappa_k^2 / 4) theta_s^2(E) F_k``.               (1)

So scattering neither couples modes nor couples energies -- it is one number
per ``(mode, species, energy)``, and it is added to the diagonal of the
interaction operator. That placement is the point: the diagonal is the part
ETD2RK integrates through ``exp(h D)``, so the damping is exact at any step
size and needs no operator split. The paraxial mode is untouched, ``kappa = 0``
in (1) giving zero: broadening the beam does not change its axis-integrated
content.

``theta_s`` is the Gauss approximation CORSIKA uses (Heck & Pierog handbook
p. 12): the Gaussian core only, with no Moliere tail, so wide-angle single
scatters are not described.

Only muons are treated here. Hadronic species are dominated by nuclear
interaction well before scattering matters, and the e+- angular distribution
is set by the EM cascade's own kinematics rather than by Coulomb scattering
of a through-going track.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "E_S",
    "LAMBDA_S",
    "MUON_MASS",
    "mode_damping",
    "muon_state_offsets",
    "theta_s_squared",
]

#: Scattering length of the CORSIKA Gauss approximation, g/cm^2.
LAMBDA_S = 37.7

#: Scattering energy of the CORSIKA Gauss approximation, GeV (21 MeV).
E_S = 0.021

#: PDG muon mass, GeV. Fallback only -- a caller that has a particle manager
#: passes the mass the tables were built with.
MUON_MASS = 0.10566

#: PDG ids and helicity states of every muon species the cascade can carry.
#: Helicity 0 is the unpolarised species; +-1 appear with
#: ``config.muon_helicity_dependence``.
MUON_KEYS = tuple((pdg, hel) for pdg in (13, -13) for hel in (0, 1, -1))


def theta_s_squared(e_kin, muon_mass=MUON_MASS):
    r"""Squared multiple-scattering angle per unit depth, per energy bin.

    ``theta_s^2(E) = (1/lambda_s) (E_s / (E beta^2))^2`` with ``E`` the total
    lab energy and ``beta`` the velocity -- the Gaussian-core width grows as
    ``theta_s^2 X`` in slant depth. Steeply falling in energy, so the damping
    is a low-energy effect.

    ``e_kin`` is kinetic energy, the MCEq grid convention; ``muon_mass`` turns
    it into ``E`` and ``beta``. Below the mass the momentum is floored at a
    positive tiny value so ``beta`` stays finite rather than returning nan for
    a grid that extends under threshold.
    """
    e_lab = np.asarray(e_kin) + muon_mass
    p2 = e_lab**2 - muon_mass**2
    p2 = np.where(p2 > 0, p2, 1e-30)
    beta = np.sqrt(p2) / e_lab
    return (1.0 / LAMBDA_S) * (E_S / (e_lab * beta**2)) ** 2


def mode_damping(theta_s_sq, kappa):
    """The diagonal contribution ``-kappa^2 theta_s^2 / 4`` of one Hankel mode.

    Eq. (1) of the module docstring, one value per energy bin, to be added to
    the muon diagonals of that mode. Negative definite, so it only damps.

    Scales as ``kappa^2`` at fixed energy and vanishes identically at
    ``kappa = 0``; a caller that assembles sparse entries skips that mode
    rather than adding a row of zeros to it.
    """
    return -(kappa**2) * np.asarray(theta_s_sq) / 4.0


def muon_state_offsets(pdg2pref):
    """State-vector row offsets of every muon species present in ``pdg2pref``.

    ``pdg2pref`` is a particle manager's ``(pdg, helicity) -> particle`` map.
    A species is taken when it has been assigned a slot in the state vector,
    which is what ``lidx`` and a non-negative ``mceqidx`` say; tracking-only
    and disabled species have neither.
    """
    offsets = []
    for key in MUON_KEYS:
        if key in pdg2pref:
            particle = pdg2pref[key]
            if hasattr(particle, "lidx") and getattr(particle, "mceqidx", -1) >= 0:
                offsets.append(particle.lidx)
    return offsets
