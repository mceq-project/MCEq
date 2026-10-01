r"""Multiple Coulomb scattering of muons in the 2D transport.

In the 2D (Hankel) transport a state is a set of ``n_k`` modes, mode ``k``
carrying the angular scale ``kappa_k``; ``kappa = 0`` is the axis-integrated
(paraxial) mode. Scattering couples neither modes nor energies: it adds a rate
``-Gamma(E, kappa_k)`` to the muon diagonal of each mode, which ETD2RK
integrates exactly through ``exp(h D)``. ``Gamma`` is the single-scattering
rate in mode space; multiple scattering follows from the exponential.
``Gamma(E, 0) = 0``, so the axis-integrated flux is unchanged.

Two models:

- Gaussian, ``Gamma = kappa^2 theta_s^2(E) / 4`` with the Gauss approximation
  of CORSIKA (Heck & Pierog handbook p. 12). Gaussian core only, no
  single-scattering tail.
- Screened Coulomb, ``Gamma = sum_i R_i [1 - kappa a_i K_1(kappa a_i)]``: the
  Hankel transform of the screened Rutherford cross section in the small-angle
  plane, summed over the elements ``i`` of the medium. Wentzel screening angle
  ``a_i`` from the Thomas-Fermi radius with the Moliere correction (Geant4
  Physics Reference Manual, single scattering, Eqs. 97-98); atomic electrons
  enter as ``Z(Z + 1)``. Point nuclei, no recoil, no charge or helicity
  dependence. The angular distribution has no finite second moment.

Only muons are treated. Hadrons interact before scattering matters, and the
e+- angular distribution is set by the EM cascade kinematics.
"""

from __future__ import annotations

import numpy as np
from scipy import constants
from scipy.special import k1

__all__ = [
    "E_S",
    "LAMBDA_S",
    "MUON_MASS",
    "mode_damping",
    "screened_coulomb_rate",
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
    it into ``E`` and ``beta``. Nonpositive computed momentum-squared values
    are floored before the division, so ``beta`` stays finite rather than
    returning nan.
    """
    e_lab = np.asarray(e_kin) + muon_mass
    p2 = e_lab**2 - muon_mass**2
    p2 = np.where(p2 > 0, p2, 1e-30)
    beta = np.sqrt(p2) / e_lab
    return (1.0 / LAMBDA_S) * (E_S / (e_lab * beta**2)) ** 2


def mode_damping(theta_s_sq, kappa):
    """The diagonal contribution ``-kappa^2 theta_s^2 / 4`` of one Hankel mode.

    Eq. (1) of the module docstring, one value per energy bin, to be added to
    the muon diagonals of that mode. Nonpositive, so it only damps (exactly
    zero for mode zero).

    Scales as ``kappa^2`` at fixed energy and vanishes identically at
    ``kappa = 0``; a caller that assembles sparse entries skips that mode
    rather than adding a row of zeros to it.
    """
    return -(kappa**2) * np.asarray(theta_s_sq) / 4.0


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


def _one_minus_z_k1(z):
    """``1 - z K_1(z)``, with its series below ``z = 1e-3``; zero at ``z = 0``."""
    out = np.zeros_like(z)
    small = (z > 0) & (z < 1e-3)
    x = z[small]
    log = np.log(x / 2) + np.euler_gamma
    out[small] = -x * x / 2 * (log - 0.5) - x**4 / 16 * (log - 1.25)
    large = z >= 1e-3
    out[large] = 1 - z[large] * k1(z[large])
    return out


def screened_coulomb_rate(e_kin, kappa, composition, muon_mass=MUON_MASS):
    """Screened Coulomb rate ``Gamma`` [cm^2/g], shape ``(n_k, n_energy)``.

    ``e_kin`` is the kinetic energy in GeV, ``kappa`` the mode scale in 1/rad.
    ``composition`` has rows ``(Z, A [g/mol], atom fraction)``; the fractions
    give ``N_A f / sum(f A)`` atoms per gram.
    """
    e_kin = np.asarray(e_kin, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    comp = np.asarray(composition, dtype=float)
    hbarc = constants.hbar * constants.c / constants.electron_volt * 1e-7  # GeV cm
    bohr = constants.physical_constants["Bohr radius"][0] * 100  # cm
    alpha = constants.alpha
    p2 = e_kin * (e_kin + 2 * muon_mass)
    beta2 = p2 / (e_kin + muon_mass) ** 2
    mean_a = np.sum(comp[:, 1] * comp[:, 2])
    rate = np.zeros((kappa.size, e_kin.size))
    for z, _, fraction in comp:
        a_tf = 0.5 * (3 * np.pi / 4) ** (2 / 3) * bohr / z ** (1 / 3)
        a2 = hbarc**2 / (p2 * a_tf**2) * (1.13 + 3.76 * (z * alpha) ** 2 / beta2)
        strength = (
            4 * np.pi * constants.Avogadro * fraction / mean_a
            * (alpha * hbarc) ** 2 * z * (z + 1) / (p2 * beta2)
        )  # fmt: skip
        rate += strength / a2 * _one_minus_z_k1(kappa[:, None] * np.sqrt(a2))
    return rate
