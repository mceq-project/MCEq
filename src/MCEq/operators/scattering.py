r"""Multiple Coulomb scattering of muons in the 2D transport.

A 2D state is a set of ``n_k`` angular modes, the Legendre moments
``<P_l(cos theta)>``. Scattering couples neither modes nor energies: it adds a
rate ``-Gamma(E, l)`` to the muon diagonal of each mode, which ETD2RK
integrates exactly through ``exp(h D)``. ``Gamma`` is the single-scattering
rate in mode space; multiple scattering follows from the exponential
(Goudsmit-Saunderson). ``Gamma(E, 0) = 0``, so the angle-integrated flux is
unchanged.

Two models:

- Gaussian, the Gauss approximation of CORSIKA (Heck & Pierog handbook
  p. 12): the diffusion eigenvalue ``Gamma = l (l + 1) theta_s^2(E) / 4`` on
  the sphere. Gaussian core only, no single-scattering tail.
- Screened Coulomb, the Wentzel cross section ``C / (1 - cos chi + 2 A)^2``
  summed over the elements of the medium. Screening ``A = chi_a^2 / 4`` from
  the Thomas-Fermi radius with the Moliere correction (Geant4 Physics
  Reference Manual, single scattering, Eqs. 97-98); atomic electrons enter as
  ``Z(Z + 1)``. ``Gamma`` is the sum of the exact transport cross sections
  ``sigma_l`` (:func:`transport_cross_sections`); its small-angle limit at
  ``kappa = l + 1/2`` is ``sum_i R_i [1 - kappa a_i K_1(kappa a_i)]``.
  Point nuclei, no recoil, no charge or helicity dependence. The angular
  distribution has no finite second moment.

Only muons are treated. Hadrons interact before scattering matters, and the
e+- angular distribution is set by the EM cascade kinematics.
"""

from __future__ import annotations

import numpy as np
from scipy import constants

__all__ = [
    "E_S",
    "LAMBDA_S",
    "MUON_MASS",
    "mode_rates",
    "muon_state_offsets",
    "screened_coulomb_rate_legendre",
    "theta_s_squared",
    "transport_cross_sections",
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

_HBARC = constants.hbar * constants.c / constants.electron_volt * 1e-7  # GeV cm
_BOHR = constants.physical_constants["Bohr radius"][0] * 100  # cm


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


def mode_rates(model, e_kin, l_grid, layout, muon_mass=MUON_MASS):
    """Rate ``Gamma >= 0`` [cm^2/g] of ``model``, shape ``(n_l, n_energy)``,
    at the Legendre degrees ``l_grid``.

    ``layout.scattering_composition()`` is read only by the screened Coulomb
    model.
    """
    ell = np.asarray(l_grid, dtype=float)
    if model == "gaussian":
        return (ell * (ell + 1))[:, None] * theta_s_squared(e_kin, muon_mass) / 4.0
    if model == "screened-coulomb":
        return screened_coulomb_rate_legendre(
            e_kin, ell, layout.scattering_composition(), muon_mass
        )
    raise ValueError(
        f"Unknown muon_scattering_model {model!r}; "
        "use 'screened-coulomb' or 'gaussian'."
    )


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


def _elements(e_kin, composition, muon_mass):
    """Per element: atoms per gram ``n``, Wentzel screening ``A`` and strength
    ``C`` [cm^2/sr] of ``dsigma/dOmega = C / (1 - cos chi + 2 A)^2``.

    ``composition`` has rows ``(Z, A [g/mol], atom fraction)``; the fractions
    give ``N_A f / sum(f A)`` atoms per gram.
    """
    comp = np.asarray(composition, dtype=float)
    alpha = constants.alpha
    p2 = e_kin * (e_kin + 2 * muon_mass)
    beta2 = p2 / (e_kin + muon_mass) ** 2
    mean_a = np.sum(comp[:, 1] * comp[:, 2])
    for z, _, fraction in comp:
        a_tf = 0.5 * (3 * np.pi / 4) ** (2 / 3) * _BOHR / z ** (1 / 3)
        chi2 = _HBARC**2 / (p2 * a_tf**2) * (1.13 + 3.76 * (z * alpha) ** 2 / beta2)
        strength = (alpha * _HBARC) ** 2 * z * (z + 1) / (p2 * beta2)
        yield constants.Avogadro * fraction / mean_a, chi2 / 4, strength


def screened_coulomb_rate_legendre(e_kin, l_grid, composition, muon_mass=MUON_MASS):
    """Screened Coulomb rate ``sum_i n_i sigma_l,i`` [cm^2/g], shape
    ``(n_l, n_energy)``, at the integer degrees ``l_grid``. ``e_kin`` is the
    kinetic energy in GeV, ``composition`` as in :func:`_elements`."""
    e_kin = np.asarray(e_kin, dtype=float)
    l_grid = np.asarray(l_grid, dtype=float)
    ls = np.rint(l_grid).astype(int)
    if np.any(ls != l_grid) or np.any(ls < 0):
        raise ValueError(f"Legendre modes need integer degrees l >= 0, got {l_grid}")
    rate = np.zeros((ls.size, e_kin.size))
    for n, screening, strength in _elements(e_kin, composition, muon_mass):
        sigma = transport_cross_sections(screening, int(ls.max(initial=0)))
        rate += np.pi * n * strength * sigma[ls]
    return rate


def transport_cross_sections(screening, l_max):
    r"""``sigma_l / (pi C)`` of the Wentzel kernel for ``l = 0 .. l_max``.

    ``sigma_l = 2 pi int (1 - P_l(mu)) C / (1 - mu + 2A)^2 dmu`` is
    ``sigma_tot (1 - <P_l>)`` with ``sigma_tot = pi C / (A (1 + A))``; the
    result has shape ``(l_max + 1,) + screening.shape`` and ``sigma_0 = 0``.

    With ``x = 1 + 2A`` and ``Q_l = P_l ln((x+1)/(x-1)) / 2 - W_{l-1}``,
    Neumann's integral gives ``1/(A(1+A)) + 4 Q_l'(x)``, which cancels to
    ``1 - <P_l>`` for small ``A``. Where ``l sqrt(A) <= 1`` the rate is
    evaluated as ``2 P_l'(x) ln((1+A)/A) + D_l / (A(1+A)) - 4 W'_{l-1}(x)``,
    ``D_l = 1 - P_l(x)``, by forward recurrences free of that cancellation.
    Above, ``P_l(x)`` grows as ``exp(2 l sqrt(A))`` and the polynomial terms
    cancel instead; there ``<P_l>`` is no longer close to one and the Neumann
    form is used with ``Q_l`` from :func:`_q_ratios`.
    """
    a = np.atleast_1d(np.asarray(screening, dtype=float))
    out = np.zeros((l_max + 1,) + a.shape)
    x, t, log, norm = 1 + 2 * a, -2 * a, np.log1p(1 / a), a * (1 + a)
    d_prev, d = np.zeros_like(a), t  # D_0, D_1
    dp_prev, dp = np.zeros_like(a), np.ones_like(a)  # P_0', P_1'
    w_prev, w = np.zeros_like(a), np.ones_like(a)  # W_{-1}, W_0
    dw_prev, dw = np.zeros_like(a), np.zeros_like(a)
    if l_max >= 1:
        out[1] = 2 * log + d / norm
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(1, l_max):  # l = n -> n + 1
            dp_prev, dp = dp, dp_prev + (2 * n + 1) * (1 - d)
            d_prev, d = d, ((2 * n + 1) * (t + (1 - t) * d) - n * d_prev) / (n + 1)
            w_prev, w, dw_prev, dw = (
                w,
                ((2 * n + 1) * x * w - n * w_prev) / (n + 1),
                dw,
                ((2 * n + 1) * (w + x * dw) - n * dw_prev) / (n + 1),
            )
            out[n + 1] = 2 * dp * log + d / norm - 4 * dw
    wide = l_max * np.sqrt(a) > 1
    if np.any(wide):
        aw, ls = a[wide], np.arange(1, l_max + 1)[:, None]
        r = _q_ratios(aw, l_max)
        q_prev = (
            np.log1p(1 / aw)
            / 2
            * np.cumprod(np.vstack([np.ones_like(aw), r[:-1]]), axis=0)
        )
        neumann = (1 - ls * q_prev * (1 - (1 + 2 * aw) * r)) / (aw * (1 + aw))
        out[1:, wide] = np.where(ls * np.sqrt(aw) > 1, neumann, out[1:, wide])
    return out.reshape((l_max + 1,) + np.shape(screening))


def _q_ratios(screening, l_max):
    """``Q_l(x) / Q_{l-1}(x)`` for ``l = 1 .. l_max``, ``x = 1 + 2A``.

    ``Q_l`` is the minimal solution of the Legendre recurrence for ``x > 1``,
    so the ratios are stable run backward (Miller), started at the asymptotic
    ratio ``x - sqrt(x^2 - 1)`` far enough above ``l_max`` that the start
    error has decayed by ``exp(-40)``.
    """
    x = 1 + 2 * screening
    r = x - np.sqrt((x - 1) * (x + 1))
    top = l_max + int(np.ceil(10 / np.sqrt(screening.min())))
    out = np.empty((l_max,) + screening.shape)
    for n in range(top, 0, -1):
        r = n / ((2 * n + 1) * x - (n + 1) * r)
        if n <= l_max:
            out[n - 1] = r
    return out
