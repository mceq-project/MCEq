"""Analytic Z-factors of the interaction and decay tables of an MCEq run."""

import numpy as np

from MCEq.misc import info

__all__ = ["decay_z_factor", "z_factor"]


def z_factor(
    proj,
    sec,
    e_grid,
    pmodel,
    definition="primary_e",
    min_energy=0.3,
    use_cs_scaling=True,
):
    r"""Energy-dependent Z-factor of ``projectile -> secondary`` interactions.

    ``definition="thunman"`` (any value but ``"primary_e"``) is the moment
    of Thunman, Ingelman and Gondolo (1996) at the secondary energy,

    .. math::

        Z(E) = \int_E^\infty dE'\,\frac{\phi_N(E')}{\phi_N(E)}
            \frac{\lambda_N(E)}{\lambda_N(E')}\frac{dn(E', E)}{dE},

    as a matrix row :math:`Z_i = \sum_j C_{ij}\,\sigma_j\phi_j /
    (\sigma_i\phi_i)`, with :math:`\phi` the proton (for a neutron
    projectile the neutron) flux of the primary model and :math:`\sigma`
    the production cross section, which ``use_cs_scaling=False`` drops.
    Under Feynman scaling and a power law :math:`E^{-(\gamma+1)}` it is
    :math:`\int_0^1 x^{\gamma-1}F(x)dx`.

    ``definition="primary_e"`` is the column at the projectile energy
    :math:`E_j`, weighted by :math:`\phi_N(E_j)/\phi_N(E_i)` over the
    secondaries above ``min_energy`` (:meth:`MCEqParticle.parent_column`
    for stencil-averaged tables). Values below ``min_energy`` are 0.

    Args:
      proj, sec: projectile and secondary :class:`~MCEq.particlemanager.MCEqParticle`.
      e_grid: kinetic-energy grid.
      pmodel: primary model (``p_and_n_flux``).

    Returns:
      (numpy.ndarray): Z on ``e_grid``
    """
    if not proj.is_projectile:
        raise Exception(f"{proj.name} is not a projectile particle.")
    info(10, f"Computing e-dependent Zfactor for {proj.name} -> {sec.name}")
    if sec not in proj.hadr_yields:
        raise Exception(f"{sec.name} is not a secondary particle of {proj.name}.")

    p_flux, n_flux = pmodel.p_and_n_flux(e_grid)[1:]
    nuc_flux = np.asarray(
        n_flux if proj.pdg_id[0] == 2112 else p_flux, dtype=np.float64
    )
    smat = np.asarray(proj.hadr_yields[sec], dtype=np.float64)
    proj_cs = np.asarray(proj.prod_cross_section(), dtype=np.float64)
    e = np.asarray(e_grid, dtype=np.float64)
    zfac = np.zeros_like(e)

    if definition == "primary_e":
        below = np.flatnonzero(e < min_energy)
        min_idx = int(below[-1]) if len(below) else 0
        for p_eidx in np.flatnonzero(e >= min_energy):
            col = proj.parent_column(smat, p_eidx)
            nuc_fac = nuc_flux[p_eidx] / nuc_flux[min_idx : p_eidx + 1]
            zfac[p_eidx] = np.sum(col[min_idx : p_eidx + 1] * nuc_fac)
        return zfac

    # Thunman et al. 1996: a row of the table against the projectile flux
    weight = nuc_flux * proj_cs if use_cs_scaling else nuc_flux.copy()
    source = smat @ weight
    ok = (e >= min_energy) & (weight > 0)
    zfac[ok] = source[ok] / weight[ok]
    return zfac


def decay_z_factor(proj, sec, e_grid, pmodel, gamma=None):
    r"""Energy-dependent decay Z-factor (Lipari 1993) at the daughter energy.

    The matrix row :math:`Z_i = \sum_j D_{ij} S_j / S_i` against a parent
    spectrum :math:`S(E) \propto E^{\gamma}`; ``gamma`` defaults to the
    nucleon index of the primary model at the daughter energy. Under scaling
    it is :math:`\int_0^1 x^{-\gamma-1} dN/dx\,dx`.

    Args:
      proj, sec: parent and daughter :class:`~MCEq.particlemanager.MCEqParticle`.
      e_grid: kinetic-energy grid.
      pmodel: primary model (``nucleon_gamma``), used when ``gamma`` is None.

    Returns:
      (numpy.ndarray): Z on ``e_grid``
    """
    if proj.is_stable:
        raise Exception(f"{proj.name} does not decay.")
    info(10, f"Computing e-dependent decay Zfactor for {proj.name} -> {sec.name}")
    if sec not in proj.decay_dists:
        raise Exception(f"{sec.name} is not a a child particle of {proj.name}.")

    e = np.asarray(e_grid, dtype=np.float64)
    if gamma is None:
        gamma = pmodel.nucleon_gamma(e)
    gamma = np.broadcast_to(np.asarray(gamma, dtype=np.float64), e.shape)
    dmat = np.asarray(proj.decay_dists[sec], dtype=np.float64)
    zfac = np.zeros_like(e)
    for i in range(len(e)):
        zfac[i] = dmat[i] @ ((e / e[i]) ** gamma[i])
    return zfac
