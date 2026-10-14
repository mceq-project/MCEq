"""Particle numbers and Z-factors from a solved cascade."""

import numpy as np

from MCEq.misc import info


def n_particles(run, label, grid_idx=None, min_energy_cutoff=1e-1):
    """Returns number of particles of type `label` at a grid step above
    an energy threshold for counting.

    Args:
        label (str): Particle name
        grid_idx (int): Depth grid index (for profiles)
        min_energy_cutoff (float): Energy threshold > mceq_config.e_min
    """
    ie_min = np.argmin(
        np.abs(run.e_bins - run.e_bins[run.e_bins >= min_energy_cutoff][0])
    )
    _e = run.e_bins[ie_min]
    _e_n = run.e_bins[ie_min + 1]
    _e_m = run.e_grid[ie_min]
    info(
        10,
        f"Energy cutoff for particle number calculation {_e:4.3e} GeV",
    )
    info(
        15,
        f"First bin is between {_e:3.2e} and {_e_n:3.2e} with midpoint {_e_m:3.2e}",
    )
    return np.sum(
        run.get_solution(label, mag=0, integrate=True, grid_idx=grid_idx)[ie_min:]
    )


def n_mu(run, grid_idx=None, min_energy_cutoff=1e-1):
    """Returns the number of positive and negative muons at a grid step above
    `min_energy_cutoff`.

    Args:
        grid_idx (int): Depth grid index (for profiles)
        min_energy_cutoff (float): Energy threshold > mceq_config.e_min

    """
    return run.n_particles(
        "total_mu+", grid_idx=grid_idx, min_energy_cutoff=min_energy_cutoff
    ) + run.n_particles(
        "total_mu-", grid_idx=grid_idx, min_energy_cutoff=min_energy_cutoff
    )


def n_e(run, grid_idx=None, min_energy_cutoff=1e-1):
    """Returns the number of electrons plus positrons at a grid step above
    `min_energy_cutoff`.

    Args:
        grid_idx (int): Depth grid index (for profiles)
        min_energy_cutoff (float): Energy threshold > mceq_config.e_min
    """
    return run.n_particles(
        "e+", grid_idx=grid_idx, min_energy_cutoff=min_energy_cutoff
    ) + run.n_particles("e-", grid_idx=grid_idx, min_energy_cutoff=min_energy_cutoff)


def z_factor(
    run,
    projectile_pdg,
    secondary_pdg,
    definition="primary_e",
    min_energy=0.3,
    use_cs_scaling=True,
):
    r"""Energy-dependent Z-factor of ``projectile -> secondary`` interactions.

    ``definition="thunman"`` (any value other than ``"primary_e"``) is the
    spectrum-weighted moment of Thunman, Ingelman and Gondolo (1996) at the
    secondary energy :math:`E`,

    .. math::

        Z(E) = \int_E^\infty dE'\,\frac{\phi_N(E')}{\phi_N(E)}
            \frac{\lambda_N(E)}{\lambda_N(E')}\frac{dn(E', E)}{dE},

    evaluated as a matrix row, :math:`Z_i = \sum_j C_{ij}\,\sigma_j\phi_j /
    (\sigma_i\phi_i)` with :math:`\phi` the proton (neutron for a neutron
    projectile) flux of the primary model at the bin centres and
    :math:`\sigma` the production cross section (``use_cs_scaling=False``
    drops the ratio). Under Feynman scaling and a power law
    :math:`E^{-(\gamma+1)}` this is :math:`\int_0^1 x^{\gamma-1}F(x)dx`.
    The row form is the table acting on a smooth flux, which is what
    stencil-averaged tables are built for. Values below ``min_energy`` are 0.

    ``definition="primary_e"`` is MCEq's own column definition with respect
    to the cosmic-ray energy (x-axis = projectile energy): the yield column of
    a projectile at :math:`E_j`, weighted by
    :math:`\phi_N(E_j)/\phi_N(E_i)` over secondaries above ``min_energy``.
    For stencil-averaged tables the column is that of a projectile at the bin
    centre (:meth:`MCEqParticle.parent_column`).

    Returns:
      (numpy.ndarray): Z on ``run.e_grid``
    """

    proj = run.pman[projectile_pdg]
    sec = run.pman[secondary_pdg]

    if not proj.is_projectile:
        raise Exception(f"{proj.name} is not a projectile particle.")
    info(10, f"Computing e-dependent Zfactor for {proj.name} -> {sec.name}")
    if not proj.is_secondary(sec):
        raise Exception(f"{sec.name} is not a secondary particle of {proj.name}.")

    p_flux, n_flux = run.pmodel.p_and_n_flux(run.e_grid)[1:]
    nuc_flux = np.asarray(
        n_flux if proj.pdg_id[0] == 2112 else p_flux, dtype=np.float64
    )

    smat = proj.hadr_yields[sec]
    proj_cs = proj.prod_cross_section()
    smat = np.asarray(smat, dtype=np.float64)
    proj_cs = np.asarray(proj_cs, dtype=np.float64)
    e = np.asarray(run.e_grid, dtype=np.float64)
    zfac = np.zeros_like(e)

    if definition == "primary_e":
        assert use_cs_scaling is False, f"cs_scaling has when definition = {definition}"
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


def decay_z_factor(run, parent_pdg, child_pdg, gamma=None):
    r"""Energy-dependent decay Z-factor (Lipari 1993) at the daughter energy.

    A matrix row against a parent decay spectrum :math:`S(E) \propto
    E^{\gamma}` (``gamma`` the differential index; default: the nucleon index
    of the primary model at the daughter energy, as the parent spectrum of
    mesons that decay rather than interact):
    :math:`Z_i = \sum_j D_{ij} S_j / S_i`. Under scaling this is Lipari's
    moment :math:`\int_0^1 x^{-\gamma-1} dN/dx\,dx`. As a row it is valid
    for stencil-averaged decay tables, which a single parent column is not.

    Returns:
      (numpy.ndarray): Z on ``run.e_grid``
    """

    proj = run.pman[parent_pdg]
    sec = run.pman[child_pdg]

    if proj.is_stable:
        raise Exception(f"{proj.name} does not decay.")
    info(
        10,
        f"Computing e-dependent decay Zfactor for {proj.name} -> {sec.name}",
    )
    if not proj.is_child(sec):
        raise Exception(f"{sec.name} is not a a child particle of {proj.name}.")

    e = np.asarray(run.e_grid, dtype=np.float64)
    if gamma is None:
        gamma = run.pmodel.nucleon_gamma(e)
    gamma = np.broadcast_to(np.asarray(gamma, dtype=np.float64), e.shape)
    dmat = proj.decay_dists[sec]
    dmat = np.asarray(dmat, dtype=np.float64)
    zfac = np.zeros_like(e)
    for i in range(len(e)):
        spec = (e / e[i]) ** gamma[i]
        zfac[i] = dmat[i] @ spec
    return zfac
