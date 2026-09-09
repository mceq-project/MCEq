"""Particle-count and Z-factor observables read from a solved cascade.

Observable functions bound as :class:`MCEqRun` methods
(``n_particles = observables.n_particles`` etc.); their first argument is
the run.
"""

import numpy as np

# trapz was finally removed with numpy 2.4
if hasattr(np, "trapezoid"):
    trapz = np.trapezoid
else:
    trapz = np.trapz

from MCEq.misc import info  # noqa: E402


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
    """Energy dependent Z-factor according to Thunman et al. (1996)"""

    proj = run.pman[projectile_pdg]
    sec = run.pman[secondary_pdg]

    if not proj.is_projectile:
        raise Exception(f"{proj.name} is not a projectile particle.")
    info(10, f"Computing e-dependent Zfactor for {proj.name} -> {sec.name}")
    if not proj.is_secondary(sec):
        raise Exception(f"{sec.name} is not a secondary particle of {proj.name}.")

    if proj == 2112:
        nuc_flux = run.pmodel.p_and_n_flux(run.e_grid)[2]
    else:
        nuc_flux = run.pmodel.p_and_n_flux(run.e_grid)[1]
    zfac = np.zeros(run.dim)

    smat = proj.hadr_yields[sec]
    proj_cs = proj.prod_cross_section()
    zfac = np.zeros_like(run.e_grid)

    if run._cfg.has_cuda:
        import cupy

        smat = cupy.asnumpy(smat)
        proj_cs = cupy.asnumpy(proj_cs)
    # Definition wrt CR energy (different from Thunman) on x-axis
    min_idx = 0
    if definition == "primary_e":
        for p_eidx, e in enumerate(run.e_grid):
            if e < min_energy:
                min_idx = p_eidx
                continue
            nuc_fac = nuc_flux[p_eidx] / nuc_flux[min_idx : p_eidx + 1]
            assert use_cs_scaling is False, (
                f"cs_scaling has when definition = {definition}"
            )
            cs_fac = 1.0
            zfac[p_eidx] = np.sum(smat[min_idx : p_eidx + 1, p_eidx] * nuc_fac * cs_fac)
        return zfac
    else:
        # Like in Thunman et al. 1996
        for p_eidx, e in enumerate(run.e_grid):
            if e < min_energy:
                continue
            min_idx = p_eidx
            nuc_fac = nuc_flux[p_eidx] / nuc_flux[min_idx : p_eidx + 1]
            if use_cs_scaling:
                cs_fac = np.zeros(p_eidx - min_idx + 1)
                old_settings = np.seterr(all="ignore")
                res = proj_cs[p_eidx] / proj_cs[min_idx : p_eidx + 1]
                np.seterr(**old_settings)
                cs_fac[(res > 0) & np.isfinite(res)] = res[(res > 0) & np.isfinite(res)]
            else:
                cs_fac = 1.0
            zfac[p_eidx] = np.sum(smat[p_eidx, p_eidx:] * nuc_fac * cs_fac)
        return zfac


def decay_z_factor(run, parent_pdg, child_pdg):
    """Energy dependent Z-factor according to Lipari (1993)."""

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

    cr_gamma = run.pmodel.nucleon_gamma(run.e_grid)
    zfac = np.zeros(run.dim)

    zfac = np.zeros_like(run.e_grid)
    for p_eidx, e in enumerate(run.e_grid):
        # if e < min_energy:
        #     min_idx = p_eidx + 1
        #     continue
        xlab, xdist = proj.dNdec_dxlab(e, sec)
        zfac[p_eidx] = trapz(xlab ** (-cr_gamma[p_eidx] - 2.0) * xdist, x=xlab)
    return zfac
