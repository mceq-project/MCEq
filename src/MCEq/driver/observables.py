"""Particle-count, longitudinal-profile and Z-factor observables of a solved cascade.

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


def _grid_states(run):
    """Depth grid and stored intermediate states of the last solve.

    Returns ``(X, states)`` with ``X`` in g/cm² and ``states`` of shape
    ``(len(X), dim_states)``; for a 2-D database the ``k = 0``
    (angle-integrated) mode block, which is the block
    :func:`MCEqRun.get_solution` reads.
    """
    grid_sol = getattr(run, "grid_sol", None)
    if grid_sol is None or len(grid_sol) == 0 or run.int_grid is None:
        raise RuntimeError(
            "Depth profiles need intermediate solutions: call "
            "solve(int_grid=<depths in g/cm^2>) first."
        )
    X = np.asarray(run.int_grid, dtype=np.float64)
    return X, np.asarray(grid_sol)[:, : run.dim_states]


def _energy_bin_mask(run, min_energy_cutoff, max_energy_cutoff):
    """Boolean mask of the kinetic-energy bins a profile counts: lower edge
    at or above ``min_energy_cutoff`` and upper edge at or below
    ``max_energy_cutoff`` (``None`` for no upper cut)."""
    mask = run.e_bins[:-1] >= min_energy_cutoff
    if max_energy_cutoff is not None:
        mask &= run.e_bins[1:] <= max_energy_cutoff
    if not np.any(mask):
        raise ValueError(
            "No energy bin lies between the cutoffs "
            f"{min_energy_cutoff} and {max_energy_cutoff} GeV; the grid spans "
            f"{run.e_bins[0]:.3g} to {run.e_bins[-1]:.3g} GeV."
        )
    return mask


def depth_profile(
    run,
    particle_name,
    min_energy_cutoff=1e-1,
    max_energy_cutoff=None,
    per_energy=False,
):
    """Longitudinal profile of a particle species: its number along the
    depth grid of the last :meth:`MCEqRun.solve` call.

    The profile is :func:`n_particles` at every grid depth, with an optional
    upper energy cut. The run must have been solved with an ``int_grid``,
    which sets the resolution of the profile.

    Args:
      particle_name (str | sequence of str): a name accepted by
        :meth:`MCEqRun.get_solution`, prefixes included (``total_mu+``,
        ``conv_mu-``, ``pi_numu`` ...). A sequence of names is summed.
      min_energy_cutoff (float): lower kinetic-energy cut in GeV; bins whose
        lower edge is below it are not counted.
      max_energy_cutoff (float | None): upper kinetic-energy cut in GeV; bins
        whose upper edge is above it are not counted. ``None`` for no cut.
      per_energy (bool): return the spectrum on :attr:`MCEqRun.e_grid`
        at every depth, differential in kinetic energy, instead of the
        energy-integrated number. The energy cuts are not applied.

    Returns:
      (X, profile): the depth grid in g/cm² and the particle number, shape
      ``(len(X),)``; with ``per_energy=True`` the spectra in 1/GeV, shape
      ``(len(X), dim)``. The normalisation of the flux (per cm², s, sr for a
      primary flux model) carries through.
    """
    X, states = _grid_states(run)
    if isinstance(particle_name, str):
        particle_name = [particle_name]

    spectra = np.zeros((len(X), run.dim))
    for i, state in enumerate(states):
        for name in particle_name:
            spectra[i] += run._get_solution_from_state(
                state,
                name,
                mag=0,
                integrate=not per_energy,
                return_as="kinetic energy",
            )
    if per_energy:
        return X, spectra
    mask = _energy_bin_mask(run, min_energy_cutoff, max_energy_cutoff)
    return X, spectra[:, mask].sum(axis=1)


def xmax(
    run,
    particle_name,
    min_energy_cutoff=1e-1,
    max_energy_cutoff=None,
    refine=True,
):
    r"""Depth of maximum of a species' longitudinal profile, in g/cm².

    The maximum of :func:`depth_profile` on the depth grid of the last
    :meth:`MCEqRun.solve` call, :math:`X_{max}` in the shower-maximum
    sense. With ``refine=True`` a parabola through the grid maximum and
    its two neighbours locates the peak between grid points. A maximum on
    the first or last grid point is returned as is, with a message at debug
    level 1: either the grid does not bracket the peak, or the profile is
    monotonic along the path.

    Args:
      particle_name (str | sequence of str): see :func:`depth_profile`.
      min_energy_cutoff (float): lower kinetic-energy cut in GeV.
      max_energy_cutoff (float | None): upper kinetic-energy cut in GeV.
      refine (bool): interpolate the peak between grid points.

    Returns:
      float: depth of the maximum in g/cm²; ``nan`` for an all-zero profile.
    """
    X, profile = depth_profile(
        run,
        particle_name,
        min_energy_cutoff=min_energy_cutoff,
        max_energy_cutoff=max_energy_cutoff,
    )
    if not np.any(profile > 0.0):
        info(1, f"The profile of {particle_name} is zero on the depth grid.")
        return np.nan
    imax = int(np.argmax(profile))
    if imax in (0, len(X) - 1):
        info(
            1,
            f"Maximum of the {particle_name} profile on the edge of the depth "
            f"grid (X = {X[imax]:.4g} g/cm^2): the grid does not bracket a peak.",
        )
        return float(X[imax])
    if not refine:
        return float(X[imax])
    sl = slice(imax - 1, imax + 2)
    a, b, _ = np.polyfit(X[sl] - X[imax], profile[sl], 2)
    if a >= 0.0:
        return float(X[imax])
    return float(np.clip(X[imax] - b / (2.0 * a), X[imax - 1], X[imax + 1]))


def muon_depth_profile(
    run,
    min_energy_cutoff=1e-1,
    max_energy_cutoff=None,
    prefix="total_",
    per_energy=False,
):
    """Longitudinal profile of positive plus negative muons.

    :func:`depth_profile` for ``<prefix>mu+`` and ``<prefix>mu-``;
    ``prefix`` selects the parent category (``total_``, ``conv_``, ``pr_``,
    ``pi_``, ``k_``).

    Returns:
      (X, n_mu): the depth grid in g/cm² and the number of muons at each
      depth.
    """
    return depth_profile(
        run,
        [prefix + "mu+", prefix + "mu-"],
        min_energy_cutoff=min_energy_cutoff,
        max_energy_cutoff=max_energy_cutoff,
        per_energy=per_energy,
    )


def muon_xmax(
    run,
    min_energy_cutoff=1e-1,
    max_energy_cutoff=None,
    prefix="total_",
    refine=True,
):
    r"""Muon :math:`X_{max}`: the slant depth in g/cm² at which the number
    of muons (positive plus negative) in the chosen energy range is largest.

    For an inclusive flux the muon number at GeV energies rises, peaks in
    the upper troposphere and decays away toward the ground; with rising
    energy the muons survive longer and the peak moves toward the ground.
    The peak is located with :func:`xmax`; the profile is
    :func:`muon_depth_profile`. Requires a preceding
    ``solve(int_grid=...)``::

        X_grid = np.linspace(0, mceq.density_model.max_X, 300)
        mceq.solve(int_grid=X_grid)
        xmax = mceq.muon_xmax(min_energy_cutoff=1.0)

    Args:
      min_energy_cutoff (float): lower kinetic-energy cut in GeV.
      max_energy_cutoff (float | None): upper kinetic-energy cut in GeV.
      prefix (str): parent category, see :func:`muon_depth_profile`.
      refine (bool): interpolate the peak between grid points.

    Returns:
      float: muon :math:`X_{max}` in g/cm²
    """
    return xmax(
        run,
        [prefix + "mu+", prefix + "mu-"],
        min_energy_cutoff=min_energy_cutoff,
        max_energy_cutoff=max_energy_cutoff,
        refine=refine,
    )


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
