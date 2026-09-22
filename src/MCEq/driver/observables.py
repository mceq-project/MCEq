"""Particle-count, longitudinal-profile and Z-factor observables of a solved cascade.

Observable functions bound as :class:`MCEqRun` methods
(``n_particles = observables.n_particles`` etc.); their first argument is
the run.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix

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


def _source_operator(run, mat):
    """The production part of a cascade operator: every block that feeds one
    species from a *different* one.

    The blocks on the species diagonal hold the loss terms (interaction and
    decay lengths, continuous energy loss, muon scattering) and the same-
    species regeneration, none of which produce a new particle; they are
    dropped. For a 2-D database only the ``k = 0`` mode block is kept.
    """
    dim_states = run.dim_states
    if run.matrix_builder.is_2d:
        mat = mat[:dim_states, :dim_states]
    coo = mat.tocoo()
    keep = (coo.row // run.dim) != (coo.col // run.dim)
    return csr_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])), shape=(dim_states, dim_states)
    )


def _production_terms(run, X, states):
    """Source term of the cascade equation on every stored state: the
    production of each species by decays and interactions of *other*
    species, shape ``(len(X), dim_states)`` in 1/(g/cm²)."""
    rho_inv = np.asarray(run.density_model.r_X2rho(X), dtype=np.float64)
    src_int = _source_operator(run, run.int_m)
    src_dec = _source_operator(run, run.dec_m)
    return (src_int @ states.T).T + rho_inv[:, None] * (src_dec @ states.T).T


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


def gaisser_hillas(X, n_max, x_max, x0, lam):
    r"""The Gaisser-Hillas profile

    .. math::

        N(X) = N_{max}\left(\frac{X - X_0}{X_{max} - X_0}\right)^{(X_{max} - X_0)/\lambda}
        \exp\!\left(\frac{X_{max} - X}{\lambda}\right),

    zero for :math:`X \le X_0`. All depths in g/cm².
    """
    X = np.asarray(X, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = (X - x0) / (x_max - x0)
        out = n_max * z ** ((x_max - x0) / lam) * np.exp((x_max - X) / lam)
    return np.where(X > x0, out, 0.0)


def fit_gaisser_hillas(X, profile, x0=-45.0):
    """Least-squares fit of :func:`gaisser_hillas` to a longitudinal profile.

    ``x0`` is held fixed by default at -45 g/cm², the value the Pierre Auger
    Observatory uses for its muon production depth fits (the average of the
    preferred proton and iron values, Aab et al., PRD 90 (2014) 012012);
    pass ``x0=None`` to fit it as well.

    Args:
      X (array): depths in g/cm².
      profile (array): the profile on ``X``, e.g. from :func:`depth_profile`.
      x0 (float | None): the depth of first interaction parameter, fixed
        when a number is given.

    Returns:
      tuple: ``(n_max, x_max, x0, lam)`` in the units of ``profile`` and
      g/cm², ready to be passed to :func:`gaisser_hillas`.
    """
    X = np.asarray(X, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)
    i = int(np.argmax(profile))
    n_guess, x_guess, lam_guess = profile[i], max(X[i], 1.0), 70.0
    if x0 is None:
        popt, _ = curve_fit(
            gaisser_hillas,
            X,
            profile,
            p0=[n_guess, x_guess, min(-1.0, X[0] - 1.0), lam_guess],
            bounds=([0.0, X[0], -np.inf, 1.0], [np.inf, np.inf, X[0], 1e4]),
            maxfev=20000,
        )
        return tuple(float(p) for p in popt)
    popt, _ = curve_fit(
        lambda x, n, xm, lam: gaisser_hillas(x, n, xm, x0, lam),
        X,
        profile,
        p0=[n_guess, max(x_guess, x0 + 1.0), lam_guess],
        bounds=([0.0, x0 + 1.0, 1.0], [np.inf, np.inf, 1e4]),
        maxfev=20000,
    )
    return float(popt[0]), float(popt[1]), float(x0), float(popt[2])


def depth_profile(
    run,
    particle_name,
    definition="production",
    min_energy_cutoff=1e-1,
    max_energy_cutoff=None,
    per_energy=False,
):
    r"""Longitudinal profile of a particle species along the depth grid of
    the last :meth:`MCEqRun.solve` call.

    Two profiles are available:

    - ``definition="production"`` (default): the production rate
      :math:`dN/dX`, the source term of the cascade equation evaluated on
      the stored states,

      .. math::

          \frac{dN}{dX}(X) = \sum_{E}\, \Big[\big(\boldsymbol{C}\,\Lambda_{int}
          + \frac{1}{\rho(X)}\boldsymbol{D}\,\Lambda_{dec}\big)\,\Phi(X)\Big]_{E},

      where only the blocks that feed the species from *other* species
      enter: losses (interaction, decay, continuous energy loss) and
      same-species regeneration are not production. For muons this is the
      muon production depth distribution of air-shower studies;
    - ``definition="number"``: the number of particles :math:`N(X)` in the
      energy range, i.e. :func:`n_particles` at every grid depth.

    The run must have been solved with an ``int_grid``, which sets the
    resolution of the profile.

    Args:
      particle_name (str | sequence of str): a name accepted by
        :meth:`MCEqRun.get_solution`, prefixes included (``total_mu+``,
        ``conv_mu-``, ``pi_numu`` ...). A sequence of names is summed.
      definition (str): ``"number"`` or ``"production"``.
      min_energy_cutoff (float): lower kinetic-energy cut in GeV; bins whose
        lower edge is below it are not counted.
      max_energy_cutoff (float | None): upper kinetic-energy cut in GeV; bins
        whose upper edge is above it are not counted. ``None`` for no cut.
      per_energy (bool): return the spectrum on :attr:`MCEqRun.e_grid`
        at every depth, differential in kinetic energy, instead of the
        energy-integrated profile. The energy cuts are not applied.

    Returns:
      (X, profile): the depth grid in g/cm² and the profile, shape
      ``(len(X),)``: a particle number for ``"number"``, a rate in 1/(g/cm²)
      for ``"production"``; with ``per_energy=True`` shape ``(len(X), dim)``
      and an additional 1/GeV. The normalisation of the flux (per cm², s, sr
      for a primary flux model, per shower for a single primary) carries
      through.
    """
    X, states = _grid_states(run)
    if definition == "number":
        vectors = states
    elif definition == "production":
        vectors = _production_terms(run, X, states)
    else:
        raise ValueError(
            f"Unknown profile definition {definition!r}; use 'number' or 'production'."
        )
    if isinstance(particle_name, str):
        particle_name = [particle_name]

    spectra = np.zeros((len(X), run.dim))
    for i, vec in enumerate(vectors):
        for name in particle_name:
            spectra[i] += run._get_solution_from_state(
                vec,
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
    definition="production",
    min_energy_cutoff=1e-1,
    max_energy_cutoff=None,
    method=None,
    x0=-45.0,
):
    r"""Depth of the maximum of a species' longitudinal profile, in g/cm².

    The maximum of :func:`depth_profile` on the depth grid of the last
    :meth:`MCEqRun.solve` call: the depth of maximum particle number
    (``definition="number"``, :math:`X_{max}` in the shower-maximum sense)
    or of maximum production (``definition="production"``, the
    :math:`X^\mu_{max}` of muon production depth measurements).

    Three ways of locating the maximum:

    - ``"gaisser-hillas"``: the profile is fitted with
      :func:`fit_gaisser_hillas` and the fitted :math:`X_{max}` is
      returned, the procedure of the Pierre Auger Observatory's muon
      production depth analysis. The default for ``"production"``, whose
      profile has that shape; a fit that does not converge falls back to
      the grid maximum with a message at debug level 1.
    - ``"parabola"``: a parabola through the grid maximum and its two
      neighbours. The default for ``"number"``, whose profile carries the
      accumulated particles in a long tail and is not Gaisser-Hillas shaped.
    - ``"grid"``: the grid point of the maximum.

    A maximum on the first or last grid point is reported at debug level 1:
    either the grid does not bracket the peak, or the profile is monotonic
    along the path; ``"parabola"`` then returns the edge point.

    Args:
      particle_name (str | sequence of str): see :func:`depth_profile`.
      definition (str): ``"number"`` or ``"production"``.
      min_energy_cutoff (float): lower kinetic-energy cut in GeV.
      max_energy_cutoff (float | None): upper kinetic-energy cut in GeV.
      method (str | None): ``"gaisser-hillas"``, ``"parabola"`` or
        ``"grid"``; ``None`` picks the default of the definition.
      x0 (float | None): the :math:`X_0` of the Gaisser-Hillas fit, fixed
        when a number is given (default -45 g/cm²), fitted when ``None``.

    Returns:
      float: depth of the maximum in g/cm²; ``nan`` for an all-zero profile.
    """
    X, profile = depth_profile(
        run,
        particle_name,
        definition=definition,
        min_energy_cutoff=min_energy_cutoff,
        max_energy_cutoff=max_energy_cutoff,
    )
    if not np.any(profile > 0.0):
        info(1, f"The {definition} profile of {particle_name} is zero on the grid.")
        return np.nan
    imax = int(np.argmax(profile))
    if imax in (0, len(X) - 1):
        info(
            1,
            f"Maximum of the {definition} profile of {particle_name} on the edge "
            f"of the depth grid (X = {X[imax]:.4g} g/cm^2): the grid does not "
            "bracket a peak.",
        )
    if method is None:
        method = "gaisser-hillas" if definition == "production" else "parabola"
    if method == "grid":
        return float(X[imax])
    if method == "parabola":
        if imax in (0, len(X) - 1):
            return float(X[imax])
        sl = slice(imax - 1, imax + 2)
        a, b, _ = np.polyfit(X[sl] - X[imax], profile[sl], 2)
        if a >= 0.0:
            return float(X[imax])
        return float(np.clip(X[imax] - b / (2.0 * a), X[imax - 1], X[imax + 1]))
    if method != "gaisser-hillas":
        raise ValueError(
            f"Unknown method {method!r}; use 'gaisser-hillas', 'parabola' or 'grid'."
        )
    try:
        _, x_max, _, _ = fit_gaisser_hillas(X, profile, x0=x0)
    except (RuntimeError, ValueError, TypeError) as exc:
        info(1, f"Gaisser-Hillas fit failed ({exc}); returning the grid maximum.")
        return float(X[imax])
    return x_max


def muon_depth_profile(
    run,
    definition="production",
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
      (X, profile): the depth grid in g/cm² and the muon number, or with
      ``definition="production"`` the muon production rate dN/dX in
      1/(g/cm²).
    """
    return depth_profile(
        run,
        [prefix + "mu+", prefix + "mu-"],
        definition=definition,
        min_energy_cutoff=min_energy_cutoff,
        max_energy_cutoff=max_energy_cutoff,
        per_energy=per_energy,
    )


def muon_xmax(
    run,
    definition="production",
    min_energy_cutoff=1e-1,
    max_energy_cutoff=None,
    prefix="total_",
    method=None,
    x0=-45.0,
):
    r"""Muon :math:`X^\mu_{max}`: the slant depth in g/cm² at which the
    production rate of muons (positive plus negative) in the chosen energy
    range peaks, located by a Gaisser-Hillas fit as in the muon production
    depth analyses of air-shower arrays; or, with ``definition="number"``,
    the depth at which the number of muons is largest.

    The production profile of a single primary
    (:meth:`MCEqRun.set_single_primary_particle`) peaks in the middle of
    the atmosphere and is the observable those analyses measure. For an
    inclusive flux it peaks within the first few g/cm² at every energy,
    because the steep primary spectrum weights the first interactions, and
    the fit carries no meaning there; the muon *number* of an inclusive flux
    rises, peaks in the upper troposphere at GeV energies and moves toward
    the ground with rising energy. The profiles are
    :func:`muon_depth_profile`. Requires a preceding
    ``solve(int_grid=...)``::

        X_grid = np.linspace(0, mceq.density_model.max_X, 300)
        mceq.set_single_primary_particle(1e10, corsika_id=14)
        mceq.solve(int_grid=X_grid)
        xmax = mceq.muon_xmax(min_energy_cutoff=1.0)

    Args:
      definition (str): ``"number"`` or ``"production"``.
      min_energy_cutoff (float): lower kinetic-energy cut in GeV.
      max_energy_cutoff (float | None): upper kinetic-energy cut in GeV.
      prefix (str): parent category, see :func:`muon_depth_profile`.
      method (str | None): ``"gaisser-hillas"``, ``"parabola"`` or
        ``"grid"``, see :func:`xmax`; ``None`` picks the definition's default.
      x0 (float | None): the :math:`X_0` of the Gaisser-Hillas fit.

    Returns:
      float: muon :math:`X_{max}` in g/cm²
    """
    return xmax(
        run,
        [prefix + "mu+", prefix + "mu-"],
        definition=definition,
        min_energy_cutoff=min_energy_cutoff,
        max_energy_cutoff=max_energy_cutoff,
        method=method,
        x0=x0,
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
