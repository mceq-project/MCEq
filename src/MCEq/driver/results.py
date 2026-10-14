"""Spectrum extraction and the batch-solve result container.

In 2D runs the state vector stacks the Legendre modes
``a_l = <P_l(cos theta)>``; the ``l = 0`` mode is the angle-integrated flux
returned by :func:`get_solution`. Angular densities come from
:func:`convert_to_theta_space`.
"""

import numpy as np

from MCEq import config
from MCEq.misc import info


class MCEqBatchResult:
    """Result of a batched (multi-RHS) solve.

    Returned by :meth:`MCEqRun.solve_batch` and
    :meth:`MCEqRun.solve_fullsky`. Holds the final states, shape
    ``(n_k * dim_states, K)`` with ``n_k = 1`` in 1D runs, and the batch
    metadata; :meth:`get_solution` extracts named spectra per column as
    :meth:`MCEqRun.get_solution` does.

    Attributes:
      sol (np.ndarray[n_k * dim_states, K]): final state, one column per
        batch member.
      grid_sol (np.ndarray[len(int_grid), n_k * dim_states, K] | None):
        stacked depth snapshots when ``int_grid`` was requested
        (shared-path batches only), else ``None``.
      nsteps_per_col (np.ndarray[K] | None): integration steps per column.
      shared_path (bool): True when every column was solved along one
        path (the multi-RHS route). Only such results can have ``grid_sol``.
      conditions (list | None): the per-column conditions passed to
        :meth:`MCEqRun.solve_batch`, if any.
      pixel_index (np.ndarray[K, 2] | None): ``(i_zen, i_az)`` per column
        for :meth:`MCEqRun.solve_fullsky` results.
      zenith_grid, azimuth_grid (np.ndarray | None): the sky grid for
        :meth:`MCEqRun.solve_fullsky` results (azimuth is the inner axis
        of the column order).

    Note:
      The result holds a reference to the producing :class:`MCEqRun`
      instance for particle-name lookups. Changing that instance's
      interaction model or particle list after the solve invalidates
      :meth:`get_solution` on this result.
    """

    def __init__(
        self,
        mceq,
        sol,
        grid_sol=None,
        int_grid=None,
        nsteps_per_col=None,
        conditions=None,
        pixel_index=None,
        zenith_grid=None,
        azimuth_grid=None,
        shared_path=None,
    ):
        self._mceq = mceq
        self.sol = sol
        self.grid_sol = grid_sol
        self.int_grid = int_grid
        self.nsteps_per_col = nsteps_per_col
        self.conditions = conditions
        self.pixel_index = pixel_index
        self.zenith_grid = zenith_grid
        self.azimuth_grid = azimuth_grid
        self.shared_path = None if shared_path is None else bool(shared_path)

    @property
    def K(self):
        """Number of batch members (columns)."""
        return self.sol.shape[1]

    @property
    def n_azimuth(self):
        """Length of the azimuth (inner) axis of the column order."""
        return self.azimuth_grid.size if self.azimuth_grid is not None else 1

    def column_index(self, k=None, pixel=None, zenith=None, azimuth=None):
        """Resolve a batch member to its column index.

        Exactly one selector should be provided: ``k`` (direct column
        index), ``pixel`` (``(i_zen, i_az)`` grid coordinates, sky-grid
        results only) or ``zenith`` (+ optional ``azimuth``) in degrees,
        matched against the sky grid with ``np.isclose``. With ``K == 1``
        all selectors may be omitted.
        """
        n_selected = sum(x is not None for x in (k, pixel, zenith))
        if n_selected > 1:
            raise ValueError("column_index: provide only one of k, pixel, or zenith")
        if k is not None:
            k = int(k)
            if not -self.K <= k < self.K:
                raise IndexError(f"column_index: k={k} out of range for K={self.K}")
            return k % self.K

        if pixel is not None or zenith is not None:
            if self.zenith_grid is None:
                raise ValueError(
                    "column_index: pixel/zenith selection requires a "
                    "solve_fullsky result (no sky grid attached)"
                )
        if pixel is not None:
            i_zen, i_az = pixel
            n_zen = self.zenith_grid.size
            if not (0 <= i_zen < n_zen and 0 <= i_az < self.n_azimuth):
                raise IndexError(
                    f"column_index: pixel {pixel} outside grid "
                    f"({n_zen} x {self.n_azimuth})"
                )
            return int(i_zen) * self.n_azimuth + int(i_az)
        if zenith is not None:
            match = np.flatnonzero(np.isclose(self.zenith_grid, zenith))
            if match.size == 0:
                raise ValueError(
                    f"column_index: zenith {zenith} not in grid {self.zenith_grid}"
                )
            i_zen = int(match[0])
            if azimuth is None:
                if self.n_azimuth > 1:
                    raise ValueError(
                        "column_index: azimuth required (grid has "
                        f"{self.n_azimuth} azimuth pixels)"
                    )
                i_az = 0
            else:
                match_az = np.flatnonzero(np.isclose(self.azimuth_grid, azimuth))
                if match_az.size == 0:
                    raise ValueError(
                        f"column_index: azimuth {azimuth} not in grid "
                        f"{self.azimuth_grid}"
                    )
                i_az = int(match_az[0])
            return i_zen * self.n_azimuth + i_az

        if self.K == 1:
            return 0
        raise ValueError(
            f"column_index: batch has K={self.K} members - select one "
            "with k=, pixel=, or zenith="
        )

    def get_solution(
        self,
        particle_name,
        k=None,
        *,
        pixel=None,
        zenith=None,
        azimuth=None,
        mag=0.0,
        grid_idx=None,
        integrate=False,
        return_as=None,
        dont_sum_helicities=False,
    ):
        """Retrieve a named spectrum for one batch member.

        Same ``particle_name`` semantics, helicity summation, ``mag``,
        ``integrate`` and ``return_as`` behaviour as
        :meth:`MCEqRun.get_solution`; the column is selected with ``k``,
        ``pixel`` or ``zenith``/``azimuth`` (see :meth:`column_index`).

        Args:
          particle_name (str): e.g. ``conv_numu``, ``total_mu+``.
          k (int, optional): column index.
          pixel (tuple, optional): ``(i_zen, i_az)`` sky-grid coordinates.
          zenith, azimuth (float, optional): angles in degrees, matched
            against the sky grid.
          grid_idx (int, optional): snapshot index when the batch was
            solved with ``int_grid`` (shared-path batches only).
          mag, integrate, return_as, dont_sum_helicities: see
            :meth:`MCEqRun.get_solution`.

        Returns:
          (np.ndarray): flux on the energy grid.
        """
        col = self.column_index(k=k, pixel=pixel, zenith=zenith, azimuth=azimuth)
        if grid_idx is None:
            state = self.sol[:, col]
        else:
            if self.grid_sol is None or len(self.grid_sol) == 0:
                raise Exception(
                    "Solution has not been computed on a grid. Re-run with int_grid."
                )
            if grid_idx >= len(self.grid_sol):
                state = self.grid_sol[-1][:, col]
            else:
                state = self.grid_sol[grid_idx][:, col]
        return self._mceq._get_solution_from_state(
            state,
            particle_name,
            mag=mag,
            integrate=integrate,
            return_as=return_as,
            dont_sum_helicities=dont_sum_helicities,
        )

    def skymap(self, particle_name, kin_energy, mag=0.0):
        """Return the ``(n_zen, n_az)`` flux map at one kinetic energy.

        Extracts ``particle_name`` for every pixel and interpolates
        linearly in log(E_kin) between the two bracketing grid points.
        Only available on :meth:`MCEqRun.solve_fullsky` results.

        Args:
          particle_name (str): see :meth:`get_solution`.
          kin_energy (float): kinetic energy in GeV; must lie within the
            energy grid.
          mag (float): energy magnification exponent, as in
            :meth:`get_solution`.

        Returns:
          (np.ndarray[n_zen, n_az]): flux map, kinetic-energy units.
        """
        if self.zenith_grid is None:
            raise ValueError("skymap() is only available on solve_fullsky results")
        e_grid = self._mceq.e_grid
        if not e_grid[0] <= kin_energy <= e_grid[-1]:
            raise ValueError(
                f"skymap: kin_energy {kin_energy} outside energy grid "
                f"[{e_grid[0]:.3g}, {e_grid[-1]:.3g}] GeV"
            )
        i_hi = int(np.clip(np.searchsorted(e_grid, kin_energy), 1, e_grid.size - 1))
        i_lo = i_hi - 1
        w = (np.log(kin_energy) - np.log(e_grid[i_lo])) / (
            np.log(e_grid[i_hi]) - np.log(e_grid[i_lo])
        )
        n_zen = self.zenith_grid.size
        flux = np.empty((n_zen, self.n_azimuth))
        for k in range(self.K):
            f = self.get_solution(
                particle_name, k=k, mag=mag, return_as="kinetic energy"
            )
            i_zen, i_az = divmod(k, self.n_azimuth)
            flux[i_zen, i_az] = (1.0 - w) * f[i_lo] + w * f[i_hi]
        return flux

    def __repr__(self):
        parts = [f"K={self.sol.shape[1]}"]
        if self.zenith_grid is not None:
            parts.append(f"sky_grid={self.zenith_grid.size}x{self.n_azimuth}")
        if self.grid_sol is not None and len(self.grid_sol):
            parts.append(f"n_snapshots={len(self.grid_sol)}")
        return f"MCEqBatchResult({', '.join(parts)})"


def get_solution(
    run,
    particle_name,
    mag=0.0,
    grid_idx=None,
    integrate=False,
    return_as=config.return_as,
    dont_sum_helicities=False,
):
    """Retrieves solution of the calculation on the energy grid.

    Some special prefixes are accepted for lepton names:

    - the total flux of muons, muon neutrinos etc. from all sources/mothers
      can be retrieved without a prefix ``mu+`` or with the prefix ``total_mu+``,
      ``total_numu``
    - the conventional flux of muons, muon neutrinos etc. from all sources
      can be retrieved by the prefix ``conv_``, i.e. ``conv_numu``
    - the prompt flux of muons, muon neutrinos etc. from all sources
      can be retrieved by the prefix ``pr_``, i.e. ``pr_numu``
    - correspondigly, the flux of leptons which originated from the decay
      of a charged pion carries the prefix ``pi_`` and from a kaon ``k_``

    Args:
      particle_name (str): The name of the particle such, e.g.
        ``total_mu+`` for the total flux spectrum of positive muons or
        ``pr_antinumu`` for the flux spectrum of prompt anti muon neutrinos
      mag (float, optional): 'magnification factor': the solution is
        multiplied by ``sol`` :math:`= \\Phi \\cdot E^{mag}`
      grid_idx (int, optional): if the integrator has been configured to save
        intermediate solutions on a depth grid, then ``grid_idx`` specifies
        the index of the depth grid for which the solution is retrieved. If
        not specified the flux at the surface is returned
      integrate (bool, optional): return averge particle number instead of
      flux (multiply by bin width)
      return_as (str, optional): the flux can be returned as ``total energy``, ``kinetic energy``,
        or ``total momentum`` flux. The default is ``config.return_as`` at import time;
        ``None`` uses the run's output setting
      dont_sum_helicities (bool, optional): Per default the lepton flux is summed over the available helicities,
        e.g. ``total_mu+`` is the muon flux from (-1, 0, +1) helicity for mu+.

    Returns:
      (:func: numpy.array): flux of particles on energy grid :attr:`e_grid`
    """

    if grid_idx is not None and len(run.grid_sol) == 0:
        raise Exception("Solution not has not been computed on grid. Check input.")
    if grid_idx is None:
        sol = np.copy(run._solution)
    elif grid_idx >= len(run.grid_sol):
        sol = run.grid_sol[-1, :]
    else:
        sol = run.grid_sol[grid_idx, :]

    return run._get_solution_from_state(
        sol,
        particle_name,
        mag=mag,
        integrate=integrate,
        return_as=return_as,
        dont_sum_helicities=dont_sum_helicities,
    )


def _get_solution_from_state(
    run,
    sol,
    particle_name,
    mag=0.0,
    integrate=False,
    return_as=None,
    dont_sum_helicities=False,
):
    """Extract a named spectrum from an explicit state vector.

    Same particle names, helicity summation and ``return_as`` conversions
    as :meth:`get_solution`, applied to ``sol`` instead of ``run._solution``.

    Args:
      sol (np.ndarray[dim_states]): state vector to extract from.
      particle_name (str): see :meth:`get_solution`.
      mag, integrate, return_as, dont_sum_helicities: see
        :meth:`get_solution`. ``return_as=None`` resolves to
        ``run._cfg.output.return_as``.

    Returns:
      (np.ndarray): flux of particles on energy grid :attr:`e_grid`
    """
    if return_as is None:
        return_as = run._cfg.output.return_as

    res = np.zeros(run._energy_grid.d)
    ref = run.pman.pname2pref

    def sum_lr(lep_str, prefix):
        result = np.zeros(run.dim)
        nsuccess = 0

        if dont_sum_helicities:
            sum_over = [lep_str]
        else:
            sum_over = [lep_str, lep_str + "_l", lep_str + "_r"]

        for ls in sum_over:
            if prefix + ls not in ref:
                info(
                    15,
                    "No separate left and right handed particles,",
                    f"or, unavailable particle prefix {prefix + ls}.",
                )
                continue
            result += sol[ref[prefix + ls].lidx : ref[prefix + ls].uidx]
            nsuccess += 1
        if nsuccess == 0 and run._cfg.output.excpt_on_missing_particle:
            raise Exception(f"Requested particle {particle_name} not found.")
        return result

    lep_str = particle_name.split("_")[1] if "_" in particle_name else particle_name

    default_tracking_prefixes = [
        "conv_",
        "pr_",
        "pi_",
        "k_",
        "K0_",
        "mulr_",
        "mu_h0_",
        "prcas_",
        "prres_",
    ]
    if not run._cfg.physics.enable_default_tracking:
        for track_pref in default_tracking_prefixes:
            if particle_name.startswith(track_pref):
                raise Exception(
                    "Tracking category requested but "
                    + "enable_default_tracking is off in config."
                )

    if particle_name.startswith("total_"):
        # pi_ and k_ prefixes are tracking counters, not species
        res = sum_lr(lep_str, prefix="")

    elif particle_name.startswith("conv_"):
        # conventional = total - prompt
        res = run._get_solution_from_state(
            sol,
            "total_" + lep_str,
            mag=0,
            integrate=False,
            return_as="kinetic energy",
        ) - run._get_solution_from_state(
            sol,
            "pr_" + lep_str,
            mag=0,
            integrate=False,
            return_as="kinetic energy",
        )

    elif particle_name.startswith("pr_"):
        if "prcas_" + lep_str in ref:
            res += sum_lr(lep_str, prefix="prcas_")
        if "prres_" + lep_str in ref:
            res += sum_lr(lep_str, prefix="prres_")
        if "em_" + lep_str in ref:
            res += sum_lr(lep_str, prefix="em_")
    else:
        try:
            res = sum_lr(particle_name, prefix="")
        except KeyError:
            if run._cfg.output.excpt_on_missing_particle:
                raise Exception(f"Requested particle {particle_name} not found.")
            else:
                info(1, f"Requested particle {particle_name} not found.")

    # When returning in Etot, interpolate on different grid
    if return_as == "total energy":
        etot_grid = run.etot_grid(lep_str)
        if not integrate:
            return res * etot_grid**mag
        return res * etot_grid**mag * run.e_widths

    if return_as == "kinetic energy":
        if not integrate:
            return res * run._energy_grid.c**mag
        return res * run._energy_grid.c**mag * run.e_widths

    if return_as == "total momentum":
        ptot_bins, ptot_grid = run.ptot_grid(lep_str, return_bins=True)
        dEkindp = np.diff(ptot_bins) / run.e_widths
        if not integrate:
            return dEkindp * res * ptot_grid**mag
        return dEkindp * res * ptot_grid**mag * np.diff(ptot_bins)

    raise Exception(
        "Unknown 'return_as' variable choice.",
        'the options are "kinetic energy", "total energy", "total momentum"',
    )


def convert_to_theta_space(run, modes, pdg_id, hel, theta_res=1200, log_theta=False):
    """Angular densities of one species from the 2D solver's mode amplitudes.

    The moments are interpolated in ``l`` between the sparse modes and summed
    as a Legendre series with Jackson damping
    (:func:`MCEq.operators.legendre.theta_density`).

    Args:
        modes (list of np.arrays): mode amplitudes at the requested slant
            depths, each reshaped to ``(n_k, dim_states)``
        pdg_id (int): PDG ID of the particle whose angular density is being
            requested (e.g. 14 for NuMu)
        hel (int): helicity of the particle whose angular density is being
            requested (e.g. 0 or ±1 for polarized muons)
        theta_res (int): number of points of the angular (theta) grid
        log_theta (bool): whether to return logarithmic angular grid
            (True=logarithmic, False=linear)

    Returns:
        tuple: ``(l_grid, amps, theta_range, f_theta)``. ``amps[depth][eidx]``
        are the moments of the species on ``l_grid``; ``f_theta[depth][eidx]``
        is the density in the planar measure ``theta dtheta``, normalised as
        ``int_0^pi f theta dtheta = a_0``.
    """
    from MCEq.operators.legendre import theta_density

    if log_theta:
        theta_range = np.logspace(-5, np.log10(np.pi / 2), theta_res)
    else:
        theta_range = np.linspace(0, np.pi / 2, theta_res)
    l_grid = np.asarray(run._mceq_db.k_grid)
    n_e = len(run.e_grid)
    ref = run.pman.pdg2mceqidx[(pdg_id, hel)] * n_e
    amps = [list(s[:, ref : ref + n_e].T) for s in modes]
    f = [list(theta_density(np.asarray(a), l_grid, theta_range)) for a in amps]
    return l_grid, amps, theta_range, f
