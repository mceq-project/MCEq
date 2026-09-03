"""The atmospheric column: slant-depth integration and the :math:`\\rho(X)` splines.

Every ``calculate_density_spline`` implementation ends in the same tail -- integrate
the sampled density along the slant path, then fit the three splines the solver
reads. What differs between them is only how the density and the heights are
obtained. That tail lives here so the contract is stated once.

Columns belong here; steps belong to :mod:`MCEq.solvers.path`, which owns the
step-size policy and the depth grid.
"""

from __future__ import annotations

import numpy as np


def fit_column_splines(atmosphere, rho_vec, dl_vec, h_intp, *, max_den):
    """Integrate the column and fit the three splines every atmosphere shares.

    Args:
      atmosphere: the :class:`~MCEq.geometry.density_profiles.EarthsAtmosphere`
        to store the result on. ``_max_X``, ``_max_den`` and the three splines
        are all set here.
      rho_vec: density at each of the ``n_steps`` samples of *dl_vec*.
      dl_vec: slant path from the top of the atmosphere to the observation
        level, ``linspace(0, path_len, n_steps)``.
      h_intp: height at ``dl_vec[2:]``, in reverse path order. A list or an
        array; the two are bitwise equivalent here. It is an argument rather
        than something computed from *dl_vec* because the scalar-``geom.h`` and
        array-``geom.h`` spellings differ by one ULP of ``r_E`` at 258 of 901
        zeniths, and both are in use -- ``tests/geometry/test_environment_pins.py``
        pins that difference and bounds it at two ULP.
      max_den: the caller's atmosphere-top density. Keyword-only and required
        on purpose: it is the one stored quantity the four callers derive
        differently, and ``EarthsAtmosphere.__init__`` pre-sets ``_max_den``
        from ``config.environment.max_density``, so a caller that simply forgot
        it would inherit a plausible 0.001225 with no exception raised.

    Returns:
      ``X_int = cumulative_trapezoid(rho_vec, dl_vec)``, one element shorter
      than *dl_vec*. The node contract is ``X_int[i]`` = depth accumulated up
      to ``dl_vec[i+1]``, so ``X_int[0]`` is the first non-zero depth and not
      the depth at ``dl_vec[0]``. Pairing it with ``rho_vec[1:]`` below is
      therefore aligned, not off by one.

    The three fits, all quadratic without smoothing:

    * ``_s_h2X`` -- height to ``log X``, over ``X_int[1:]`` in the same reverse
      path order as *h_intp*.
    * ``_s_X2rho`` -- ``X`` to density, over all of ``X_int``.
    * ``_s_lX2h`` -- ``log X`` to height, the inverse of the first.

    **Reverse path order is not the same as increasing height, and FITPACK
    requires the latter.** It coincides for any downgoing column and for the
    ``*LocationCentered`` classes at every zenith, since those hand the
    geometry an effective *local* zenith of at most 90 degrees. It does not
    coincide for a genuinely upgoing column above a raised observation level:
    with ``h_obs > 0`` and theta > 90 the height is V-shaped along the path, and
    ``UnivariateSpline`` then rejects *h_intp* outright with "x must be strictly
    increasing if s = 0" -- 229 of 1998 samples are non-increasing at
    theta = 91.4 deg, ``h_obs`` = 2 km. ``set_h_obs`` admits those angles
    (``max_theta = geom.theta_max_deg`` = 91.435 deg there), so the class accepts
    zeniths this fit cannot serve. Nothing in the tree reaches it; it is a real
    latent defect rather than a handled case, and reversing the arrays is no
    workaround for it.

    ``_s_X2rho`` is fitted on ``X_int``, so its domain starts at
    ``X_int[0] > 0`` and every solve extrapolates it down to
    ``config.X_start == 0``. That extrapolation is load-bearing, not a hazard:
    it is the top-of-atmosphere saturation of ``1/rho`` that
    ``etd2_nonuniform_path`` sizes its first steps against, and
    ``_s_X2rho.get_knots()[0]`` recovers the fitted lower bound exactly for
    anyone who needs it.
    """
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import UnivariateSpline

    h_intp = np.asarray(h_intp)
    X_int = cumulative_trapezoid(rho_vec, dl_vec)

    # Assigned before anything that can raise, which is the order the four
    # callers had. It means a rejected ``h_intp`` leaves ``_max_X`` describing
    # the new zenith while the splines still describe the old one -- see B28 --
    # but changing that is a behaviour change, not a relocation.
    atmosphere._max_X = X_int[-1]
    atmosphere._max_den = max_den

    # ``np.log`` of the CONTIGUOUS slice, reversed after, rather than of the
    # reversed view. Elementwise, so the two are equal on this host; they are
    # not equal by construction, because numpy dispatches the float64 ``log``
    # SVML loop only on AVX512-SKX and conditions that dispatch on contiguity.
    # The base tail fed ``np.log`` a Python list, hence contiguous, and the
    # MSIS21 tails fed it a negative-stride view -- so the two families already
    # disagreed in this respect and a shared helper has to pick one. It picks
    # the contiguous form, which is what the majority of atmospheres had.
    log_X_intp = np.log(X_int[1:])[::-1]
    atmosphere._s_h2X = UnivariateSpline(h_intp, log_X_intp, k=2, s=0.0)
    atmosphere._s_X2rho = UnivariateSpline(X_int, rho_vec[1:], k=2, s=0.0)
    atmosphere._s_lX2h = UnivariateSpline(log_X_intp[::-1], h_intp[::-1], k=2, s=0.0)

    return X_int
