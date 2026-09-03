"""The atmospheric column: slant-depth integration and the :math:`\\rho(X)` splines.

Every ``calculate_density_spline`` implementation ends in the same tail -- integrate
the sampled density along the slant path, then fit the three splines the solver
reads. What differs between them is only how the density and the heights are
obtained. That tail lives here so the contract is stated once.

Destined for ``MCEq.environment`` (plan §4), which owns columns while
``solvers/path.py`` owns steps.
"""

from __future__ import annotations

import numpy as np


def fit_column_splines(atmosphere, rho_vec, dl_vec, h_intp):
    """Integrate the column and fit the three splines every atmosphere shares.

    Args:
      atmosphere: the :class:`~MCEq.geometry.density_profiles.EarthsAtmosphere`
        to store the result on. ``_max_X``, ``_s_h2X``, ``_s_X2rho`` and
        ``_s_lX2h`` are set; ``_max_den`` is **not**, because callers derive it
        differently.
      rho_vec: density at each of the ``n_steps`` samples of *dl_vec*.
      dl_vec: slant path from the top of the atmosphere to the observation
        level, ``linspace(0, path_len, n_steps)``.
      h_intp: height at ``dl_vec[2:]``, reversed -- so, increasing height. A
        list or an array; the two are bitwise equivalent here. It is an
        argument rather than something computed from *dl_vec* because the
        scalar-``geom.h`` and array-``geom.h`` spellings differ by one ULP of
        ``r_E`` at a fifth of the zenith range, and both are in use --
        ``tests/geometry/test_environment_pins.py`` pins that difference.

    Returns:
      ``X_int = cumulative_trapezoid(rho_vec, dl_vec)``, one element shorter
      than *dl_vec* and starting at the first non-zero depth.

    The three fits, all quadratic without smoothing:

    * ``_s_h2X`` -- height to ``log X``, over ``X_int[1:]`` reversed so it runs
      with *h_intp*. Reversing rather than sorting is an incomplete workaround
      for the non-monotonic elevations of upgoing trajectories.
    * ``_s_X2rho`` -- ``X`` to density, over all of ``X_int``.
    * ``_s_lX2h`` -- ``log X`` to height, the inverse of the first.

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
    log_X_intp = np.log(X_int[1:][::-1])

    atmosphere._max_X = X_int[-1]
    atmosphere._s_h2X = UnivariateSpline(h_intp, log_X_intp, k=2, s=0.0)
    atmosphere._s_X2rho = UnivariateSpline(X_int, rho_vec[1:], k=2, s=0.0)
    atmosphere._s_lX2h = UnivariateSpline(log_X_intp[::-1], h_intp[::-1], k=2, s=0.0)

    return X_int
